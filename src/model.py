import numpy as np
from numpy import linalg as la
from scipy.linalg import expm

class Nonneg_dyn_dag_learning():
    def __init__(self, primal_opt='pgd', acyclicity='logdet', restart=True):
        self.acyc_const = acyclicity
        if acyclicity == 'logdet':
            self.dagness = self.logdet_acyc_
            self.gradient_acyclic = self.logdet_acyclic_grad_
        elif acyclicity == 'matexp':
            self.dagness = self.matexp_acyc_
            self.gradient_acyclic = self.matexp_acyclic_grad_
        else:
            raise ValueError('Unknown acyclicity constraint')

        self.restart = restart
        self.opt_type = primal_opt
        if primal_opt in ['pgd', 'adam']:
            self.minimize_primal = self.proj_grad_desc_
        elif primal_opt == 'fista':
            self.minimize_primal = self.acc_proj_grad_desc_
        elif 'sca' in primal_opt:
            self.sca_adam = True if primal_opt == 'sca-adam' else False
            self.minimize_primal = self.succ_conv_approx_
        else:
            raise ValueError('Unknown solver type for primal problem')   

    def init_variables_(self, X, Y, rho_init, alpha_init, track_seq, s, beta1, beta2, delta, verb,
                        ridge_cyy=1e-6, use_pinv=False):
        """
        Initialize internal variables and precompute covariance-like matrices.

        Args:
            X: (M, N) design matrix.
            Y: (M, N*P) target matrix with P lags concatenated along columns.
            rho_init: initial rho for method of multipliers.
            alpha_init: initial alpha for method of multipliers.
            track_seq: whether to store parameter trajectories.
            s, delta: hyperparameters for the log-det acyclicity term.
            beta1, beta2: Adam hyperparameters.
            verb: verbosity flag.
            ridge_cyy: ridge regularization added to Cyy for numerical stability.
            use_pinv: use pseudo-inverse instead of solve/inverse for Cyy.
        """

        # --- Reset objective handles (useful when restarting) ---
        self.Gw_obj_func = None
        self.Ga_obj_func = None

        # --- Ensure consistent floating dtype ---
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # --- Dimensions and basic validation ---
        self.M, self.N = X.shape
        if Y.shape[0] != self.M:
            raise ValueError(f"Y must have M={self.M} rows; got {Y.shape[0]}")
        if Y.shape[1] % self.N != 0:
            raise ValueError(f"Y.shape[1]={Y.shape[1]} is not a multiple of N={self.N}")
        self.P = Y.shape[1] // self.N  # number of lags

        # --- Precompute second-order statistics (normalized by M) ---
        # Shapes: Cxx -> (N, N), Cxy -> (N, N*P), Cyy -> (N*P, N*P)
        self.Cxx = (X.T @ X) / self.M
        self.Cxy = (X.T @ Y) / self.M
        self.Cyy = (Y.T @ Y) / self.M
        self.Cyy_inv = np.linalg.inv(self.Cyy)

        # # --- Improve conditioning of Cyy (ridge) before inversion/solve ---
        # if ridge_cyy and ridge_cyy > 0.0:
        #     self.Cyy = self.Cyy + ridge_cyy * np.eye(self.Cyy.shape[0])

        # # --- Compute Cyy^{-1} in a numerically stable way ---
        # if use_pinv:
        #     # Pseudo-inverse is safer for rank-deficient matrices
        #     self.Cyy_inv = np.linalg.pinv(self.Cyy)
        # else:
        #     # Solve Cyy * X = I is typically more stable than np.linalg.inv(Cyy)
        #     self.Cyy_inv = np.linalg.solve(self.Cyy, np.eye(self.Cyy.shape[0]))

        # --- Initialize parameters to estimate ---
        self.W_est = np.zeros_like(self.Cxx)                 # (N, N)
        self.A_est = np.zeros((self.N * self.P, self.N))     # (N*P, N)
        self.verb = bool(verb)

        # --- Method of multipliers parameters ---
        self.rho = float(rho_init)
        self.alpha = float(alpha_init)

        # --- Acyclicity (log-det) components ---
        self.Id = np.eye(self.N)
        self.s = float(s)
        self.delta = float(delta)

        # --- Adam optimizer states (match shapes to parameters to avoid broadcasting bugs) ---
        # For W
        self.opt_m = np.zeros_like(self.W_est)
        self.opt_v = np.zeros_like(self.W_est)
        # For A
        self.opt_m_a = np.zeros_like(self.A_est)
        self.opt_v_a = np.zeros_like(self.A_est)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        # Optional step counters for bias correction
        self.adam_t_w = 0
        self.adam_t_a = 0

        # --- Tracking sequences for diagnostics/plots ---
        self.acyclicity = []  # e.g., store h(W) values over iterations
        self.diff_W = []      # e.g., ||W^{t} - W^{t-1}||
        self.diff_A = []      # e.g., ||A^{t} - A^{t-1}||
        self.seq_W = [] if track_seq else None
        self.seq_A = [] if track_seq else None

    def track_variables_(self, W, W_prev, A, A_prev, track_seq):
        norm_W_prev = la.norm(W_prev)
        norm_W_prev = norm_W_prev if norm_W_prev != 0 else 1
        self.diff_W.append(la.norm(W - W_prev) / norm_W_prev)
        if track_seq:
            self.seq_W.append(W)
            self.acyclicity.append(self.dagness(W))

        norm_A_prev = la.norm(A_prev)
        norm_A_prev = norm_A_prev if norm_A_prev != 0 else 1
        self.diff_A.append(la.norm(A - A_prev) / norm_A_prev)
        if track_seq:
            self.seq_A.append(A)

    def logdet_acyc_(self, W):
        """
        Compute the log-determinant based acyclicity constraint.

        This constraint equals: N * log(s) - logdet(sI - W),
        where W is the weighted adjacency matrix, s > 0 a scaling 
        factor, and I the identity matrix. 
        It is equal to zero if and only if W corresponds to a DAG.
        """
        return self.N * np.log(self.s) - la.slogdet(self.s*self.Id - W)[1]
    
    def logdet_acyclic_grad_(self, W):
        """
        Gradient of the log-determinant acyclicity constraint 
        with respect to W.
        """
        return la.inv(self.s*self.Id - W).T
    
    def matexp_acyc_(self, W):
        """
        Compute the matrix-exponential based acyclicity constraint.

        This constraint equals: tr(exp(W)) - N,
        where W is the weighted adjacency matrix.
        It is zero if and only if W corresponds to a DAG.

        W is clipped to prevent numerical overflow.
        """
        entry_limit = np.maximum(10, 5e2/W.shape[0])
        W = np.clip(W, -entry_limit, entry_limit)
        return np.trace(expm(W)) - self.N

    def matexp_acyclic_grad_(self, W):
        """
        Gradient of the matrix-exponential acyclicity constraint 
        with respect to W.

        The gradient is clipped elementwise to avoid overflow.
        """
        return np.clip(expm(W).T, -1e6, 1e6)
    
    def compute_gradient_W_(self, W, A, lamb_W, alpha):
        G_loss = self.Cxx @(W - self.Id) + self.Cxy @ A + lamb_W
        acyc_val = self.dagness(W)
        G_acyc = self.gradient_acyclic(W)
        return G_loss + (alpha + self.rho*acyc_val)*G_acyc
    
    def compute_adam_grad_(self, grad, opt_m, opt_v, iter):
        opt_m = opt_m * self.beta1 + (1 - self.beta1) * grad
        opt_v = opt_v * self.beta2 + (1 - self.beta2) * (grad ** 2)
        m_hat = opt_m / (1 - self.beta1 ** iter)
        v_hat = opt_v / (1 - self.beta2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad, opt_m, opt_v
    
    def proj_grad_step_W_(self, W, A, alpha, lamb_W, stepsize, iter):
        self.Gw_obj_func = self.compute_gradient_W_(W, A, lamb_W, alpha)
        if self.opt_type == 'adam':
            self.Gw_obj_func, self.opt_m, self.opt_v = self.compute_adam_grad_(self.Gw_obj_func, self.opt_m, 
                                                                              self.opt_v, iter+1)
        W_est = np.maximum(W - stepsize*self.Gw_obj_func, 0)

        # Ensure non-negative acyclicity
        if self.acyc_const == 'logdet':
            acyc = self.dagness(W_est)        
            if acyc < -1e-12:
                eigenvalues, _ = np.linalg.eig(W_est)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                W_est = (self.s - self.delta) * W_est / max_eigenvalue
                acyc = self.dagness(W_est)

                stepsize /= 2
                if self.verb:
                    print('Negative acyclicity. Projecting and reducing stepsize to: ', stepsize)

                assert acyc > -1e-12, f'Acyclicity is negative: {acyc}'
        
        return W_est, stepsize
    
    def proj_grad_step_A_(self, A, W, lamb_A, stepsize, iter):
        # Compute the gradient
        self.Ga_obj_func = self.Cxy.T @ (W - self.Id) + self.Cyy @ A + lamb_A
        if self.opt_type == 'adam':
            self.Ga_obj_func, self.opt_m_a, self.opt_v_a \
                = self.compute_adam_grad_(self.Ga_obj_func, self.opt_m_a, self.opt_v_a, iter+1)

        A_est = np.maximum(A - stepsize*self.Ga_obj_func, 0)        
        return A_est

    def proj_grad_desc_(self, W, A, lamb_W, lamb_A, alpha, stepsize, max_iters, checkpoint, tol,
                        track_seq):
        W_prev = W.copy()
        A_prev = A.copy()
        for i in range(max_iters):
            # Gradient step for W
            W, stepsize = self.proj_grad_step_W_(W_prev, A_prev, alpha, lamb_W, stepsize, i)

            # Gradient step for A
            A = self.proj_grad_step_A_(A_prev, W_prev, lamb_A, stepsize, i)

            # Update tracking variables
            self.track_variables_(W, W_prev, A, A_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and (self.diff_W[-1] + self.diff_A[-1]) / 2 <= tol:
                if self.verb:
                    print("Convergence achieved at iter", i+1)
                break
    
            W_prev = W.copy()
            A_prev = A.copy()

        return W, A, stepsize

    def acc_proj_grad_desc_(self, W, A, lamb_W, lamb_A, alpha, stepsize, max_iters, checkpoint, tol,
                            track_seq):
        W_prev, W_fista = W.copy(), W.copy()
        A_prev, A_fista = A.copy(), A.copy()
        t_k = 1
        for i in range(max_iters):
            W, stepsize = self.proj_grad_step_W_(W_fista, A_fista, alpha, lamb_W, stepsize, i)
            diff_W = W - W_prev

            A = self.proj_grad_step_A_(A_fista, W_fista, lamb_A, stepsize, i)
            diff_A = A - A_prev
    
            # Update tracking variables
            self.track_variables_(W, W_prev, A, A_prev, track_seq)

            # Check if restarting condition is met
            inner_prod_grad_W = self.Gw_obj_func.flatten().T @ diff_W.flatten()
            inner_prod_grad_A = self.Ga_obj_func.flatten().T @ diff_A.flatten()

            if self.restart and (inner_prod_grad_W + inner_prod_grad_A) > 1e-6:
                W = W_prev.copy()      
                W_fista = W.copy()
                A = A_prev.copy()
                A_fista = A.copy()
                t_k = 1
                continue

            t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
            W_fista = W + (t_k - 1)/t_next*(diff_W)
            A_fista = A + (t_k - 1)/t_next*(diff_A)

            
            # Check convergence
            if i % checkpoint == 0 and (self.diff_W[-1] + self.diff_A[-1]) / 2 <= tol:
                if self.verb:
                    print("Convergence achieved at iter", i+1)
                break

            W_prev = W
            A_prev = A
            t_k = t_next

        return W, A, stepsize

    def succ_conv_approx_(self, W, A, lamb_W, lamb_A, alpha, stepsize, max_iters, checkpoint, tol,
                          track_seq):
        """
        Succesive Convex Approximation algorithm that estiamtes W and A via an alternating
        minimization scheme where at each iteration minimizes an upper bound with closed form solution
        """
        W_prev = W.copy()
        A_prev = A.copy()

        self.opt_type = "adam" if self.sca_adam else self.opt_type
        for i in range(max_iters):
            # Closed form solution of upperbound of W (coincides with a single gradient step) 
            W, stepsize = self.proj_grad_step_W_(W_prev, A_prev, alpha, lamb_W, stepsize, 0)
            # W, stepsize = self.proj_grad_step_W_(W_prev, A_prev, alpha, lamb_W, stepsize, i)

            # self.Ga_obj_func = self.Cxy.T @ (W - self.Id) + self.Cyy @ A + lamb_A
            
            # Closed form solucion of A
            A_aux = self.Cyy_inv @ ( self.Cxy.T @ (self.Id - W) - lamb_A )
            A = np.maximum( A_aux, 0 )

            # Update tracking variables
            self.track_variables_(W, W_prev, A, A_prev, track_seq)

            # Check convergence
            if (checkpoint and i % checkpoint == 0) and (self.diff_W[-1] + self.diff_A[-1]) / 2 <= tol:
                break
    
            W_prev = W.copy()
            A_prev = A.copy()

        return W, A, stepsize
    
    def fit(self, X, Y, lamb_W, lamb_A, stepsize, s=1, iters_in=1000, iters_out=10, checkpoint=250, tol=1e-6,
            beta=5, gamma=.25, rho_0=1, alpha_0=.1, track_seq=False, dec_step=None,
            beta1=.99, beta2=.999, delta=.01, verb=False):
        
        self.init_variables_(X, Y, rho_0, alpha_0, track_seq, s, beta1, beta2, delta, verb)        
        
        dagness_prev = self.dagness(self.W_est)
        for i in range(iters_out):
            # Minimize augmented Lagrangian to estimate W
            self.W_est, self.A_est, stepsize = self.minimize_primal(self.W_est, self.A_est, lamb_W, lamb_A, self.alpha,
                                                                    stepsize, iters_in, checkpoint, tol, track_seq)

            # Update Lagrange multiplier
            dagness = self.dagness(self.W_est)
            self.alpha += self.rho*dagness
            
            # Update augmented Lagrangian parameters
            self.rho = beta*self.rho if dagness > gamma*dagness_prev else self.rho

            dagness_prev = dagness

            if dec_step:
                stepsize *= dec_step
            
            if verb:
                print(f'- {i+1}/{iters_out}. Diff W: {self.diff_W[-1]:.6f} | Diff A: {self.diff_A[-1]:.6f}' +
                      f' | Acycl: {dagness:.6f} |Rho: {self.rho:.3f} - Alpha: {self.alpha:.3f} - Step: {stepsize:.4f}')
        
        return self.W_est, self.A_est
    
