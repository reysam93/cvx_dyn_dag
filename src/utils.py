import numpy as np
from numpy import linalg as la
import networkx as nx
import time
from pandas import DataFrame
from IPython.display import display

import matplotlib.pyplot as plt

def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times 

def is_dag(W):
    """Return True iff the adjacency matrix W defines a DAG (nonzero = edge)."""
    return nx.is_directed_acyclic_graph(nx.DiGraph(W))

def _sample_weights(W, edge_type, w_range):
    """
    Assign weights on top of a 0/1 structure matrix W.
    """
    if edge_type == 'binary':
        return (W > 0).astype(float)

    elif edge_type == 'positive':
        low, high = w_range
        if low < 0 or high <= 0:
            raise ValueError("For 'positive', w_range must be nonnegative.")
        weights =  np.random.uniform(low, high, size=W.shape)
        return weights * W

    elif edge_type == 'weighted':
        # Default range: w_range=((-2.0, -0.5), (0.5, 2.0))
        W_weighted = np.zeros(W.shape)
        S =  np.random.randint(len(w_range), size=W.shape)
        for i, (low, high) in enumerate(w_range):
            weights =  np.random.uniform(low=low, high=high, size=W.shape)
            W_weighted += W * (S == i) * weights

        return W_weighted
        
    else:
        raise ValueError("Unknown edge_type. Use 'binary', 'positive', or 'weighted'")
    

def create_dag(n_nodes, graph_type, edges, permute=True, edge_type='positive', w_range=(.5, 1.5),
               rew_prob=.1):
    """
    Create a DAG adjacency matrix W (d x d) and its DiGraph.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    graph_type : str
        Graph generator type: 'er', 'sf'/'sf_t', 'sw'/'sw_t'.
        Suffix '_t' => use strictly lower triangular (reverse topological order).
    edges : int
        Approximate number of expected edges.
    permute : bool
        If True, apply a random node permutation to the DAG.
    edge_type : str
        One of {'binary', 'positive', 'weighted'}.
        - 'binary': edges are 0/1.
        - 'positive': edge weights are uniform in [low, high].
        - 'weighted': edge weights sampled from multiple intervals.
    w_range : tuple or list of tuples
        - For 'positive': (low, high), must be >= 0.
        - For 'weighted': list of (low, high) intervals.
    rew_prob : float
        Rewiring probability for small-world graphs.

    Returns
    -------
    W_weighted : np.ndarray
        Weighted adjacency matrix (n_nodes x n_nodes).
    dag : networkx.DiGraph
        Directed acyclic graph with edge weights.
    """

    # Create the graph 
    if graph_type == 'er':
        prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        G = nx.erdos_renyi_graph(n_nodes, prob)
        W = np.triu(nx.to_numpy_array(G), k=1)

    elif graph_type in ('sf', 'sf_t'):
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m)
        adj = nx.to_numpy_array(G)
        W = np.triu(adj, k=1) if graph_type == 'sf' else np.tril(adj, k=-1)

    elif graph_type == 'sw' or graph_type == 'sw_t':
        G = nx.watts_strogatz_graph(n_nodes, int(2*round(edges/n_nodes)), rew_prob)
        adj = nx.to_numpy_array(G)
        W = np.triu(adj, k=1) if graph_type == 'sw' else np.tril(adj, k=-1)

    else:
        raise ValueError('Unknown graph type')

    assert nx.is_weighted(G) == False
    assert nx.is_empty(G) == False
    
    # Optional node permutation (preserves acyclicity)
    if permute:
        P = np.eye(n_nodes)
        P = P[:, np.random.permutation(n_nodes)]
        W = P @ W @ P.T

    # --- assign weights ---
    W_weighted = _sample_weights(W, edge_type, w_range)

    # Build DAG and check acyclicity
    dag = nx.DiGraph(W_weighted)
    assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"

    return W_weighted, nx.DiGraph(dag)


def create_interslice_graph(n_nodes, n_lags, graph_type, *,
                      # ER controls
                      er_edges=None,
                      # SBM controls
                      sbm_block_sizes=None, sbm_block_p=None,
                      # weights
                      edge_type='positive', w_range=(0.5, .9), exp_decay=1):
    """
    Create temporal dependency graphs A^{(k)} (not necessarily DAGs), k = 1..n_lags.
    Each A^{(k)} is generated using either an Erdős–Rényi (ER) or stochastic block model (SBM).
    Returns the horizontal concatenation A = [A^(1) A^(2) ... A^(n_lags)] and the list of graphs.

    Parameters
    ----------
    n_nodes : int
        Number of variables (nodes), d.
    n_lags : int
        Number of lags, p.
    graph_type : {'er', 'sbm'}
        Graph generator type.
    er_edges : int, optional
        Target expected number of directed er_edges for ER.
    sbm_block_sizes : list[int], optional
        Community sizes for SBM (must sum to n_nodes).
    sbm_block_p : float | (float,float) | array-like, optional
        Block probability specification for SBM:
          - scalar q : uniform edge probability across all block pairs.
          - (p_intra, p_inter) : same intra-/inter-block probabilities.
          - (K x K) matrix : full block-to-block probability matrix.
    edge_type : {'binary','positive','weighted'}, default='positive'
        Weighting scheme:
          - 'binary': er_edges are {0,1}.
          - 'positive': edge weights sampled uniformly in [low, high].
          - 'weighted': edge weights sampled from multiple intervals.
    w_range : tuple or list of tuples, default=(0.5, 1.5)
        Range(s) of weights depending on edge_type.

    Returns
    -------
    A : np.ndarray, shape (n_nodes, n_lags * n_nodes)
        Vertical concatenation [A^(1) ... A^(p)].
    graphs : list of networkx.DiGraph
        One directed graph per lag, with edge weights.
    """

    graphs = []
    A_list = []

    if n_lags < 1:
        return np.array(A_list), graphs

    for k in range(n_lags):
        # Structure
        if graph_type == 'er':
            if er_edges is None:
                er_edges = n_nodes
            prob = float(er_edges)/float(n_nodes**2 - n_nodes)/2
            G = nx.erdos_renyi_graph(n_nodes, prob, directed=True)
            W = nx.to_numpy_array(G)

        elif graph_type == 'sbm':
            assert n_nodes == None or sum(sbm_block_sizes) == n_nodes, \
                "Sum of sbm_block_sizes must equal n_nodes"
            
            G = nx.stochastic_block_model(sbm_block_sizes, sbm_block_p, directed=True)
            W = nx.to_numpy_array(G)
            
        else:
            raise ValueError("graph_type must be 'er' or 'sbm'.")

        # Weights
        factor = 1.0 / (exp_decay ** k)
        decayed_range = tuple(np.array(w_range) * factor)
        A_k = _sample_weights(W, edge_type=edge_type, w_range=decayed_range)

        # Build DiGraph with weighted edges
        graphs.append(nx.DiGraph(A_k))
        A_list.append(A_k)

    # Horizontal concatenation: d x (p*d)
    A = np.vstack(A_list) if A_list else np.zeros((n_nodes, 0))

    return A, graphs

def create_svar_signals(n_samples, dag, lag_graphs, noise_type='normal', var=1):
    """
    Sample a time series X (T x n_nodes) from the SVAR model:
        x_t = x_t W + sum_{k=1}^p x_{t-k} A^{(k)} + z_t
    where W is a DAG (instantaneous effects) and A^{(k)} are lagged effects.

    Parameters
    ----------
    n_samples : int
        Number of time steps T to simulate.
    dag : nx.DiGraph
        Directed acyclic graph with edge weights for the instantaneous matrix W.
    lag_A : np.ndarray | list[np.ndarray] | list[nx.DiGraph]
        Lag dependencies. Supported forms:
          - ndarray of shape (n_nodes, p*n_nodes): horizontal concat [A^(1) ... A^(p)]
          - list of n_nodes x n_nodes ndarrays: [A^(1), ..., A^(p)]
          - list of nx.DiGraph objects (weights taken from adjacency matrices)
    noise_type : {'normal','exp','laplace','gumbel'}, default 'normal'
        Distribution of z_t (i.i.n_nodes. across t).
    var : float or array-like of length n_nodes, default 1
        Per-node noise variance(s).

    Returns
    -------
    X : np.ndarray, shape (T, n_nodes)
        Simulated time series.
    Y_list : list of np.ndarray
        List [Y_1, ..., Y_p], each with shape (T, n_nodes),
        where Y_k[t] = X[t-k] (zero if t-k < 0).
    """

    # --- DAG to matrix W ---
    n_nodes = dag.number_of_nodes()
    ordered = list(nx.topological_sort(dag))
    assert nx.is_directed_acyclic_graph(dag), "W must be a DAG."
    W = nx.to_numpy_array(dag)

    # --- parse lag_A ---
    A_list = [nx.to_numpy_array(Ak) for Ak in lag_graphs]
    p = len(lag_graphs)

    # --- noise variances ---
    if np.isscalar(var):
        var = var*np.ones(n_nodes)

    # --- simulate ---
    X = np.zeros((n_samples, n_nodes))
    Y_list = [np.zeros((n_samples, n_nodes)) for _ in range(p)]

    parent_idx = [list(dag.predecessors(j)) for j in range(n_nodes)]

    for t in range(n_samples):
        # Contribution from lags
        lag_vec = np.zeros(n_nodes)
        for k, Ak in enumerate(A_list, start=1):
            if t - k >= 0:
                lag_vec += X[t-k, :].dot(Ak)
                Y_list[k-1][t, :] = X[t-k, :]

        # Contemporaneous part via DAG order
        for j in ordered:
            eta = X[t, parent_idx[j]] @ W[parent_idx[j], j] + lag_vec[j]

            # noise
            if noise_type == 'normal':
                scale = np.sqrt(var[j])
                noise = np.random.normal(0.0, scale)
            elif noise_type == 'exp':
                scale = np.sqrt(var[j])
                noise = np.random.exponential(scale=scale)
            elif noise_type == 'laplace':
                scale = np.sqrt(var[j]/2.0)
                noise = np.random.laplace(0.0, scale)
            elif noise_type == 'gumbel':
                scale = np.sqrt(6.0*var[j]) / np.pi
                noise = np.random.gumbel(0.0, scale)
            else:
                raise ValueError("Unknown noise_type.")

            X[t, j] = eta + noise

        # print(f't={t}, lag_vec={np.linalg.norm(lag_vec)}, X[t]={np.linalg.norm(X[t,:])}, noise={np.linalg.norm(noise)}')  # Debug print

    Y = np.hstack(Y_list) if p > 0 else np.zeros((n_samples, n_nodes))
    return X, Y

def simulate_svar(n_nodes, n_samples,
    # --- DAG (instantaneous W) ---
    dag_graph_type, dag_edges,
    dag_permute=True, dag_edge_type='positive',
    dag_w_range=(.5, 1.5), dag_rew_prob=.1,
    # --- Lags A^(k) ---
    n_lags=1, lag_graph_type='er', er_edges=None,
    sbm_block_sizes=None, sbm_block_p=None,
    lag_edge_type='positive', lag_w_range=(0.5, .9), exp_decay=1,
    # --- Noise ---
    noise_type='normal', var=1, max_trials=100):
    """
    Simulate data from an SVAR model in a single call:
        x_t = x_t W + sum_{k=1}^p x_{t-k} A^{(k)} + z_t

    Parameters
    ----------
    n_nodes : int
        Number of variables (nodes).
    n_samples : int
        Time series length (T).
    dag_* :
        Controls passed to `create_dag` for the instantaneous graph W.
    n_lags : int
        Number of lags p.
    lag_graph_type : {'er','sbm'}
        Graph type for lag matrices A^{(k)}.
    er_edges, sbm_block_sizes, sbm_block_p :
        Structure controls passed to `create_interslice_graph`.
    lag_edge_type, lag_w_range :
        Weight controls for `create_interslice_graph`.
    noise_type, var :
        Noise settings forwarded to `create_svar_signals`.

    Returns
    -------
    W : np.ndarray
        Weighted adjacency matrix of the instantaneous DAG.
    dag : nx.DiGraph
        Instantaneous DAG with edge weights.
    A_concat : np.ndarray
        Concatenation of lag matrices returned by `create_interslice_graph`.
    lag_graphs : list[nx.DiGraph]
        One directed graph per lag (A^{(k)}).
    X : np.ndarray, shape (T, n_nodes)
        Simulated time series.
    Y : np.ndarray
        Stacked lag regressors as returned by `create_svar_signals`.
    """
    for _ in range(max_trials):
        # --- 1) Build instantaneous graph (W) ---
        W, dag = create_dag(n_nodes=n_nodes, graph_type=dag_graph_type, edges=dag_edges,
                            permute=dag_permute, edge_type=dag_edge_type, w_range=dag_w_range,
                            rew_prob=dag_rew_prob)

        # --- 2) Build lag graphs A^{(k)} ---
        A_concat, lag_graphs = create_interslice_graph(n_nodes=n_nodes, n_lags=n_lags, graph_type=lag_graph_type,
                                                    er_edges=er_edges, sbm_block_sizes=sbm_block_sizes,
                                                    sbm_block_p=sbm_block_p, edge_type=lag_edge_type,
                                                    w_range=lag_w_range, exp_decay=exp_decay)
        
        # --- Check stability ---
        stable, _ = svar_companion_stability(W, A_concat, verbose=False)
        if stable:
            break
    
    if not stable:
        raise ValueError(f"Could not generate a stable SVAR model in {max_trials} trials.")

    # --- 3) Simulate signals from the SVAR model ---
    X, Y = create_svar_signals(n_samples=n_samples, dag=dag, lag_graphs=lag_graphs, noise_type=noise_type,
                               var=var)

    return W, dag, A_concat, lag_graphs, X, Y

def svar_companion_stability(W, A_vert, tol=1e-12, verbose=False):
    d = W.shape[0]
    assert W.shape == (d, d), "W must be square (d,d)."
    assert A_vert.shape[1] == d and A_vert.shape[0] % d == 0, \
        "A must have shape ((p*d), d) with blocks A_k^T stacked by rows."

    p = A_vert.shape[0] // d

    # Recover A^(k) from vertical stack of A_k^T blocks
    A_list = [A_vert[k*d:(k+1)*d, :] for k in range(p)]

    # Compute (I - W)^{-1}
    IW_inv = np.linalg.inv(np.eye(d) - W)

    # B^(k) = (I - W)^{-1} A^(k)
    B_list = [IW_inv @ Ak for Ak in A_list]

    # Build companion matrix C_B of size (p*d, p*d)
    C = np.zeros((p*d, p*d), dtype=float)
    # top block row: [B^(1) ... B^(p)]
    C[:d, :p*d] = np.hstack(B_list)
    # sub-diagonal identity blocks
    if p > 1:
        C[d:, :-d] = np.eye((p-1)*d)

    # Spectral radius
    eigvals = np.linalg.eigvals(C)
    rho = float(np.max(np.abs(eigvals)))

    stable = (rho < 1.0 - tol)

   
    if verbose:
        print(f'Stable: {stable}, spectral radius: {rho:.4f}')

    return stable, rho

def to_bin(W, thr=0.1):
    W_bin = np.copy(W)
    W_bin[np.abs(W_bin) < thr] = 0
    W_bin[np.abs(W_bin) >= thr] = 1

    return W_bin

def compute_norm_sq_err(W_true, W_est, norm_W_true=None):
    norm_W_true = norm_W_true if norm_W_true is not None else la.norm(W_true)
    norm_W_est = la.norm(W_est) if la.norm(W_est) > 0 else 1
    return (la.norm(W_true/norm_W_true - W_est/norm_W_est))**2

def count_accuracy(W_bin_true, W_bin_est):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1}, 
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    pred_und = np.flatnonzero(W_bin_est == -1)
    pred = np.flatnonzero(W_bin_est == 1)
    cond = np.flatnonzero(W_bin_true)
    cond_reversed = np.flatnonzero(W_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # Compute SHD
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    pred_lower = np.flatnonzero(np.tril(W_bin_est + W_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(W_bin_true + W_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # Compute TPR
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    tpr = float(len(true_pos)) / max(len(cond), 1)

    # Compute FDR
    pred_size = len(pred) + len(pred_und)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)

    return shd, tpr, fdr

def display_results(exps_leg, metrics, agg='mean', file_name=None):
    
    metric_str = {'leg': exps_leg}
    for key, value in metrics.items():
        metric_str[key] = []
        
        agg_metric = np.median(value, axis=0) if agg == 'median' else np.mean(value, axis=0)
        std_metric = np.std(value, axis=0)
        for i, _ in enumerate(exps_leg):
            text = f'{agg_metric[i]:.4f}  \u00B1 {std_metric[i]:.4f}'
            metric_str[key].append(text)
        
    df = DataFrame(metric_str)
    display(df)

    if file_name:
        df.to_csv(f'{file_name}.csv', index=False)
        print(f'DataFrame saved to {file_name}.csv')

def standarize(X):
    return (X - X.mean(axis=0))/X.std(axis=0)

def plot_data(axes, data, exps, x_vals, xlabel, ylabel, skip_idx=[], agg='mean', deviation=None,
              alpha=.25, plot_func='semilogx'):
    if agg == 'median':
        agg_data = np.median(data, axis=0)
    else:
        agg_data = np.mean(data, axis=0)

    std = np.std(data, axis=0)
    prctile25 = np.percentile(data, 25, axis=0)
    prctile75 = np.percentile(data, 75, axis=0)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue
        getattr(axes, plot_func)(x_vals, agg_data[:,i], exp['fmt'], label=exp['leg'])

        if deviation == 'prctile':
            up_ci = prctile25[:,i]
            low_ci = prctile75[:,i]
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha)
        elif deviation == 'std':
            up_ci = agg_data[:,i] + std[:,i]
            low_ci = np.maximum(agg_data[:,i] - std[:,i], 0)
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True)
    axes.legend()

def plot_all_metrics(shd, fs_W, err_W, fs_A, err_A, acyc, runtime, dag_count, x_vals, exps, 
                     agg='mean', skip_idx=[], dev=False, alpha=.25, xlabel='Number of samples'):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_data(axes[0], shd, exps, x_vals, xlabel, 'SHD-W', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[1], fs_W, exps, x_vals, xlabel, 'F1-W', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[2], fs_A, exps, x_vals, xlabel, 'F1-A', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[3], acyc, exps, x_vals, xlabel, 'Acyclicity', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_data(axes[0], err_W, exps, x_vals, xlabel, 'Fro Error W', skip_idx, agg=agg,
              deviation=dev, alpha=alpha, plot_func='loglog')
    plot_data(axes[1], err_A, exps, x_vals, xlabel, 'Fro Error A', skip_idx, agg=agg,
              deviation=dev, plot_func='loglog')
    plot_data(axes[2], runtime, exps, x_vals, xlabel, 'Running time (seconds)',
              skip_idx, agg=agg, deviation=dev, alpha=alpha, plot_func='loglog')
    plot_data(axes[3], dag_count, exps, x_vals, xlabel, 'Graph is DAG', skip_idx,
              agg=agg)
    plt.tight_layout()


def data_to_csv(fname, models, xaxis, error, agg='mean', dev='std'):
    data = np.concatenate((xaxis.reshape([xaxis.size, 1]), error), axis=1)
    header = 'xaxis; '  

    for i, model in enumerate(models):
        header += model['leg']
        if i < len(models)-1:
            header += '; '

    np.savetxt(fname, data, delimiter=';', header=header, comments='')
    print('SAVED as:', fname)


def _aggregate(data, agg):
    """
    Compute aggregated statistics (mean/median, std, percentiles) along the first axis.
    """
    if agg == 'median':
        agg_data = np.median(data, axis=0)
    else:
        agg_data = np.mean(data, axis=0)

    stats = {
        'agg': agg_data,
        'std': np.std(data, axis=0),
        'p25': np.percentile(data, 25, axis=0),
        'p75': np.percentile(data, 75, axis=0),
    }
    return stats


def _apply_linestyle(fmt, ls):
    """
    Replace the leading linestyle part of a matplotlib fmt string with `ls`,
    while keeping the marker (and any trailing chars) intact.

    Examples:
      fmt='-o'  -> with ls='-'  -> '-o'
      fmt='-o'  -> with ls='--' -> '--o'
      fmt='--x' -> with ls='-'  -> '-x'
      fmt=':s'  -> with ls='--' -> '--s'
      fmt='o'   -> with ls='--' -> '--o'
    """
    i = 0
    while i < len(fmt) and fmt[i] in ['-', ':', '.']:
        i += 1
    # fmt[i:] now holds marker/etc. Prepend desired linestyle.
    return ls + fmt[i:]


def plot_merged_data(axes, data_W, data_A, exps, x_vals, xlabel, ylabel, skip_idx=[],
                     agg='mean', deviation=None, alpha=.25, plot_func='semilogx'):
    """
    Plot aggregated results for W (solid line) and A (dashed line) on the same axes.
    The marker is taken from each experiment's 'fmt'.
    """

    # Aggregate W and A results separately
    stats_W = _aggregate(data_W, agg)
    stats_A = _aggregate(data_A, agg)

    # Get plotting function dynamically
    plot_fn = getattr(axes, plot_func)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue

        base_fmt = exp.get('fmt', '-')

        # ---- W: solid line ----
        fmt_W = _apply_linestyle(base_fmt, '-')
        line_W, = plot_fn(x_vals, stats_W['agg'][:, i], fmt_W,
                          label=f"{exp.get('leg', f'Exp {i}')} W")

        if deviation == 'prctile':
            up_ci = stats_W['p75'][:, i]
            low_ci = stats_W['p25'][:, i]
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha, color=line_W.get_color())
        elif deviation == 'std':
            up_ci = stats_W['agg'][:, i] + stats_W['std'][:, i]
            low_ci = np.maximum(stats_W['agg'][:, i] - stats_W['std'][:, i], 0)
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha, color=line_W.get_color())

        # ---- A: dashed line ----
        fmt_A = _apply_linestyle(base_fmt, '--')
        line_A, = plot_fn(x_vals, stats_A['agg'][:, i], fmt_A,
                          label=f"{exp.get('leg', f'Exp {i}')} A",
                          color=line_W.get_color())  # opcional: mismo color que W

        if deviation == 'prctile':
            up_ci = stats_A['p75'][:, i]
            low_ci = stats_A['p25'][:, i]
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha, color=line_A.get_color())
        elif deviation == 'std':
            up_ci = stats_A['agg'][:, i] + stats_A['std'][:, i]
            low_ci = np.maximum(stats_A['agg'][:, i] - stats_A['std'][:, i], 0)
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha, color=line_A.get_color())

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True)
    axes.legend()
