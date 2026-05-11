import numpy as np
from scipy.optimize import brentq
from ripser import ripser


# -------------------------
# Hyperbolic sampling (Poincaré ball, curvature -1), d=3
# -------------------------

def sample_uniform_hyperbolic_ball_d3(N, Rmax, rng=None):
    """
    Sample N points uniformly (w.r.t hyperbolic volume) in a 3D hyperbolic ball
    of geodesic radius Rmax, returned in Poincaré ball coordinates (||x||<1).
    Curvature assumed -1.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Sample directions uniformly on S^2
    dirs = rng.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # CDF for radius in d=3: integral sinh^2(t) dt = (sinh(2t) - 2t)/4
    def I(r):
        return (np.sinh(2 * r) - 2 * r) / 4.0

    Imax = I(Rmax)

    # Inverse-CDF by root finding
    u = rng.uniform(0.0, 1.0, size=N)
    targets = u * Imax

    radii = np.empty(N)
    for i, T in enumerate(targets):
        radii[i] = brentq(lambda r: I(r) - T, 0.0, Rmax)

    # Map geodesic radius r to Poincaré radius a = tanh(r/2)
    a = np.tanh(radii / 2.0)

    pts = dirs * a[:, None]
    return pts


def hyperbolic_distance_poincare(x, y, eps=1e-10):
    """
    Hyperbolic distance in Poincaré ball (curvature -1).
    d(x,y) = arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)) )
    """
    x2 = float(np.dot(x, x))
    y2 = float(np.dot(y, y))
    xy2 = float(np.dot(x - y, x - y))
    denom = (1.0 - x2) * (1.0 - y2)
    denom = max(denom, eps)
    arg = 1.0 + 2.0 * xy2 / denom
    arg = max(arg, 1.0 + eps)
    return float(np.arccosh(arg))


def pairwise_hyperbolic_distances_poincare(X):
    N = X.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = hyperbolic_distance_poincare(X[i], X[j])
            D[i, j] = D[j, i] = d
    return D


# -------------------------
# 1D "linear track" trajectory in hyperbolic space
# -------------------------

def sample_radial_geodesic_track(T, Rtrack, rng=None):
    """
    Returns a 1D trajectory x(t) in the Poincaré ball (curvature -1),
    realized as a geodesic ray from the origin out to geodesic radius Rtrack.

    In the Poincaré model, this is a straight line through the origin.
    We randomize the direction.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.normal(size=(3,))
    u /= np.linalg.norm(u)

    # geodesic radius values along the track
    rs = np.linspace(0.0, Rtrack, T)

    # Poincaré radius: a = tanh(r/2)
    a = np.tanh(rs / 2.0)  # shape (T,)
    X = a[:, None] * u[None, :]
    return X  # (T,3)


# -------------------------
# Tuning-curve responses and correlation similarity
# -------------------------

def tuning_responses_gaussian(track_X, centers_S, sigma):
    """
    track_X: (T,3) Poincaré points
    centers_S: (N,3) Poincaré points
    returns R: (T,N) responses
      R[t,i] = exp( - d(track_X[t], centers_S[i])^2 / (2 sigma^2) )
    """
    T = track_X.shape[0]
    N = centers_S.shape[0]
    R = np.zeros((T, N), dtype=np.float64)

    # compute distances track->centers
    for i in range(N):
        for t in range(T):
            d = hyperbolic_distance_poincare(track_X[t], centers_S[i])
            R[t, i] = np.exp(-(d * d) / (2.0 * sigma * sigma))
    return R


def correlation_similarity(R, eps=1e-12):
    """
    Pearson correlation across time for each pair of neurons.
    R: (T,N)
    returns A: (N,N), symmetric, diag=1
    """
    # center and standardize columns
    Rc = R - R.mean(axis=0, keepdims=True)
    std = R.std(axis=0, ddof=1, keepdims=True)
    std = np.maximum(std, eps)
    Z = Rc / std
    A = (Z.T @ Z) / (R.shape[0] - 1)
    # numerical safety
    A = np.clip(A, -1.0, 1.0)
    np.fill_diagonal(A, 1.0)
    return A


def similarity_to_dissimilarity(A):
    """
    Convert similarity to a dissimilarity matrix suitable for ripser VR filtration.
    We use D = Amax - A so that higher similarity => smaller distance.
    """
    Amax = float(np.max(A))
    D = (Amax - A).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    return D


# -------------------------
# Betti curve vs edge density rho using one PH run
# -------------------------

def betti_curves_vs_rho_from_distance(D, rhos, maxdim=3):
    """
    Compute Betti_k(rho) for k=1..maxdim using persistent homology of VR filtration on D.
    rhos: array of edge densities in (0,1).
    Returns dict k -> betti array aligned with rhos.
    """
    N = D.shape[0]
    tri = D[np.triu_indices(N, k=1)]
    tri_sorted = np.sort(tri)

    def eps_of_rho(rho):
        idx = int(np.clip(np.floor(rho * (len(tri_sorted) - 1)), 0, len(tri_sorted) - 1))
        return tri_sorted[idx]

    epsilons = np.array([eps_of_rho(r) for r in rhos])

    out = ripser(D, distance_matrix=True, maxdim=maxdim)
    dgms = out["dgms"]

    bettis = {}
    for k in range(1, maxdim + 1):
        if k >= len(dgms):
            bettis[k] = np.zeros_like(epsilons, dtype=int)
            continue
        intervals = dgms[k]
        births = intervals[:, 0]
        deaths = intervals[:, 1]
        deaths = np.where(np.isfinite(deaths), deaths, np.inf)

        bk = np.array([(births <= e).sum() - (deaths <= e).sum() for e in epsilons], dtype=int)
        bettis[k] = bk

    return bettis


# -------------------------
# Simulation: centers + track + correlation similarity
# -------------------------

def simulate_one_betti_curve_corr(
    *,
    N=41,
    Rmax=15.5,
    Rtrack=15.5,
    sigma=2.0,
    T=800,
    rhos=None,
    seed=0,
):
    """
    Regime-1 with a realistic measurement model:
      - sample neuron centers uniformly in hyperbolic ball radius Rmax
      - generate a 1D geodesic track of length Rtrack
      - compute Gaussian tuning responses along track
      - build Pearson correlation similarity matrix
      - convert to dissimilarity and compute Betti curves vs edge density

    Returns: rhos, bettis (dict k->array)
    """
    if rhos is None:
        rhos = np.linspace(0.0, 1.0, 101)[1:-1]

    rng = np.random.default_rng(seed)

    # neuron centers
    S = sample_uniform_hyperbolic_ball_d3(N=N, Rmax=Rmax, rng=rng)

    # 1D linear track (geodesic)
    Xtrack = sample_radial_geodesic_track(T=T, Rtrack=Rtrack, rng=rng)

    # tuning responses and correlations
    Rresp = tuning_responses_gaussian(Xtrack, S, sigma=sigma)
    A = correlation_similarity(Rresp)

    # similarity -> dissimilarity for VR filtration
    D = similarity_to_dissimilarity(A)

    bettis = betti_curves_vs_rho_from_distance(D, rhos=rhos, maxdim=3)
    return rhos, bettis


def simulate_many_corr(
    *,
    B=300,
    N=41,
    Rmax=15.5,
    Rtrack=15.5,
    sigma=2.0,
    T=800,
    rhos=None,
    seed0=0,
):
    """
    Returns:
      rhos, summary where summary[k] = (mean_curve, sd_curve),
      and raw where raw[k] = (B,T) array for statistical tests.
    """
    if rhos is None:
        rhos = np.linspace(0.0, 1.0, 101)[1:-1]

    all_b = {1: [], 2: [], 3: []}
    for b in range(B):
        _, bettis = simulate_one_betti_curve_corr(
            N=N, Rmax=Rmax, Rtrack=Rtrack, sigma=sigma, T=T, rhos=rhos, seed=seed0 + b
        )
        for k in [1, 2, 3]:
            all_b[k].append(bettis[k])

    raw = {k: np.stack(all_b[k], axis=0) for k in [1, 2, 3]}
    summary = {k: (raw[k].mean(axis=0), raw[k].std(axis=0, ddof=1)) for k in [1, 2, 3]}
    return rhos, summary, raw
# rhos_out, summary, raw = simulate_many_corr(
#     B=300, N=41, Rmax=15.5, Rtrack=15.5, sigma=2.0, T=800, rhos=rhos, seed0=0
# )