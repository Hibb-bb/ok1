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
        # solve I(r) - T = 0 on [0, Rmax]
        radii[i] = brentq(lambda r: I(r) - T, 0.0, Rmax)

    # Map geodesic radius r to Poincaré radius a = tanh(r/2)
    a = np.tanh(radii / 2.0)

    pts = dirs * a[:, None]
    return pts


def hyperbolic_distance_poincare(x, y, eps=1e-8):
    """
    Hyperbolic distance in Poincaré ball (curvature -1), for x,y in R^d, ||x||,||y||<1.
    d(x,y) = arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)) )
    """
    x2 = np.dot(x, x)
    y2 = np.dot(y, y)
    xy2 = np.dot(x - y, x - y)
    denom = (1.0 - x2) * (1.0 - y2)
    denom = max(denom, eps)
    arg = 1.0 + 2.0 * xy2 / denom
    arg = max(arg, 1.0 + eps)
    return np.arccosh(arg)


def pairwise_hyperbolic_distances_poincare(X):
    N = X.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = hyperbolic_distance_poincare(X[i], X[j])
            D[i, j] = D[j, i] = d
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

    # Map rho -> epsilon via distance quantile
    # rho corresponds to fraction of edges included: #{D_ij <= eps} / (N choose 2) = rho
    def eps_of_rho(rho):
        idx = int(np.clip(np.floor(rho * (len(tri_sorted) - 1)), 0, len(tri_sorted) - 1))
        return tri_sorted[idx]

    epsilons = np.array([eps_of_rho(r) for r in rhos])

    # Persistent homology once
    out = ripser(D, distance_matrix=True, maxdim=maxdim)
    dgms = out["dgms"]  # list: dgms[k] is array [[birth, death], ...]

    bettis = {}
    for k in range(1, maxdim + 1):
        if k >= len(dgms):
            bettis[k] = np.zeros_like(epsilons, dtype=int)
            continue
        intervals = dgms[k]
        births = intervals[:, 0]
        deaths = intervals[:, 1]
        # Treat infinite deaths as +inf
        deaths = np.where(np.isfinite(deaths), deaths, np.inf)

        # Betti_k(eps) = count(birth <= eps < death)
        bk = np.array([(births <= e).sum() - (deaths <= e).sum() for e in epsilons], dtype=int)
        bettis[k] = bk

    return bettis


def simulate_one_betti_curve(N=41, Rmax=15.5, rhos=None, seed=0):
    if rhos is None:
        rhos = np.linspace(0.0, 1.0, 101)[1:-1]  # avoid 0 and 1
    rng = np.random.default_rng(seed)
    X = sample_uniform_hyperbolic_ball_d3(N=N, Rmax=Rmax, rng=rng)
    D = pairwise_hyperbolic_distances_poincare(X)
    bettis = betti_curves_vs_rho_from_distance(D, rhos=rhos, maxdim=3)
    return rhos, bettis


def simulate_many(B=300, N=41, Rmax=15.5, rhos=None, seed0=0):
    if rhos is None:
        rhos = np.linspace(0.0, 1.0, 101)[1:-1]
    all_b1 = []
    all_b2 = []
    all_b3 = []
    for b in range(B):
        _, bettis = simulate_one_betti_curve(N=N, Rmax=Rmax, rhos=rhos, seed=seed0 + b)
        all_b1.append(bettis[1])
        all_b2.append(bettis[2])
        all_b3.append(bettis[3])
    all_b1 = np.stack(all_b1, axis=0)
    all_b2 = np.stack(all_b2, axis=0)
    all_b3 = np.stack(all_b3, axis=0)

    summary = {
        1: (all_b1.mean(0), all_b1.std(0)),
        2: (all_b2.mean(0), all_b2.std(0)),
        3: (all_b3.mean(0), all_b3.std(0)),
    }
    return rhos, summary


