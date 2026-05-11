import numpy as np
import geomstats.backend as gs

def sample_tangent_sphere_at_origin(dim: int, M: int, r: float, seed: int = 0):
    """
    Sample M tangent vectors uniformly from the Euclidean sphere of radius r
    in T_o H^dim_kappa at the origin o.

    Returns: (M, dim+1) array in extrinsic Minkowski coords, of the form (0, u).
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(M, dim))                     # Gaussian in R^dim
    u /= np.linalg.norm(u, axis=1, keepdims=True)     # normalize -> uniform on S^{dim-1}
    u *= r
    v = np.zeros((M, dim + 1))
    v[:, 1:] = u                                      # time component 0
    return gs.array(v)

def origin_point(dim: int, kappa: float):
    """
    Return the canonical origin o = (rho, 0, ..., 0) on the hyperboloid
    satisfying <o,o>_L = -rho^2, where rho = 1/sqrt(-kappa).
    """
    if kappa >= 0:
        raise ValueError("kappa must be < 0")
    rho = 1.0 / np.sqrt(-kappa)
    o = np.zeros((dim + 1,))
    o[0] = rho
    return gs.array(o)

def sample_hyperboloid_points(dim: int, M: int, r: float, kappa: float, H):
    """
    Given a HyperboloidKappa instance H (dim, curvature=kappa),
    sample M points via: v_i ~ Unif sphere radius r in T_o, x_i = Exp_o(v_i).
    """
    o = origin_point(dim, kappa)                      # (dim+1,)
    v = sample_tangent_sphere_at_origin(dim, M, r)     # (M, dim+1)

    # broadcast base point to match batch
    o_batch = gs.tile(o, (M, 1))                       # (M, dim+1)

    x = H.metric.exp(v, o_batch)                       # (M, dim+1)
    return x, v, o
