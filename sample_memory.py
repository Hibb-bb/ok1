from hyperboloid import HyperboloidKappa
import numpy as np
import geomstats.backend as gs


def origin_point(dim: int, kappa: float):
    """
    Canonical origin o = (rho, 0, ..., 0) on hyperboloid with rho = 1/sqrt(-kappa).
    """
    if kappa >= 0:
        raise ValueError("kappa must be < 0 for hyperbolic space.")
    rho = 1.0 / np.sqrt(-kappa)
    o = np.zeros((dim + 1,), dtype=float)
    o[0] = rho
    return gs.array(o)


def sample_tangent_ball_at_origin(dim: int, M: int, R: float, rng: np.random.Generator):
    u = rng.normal(size=(M, dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    rad = R * (rng.random(M) ** (1.0 / dim))
    u = u * rad[:, None]

    v = np.zeros((M, dim + 1), dtype=float)
    v[:, 1:] = u
    return gs.array(v)



def sample_hyperboloid_points_from_tangent_ball(
    geometry: HyperboloidKappa, M: int, R: float, rng: np.random.Generator
):

    dim = geometry.dim
    kappa = geometry.curvature

    o = origin_point(dim, kappa)  # (dim+1,)
    v = sample_tangent_ball_at_origin(dim, M, R, rng)  # (M, dim+1)

    o_batch = gs.tile(o, (M, 1))  # (M, dim+1)
    x = geometry.metric.exp(v, o_batch)  # (M, dim+1)

    return x, v