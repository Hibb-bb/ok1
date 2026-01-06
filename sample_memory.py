from hyperboloid import HyperboloidKappa
import numpy as np
import geomstats.backend as gs
import math

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


def origin_batch(H, batch):
    d1 = H.dim + 1
    o = gs.zeros((batch, d1))
    o[:, 0] = H.radius
    return o


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


def sanity_check_tangent_at_origin(H, v, atol=1e-6):
    """
    H: HyperboloidKappa instance
    v: (M, d+1) tangent vectors you constructed (should satisfy <v,o>=0)
    """
    v = gs.array(v)
    d1 = v.shape[-1]
    d = d1 - 1

    # canonical origin
    o = gs.zeros((d1,))
    o = gs.array(o)
    o = gs.assignment(o, H.radius, [0])  # o[0] = radius
    o_batch = gs.tile(o, (v.shape[0], 1))

    # check Lorentz orthogonality <v,o>_L = 0
    ip = H.embedding_space.metric.inner_product(v, o_batch)
    max_abs_ip = float(gs.amax(gs.abs(ip)))
    print("max |<v,o>_L|:", max_abs_ip)
    assert max_abs_ip <= atol, "Tangent check failed: <v,o>_L not ~ 0"

    # optional: check your v is unchanged by projection to tangent
    v_proj = H.to_tangent(v, o_batch)
    diff = gs.amax(gs.abs(v - v_proj))
    diff = float(diff)
    print("max |v - Proj_T(v)|:", diff)
    assert diff <= 10 * atol, "v is not in tangent space (projection changed it noticeably)."

def sanity_check_exp_lands_on_hyperboloid(H, x, atol=1e-5):
    """
    x: (M, d+1) points after exp
    """
    x = gs.array(x)
    sq = H.embedding_space.metric.squared_norm(x)  # should be -R^2
    target = -(H.radius ** 2)
    err = gs.amax(gs.abs(sq - target))
    err = float(err)
    print("max |<x,x>_L + R^2|:", err)
    assert err <= atol, "Exp output is off the hyperboloid (Lorentz norm mismatch)."

def sanity_check_exp_zero_is_origin(H, atol=1e-7):
    d1 = H.dim + 1
    o = gs.zeros((1, d1))
    o[:, 0] = H.radius

    v0 = gs.zeros((1, d1))
    x0 = H.metric.exp(v0, o)

    err = float(gs.amax(gs.abs(x0 - o)))
    print("max |Exp_o(0) - o|:", err)
    assert err <= atol, "Exp_o(0) != origin"

def sanity_check_dist_matches_tangent_norm(H, v, atol=1e-4):
    """
    Check: dist(o, Exp_o(v)) == ||v||_o
    where ||v||_o is the Riemannian norm at the origin induced by the Minkowski metric
    restricted to the tangent space.
    """
    v = gs.array(v)
    M = int(v.shape[0])
    d1 = int(H.dim + 1)

    # Canonical origin batch (avoid gs.assignment)
    o = gs.zeros((M, d1))
    o[:, 0] = H.radius

    # Map to manifold
    x = H.metric.exp(v, o)      # (M, d+1)
    d_ox = H.metric.dist(o, x)  # (M,)

    # Tangent norm at origin: sqrt( <v,v>_o ) using the metric inner product
    v_sq = H.metric.inner_product(v, v, base_point=o)  # (M,)
    v_sq = gs.clip(v_sq, 0.0, math.inf)
    v_norm = gs.sqrt(v_sq)

    err = float(gs.amax(gs.abs(d_ox - v_norm)))
    print("max |dist(o,Exp(v)) - ||v||_o|:", err)
    assert err <= atol, "Distance-radius identity failed (check exp scaling or tangent construction)."

def run_all_sampling_sanity_checks(H, x, v):
    sanity_check_tangent_at_origin(H, v)
    sanity_check_exp_lands_on_hyperboloid(H, x)
    sanity_check_exp_zero_is_origin(H)
    sanity_check_dist_matches_tangent_norm(H, v)
    print("All sanity checks passed.")

# run_all_sampling_sanity_checks(geometry, x, v)