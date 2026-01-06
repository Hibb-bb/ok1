from hyperboloid import HyperboloidKappa
import numpy as np
import geomstats.backend as gs
from sample_memory import sample_hyperboloid_points_from_tangent_ball
from scipy.special import softmax


def hyperbolic_distance_from_lorentz_ip(lorentz_ip, kappa):

    if kappa >= 0:
        raise ValueError("kappa must be < 0.")
    R = 1.0 / np.sqrt(-kappa)

    # cosh(d/R) = - <x,y>_L / R^2
    z = -lorentz_ip / (R**2)

    # numerical safety: z must be >= 1
    z = np.clip(z, 1.0, 1e24)

    return R * np.arccosh(z)


def foo(x, kappa=None):
    return -x


def geo_distance(lorentz_ip, kappa):
    d = hyperbolic_distance_from_lorentz_ip(lorentz_ip, kappa)
    return d  # or -d if you want closer => higher score


def square_distance(lorentz_ip, kappa):
    d = hyperbolic_distance_from_lorentz_ip(lorentz_ip, kappa)
    return d**2


def update(geometry, query, memory, phi=foo):

    tangent_memory = geometry.metric.log(memory, base_point=query)
    lorentz_inner = geometry.embedding_space.metric.inner_product(query, memory)
    score = phi(lorentz_inner, geometry.curvature)
    weights = softmax(-score)
    tangent_query = weights @ tangent_memory
    return geometry.metric.exp(tangent_query, query)


def make_query_from_target(geometry, target, sigma, rng):
    d1 = geometry.dim + 1
    eps = rng.normal(scale=sigma, size=(d1,))
    eps = gs.array(eps)
    eps = geometry.to_tangent(eps, target)
    return geometry.metric.exp(eps, target)


def run_recall_hyperbolic(args, phi_choice, M, seed):
    # pick phi
    if phi_choice == "identity":
        phi = foo
    elif phi_choice == "geo_distance":
        phi = geo_distance
    elif phi_choice == "square_distance":
        phi = square_distance
    else:
        raise ValueError(f"Unknown phi: {args.phi}")

    rng = np.random.default_rng(seed)

    geometry = HyperboloidKappa(dim=args.d, curvature=args.kappa)

    memory, _ = sample_hyperboloid_points_from_tangent_ball(
        geometry, M, args.mem_R, rng
    )

    correct_recall = 0

    for t in range(M):
        target = memory[t]
        query = make_query_from_target(geometry, target, args.noise_sigma, rng)

        for _ in range(args.max_steps):
            new = update(geometry, query, memory, phi=phi)
            if geometry.metric.dist(new, target) < args.tol:
                correct_recall += 1
                break
            query = new

    return correct_recall / M
