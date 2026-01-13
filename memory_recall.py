from hyperboloid import HyperboloidKappa
import numpy as np
import geomstats.backend as gs
from sample_memory import sample_hyperboloid_points_from_tangent_ball
from scipy.special import softmax
import json
import os


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


def geo_distance(d):
    return d 


def square_distance(d):
    return d**2


def update(geometry, query, memory, phi=foo):
    query = geometry.regularize(query)
    memory = geometry.regularize(memory)

    tangent_memory = geometry.metric.log(memory, base_point=query)

    if phi == foo:
        lorentz_inner = geometry.embedding_space.metric.inner_product(query, memory)
        # dist = np.arccosh(-lorentz_inner)
        score = phi(lorentz_inner, geometry.curvature)

    else:
    #     if phi == geo_distance:
    #         # print(geometry.metric.dist(query, memory), "geo")
    #         # raise Exception
    #     else:
    #         print(geometry.metric.dist(query, memory), "square")
    #         raise Exception

        score = phi(geometry.metric.dist(query, memory))
    weights = softmax(-score)

    # if np.isnan(weights).any() is True:
    #     print("w", weights)
    #     print("s", score)
    #     raise Exception("bad weights")

    if phi == geo_distance:
        print("geo distance", weights)

    # Check if weights are close to one-hot
    max_weight_idx = np.argmax(weights)
    max_weight = weights[max_weight_idx]
    # If the max weight is very close to 1 (e.g., > 0.99), treat as one-hot
    if max_weight > 0.99:
        result = memory[max_weight_idx]
    else:
        tangent_query = weights @ tangent_memory
        result = geometry.metric.exp(tangent_query, query)
    # result = geometry.regularize(result)

    # if gs.any(gs.isnan(result)):
    #     print("result", result)
    #     print("score", score)
    #     print("weights", weights)
    #     raise Exception

    return result


def make_query_from_target(geometry, target, sigma, rng):
    d1 = geometry.dim + 1
    eps = rng.normal(scale=sigma, size=(d1,))
    eps = gs.array(eps)
    eps = geometry.to_tangent(eps, target)
    return geometry.metric.exp(eps, target)


def map_euclidean_to_hyperboloid(geometry, euclidean_vectors):
    M, dim = euclidean_vectors.shape
    assert dim == geometry.dim, f"Dimension mismatch: {dim} != {geometry.dim}"
    tangent_vectors = np.zeros((M, dim + 1), dtype=float)
    tangent_vectors[:, 1:] = euclidean_vectors  # First component is 0
    tangent_vectors = gs.array(tangent_vectors)
    origin_batch = gs.zeros((M, dim + 1))
    origin_batch[:, 0] = geometry.radius
    hyperboloid_points = geometry.metric.exp(tangent_vectors, origin_batch)
    hyperboloid_points = geometry.regularize(hyperboloid_points)

    return hyperboloid_points


def generate_image_queries(geometry, euclidean_vectors, sigma, rng):

    M, dim = euclidean_vectors.shape
    eps = rng.normal(scale=sigma, size=(M,dim))
    euclidean_vectors += eps
    assert dim == geometry.dim, f"Dimension mismatch: {dim} != {geometry.dim}"
    tangent_vectors = np.zeros((M, dim + 1), dtype=float)
    tangent_vectors[:, 1:] = euclidean_vectors  # First component is 0
    tangent_vectors = gs.array(tangent_vectors)
    origin_batch = gs.zeros((M, dim + 1))
    origin_batch[:, 0] = geometry.radius
    hyperboloid_points = geometry.metric.exp(tangent_vectors, origin_batch)
    hyperboloid_points = geometry.regularize(hyperboloid_points)

    return hyperboloid_points


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

    # Choose sampler based on dataset
    dataset = getattr(args, "dataset", "synthetic")
    if dataset == "synthetic":
        dim = args.d
        geometry = HyperboloidKappa(dim=dim, curvature=args.kappa)
        memory, _ = sample_hyperboloid_points_from_tangent_ball(
            geometry, M, args.mem_R, rng
        )
    elif dataset in ["mnist", "cifar10"]:
        from sample_image_memory import sample_images_from_dataset
        requested_dim = getattr(args, "pca_dim", None) or args.d
        memory_euclidean, _ = sample_images_from_dataset(
            dataset_name=dataset, M=M, dim=requested_dim, rng=rng
        )
        actual_dim = memory_euclidean.shape[1]
        geometry = HyperboloidKappa(dim=actual_dim, curvature=args.kappa)
        memory = map_euclidean_to_hyperboloid(geometry, memory_euclidean)
        queries = generate_image_queries(geometry, memory_euclidean, args.noise_sigma, rng)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    correct_recall = 0
    for t in range(M):
        target = memory[t]

        if dataset == "synthetic":
            query = make_query_from_target(geometry, target, args.noise_sigma, rng)
        else:
            query = queries[t]

        for step in range(args.max_steps):
            new = update(geometry, query, memory, phi=phi)

            if geometry.metric.dist(new, target) < args.tol:
                correct_recall += 1
                break
            query = new

    return correct_recall / M