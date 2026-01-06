import numpy as np
from scipy.special import softmax
from sample_euclidean_memory import sample_euclidean_points_from_tangent_ball


def update_mhn(query, memory, beta=1.0):
    score = beta * (memory @ query)  # (M,)
    w = softmax(score, axis=0)  # (M,)
    return w @ memory  # (n,)


def update_dam(query, memory, n_order=3, beta=1.0):
    sim = memory @ query  # (M,)
    score = beta * (sim**n_order)  # (M,)
    w = softmax(score, axis=0)  # (M,)
    return w @ memory  # (n,)


def make_query_from_target_euclidean(target, sigma, rng):
    return target + rng.normal(scale=sigma, size=target.shape)


def run_recall_mhn(args):
    rng = np.random.default_rng(getattr(args, "seed", None))

    n = int(getattr(args, "n_dim", 20))
    M = int(getattr(args, "M", 30))
    mem_R = float(getattr(args, "mem_R", getattr(args, "R", 3.0)))
    beta = float(getattr(args, "beta", 1.0))

    # memory from Euclidean tangent ball
    memory, _ = sample_euclidean_points_from_tangent_ball(M=M, dim=n, R=mem_R, rng=rng)
    correct_recall = 0

    for t in range(M):

        target = memory[t]
        query = make_query_from_target_euclidean(target, args.noise_sigma, rng)

        for _ in range(args.max_steps):
            new = update_mhn(query, memory, beta=beta)

            if np.linalg.norm(new - target) <= args.tol:
                correct_recall += 1
                break
            query = new

        return correct_recall / M


def run_recall_dam(args, M, seed):
    rng = np.random.default_rng(seed)

    n = int(getattr(args, "n_dim", 20))
    mem_R = float(getattr(args, "mem_R", getattr(args, "R", 3.0)))
    beta = float(getattr(args, "beta", 1.0))
    n_order = int(getattr(args, "n_order", 5))

    memory, _ = sample_euclidean_points_from_tangent_ball(M=M, dim=n, R=mem_R, rng=rng)
    correct_recall = 0

    for t in range(M):
        target = memory[t]
        query = make_query_from_target_euclidean(target, args.noise_sigma, rng)

        for _ in range(args.max_steps):
            new = update_dam(query, memory, n_order=n_order, beta=beta)

            if np.linalg.norm(new - target) <= args.tol:
                correct_recall += 1
                break
            query = new

        return correct_recall / M
