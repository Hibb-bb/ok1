import numpy as np
from scipy.special import softmax


def update_mhn(query, memory, beta=1.0):
    score = beta * (memory @ query)  # (M,)
    w = softmax(score, axis=0)  # (M,)
    return memory.T @ w  # (n,) - equivalent to w @ memory but more explicit

def update_dam(query, memory, n_order=10, beta=1.0):
    sim = memory @ query  # (M,)
    score = beta * (sim**n_order)  # (M,)
    w = score / np.linalg.norm(score, axis=0) # (M,)
    return w @ memory  # (n,)


def make_query_from_target_euclidean(target, sigma, rng):
    return target + rng.normal(scale=sigma, size=target.shape)


def run_recall_mhn(args, M, seed):
    rng = np.random.default_rng(seed)

    n = int(getattr(args, "d", getattr(args, "n_dim", 20)))
    mem_R = float(getattr(args, "mem_R", getattr(args, "R", 3.0)))
    beta = float(getattr(args, "beta", 1.0))

    # Choose sampler based on dataset
    dataset = getattr(args, "dataset", "synthetic")
    if dataset == "synthetic":
        from sample_euclidean_memory import sample_euclidean_points_from_tangent_ball, uniform_sampling_in_ball
        memory = uniform_sampling_in_ball(M=M, dim=n, R=mem_R)
        # memory, _ = sample_euclidean_points_from_tangent_ball(M=M, dim=n, R=mem_R, rng=rng)
    elif dataset in ["mnist", "cifar10"]:
        from sample_image_memory import sample_images_from_dataset
        pca_dim = getattr(args, "pca_dim", None) or n
        memory, _ = sample_images_from_dataset(
            dataset_name=dataset, M=M, dim=pca_dim, rng=rng
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    correct_recall = 0
    
    for t in range(M):
        target = memory[t]
        query = make_query_from_target_euclidean(target, args.noise_sigma, rng)

        converged = False
        for step in range(args.max_steps):
            new = update_mhn(query, memory, beta=beta)
            dist = np.linalg.norm(new - target)

            if dist <= args.tol:
                correct_recall += 1
                converged = True
                break
            query = new
    
    return correct_recall / M


def run_recall_dam(args, M, seed):
    rng = np.random.default_rng(seed)

    n = int(getattr(args, "d", getattr(args, "n_dim", 20)))
    mem_R = float(getattr(args, "mem_R", getattr(args, "R", 3.0)))
    beta = float(getattr(args, "beta", 1.0))
    n_order = int(getattr(args, "n_order", 10))

    # Choose sampler based on dataset
    dataset = getattr(args, "dataset", "synthetic")
    if dataset == "synthetic":
        from sample_euclidean_memory import sample_euclidean_points_from_tangent_ball
        memory, _ = sample_euclidean_points_from_tangent_ball(M=M, dim=n, R=mem_R, rng=rng)
    elif dataset in ["mnist", "cifar10"]:
        from sample_image_memory import sample_images_from_dataset
        pca_dim = getattr(args, "pca_dim", None) or n
        memory, _ = sample_images_from_dataset(
            dataset_name=dataset, M=M, dim=pca_dim, rng=rng
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    correct_recall = 0

    for t in range(M):
        target = memory[t]
        query = make_query_from_target_euclidean(target, args.noise_sigma, rng)

        for step in range(args.max_steps):
            new = update_dam(query, memory, n_order=n_order, beta=beta)
            dist = np.linalg.norm(new - target)

            if dist <= args.tol:
                correct_recall += 1
                break
            query = new
    
    return correct_recall / M
