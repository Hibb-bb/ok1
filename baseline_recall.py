import numpy as np
from scipy.special import softmax


from scipy.special import softmax

def l2_normalize(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def update_mhn(query, memory, beta=1.0, scale=False, normalize=True, max_steps=10):
    # memory: (M, n), query: (n,)
    memory = np.asarray(memory, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    beta = np.float64(beta)
    tol = 0.01
    mem = l2_normalize(memory, axis=1) if normalize else memory

    for step in range(max_steps):

        q = l2_normalize(query[None, :], axis=1)[0] if normalize else query

        logits = mem @ q  # (M,)
        if scale:
            logits = logits / np.sqrt(mem.shape[1])
        w = softmax(beta * logits, axis=0)  # (M,)
        max_weight_idx = np.argmax(w)
        max_weight = w[max_weight_idx]

        if max_weight >= 1 - tol:
            return memory[max_weight_idx]
        else:
            query = w @ mem
    result = query
    return result # , w  # return weights for evaluation/debug

def update_dam(query, memory, n_order=10, beta=1.0, max_steps=10):
    memory = np.asarray(memory, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    beta = np.float64(beta)
    tol = 0.01
    for step in range(max_steps):

        sim = memory @ query  # (M,)
        score = beta * (sim**n_order)  # (M,)
        w = score / (np.linalg.norm(score, axis=0) + 1e-6)  # (M,)

        max_weight_idx = np.argmax(w)
        max_weight = w[max_weight_idx]

        if max_weight >= 1 - tol:
            return memory[max_weight_idx]
        else:
            query = score @ memory

    return query  # (n,)


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
        new = update_mhn(query, memory, beta=beta,max_steps= args.max_steps)
        dist = np.linalg.norm(new - target)

        if dist <= args.tol:
            correct_recall += 1
    
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

        new = update_dam(query, memory, n_order=n_order, beta=beta, max_steps= args.max_steps)
        dist = np.linalg.norm(new - target)

        if dist <= args.tol:
            correct_recall += 1
    
    return correct_recall / M
