import numpy as np
from scipy.special import softmax

import torch

from recall_config import resolve_pca_dim_images, resolve_torch_device, set_global_torch_seed


def l2_normalize(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def update_mhn(query, memory, beta=1, scale=False, normalize=True, max_steps=10):
    memory = np.asarray(memory, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    beta = np.float64(beta)
    tol = 0.01
    mem = l2_normalize(memory, axis=1) if normalize else memory

    for step in range(max_steps):
        q = l2_normalize(query[None, :], axis=1)[0] if normalize else query
        logits = mem @ q
        if scale:
            logits = logits / np.sqrt(mem.shape[1])
        w = softmax(beta * logits, axis=0)
        max_weight_idx = np.argmax(w)
        max_weight = w[max_weight_idx]

        if max_weight >= 1 - tol:
            return memory[max_weight_idx]
        query = w @ mem
    return query


def update_mhn_batched(
    memory: torch.Tensor,
    queries: torch.Tensor,
    *,
    beta: float,
    scale: bool = False,
    normalize: bool = True,
    max_steps: int = 10,
    tol: float = 0.01,
) -> torch.Tensor:
    """
    memory (M, n), queries (M, n) — column j uses noisy query for target memory[j].
    Returns converged / final state per row (M, n).
    """
    mem_raw = memory
    Q = queries.clone()
    converged = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
    output = Q.clone()

    n_feat = memory.shape[1]
    sqrtn = float(n_feat) ** 0.5

    for _ in range(max_steps):
        mem_use = torch.nn.functional.normalize(mem_raw, dim=1, eps=1e-12) if normalize else mem_raw
        Q_use = torch.nn.functional.normalize(Q, dim=1, eps=1e-12) if normalize else Q
        logits = mem_use @ Q_use.T
        w = torch.softmax(beta * logits, dim=0)
        max_w, argmax = w.max(dim=0)
        newly = max_w >= (1.0 - tol)
        converged = converged | newly
        output = torch.where(newly.unsqueeze(1), mem_raw[argmax], output)
        if bool(converged.all().item()):
            break
        mem_upd = mem_use if normalize else mem_raw
        Q = w.T @ mem_upd
        Q = torch.where(converged.unsqueeze(1), output, Q)

    return torch.where(converged.unsqueeze(1), output, Q)


def update_dam(query, memory, n_order=10, beta=1, max_steps=10):
    memory = np.asarray(memory, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    beta = np.float64(beta)
    tol = 0.01
    for step in range(max_steps):
        sim = memory @ query
        score = beta * (sim**n_order)
        w = score / (np.linalg.norm(score, axis=0) + 1e-6)
        max_weight_idx = np.argmax(w)
        max_weight = w[max_weight_idx]

        if max_weight >= 1 - tol:
            return memory[max_weight_idx]
        query = w @ memory
    return query


def update_dam_batched(
    memory: torch.Tensor,
    queries: torch.Tensor,
    *,
    n_order: int,
    beta: float,
    max_steps: int = 10,
    tol: float = 0.01,
) -> torch.Tensor:
    mem_raw = memory
    Q = queries.clone()
    converged = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
    output = Q.clone()

    for _ in range(max_steps):
        sim = mem_raw @ Q.T
        score = beta * (sim**n_order)
        col_norm = torch.linalg.norm(score, dim=0, keepdim=True) + 1e-6
        w = score / col_norm
        max_w, argmax = w.max(dim=0)
        newly = max_w >= (1.0 - tol)
        converged = converged | newly
        output = torch.where(newly.unsqueeze(1), mem_raw[argmax], output)
        if bool(converged.all().item()):
            break
        Q = score.T @ mem_raw
        Q = torch.where(converged.unsqueeze(1), output, Q)

    return torch.where(converged.unsqueeze(1), output, Q)


def make_query_from_target_euclidean(target, sigma, rng):
    return target + rng.normal(scale=sigma, size=target.shape)


def _load_memory_numpy(args, M, seed):
    rng = np.random.default_rng(seed)
    n = int(getattr(args, "d", 20))
    mem_R = float(getattr(args, "mem_R", 3.0))
    dataset = getattr(args, "dataset", "synthetic")
    if dataset == "synthetic":
        from sample_euclidean_memory import uniform_sampling_in_ball

        memory = uniform_sampling_in_ball(M=M, dim=n, R=mem_R)
    elif dataset in ["mnist", "cifar10"]:
        from sample_image_memory import sample_images_from_dataset

        pca_dim = resolve_pca_dim_images(args)
        memory, _ = sample_images_from_dataset(
            dataset_name=dataset, M=M, dim=pca_dim, R=mem_R, rng=rng
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return memory, rng


def run_recall_mhn(args, M, seed):
    beta = float(getattr(args, "beta", 1.0))
    device = resolve_torch_device(getattr(args, "device", "cpu"))
    use_batch = getattr(args, "use_batch", True)
    set_global_torch_seed(int(seed), device)

    memory, rng = _load_memory_numpy(args, M, seed)
    targets = torch.from_numpy(memory.astype(np.float64)).to(device)
    noise = torch.from_numpy(
        rng.normal(scale=args.noise_sigma, size=memory.shape).astype(np.float64)
    ).to(device)
    queries = targets + noise

    if use_batch:
        final = update_mhn_batched(
            targets,
            queries,
            beta=beta,
            scale=False,
            normalize=True,
            max_steps=args.max_steps,
            tol=0.01,
        )
        dist = torch.linalg.norm(final - targets, dim=1)
        correct_recall = int((dist <= args.tol).sum().item())
    else:
        correct_recall = 0
        memory_f64 = memory.astype(np.float64)
        for t in range(M):
            target = memory_f64[t]
            query = make_query_from_target_euclidean(target, args.noise_sigma, rng)
            new = update_mhn(query, memory_f64, beta=beta, max_steps=args.max_steps)
            if np.linalg.norm(new - target) <= args.tol:
                correct_recall += 1

    return correct_recall / M


def run_recall_dam(args, M, seed):
    beta = float(getattr(args, "beta", 1.0))
    n_order = int(getattr(args, "n_order", 10))
    device = resolve_torch_device(getattr(args, "device", "cpu"))
    use_batch = getattr(args, "use_batch", True)
    set_global_torch_seed(int(seed), device)

    memory, rng = _load_memory_numpy(args, M, seed)
    targets = torch.from_numpy(memory.astype(np.float64)).to(device)
    noise = torch.from_numpy(
        rng.normal(scale=args.noise_sigma, size=memory.shape).astype(np.float64)
    ).to(device)
    queries = targets + noise

    if use_batch:
        final = update_dam_batched(
            targets,
            queries,
            n_order=n_order,
            beta=beta,
            max_steps=args.max_steps,
            tol=0.01,
        )
        dist = torch.linalg.norm(final - targets, dim=1)
        correct_recall = int((dist <= args.tol).sum().item())
    else:
        correct_recall = 0
        memory_f64 = memory.astype(np.float64)
        for t in range(M):
            target = memory_f64[t]
            query = make_query_from_target_euclidean(target, args.noise_sigma, rng)
            new = update_dam(
                query, memory_f64, n_order=n_order, beta=beta, max_steps=args.max_steps
            )
            if np.linalg.norm(new - target) <= args.tol:
                correct_recall += 1

    return correct_recall / M
