import numpy as np
import geomstats.backend as gs
from scipy.special import softmax

import torch

from icml_hyp.geom.hyperboloid import HyperboloidKappa
from icml_hyp.data.sample_memory import sample_hyperboloid_points_from_tangent_ball
from icml_hyp.config.recall_config import (
    resolve_pca_dim_images,
    resolve_torch_device,
    set_global_torch_seed,
)


def identity_phi(x, kappa=None):
    return -x


def geo_distance(d):
    return d


def square_distance(d):
    return d**2


def _gs_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax_weights(score):
    """Softmax over axis/dim 0; works on both torch tensors and numpy arrays."""
    if isinstance(score, torch.Tensor):
        return torch.softmax(score, dim=0)
    return softmax(np.asarray(score), axis=0)


def update(geometry, query, memory, phi=identity_phi, max_steps=10, beta=10.0):
    query = geometry.regularize(query)
    memory = geometry.regularize(memory)
    tol = 0.01
    beta = float(beta)

    for step in range(max_steps):

        tangent_memory = geometry.metric.log(memory, base_point=query)

        if phi == identity_phi:
            lorentz_inner = geometry.embedding_space.metric.inner_product(query, memory)
            score = phi(lorentz_inner, geometry.curvature)

        else:
            score = phi(geometry.metric.dist(query, memory))

        weights = _softmax_weights(beta * score)

        if isinstance(weights, torch.Tensor):
            max_weight_idx = int(torch.argmax(weights).item())
            max_weight = float(weights[max_weight_idx].item())
            tangent_query = weights @ tangent_memory
        else:
            max_weight_idx = int(np.argmax(weights))
            max_weight = float(weights[max_weight_idx])
            tangent_query = weights @ _gs_np(tangent_memory)

        if max_weight >= 1 - tol:
            return memory[max_weight_idx]

        query = geometry.metric.exp(tangent_query, query)

    return query


def update_karcher_batched(
    geometry,
    queries,
    memory,
    max_steps: int = 10,
    tol: float = 0.01,
    beta: float = 10.0,
):
    """
    Batched Karcher-flow dynamics with identity score (negative Lorentz inner product).

    queries: (B, dim+1), memory: (M, dim+1), gs-arrays (NumPy backend recommended).
    Row j uses the same fixed memory set; typically B=M and query j is a noisy copy of
    memory[j] (as in recall experiments).

    Returns gs-array (B, dim+1) of terminal points.
    """
    beta = float(beta)
    B = int(gs.shape(queries)[0])
    Mdim = int(gs.shape(memory)[0])
    d1 = int(gs.shape(queries)[1])

    Q = geometry.regularize(gs.array(queries))
    mem = geometry.regularize(memory)
    converged = np.zeros(B, dtype=bool)
    mem_np = np.asarray(mem, dtype=np.float64)
    out_np = np.asarray(Q, dtype=np.float64).copy()

    for _ in range(max_steps):
        m_flat = gs.tile(mem, (B, 1))
        q_flat = gs.repeat(Q, Mdim, axis=0)
        tangent_flat = geometry.metric.log(m_flat, base_point=q_flat)
        tangent_bm = gs.reshape(tangent_flat, (B, Mdim, d1))
        ip = geometry.embedding_space.metric.inner_product(q_flat, m_flat)
        inner_mb = gs.reshape(ip, (B, Mdim)).T
        score_mb = -inner_mb
        w = _softmax_weights(beta * score_mb)
        w_np = np.asarray(w, dtype=np.float64)
        max_w = w_np.max(axis=0)
        argmax = np.argmax(w_np, axis=0)
        newly = max_w >= (1.0 - tol)
        converged |= newly
        out_np[newly] = mem_np[argmax[newly]]
        if converged.all():
            break

        tangent_q = gs.einsum("mb,bmd->bd", w, tangent_bm)
        Q = geometry.regularize(geometry.metric.exp(tangent_q, Q))
        Q_np = np.asarray(Q, dtype=np.float64)
        Q_np[converged] = out_np[converged]
        Q = gs.array(Q_np)

    Q_final_np = np.asarray(Q, dtype=np.float64)
    fin = out_np.copy()
    fin[~converged] = Q_final_np[~converged]
    return gs.array(fin)


def make_query_from_target(geometry, target, sigma, rng):
    d1 = geometry.dim + 1
    eps = rng.normal(scale=sigma, size=(d1,))
    eps = gs.array(eps)
    eps = geometry.to_tangent(eps, target)
    return geometry.metric.exp(eps, target)


def map_euclidean_to_hyperboloid(geometry, euclidean_vectors):
    M, dim = euclidean_vectors.shape
    assert dim == geometry.dim, f"Dimension mismatch: {dim} != {geometry.dim}"

    if isinstance(euclidean_vectors, torch.Tensor):
        dev, dt = euclidean_vectors.device, euclidean_vectors.dtype
        tangent_np = torch.zeros(M, dim + 1, device=dev, dtype=dt)
        tangent_np[:, 1:] = euclidean_vectors
        tangent_vectors = gs.array(tangent_np)
        origin = torch.zeros(M, dim + 1, device=dev, dtype=dt)
        origin[:, 0] = geometry.radius
        origin_batch = gs.array(origin)
    else:
        tangent_vectors = np.zeros((M, dim + 1), dtype=float)
        tangent_vectors[:, 1:] = np.asarray(euclidean_vectors, dtype=float)
        tangent_vectors = gs.array(tangent_vectors)
        origin_batch = gs.zeros((M, dim + 1))
        origin_batch[:, 0] = geometry.radius
    hyperboloid_points = geometry.metric.exp(tangent_vectors, origin_batch)
    hyperboloid_points = geometry.regularize(hyperboloid_points)

    return hyperboloid_points


def generate_image_queries(geometry, euclidean_vectors, sigma, rng):
    if isinstance(euclidean_vectors, torch.Tensor):
        M, dim = euclidean_vectors.shape
        dev, dt = euclidean_vectors.device, euclidean_vectors.dtype
        noise = torch.from_numpy(
            rng.normal(scale=sigma, size=(M, dim)).astype(np.float64)
        ).to(device=dev, dtype=dt)
        euc_noisy = euclidean_vectors + noise
        tangent_np = torch.zeros(M, dim + 1, device=dev, dtype=dt)
        tangent_np[:, 1:] = euc_noisy
        tangent_vectors = gs.array(tangent_np)
        origin = torch.zeros(M, dim + 1, device=dev, dtype=dt)
        origin[:, 0] = geometry.radius
        origin_batch = gs.array(origin)
    else:
        M, dim = euclidean_vectors.shape
        euc = np.asarray(euclidean_vectors, dtype=np.float64) + rng.normal(
            scale=sigma, size=(M, dim)
        )
        assert dim == geometry.dim, f"Dimension mismatch: {dim} != {geometry.dim}"
        tangent_vectors = np.zeros((M, dim + 1), dtype=float)
        tangent_vectors[:, 1:] = euc
        tangent_vectors = gs.array(tangent_vectors)
        origin_batch = gs.zeros((M, dim + 1))
        origin_batch[:, 0] = geometry.radius
    hyperboloid_points = geometry.metric.exp(tangent_vectors, origin_batch)
    hyperboloid_points = geometry.regularize(hyperboloid_points)
    return hyperboloid_points


def run_recall_hyperbolic(args, phi_choice, M, seed):
    if phi_choice == "identity":
        phi = identity_phi
    elif phi_choice == "geo_distance":
        phi = geo_distance
    elif phi_choice == "square_distance":
        phi = square_distance
    else:
        raise ValueError(f"Unknown phi: {phi_choice}")

    beta = float(getattr(args, "beta", 10.0))

    req_dev = resolve_torch_device(getattr(args, "device", "cpu"))
    set_global_torch_seed(int(seed), req_dev)

    rng = np.random.default_rng(seed)

    dataset = getattr(args, "dataset", "synthetic")
    # NumPy-based sampler; run hyperbolic dynamics on CPU for synthetic.
    hyp_dev = torch.device("cpu") if dataset == "synthetic" else req_dev

    if dataset == "synthetic":
        dim = args.d
        geometry = HyperboloidKappa(dim=dim, curvature=args.kappa)
        memory, _ = sample_hyperboloid_points_from_tangent_ball(
            geometry, M, args.mem_R, rng
        )
        queries_gs = None
    elif dataset in ["mnist", "cifar10"]:
        from icml_hyp.data.sample_image_memory import sample_images_from_dataset

        requested_dim = resolve_pca_dim_images(args)
        memory_euclidean, _ = sample_images_from_dataset(
            dataset_name=dataset, M=M, dim=requested_dim, R=float(args.mem_R), rng=rng
        )
        actual_dim = memory_euclidean.shape[1]
        geometry = HyperboloidKappa(dim=actual_dim, curvature=args.kappa)
        me = memory_euclidean.astype(np.float64)
        if str(hyp_dev) != "cpu":
            me_t = torch.from_numpy(me).to(device=hyp_dev, dtype=torch.float64)
            memory = map_euclidean_to_hyperboloid(geometry, me_t)
            queries_gs = generate_image_queries(
                geometry, me_t, args.noise_sigma, rng
            )
        else:
            memory = map_euclidean_to_hyperboloid(geometry, me)
            queries_gs = generate_image_queries(geometry, me, args.noise_sigma, rng)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    use_hyp_batch = (
        phi_choice == "identity"
        and getattr(args, "use_batch", True)
        and str(hyp_dev) == "cpu"
    )

    correct_recall = 0
    except_count = 0
    ran_hyp_batch = False

    if use_hyp_batch:
        if dataset == "synthetic":
            qrows = [
                make_query_from_target(geometry, memory[t], args.noise_sigma, rng)
                for t in range(M)
            ]
            Q0 = gs.stack(qrows, axis=0)
        else:
            Q0 = queries_gs
        try:
            final = update_karcher_batched(
                geometry,
                Q0,
                memory,
                max_steps=args.max_steps,
                beta=beta,
            )
            for t in range(M):
                target = memory[t]
                dist = geometry.metric.dist(final[t], target)
                if isinstance(dist, torch.Tensor):
                    ok = float(dist.item()) < args.tol
                else:
                    ok = float(dist) < args.tol
                if ok:
                    correct_recall += 1
            ran_hyp_batch = True
        except Exception:
            pass

    if not ran_hyp_batch:
        for t in range(M):
            target = memory[t]

            if dataset == "synthetic":
                query = make_query_from_target(geometry, target, args.noise_sigma, rng)
            else:
                query = queries_gs[t]

            try:
                new = update(
                    geometry,
                    query,
                    memory,
                    phi=phi,
                    max_steps=args.max_steps,
                    beta=beta,
                )
                dist = geometry.metric.dist(new, target)
                if isinstance(dist, torch.Tensor):
                    ok = float(dist.item()) < args.tol
                else:
                    ok = float(dist) < args.tol
                if ok:
                    correct_recall += 1
            except Exception:
                except_count += 1
                continue
    if except_count != 0:
        print("Exception count: ", except_count, "/", M)
    return correct_recall / M
