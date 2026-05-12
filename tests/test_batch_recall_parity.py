"""Compare batched vs scalar Euclidean recall on tiny random memory."""

from argparse import Namespace

import numpy as np
import torch

from icml_hyp.recall.baseline_recall import (
    update_mhn,
    update_mhn_batched,
    update_dam,
    update_dam_batched,
    make_query_from_target_euclidean,
)
from icml_hyp.data.sample_memory import sample_hyperboloid_points_from_tangent_ball
from icml_hyp.geom.hyperboloid import HyperboloidKappa
from icml_hyp.recall.memory_recall import (
    identity_phi,
    make_query_from_target,
    update,
    update_karcher_batched,
)


def test_mhn_batched_matches_scalar():
    rng = np.random.default_rng(0)
    M, n = 12, 8
    memory = rng.normal(size=(M, n))
    beta = 1.2
    max_steps = 5
    tol = 0.01

    queries_np = memory + rng.normal(scale=0.3, size=memory.shape)
    mem_t = torch.from_numpy(memory.astype(np.float64))
    q_t = torch.from_numpy(queries_np.astype(np.float64))

    batched = update_mhn_batched(
        mem_t, q_t, beta=beta, max_steps=max_steps, tol=tol
    ).numpy()

    for t in range(M):
        single = update_mhn(
            queries_np[t], memory, beta=beta, max_steps=max_steps
        )
        d = np.linalg.norm(single - batched[t])
        assert d < 1e-6, (t, d)


def test_dam_batched_matches_scalar():
    rng = np.random.default_rng(1)
    M, n = 8, 5
    memory = rng.normal(scale=0.3, size=(M, n))
    beta = 0.5
    n_order = 3
    max_steps = 6
    tol = 0.01

    queries_np = memory + rng.normal(scale=0.05, size=memory.shape)
    mem_t = torch.from_numpy(memory.astype(np.float64))
    q_t = torch.from_numpy(queries_np.astype(np.float64))

    batched = update_dam_batched(
        mem_t,
        q_t,
        n_order=n_order,
        beta=beta,
        max_steps=max_steps,
        tol=tol,
    ).numpy()

    for t in range(M):
        single = update_dam(
            queries_np[t],
            memory,
            n_order=n_order,
            beta=beta,
            max_steps=max_steps,
        )
        d = np.linalg.norm(single - batched[t])
        assert d < 1e-5, (t, d)


def test_run_recall_mhn_batch_vs_no_batch():
    args = Namespace(
        dataset="synthetic",
        d=5,
        mem_R=1.5,
        noise_sigma=0.1,
        beta=1.0,
        max_steps=8,
        tol=0.05,
        device="cpu",
        use_batch=True,
        no_pca=False,
        pca_dim=None,
    )
    from icml_hyp.recall.baseline_recall import run_recall_mhn

    r1 = run_recall_mhn(args, M=15, seed=3)
    args.use_batch = False
    r2 = run_recall_mhn(args, M=15, seed=3)
    assert abs(r1 - r2) < 1e-9


def test_karcher_batched_matches_scalar():
    import geomstats.backend as gs

    geom = HyperboloidKappa(4, -1.0)
    rng = np.random.default_rng(0)
    M = 8
    mem, _ = sample_hyperboloid_points_from_tangent_ball(geom, M, 0.8, rng)
    q_rng = np.random.default_rng(42)
    queries = gs.stack(
        [make_query_from_target(geom, mem[t], 0.15, q_rng) for t in range(M)],
        axis=0,
    )
    q_rng2 = np.random.default_rng(42)
    scalar_stack = gs.stack(
        [
            update(
                geom,
                make_query_from_target(geom, mem[t], 0.15, q_rng2),
                mem,
                phi=identity_phi,
                max_steps=12,
                beta=10.0,
            )
            for t in range(M)
        ],
        axis=0,
    )
    batched = update_karcher_batched(geom, queries, mem, max_steps=12, beta=10.0)
    for t in range(M):
        d = float(geom.metric.dist(batched[t], scalar_stack[t]))
        assert d < 1e-5, (t, d)


def test_run_recall_hyperbolic_batch_vs_no_batch():
    args = Namespace(
        dataset="synthetic",
        d=5,
        mem_R=1.5,
        noise_sigma=0.1,
        beta=10.0,
        max_steps=8,
        tol=0.05,
        kappa=-1,
        device="cpu",
        use_batch=True,
        no_pca=False,
        pca_dim=None,
    )
    from icml_hyp.recall.memory_recall import run_recall_hyperbolic

    r1 = run_recall_hyperbolic(args, "identity", M=12, seed=7)
    args.use_batch = False
    r2 = run_recall_hyperbolic(args, "identity", M=12, seed=7)
    assert abs(r1 - r2) < 1e-9
