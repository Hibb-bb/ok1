"""
In-class recall: all memory patterns come from one MNIST digit / CIFAR-10 class.
Models: Karcher-Flow (hyperbolic), MHN, DAM.
"""

import recall_config

recall_config.early_set_geomstats_backend_from_argv()

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from hyperboloid import HyperboloidKappa
from memory_recall import (
    map_euclidean_to_hyperboloid,
    generate_image_queries,
    update,
    update_karcher_batched,
    foo as identity_phi,
)
from baseline_recall import (
    update_mhn_batched,
    update_dam_batched,
    update_mhn,
    update_dam,
    make_query_from_target_euclidean,
)
from sample_image_memory import load_images_single_class
from recall_config import (
    image_feature_dir,
    image_title_dim,
    resolve_pca_dim_images,
    resolve_torch_device,
    set_global_torch_seed,
)

warnings.filterwarnings(
    "error",
    message=".*invalid value encountered in sqrt.*",
    category=RuntimeWarning,
)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
MNIST_CLASSES = [str(d) for d in range(10)]


def get_args():
    ap = argparse.ArgumentParser(
        description="In-class recall (MNIST / CIFAR-10, PCA fit per class)."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="outputs")
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="PCA dimension; defaults to --d when omitted.",
    )
    ap.add_argument("--d", type=int, default=50)
    ap.add_argument(
        "--no-pca",
        action="store_true",
        help="Raw pixels after R scaling (no PCA).",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no-batch", action="store_true")
    ap.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Class 0-9. Use with --all-classes omitted.",
    )
    ap.add_argument(
        "--all-classes",
        action="store_true",
        help="Run classes 0..9; one CSV + one plot per class.",
    )
    ap.add_argument("--M-min", type=int, default=10)
    ap.add_argument("--M-max", type=int, default=500)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--mem-R", type=float, default=2.0)
    ap.add_argument("--kappa", type=float, default=-1.0)
    ap.add_argument("--noise_sigma", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-order", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.001)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    args.use_batch = not args.no_batch
    return args


def output_paths(args, class_id: int):
    feat = image_feature_dir(args)
    base = (
        Path(args.output_dir)
        / args.dataset
        / "inclass"
        / f"class{class_id}"
        / feat
        / f"Radius{args.mem_R}"
    )
    return base / "result.csv", base / "recall_plot.png"


def class_label(dataset: str, class_id: int) -> str:
    if dataset == "mnist":
        return MNIST_CLASSES[class_id]
    return CIFAR10_CLASSES[class_id]


def run_one_trial_mhn(args, class_id, M, seed):
    rng = np.random.default_rng(seed)
    pca = resolve_pca_dim_images(args)
    images, _ = load_images_single_class(
        args.dataset, class_id, M, pca_dim=pca, R=args.mem_R, rng=rng
    )
    device = resolve_torch_device(args.device)
    set_global_torch_seed(int(seed), device)

    if args.use_batch:
        mem = torch.from_numpy(images.astype(np.float64)).to(device)
        noise = torch.from_numpy(
            rng.normal(scale=args.noise_sigma, size=images.shape).astype(np.float64)
        ).to(device)
        queries = mem + noise
        final = update_mhn_batched(
            mem, queries, beta=args.beta, max_steps=args.max_steps, tol=0.01
        )
        dist = torch.linalg.norm(final - mem, dim=1)
        return float((dist <= args.tol).float().mean())
    correct = 0
    memory = images.astype(np.float64)
    for t in range(M):
        target = memory[t]
        query = make_query_from_target_euclidean(target, args.noise_sigma, rng)
        new = update_mhn(query, memory, beta=args.beta, max_steps=args.max_steps)
        if np.linalg.norm(new - target) <= args.tol:
            correct += 1
    return correct / M


def run_one_trial_dam(args, class_id, M, seed):
    rng = np.random.default_rng(seed)
    pca = resolve_pca_dim_images(args)
    images, _ = load_images_single_class(
        args.dataset, class_id, M, pca_dim=pca, R=args.mem_R, rng=rng
    )
    device = resolve_torch_device(args.device)
    set_global_torch_seed(int(seed), device)

    if args.use_batch:
        mem = torch.from_numpy(images.astype(np.float64)).to(device)
        noise = torch.from_numpy(
            rng.normal(scale=args.noise_sigma, size=images.shape).astype(np.float64)
        ).to(device)
        queries = mem + noise
        final = update_dam_batched(
            mem,
            queries,
            n_order=args.n_order,
            beta=args.beta,
            max_steps=args.max_steps,
            tol=0.01,
        )
        dist = torch.linalg.norm(final - mem, dim=1)
        return float((dist <= args.tol).float().mean())
    correct = 0
    memory = images.astype(np.float64)
    for t in range(M):
        target = memory[t]
        query = make_query_from_target_euclidean(target, args.noise_sigma, rng)
        new = update_dam(
            query,
            memory,
            n_order=args.n_order,
            beta=args.beta,
            max_steps=args.max_steps,
        )
        if np.linalg.norm(new - target) <= args.tol:
            correct += 1
    return correct / M


def run_one_trial_hyperbolic(args, class_id, M, seed):
    rng = np.random.default_rng(seed)
    pca = resolve_pca_dim_images(args)
    images, _ = load_images_single_class(
        args.dataset, class_id, M, pca_dim=pca, R=args.mem_R, rng=rng
    )
    euc = images.astype(np.float64)
    geometry = HyperboloidKappa(dim=euc.shape[1], curvature=args.kappa)
    hyp_dev = resolve_torch_device(args.device)
    set_global_torch_seed(int(seed), hyp_dev)

    if str(hyp_dev) != "cpu":
        me_t = torch.from_numpy(euc).to(device=hyp_dev, dtype=torch.float64)
        memory = map_euclidean_to_hyperboloid(geometry, me_t)
        queries = generate_image_queries(geometry, me_t, args.noise_sigma, rng)
    else:
        memory = map_euclidean_to_hyperboloid(geometry, euc)
        queries = generate_image_queries(geometry, euc.copy(), args.noise_sigma, rng)

    correct = 0
    except_count = 0
    ran_batch = False
    if args.use_batch and str(hyp_dev) == "cpu":
        try:
            final = update_karcher_batched(
                geometry,
                queries,
                memory,
                max_steps=args.max_steps,
            )
            for t in range(M):
                dist = geometry.metric.dist(final[t], memory[t])
                if hasattr(dist, "item"):
                    ok = float(dist.item()) < args.tol
                else:
                    ok = float(dist) < args.tol
                if ok:
                    correct += 1
            ran_batch = True
        except Exception:
            pass

    if not ran_batch:
        for t in range(M):
            target = memory[t]
            query = queries[t]
            try:
                new = update(
                    geometry, query, memory, phi=identity_phi, max_steps=args.max_steps
                )
                dist = geometry.metric.dist(new, target)
                if hasattr(dist, "item"):
                    ok = float(dist.item()) < args.tol
                else:
                    ok = float(dist) < args.tol
                if ok:
                    correct += 1
            except Exception:
                except_count += 1
    if except_count:
        print(f"  [hyperbolic] {except_count}/{M} exceptions at M={M}")
    return correct / M


def make_plot(df, args, class_id: int, plot_path: Path):
    palette = {"MHN": "#37373E", "DAM": "#86928B", "Karcher-Flow": "#FF6B6B"}
    label = class_label(args.dataset, class_id)
    dim_l = image_title_dim(args)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="M",
        y="recall rate",
        hue="model",
        errorbar="sd",
        marker="o",
        markersize=10,
        alpha=0.7,
        palette=palette,
        err_style="bars",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title(
        f"In-class recall — {args.dataset}, class {class_id} ({label}), d={dim_l}, R={args.mem_R}"
    )
    ax.set_xlabel("M")
    ax.set_ylabel("Recall rate")
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs="all"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(title="Model")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout(pad=0.1)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=480)
    plt.close(fig)


def run_simulation(args, class_id: int, logger):
    csv_path, plot_path = output_paths(args, class_id)
    label = class_label(args.dataset, class_id)
    logger.info("Class %s (%s)", class_id, label)

    K = 15
    r = (args.M_max / args.M_min) ** (1 / K)
    M_values = np.round(args.M_min * r ** np.arange(K + 1)).astype(int)

    result_data = {"model": [], "recall rate": [], "M": []}
    for M in M_values:
        logger.info("  M=%s", M)
        for i in range(args.n_trials):
            s = int(i + M)
            result_data["model"].append("MHN")
            result_data["recall rate"].append(run_one_trial_mhn(args, class_id, M, s))
            result_data["M"].append(M)

            result_data["model"].append("DAM")
            result_data["recall rate"].append(run_one_trial_dam(args, class_id, M, s))
            result_data["M"].append(M)

            result_data["model"].append("Karcher-Flow")
            result_data["recall rate"].append(
                run_one_trial_hyperbolic(args, class_id, M, s)
            )
            result_data["M"].append(M)

    df = pd.DataFrame(result_data)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("  CSV -> %s", csv_path)
    make_plot(df, args, class_id, plot_path)
    logger.info("  plot -> %s", plot_path)


def run_replot(args, class_id: int):
    csv_path, plot_path = output_paths(args, class_id)
    if not csv_path.is_file():
        print(f"Missing CSV: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    make_plot(df, args, class_id, plot_path)
    print(f"Wrote {plot_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = get_args()

    sns.set_theme(
        rc={
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "legend.fontsize": 16,
            "legend.title_fontsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "font.family": "serif",
            "font.serif": ["Courier New"],
        }
    )
    sns.set_style("whitegrid")

    if args.all_classes:
        class_ids = list(range(10))
    elif args.class_id is not None:
        class_ids = [args.class_id]
    else:
        raise SystemExit("Specify --class-id N or --all-classes.")

    for cid in class_ids:
        if args.replot:
            run_replot(args, cid)
        else:
            run_simulation(args, cid, logger)


if __name__ == "__main__":
    main()
