"""
CIFAR-10 reconstruction demo in the style of Theory_Associative_Memory’s
``generate_demonstration_reconstructions``: one random memory, several noise
levels (columns), rows = ground truth | noisy query | MHN | DAM | Karcher.

Uses ``baseline_recall.update_mhn`` / ``update_dam`` and
``memory_recall.update`` + ``map_euclidean_to_hyperboloid`` (identity / Karcher
score) for the hyperbolic column.
"""

import argparse
import os
import sys

import numpy as np


def _early_set_geomstats_backend_from_argv():
    if "--device" not in sys.argv:
        return
    i = sys.argv.index("--device")
    if i + 1 >= len(sys.argv):
        return
    dev = sys.argv[i + 1].lower()
    if dev == "cuda" or dev.startswith("cuda:"):
        os.environ["GEOMSTATS_BACKEND"] = "pytorch"


_early_set_geomstats_backend_from_argv()

import geomstats.backend as gs
import matplotlib.pyplot as plt

from icml_hyp.recall.baseline_recall import update_dam, update_mhn
from icml_hyp.geom.hyperboloid import HyperboloidKappa
from icml_hyp.recall.memory_recall import map_euclidean_to_hyperboloid, update as karcher_update
from icml_hyp.data.sample_image_memory import load_cifar10_images

SAVE_FORMAT = "png"


def _features_to_rgb(x, pca, R):
    """(dim,) -> (32, 32, 3) in [0, 1] for imshow."""
    v = np.asarray(x, dtype=np.float64).reshape(1, -1)
    if pca is not None:
        flat = pca.inverse_transform(v)
    else:
        flat = v
    if flat.shape[1] != 3072:
        raise ValueError("Expected 3072 channels*H*W after inverse PCA")
    rgb = flat.reshape(3, 32, 32).transpose(1, 2, 0)
    return np.clip(rgb / float(R), 0.0, 1.0)


def _hyperboloid_to_euclidean(geometry, point_gs):
    """Single hyperboloid point -> (dim,) feature vector (log at origin, spatial)."""
    d = geometry.dim
    p = np.asarray(point_gs, dtype=np.float64).reshape(d + 1)
    origin = np.zeros((1, d + 1), dtype=np.float64)
    origin[0, 0] = geometry.radius
    tv = geometry.metric.log(gs.array(p[None, :]), gs.array(origin))
    return np.asarray(tv, dtype=np.float64)[0, 1:]


def _karcher_recon_euclidean(geometry, query_euc, memory_euc, max_steps, beta):
    mem_h = map_euclidean_to_hyperboloid(geometry, memory_euc)
    q_h = map_euclidean_to_hyperboloid(geometry, np.atleast_2d(query_euc))
    final_h = karcher_update(
        geometry, q_h[0], mem_h, max_steps=max_steps, beta=beta
    )
    return _hyperboloid_to_euclidean(geometry, final_h)


def generate_demonstration_reconstructions(
    memory,
    N,
    *,
    perturb_vals,
    rng,
    pca=None,
    mem_R=2.0,
    beta=10.0,
    n_order=10,
    max_steps=10,
    kappa=-1.0,
    sname="cifar_recon_demo.png",
    img_idx=None,
    show=True,
):
    """
    memory: (>=N, dim) float — CIFAR features (PCA or raw pixels).
    perturb_vals: noise std for Gaussian perturbation in feature space.
    """
    X = np.asarray(memory[:N], dtype=np.float64)
    d = X.shape[1]
    if img_idx is None:
        img_idx = int(rng.integers(N))
    init = X[img_idx].copy()
    show_init = _features_to_rgb(init, pca, mem_R)

    geometry = HyperboloidKappa(dim=d, curvature=float(kappa))

    perturbed_rgb = []
    mhn_rgb = []
    dam_rgb = []
    k_rgb = []

    for sigma in perturb_vals:
        query = init + rng.normal(scale=float(sigma), size=init.shape)
        perturbed_rgb.append(_features_to_rgb(query, pca, mem_R))

        mhn_out = update_mhn(query, X, beta=beta, max_steps=max_steps)
        dam_out = update_dam(query, X, n_order=n_order, beta=beta, max_steps=max_steps)
        k_out = _karcher_recon_euclidean(geometry, query, X, max_steps, beta)

        mhn_rgb.append(_features_to_rgb(mhn_out, pca, mem_R))
        dam_rgb.append(_features_to_rgb(dam_out, pca, mem_R))
        k_rgb.append(_features_to_rgb(k_out, pca, mem_R))

    n_vals = len(perturb_vals)
    nrow, ncol = 5, n_vals
    fig, ax_array = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol + 1, nrow + 1),
        gridspec_kw={
            "wspace": 0,
            "hspace": 0,
            "top": 1.0 - 0.5 / (nrow + 1),
            "bottom": 0.5 / (nrow + 1),
            "left": 0.5 / (ncol + 1),
            "right": 1.0 - 0.5 / (ncol + 1),
        },
    )
    if ncol == 1:
        ax_array = ax_array.reshape(nrow, 1)

    row_imgs = [
        [show_init] * n_vals,
        perturbed_rgb,
        mhn_rgb,
        dam_rgb,
        k_rgb,
    ]
    row_labels = ["GT", "Noisy", "MHN", "DAM", "Karcher"]

    for i in range(nrow):
        for j in range(ncol):
            ax = ax_array[i, j]
            ax.imshow(row_imgs[i][j])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(sname, format=SAVE_FORMAT, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--N", type=int, default=100, help="Memory bank size (random subset)")
    p.add_argument("--mem-R", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=100.0)
    p.add_argument("--n-order", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--kappa", type=float, default=-1.0)
    p.add_argument("--no-pca", action="store_true")
    p.add_argument("--pca-dim", type=int, default=None)
    p.add_argument("--d", type=int, default=20, help="PCA dim if --pca-dim omitted")
    p.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.35, 0.5],
        help="Gaussian noise stds in feature space (columns)",
    )
    p.add_argument("--img-idx", type=int, default=None, help="Fixed memory index")
    p.add_argument("--out", type=str, default="outputs/vis/cifar_recon_demo.png")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Only save the figure (no interactive window)",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.no_pca:
        pca_dim = None
    elif args.pca_dim is not None:
        pca_dim = args.pca_dim
    else:
        pca_dim = args.d

    memory, pca = load_cifar10_images(
        args.N, pca_dim=pca_dim, rng=rng, R=float(args.mem_R)
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    generate_demonstration_reconstructions(
        memory,
        args.N,
        perturb_vals=list(args.sigma),
        rng=rng,
        pca=pca,
        mem_R=args.mem_R,
        beta=args.beta,
        n_order=args.n_order,
        max_steps=args.max_steps,
        kappa=args.kappa,
        sname=args.out,
        img_idx=args.img_idx,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
