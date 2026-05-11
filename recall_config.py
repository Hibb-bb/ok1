"""Shared helpers for recall experiments: image PCA paths, device, seeds."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional


def early_set_geomstats_backend_from_argv() -> None:
    """Call before importing geomstats / memory_recall if hyperbolic on CUDA is needed."""
    if "--device" not in sys.argv:
        return
    i = sys.argv.index("--device")
    if i + 1 >= len(sys.argv):
        return
    dev = sys.argv[i + 1].lower()
    if dev == "cuda" or dev.startswith("cuda:"):
        os.environ["GEOMSTATS_BACKEND"] = "pytorch"


def resolve_pca_dim_images(args: Namespace) -> Optional[int]:
    """None => raw pixels (after R scaling). Otherwise PCA dimension."""
    if getattr(args, "no_pca", False):
        return None
    v = getattr(args, "pca_dim", None)
    if v is not None:
        return int(v)
    return int(getattr(args, "d", 20))


def image_feature_dir(args: Namespace) -> str:
    """Directory segment under dataset/: 'pixels' or 'pca{d}' (not used for synthetic)."""
    pca = resolve_pca_dim_images(args)
    if pca is None:
        return "pixels"
    return f"pca{pca}"


def image_title_dim(args: Namespace) -> str:
    """Short label for plot titles (image runs)."""
    if getattr(args, "dataset", "") not in ("mnist", "cifar10"):
        return str(getattr(args, "d", 20))
    if resolve_pca_dim_images(args) is None:
        d = 784 if args.dataset == "mnist" else 3072
        return f"pixels ({d})"
    return str(resolve_pca_dim_images(args))


def synthetic_csv_plot_paths(args: Namespace) -> tuple[Path, Path]:
    """Output paths for synthetic-dataset capacity runs."""
    base = Path(args.output_dir) / f"dim{args.d}" / f"Radius{args.mem_R}"
    return base / "result.csv", base / "recall_plot.png"


def image_csv_plot_paths(args: Namespace) -> tuple[Path, Path]:
    """Output paths for image-dataset capacity runs ('pixels' or 'pca{d}')."""
    feat = image_feature_dir(args)
    base = Path(args.output_dir) / args.dataset / feat / f"Radius{args.mem_R}"
    return base / "result.csv", base / "recall_plot.png"


def resolve_torch_device(device_str: str) -> "torch.device":
    import torch

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_str.startswith("cuda:"):
        return torch.device(device_str)
    return torch.device("cpu")


def set_global_torch_seed(seed: int, device) -> None:
    import torch

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
