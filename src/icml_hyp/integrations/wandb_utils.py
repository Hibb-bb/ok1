"""Shared Weights & Biases helpers for the simulation entry-point scripts."""

from __future__ import annotations

from argparse import Namespace
from typing import Optional


def parse_wandb_tags(tag_str: Optional[str]) -> list[str]:
    """Split a comma-separated tag string into a clean list of tags."""
    if not tag_str:
        return []
    return [t.strip() for t in tag_str.split(",") if t.strip()]


def init_wandb_run(
    args: Namespace,
    *,
    project: str,
    run_name: str,
    extra_tags: list[str],
):
    """Initialize a W&B run if `args.wandb` is set; otherwise return None.

    Common metric definitions used by both capacity and in-class simulations are
    registered here.
    """
    if not getattr(args, "wandb", False):
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError("W&B logging requested but wandb could not be imported") from e

    tags = parse_wandb_tags(getattr(args, "wandb_tags", None))
    tags.extend(extra_tags)

    run = wandb.init(
        project=project,
        entity=getattr(args, "wandb_entity", None),
        group=getattr(args, "wandb_group", None),
        name=run_name,
        tags=tags,
        config=vars(args),
    )

    wandb.define_metric("agg/*", step_metric="M")
    wandb.define_metric("trial/*", step_metric="step")
    return run
