import argparse
import os
from argparse import Namespace
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

import ttavlm.lib as lib
from ttavlm.datasets import return_train_val_datasets, split_open_set_dataset
from ttavlm.methods import return_tta_model
from ttavlm.models import return_base_model
from ttavlm.transforms import TransformList
from ttavlm.vit_osda import train_source


def load_known_class_names(args: argparse.Namespace, transform) -> List[str]:
    if args.source_checkpoint is not None:
        checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
        known_class_names = checkpoint.get("known_class_names")
        if known_class_names is None:
            raise KeyError("source checkpoint must contain known_class_names.")
        return known_class_names

    _, source_dataset = return_train_val_datasets(
        name=args.dataset,
        data_dir=args.dataroot,
        train_transform=transform,
        val_transform=transform,
        shift=args.source_domain,
    )
    _, _, known_class_names = split_open_set_dataset(source_dataset, args.known_class_ratio)
    return known_class_names


def build_cliptta_args(
    args: argparse.Namespace,
    *,
    k_unknown: int,
    max_iter: int,
) -> Namespace:
    return Namespace(
        dataset=args.dataset,
        base_model_name=args.base_model_name,
        save_root=args.save_root,
        adaptation="cliptta",
        update_text=False,
        update_all_params=False,
        optimizer_type=args.optimizer_type,
        steps=args.steps,
        episodic=False,
        logit_scale=args.logit_scale,
        id_score_type="max_prob",
        ood_logit_scale=1.0,
        use_ood_loss=False,
        detect_ood=False,
        score_type="max_prob",
        loss_ood_type="max_inter_var",
        use_weights=False,
        update_alpha=True,
        update_alpha_miss=False,
        alpha=args.alpha,
        beta_tta=1.0,
        beta_reg=0.1,
        beta_ood=args.beta_ood,
        gamma=args.gamma,
        beta_schedule="none",
        milestone=2,
        lr=args.lr,
        lr_miss=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        use_sam=False,
        skip_top_layers=args.skip_top_layers,
        max_iter=max_iter,
        use_batch_stats_only=False,
        distributed=False,
        sample_size=256,
        num_shots=4,
        k_unknown=k_unknown,
        tsne=False,
        measure_collapse=False,
        measure_improvement=False,
        use_softmax_entropy=False,
        use_memory=args.use_memory,
        use_scheduler=False,
        use_tent=False,
        use_clipartt_loss=False,
        K=3,
        clipartt_temp=0.01,
        queue_size=args.queue_size,
        n_neighbors=args.n_neighbors,
        beta_cluster=args.beta_div,
        beta_nl=args.beta_nlinfo,
        beta_clip=args.beta_clip,
        beta_nlcls=args.beta_nlcls,
        beta_nlinfo=args.beta_nlinfo,
        beta_div=args.beta_div,
    )


def adapt(args: argparse.Namespace) -> Dict[str, float]:
    lib.setup_logger()
    lib.fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, base_transform = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=args.dataset,
        path_to_weights=args.save_root,
        segments=0,
    )
    val_transform = TransformList([base_transform])
    known_class_names = load_known_class_names(args, base_transform)
    num_known = len(known_class_names)
    k_unknown = args.k_unknown if args.k_unknown is not None else num_known

    _, target_dataset = return_train_val_datasets(
        name=args.dataset,
        data_dir=args.dataroot,
        train_transform=val_transform,
        val_transform=val_transform,
        shift=args.target_domain,
    )
    target_known, target_private, _ = split_open_set_dataset(target_dataset, args.known_class_ratio)
    known_loader = DataLoader(
        target_known,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )
    private_loader = DataLoader(
        target_private,
        batch_size=args.ood_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )

    tta_args = build_cliptta_args(args, k_unknown=k_unknown, max_iter=len(known_loader))
    tta_model = return_tta_model(
        "cliptta",
        base_model,
        tta_args,
        template=[args.prompt_template],
        class_names=known_class_names,
    )
    lib.LOGGER.info(
        f"Running fused OSDA {args.dataset}: {args.source_domain}->{args.target_domain}, "
        f"known={num_known}, private_prototypes={k_unknown}"
    )
    acc, auc, fpr, oscr = tta_model.get_results(
        known_loader,
        private_loader,
        run_wandb=None,
        trigger_sync=None,
        display_progress=args.display_progress,
    )
    metrics = {"acc": acc, "auroc": auc, "fpr95": fpr, "oscr": oscr}
    lib.LOGGER.info(
        f"{args.dataset} {args.source_domain}->{args.target_domain}: "
        f"acc={acc * 100:.2f}, auroc={auc * 100:.2f}, fpr95={fpr * 100:.2f}, oscr={oscr * 100:.2f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=["officehome", "visda"], required=True)
    common.add_argument("--dataroot", default="/media/fnn/cliptta/ttavlm/data")
    common.add_argument("--save_root", default="/media/fnn/cliptta/ttavlm/result")
    common.add_argument("--known_class_ratio", type=float, default=0.5)
    common.add_argument("--batch_size", type=int, default=64)
    common.add_argument("--ood_batch_size", type=int, default=64)
    common.add_argument("--workers", type=int, default=4)
    common.add_argument("--seed", type=int, default=42)

    source_parser = subparsers.add_parser("source-train", parents=[common])
    source_parser.add_argument("--source_domain", required=True)
    source_parser.add_argument("--epochs", type=int, default=10)
    source_parser.add_argument("--lr", type=float, default=1e-4)
    source_parser.add_argument("--weight_decay", type=float, default=1e-4)
    source_parser.add_argument("--output", default=None)

    adapt_parser = subparsers.add_parser("adapt", parents=[common])
    adapt_parser.add_argument("--source_domain", required=True)
    adapt_parser.add_argument("--target_domain", required=True)
    adapt_parser.add_argument("--source_checkpoint", default=None)
    adapt_parser.add_argument("--base_model_name", default="clip-ViT-B/16")
    adapt_parser.add_argument("--prompt_template", default="A photo of a {}")
    adapt_parser.add_argument("--k_unknown", type=int, default=None)
    adapt_parser.add_argument("--steps", type=int, default=1)
    adapt_parser.add_argument("--lr", type=float, default=1e-3)
    adapt_parser.add_argument("--momentum", type=float, default=0.9)
    adapt_parser.add_argument("--weight_decay", type=float, default=0.0)
    adapt_parser.add_argument("--optimizer_type", choices=["sgd", "adam", "adamw"], default="sgd")
    adapt_parser.add_argument("--logit_scale", type=float, default=100.0)
    adapt_parser.add_argument("--alpha", type=float, default=0.5)
    adapt_parser.add_argument("--gamma", type=float, default=0.005)
    adapt_parser.add_argument("--queue_size", type=int, default=16384)
    adapt_parser.add_argument("--n_neighbors", type=int, default=3)
    adapt_parser.add_argument("--beta_clip", type=float, default=1.0)
    adapt_parser.add_argument("--beta_ood", type=float, default=0.1)
    adapt_parser.add_argument("--beta_nlcls", type=float, default=0.1)
    adapt_parser.add_argument("--beta_nlinfo", type=float, default=0.1)
    adapt_parser.add_argument("--beta_div", type=float, default=0.1)
    adapt_parser.add_argument("--skip_top_layers", action="store_true")
    adapt_parser.add_argument("--use_memory", action="store_true")
    adapt_parser.add_argument("--display_progress", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "source-train":
        train_source(args)
    elif args.command == "adapt":
        adapt(args)
    else:
        raise NotImplementedError(args.command)


if __name__ == "__main__":
    main()
