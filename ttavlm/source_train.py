import argparse
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ttavlm.lib as lib
from ttavlm.datasets import return_train_val_datasets, split_open_set_dataset
from ttavlm.models.clip import tokenize
from ttavlm.models import return_base_model


def collect_visual_norm_params(model: nn.Module) -> List[torch.nn.Parameter]:
    params = []
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            module.requires_grad_(True)
            params.extend([p for p in module.parameters() if p.requires_grad])
    return params


@torch.no_grad()
def build_text_features(clip_model: nn.Module, class_names: List[str], device: torch.device) -> torch.Tensor:
    prompts = [f"A photo of a {name.replace('_', ' ')}" for name in class_names]
    tokens = tokenize(prompts).to(device)
    text_features = clip_model.encode_text(tokens)
    return F.normalize(text_features, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["officehome", "visda"], required=True)
    parser.add_argument("--source_domain", required=True)
    parser.add_argument("--dataroot", default="/media/fnn/cliptta/ttavlm/data")
    parser.add_argument("--save_root", default="/media/fnn/cliptta/ttavlm/result")
    parser.add_argument("--base_model_name", default="clip-ViT-B/16", choices=["clip-ViT-B/32", "clip-ViT-B/16"])
    parser.add_argument("--known_class_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lib.setup_logger()
    lib.fix_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, transform = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=args.dataset,
        path_to_weights=None,
    )

    _, source_dataset = return_train_val_datasets(
        name=args.dataset,
        data_dir=args.dataroot,
        train_transform=transform,
        val_transform=transform,
        shift=args.source_domain,
    )
    source_dataset, _, known_class_names = split_open_set_dataset(source_dataset, args.known_class_ratio)
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )

    visual_model = clip_model.visual
    visual_model.use_local = False
    visual_model.train()
    visual_model.dtype = clip_model.dtype
    params = collect_visual_norm_params(visual_model)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    text_features = build_text_features(clip_model, known_class_names, device).detach()

    for epoch in range(args.epochs):
        total_loss, total_correct, total_count = 0.0, 0.0, 0
        for batch in lib.track(source_loader, f"Source training {args.dataset}/{args.source_domain} epoch {epoch + 1}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["target"].to(device, non_blocking=True)

            image_features = visual_model(images.type(visual_model.dtype))
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ text_features.t()
            loss = F.cross_entropy(logits.float(), labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            total_correct += (logits.argmax(dim=-1) == labels).float().sum().item()
            total_count += labels.numel()

        lib.LOGGER.info(
            f"Source epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / max(total_count, 1):.4f}, "
            f"acc={total_correct / max(total_count, 1):.4f}"
        )

    output = args.output
    if output is None:
        output = os.path.join(
            args.save_root,
            "source",
            args.dataset,
            args.source_domain.replace(" ", "_"),
            f"{args.base_model_name.replace('/', '_')}_known{args.known_class_ratio}_seed{args.seed}.pt",
        )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(
        {
            "visual_state_dict": visual_model.state_dict(),
            "known_class_names": known_class_names,
            "args": vars(args),
        },
        output,
    )
    lib.LOGGER.info(f"Saved source checkpoint to {output}")


if __name__ == "__main__":
    main()
