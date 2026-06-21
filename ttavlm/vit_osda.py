import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ViT_B_16_Weights
from torchvision.transforms.functional import InterpolationMode

import ttavlm.lib as lib
from ttavlm.datasets import return_train_val_datasets, split_open_set_dataset
from ttavlm.models.clip import load as load_clip
from ttavlm.models.clip import tokenize as clip_tokenize


def build_vit_b16(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
    return model


def vit_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_head(model: nn.Module) -> nn.Linear:
    if isinstance(model.heads, nn.Sequential):
        return model.heads[0]
    return model.heads


def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    old_heads = model.heads
    model.heads = nn.Identity()
    model.eval()
    features = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            feats = model(images)
            features.append(F.normalize(feats.float(), dim=-1).cpu())
    model.heads = old_heads
    return torch.cat(features, dim=0)


@torch.no_grad()
def build_known_text_prototypes(
    clip_model: nn.Module,
    class_names: List[str],
    device: torch.device,
    prompt_template: str = "a photo of a {}",
) -> torch.Tensor:
    prompts = [prompt_template.format(name.replace("_", " ")) for name in class_names]
    tokens = clip_tokenize(prompts).to(device)
    text_features = clip_model.encode_text(tokens)
    return F.normalize(text_features.float(), dim=-1)


@torch.no_grad()
def extract_clip_image_features(
    clip_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    clip_model.eval()
    features = []
    for batch in lib.track(loader, "Extract CLIP target features"):
        images = batch["image"].to(device, non_blocking=True)
        image_features = clip_model.encode_image(images.type(clip_model.dtype))
        features.append(F.normalize(image_features.float(), dim=-1).cpu())
    return torch.cat(features, dim=0)


def build_extended_clip_prototypes(
    clip_model: nn.Module,
    target_loader: DataLoader,
    known_class_names: List[str],
    num_private_prototypes: int,
    device: torch.device,
    prompt_template: str = "a photo of a {}",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    known_prototypes = build_known_text_prototypes(
        clip_model=clip_model,
        class_names=known_class_names,
        device=device,
        prompt_template=prompt_template,
    )
    target_features = extract_clip_image_features(clip_model, target_loader, device)
    n_clusters = len(known_class_names) + num_private_prototypes
    if len(target_features) < n_clusters:
        raise ValueError(f"K-means needs at least {n_clusters} samples, got {len(target_features)}.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    centers = torch.from_numpy(kmeans.fit(target_features.numpy()).cluster_centers_).to(device=device, dtype=torch.float32)
    centers = F.normalize(centers, dim=-1)

    similarity = centers @ known_prototypes.t()
    center_idx, class_idx = linear_sum_assignment(-similarity.cpu().numpy())

    matched_by_class = torch.empty(len(known_class_names), dtype=torch.long, device=device)
    matched_by_class[torch.as_tensor(class_idx, device=device)] = torch.as_tensor(center_idx, device=device)

    matched_mask = torch.zeros(len(centers), dtype=torch.bool, device=device)
    matched_mask[matched_by_class] = True
    private_indices = torch.arange(len(centers), device=device)[~matched_mask]
    private_prototypes = centers[private_indices]
    if len(private_prototypes) != num_private_prototypes:
        raise RuntimeError(
            f"Expected {num_private_prototypes} private prototypes, got {len(private_prototypes)}."
        )

    extended_prototypes = torch.cat((known_prototypes, private_prototypes), dim=0)
    return extended_prototypes, similarity, private_indices


def expand_classifier_head(model: nn.Module, num_known: int) -> nn.Module:
    old_head = get_head(model)
    in_features = old_head.in_features
    new_head = nn.Linear(in_features, 2 * num_known).to(old_head.weight.device)
    nn.init.normal_(new_head.weight[num_known:], mean=0.0, std=0.02)
    nn.init.zeros_(new_head.bias[num_known:])
    with torch.no_grad():
        new_head.weight[:num_known].copy_(old_head.weight)
        new_head.bias[:num_known].copy_(old_head.bias)
    model.heads = nn.Sequential(new_head)
    return model


def init_unknown_head_with_kmeans(model: nn.Module, target_loader: DataLoader, num_known: int, device: torch.device) -> None:
    features = extract_features(model, target_loader, device)
    kmeans = KMeans(n_clusters=2 * num_known, random_state=0, n_init=10)
    centers = torch.from_numpy(kmeans.fit(features.numpy()).cluster_centers_).to(device=device, dtype=torch.float32)
    centers = F.normalize(centers, dim=-1)

    head = get_head(model)
    known_weights = F.normalize(head.weight[:num_known].detach().float(), dim=-1)
    similarity = centers @ known_weights.t()
    center_idx, class_idx = linear_sum_assignment(-similarity.cpu().numpy())

    matched = torch.zeros(len(centers), dtype=torch.bool, device=device)
    matched[torch.as_tensor(center_idx, device=device)] = True
    unknown_centers = centers[~matched]
    if len(unknown_centers) < num_known:
        raise RuntimeError(f"K-means produced only {len(unknown_centers)} unmatched centers, expected {num_known}.")

    with torch.no_grad():
        head.weight[num_known:].copy_(unknown_centers[:num_known].to(head.weight.dtype))
        head.bias[num_known:].zero_()


def train_source(args: argparse.Namespace) -> str:
    lib.setup_logger()
    lib.fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = vit_transform()
    _, source_dataset = return_train_val_datasets(
        name=args.dataset,
        data_dir=args.dataroot,
        train_transform=transform,
        val_transform=transform,
        shift=args.source_domain,
    )
    source_dataset, _, known_class_names = split_open_set_dataset(source_dataset, args.known_class_ratio)
    loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
    )

    model = build_vit_b16(len(known_class_names), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for epoch in range(args.epochs):
        total_loss, total_correct, total_count = 0.0, 0.0, 0
        for batch in lib.track(loader, f"ViT source train {args.dataset}/{args.source_domain} epoch {epoch + 1}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["target"].to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            total_correct += (logits.argmax(dim=-1) == labels).float().sum().item()
            total_count += labels.numel()
        lib.LOGGER.info(
            f"epoch={epoch + 1}/{args.epochs}, "
            f"loss={total_loss / max(total_count, 1):.4f}, "
            f"acc={total_correct / max(total_count, 1):.4f}"
        )

    output = args.output
    if output is None:
        source_tag = args.source_domain.replace(" ", "_")
        output = os.path.join(args.save_root, "vit_source", args.dataset, source_tag, f"seed{args.seed}.pt")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_known": len(known_class_names),
            "known_class_names": known_class_names,
            "args": vars(args),
        },
        output,
    )
    lib.LOGGER.info(f"Saved ViT source checkpoint to {output}")
    return output


@torch.no_grad()
def evaluate_osda(model: nn.Module, known_loader: DataLoader, unknown_loader: DataLoader, num_known: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    per_class_correct = torch.zeros(num_known, dtype=torch.float64)
    per_class_total = torch.zeros(num_known, dtype=torch.float64)
    unknown_correct, unknown_total = 0.0, 0.0

    for batch in known_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)
        preds = model(images).argmax(dim=-1)
        for cls in range(num_known):
            cls_mask = labels == cls
            per_class_total[cls] += cls_mask.sum().item()
            per_class_correct[cls] += ((preds == labels) & cls_mask).sum().item()

    for batch in unknown_loader:
        images = batch["image"].to(device, non_blocking=True)
        preds = model(images).argmax(dim=-1)
        unknown_correct += (preds >= num_known).float().sum().item()
        unknown_total += preds.numel()

    valid_classes = per_class_total > 0
    os_star = (per_class_correct[valid_classes] / per_class_total[valid_classes]).mean().item() if valid_classes.any() else 0.0
    unk = unknown_correct / max(unknown_total, 1.0)
    hos = 2 * os_star * unk / (os_star + unk) if (os_star + unk) > 0 else 0.0
    return {"OS*": os_star, "UNK": unk, "HOS": hos}


@torch.no_grad()
def evaluate_clip_prototypes(
    clip_model: nn.Module,
    prototypes: torch.Tensor,
    known_loader: DataLoader,
    unknown_loader: DataLoader,
    num_known: int,
    device: torch.device,
    logit_scale: float,
) -> Dict[str, float]:
    clip_model.eval()
    prototypes = F.normalize(prototypes.float(), dim=-1).to(device)
    per_class_correct = torch.zeros(num_known, dtype=torch.float64)
    per_class_total = torch.zeros(num_known, dtype=torch.float64)
    unknown_correct, unknown_total = 0.0, 0.0

    for batch in known_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)
        features = clip_model.encode_image(images.type(clip_model.dtype))
        features = F.normalize(features.float(), dim=-1)
        preds = (logit_scale * (features @ prototypes.t())).argmax(dim=-1)
        for cls in range(num_known):
            cls_mask = labels == cls
            per_class_total[cls] += cls_mask.sum().item()
            per_class_correct[cls] += ((preds == labels) & cls_mask).sum().item()

    for batch in unknown_loader:
        images = batch["image"].to(device, non_blocking=True)
        features = clip_model.encode_image(images.type(clip_model.dtype))
        features = F.normalize(features.float(), dim=-1)
        preds = (logit_scale * (features @ prototypes.t())).argmax(dim=-1)
        unknown_correct += (preds >= num_known).float().sum().item()
        unknown_total += preds.numel()

    valid_classes = per_class_total > 0
    os_star = (per_class_correct[valid_classes] / per_class_total[valid_classes]).mean().item() if valid_classes.any() else 0.0
    unk = unknown_correct / max(unknown_total, 1.0)
    hos = 2 * os_star * unk / (os_star + unk) if (os_star + unk) > 0 else 0.0
    return {"OS*": os_star, "UNK": unk, "HOS": hos}


def adapt_and_eval(args: argparse.Namespace) -> Dict[str, float]:
    lib.setup_logger()
    lib.fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, transform = load_clip(args.clip_model, device=device)
    clip_model.eval()

    if args.source_checkpoint is not None:
        checkpoint = torch.load(args.source_checkpoint, map_location=device)
        num_known = checkpoint["num_known"]
        known_class_names = checkpoint.get("known_class_names")
        if known_class_names is None:
            raise KeyError("source checkpoint must contain known_class_names for CLIP text prototypes.")
        if len(known_class_names) != num_known:
            raise ValueError(
                f"source checkpoint has num_known={num_known}, but {len(known_class_names)} known_class_names."
            )
    else:
        _, source_dataset = return_train_val_datasets(
            name=args.dataset,
            data_dir=args.dataroot,
            train_transform=transform,
            val_transform=transform,
            shift=args.source_domain,
        )
        _, _, known_class_names = split_open_set_dataset(source_dataset, args.known_class_ratio)
        num_known = len(known_class_names)

    _, target_dataset = return_train_val_datasets(
        name=args.dataset,
        data_dir=args.dataroot,
        train_transform=transform,
        val_transform=transform,
        shift=args.target_domain,
    )
    known_dataset, unknown_dataset, _ = split_open_set_dataset(target_dataset, args.known_class_ratio)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    known_loader = DataLoader(known_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    unknown_loader = DataLoader(unknown_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    num_private = args.num_private_prototypes if args.num_private_prototypes is not None else num_known
    if num_private < 0:
        raise ValueError("num_private_prototypes must be non-negative.")
    extended_prototypes, cluster_text_similarity, private_indices = build_extended_clip_prototypes(
        clip_model=clip_model,
        target_loader=target_loader,
        known_class_names=known_class_names,
        num_private_prototypes=num_private,
        device=device,
        prompt_template=args.prompt_template,
    )
    metrics = evaluate_clip_prototypes(
        clip_model=clip_model,
        prototypes=extended_prototypes,
        known_loader=known_loader,
        unknown_loader=unknown_loader,
        num_known=num_known,
        device=device,
        logit_scale=args.logit_scale,
    )
    lib.LOGGER.info(
        f"Built CLIP extended prototypes: known_text={num_known}, private={num_private}, "
        f"kmeans_clusters={num_known + num_private}, private_centroids={private_indices.cpu().tolist()}"
    )
    lib.LOGGER.info(
        f"Mean matched centroid/text cosine={cluster_text_similarity.max(dim=0).values.mean().item():.4f}"
    )
    lib.LOGGER.info(
        f"{args.dataset} {args.source_domain}->{args.target_domain}: "
        f"OS*={metrics['OS*'] * 100:.2f}, UNK={metrics['UNK'] * 100:.2f}, HOS={metrics['HOS'] * 100:.2f}"
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
    common.add_argument("--workers", type=int, default=4)
    common.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("source-train", parents=[common])
    train_parser.add_argument("--source_domain", required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--output", default=None)

    eval_parser = subparsers.add_parser("adapt-eval", parents=[common])
    eval_parser.add_argument("--source_domain", required=True)
    eval_parser.add_argument("--target_domain", required=True)
    eval_parser.add_argument("--source_checkpoint", default=None)
    eval_parser.add_argument("--clip_model", default="ViT-B/16")
    eval_parser.add_argument("--prompt_template", default="a photo of a {}")
    eval_parser.add_argument("--num_private_prototypes", type=int, default=None)
    eval_parser.add_argument("--logit_scale", type=float, default=100.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "source-train":
        train_source(args)
    elif args.command == "adapt-eval":
        adapt_and_eval(args)
    else:
        raise NotImplementedError(args.command)


if __name__ == "__main__":
    main()
