from typing import Dict, Any, Optional, Tuple, List
from typing_extensions import TypeAlias

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from skimage.filters import threshold_otsu as sk_otsu
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.memory import CCM


import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class CLIPTTA(AbstractOpenSetTTAModel):
    """
    CLIPTTA adapts CLIP using the same loss as during the pre-training.
    """

    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        use_memory: bool = False,
        use_softmax_entropy: bool = False,
        use_scheduler: bool = False,
        use_tent: bool = False,
        use_clipartt: bool = False,
        K: int = 3,
        clipartt_temp: Optional[float] = 0.01,
        feature_bank_size: int = 16384,
        n_neighbors: int = 3,
        beta_cluster: float = 0.1,
        beta_nl: float = 0.1,
        beta_clip: float = 1.0,
        beta_nlcls: Optional[float] = None,
        beta_nlinfo: Optional[float] = None,
        beta_div: Optional[float] = None,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.template = template
        self.class_names = class_names
        self.use_memory = use_memory
        self.use_softmax_entropy = use_softmax_entropy
        self.use_scheduler = use_scheduler
        self.use_tent = use_tent
        self.use_clipartt = use_clipartt
        self.K = K
        self.clipartt_temp = clipartt_temp
        self.feature_bank_size = feature_bank_size
        self.n_neighbors = n_neighbors
        self.beta_clip = beta_clip
        self.beta_nlcls = beta_nl if beta_nlcls is None else beta_nlcls
        self.beta_nlinfo = beta_nl if beta_nlinfo is None else beta_nlinfo
        self.beta_div = beta_cluster if beta_div is None else beta_div
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        bank_dim = len(self.extended_classification_weights)
        self.register_buffer("feature_bank_features", torch.empty(0, self.class_prototypes.shape[-1]))
        self.register_buffer("feature_bank_probs", torch.empty(0, bank_dim))

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        self.memory = None
        if use_memory:
            self.memory = CCM(num_shots=self.num_shots, num_classes=len(self.class_names), sample_size=self.sample_size)

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        if self.measure_improvement:
            self.model0 = deepcopy(self.model)
            for param in self.model0.parameters():
                param.detach()

    @torch.enable_grad()
    def _forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
    ) -> Tensor:
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        extended_logits = self.get_extended_logits(image_features)

        scores = self.get_known_confidence(extended_logits)
        loss_ood = self.compute_oce_loss(scores)

        # OOD-ID separation
        if self.detect_ood:
            images, image_features = self.filter_id(images, image_features, scores, self.alpha)

        # Compute TTA losses using samples from the batch
        loss_s_cont, loss_nlcls, loss_nlinfo, loss_div = self.compute_loss_tta(image_features)

        # Compute TTA loss using samples from the memory
        if self.use_memory:
            # Updating Memory
            if step == 0:
                with torch.no_grad():
                    logits = self.get_logits(image_features)
                    _, pred = logits[0].topk(1, 1, True, True)
                    scores = self.get_scores(logits, image_features, score_type=self.id_score_type)

                self.memory.update(images[0].cpu().detach(), pred[:, 0].cpu().detach(), scores.cpu().detach())

            # Sampling from memory
            images, _, _ = self.memory.sample()
            image_features = self.get_features([images.to(image_features[0].device)])
            logits = self.get_logits(image_features)
            loss_s_cont_mem, loss_nlcls_mem, loss_nlinfo_mem, loss_div_mem = self.compute_loss_tta(
                image_features,
                update_bank=False,
            )
            loss_s_cont += loss_s_cont_mem
            loss_nlcls += loss_nlcls_mem
            loss_nlinfo += loss_nlinfo_mem
            loss_div += loss_div_mem

        # Final loss
        loss = (
            self.beta_clip * loss_s_cont
            + self.beta_ood * loss_ood
            + self.beta_nlcls * loss_nlcls
            + self.beta_nlinfo * loss_nlinfo
            + self.beta_div * loss_div
        )
        self.last_losses = {
            "scont_known": loss_s_cont.detach(),
            "oce": loss_ood.detach(),
            "negative_cls": loss_nlcls.detach(),
            "nl_infonce": loss_nlinfo.detach(),
            "div": loss_div.detach(),
            "total": loss.detach(),
        }

        loss.backward()
        return loss

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        _ = self._forward_and_adapt(images, step)

        closure = partial(self._forward_and_adapt, images=images, step=step) if self.use_sam else None

        self.optimizer.step(closure)
        self.optimizer.zero_grad(set_to_none=True)

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                image_features = self.get_features(images)
                logits = self.get_extended_logits(image_features)
                scores = self.get_known_confidence(self.get_extended_logits(image_features))
        else:
            logits, scores = None, None

        return logits, scores

    def get_known_confidence(self, extended_logits: List[Tensor]) -> Tensor:
        """Maximum known-class probability in the extended class space."""
        known_count = len(self.class_prototypes)
        probabilities = [
            self.safe_scaled_logits(logits).softmax(dim=-1)
            for logits in extended_logits
        ]
        probabilities = torch.stack(probabilities, dim=0).mean(dim=0)
        return probabilities[:, :known_count].max(dim=-1).values

    def safe_scaled_logits(self, logits: Tensor, scale: Optional[float] = None) -> Tensor:
        scale = self.logit_scale if scale is None else scale
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        return (scale * logits).clamp(-50.0, 50.0)

    def sanitize_probs(self, probs: Tensor) -> Tensor:
        probs = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        denom = probs.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
        return torch.where(denom > 1e-12, probs / denom.clamp_min(1e-12), uniform)

    def compute_oce_loss(self, scores: Tensor) -> Tensor:
        """Increase the confidence gap between alpha-filtered ID and OOD samples."""
        threshold = self.alpha.detach().to(scores.device)
        id_mask = scores >= threshold
        ood_mask = ~id_mask
        losses = []

        if id_mask.any():
            losses.append(-scores[id_mask].clamp_min(1e-12).log().mean())
        if ood_mask.any():
            losses.append(-(1.0 - scores[ood_mask]).clamp_min(1e-12).log().mean())

        return torch.stack(losses).mean() if losses else scores.sum() * 0.0

    def init_otsu(self, images: List[Tensor]) -> None:
        """Initialize alpha from known confidence in the extended class space."""
        with torch.no_grad():
            image_features = self.get_features(images)
            scores = self.get_known_confidence(self.get_extended_logits(image_features))
            threshold = torch.tensor(
                [sk_otsu(scores.detach().cpu().numpy())],
                device=images[0].device,
            )
            self.alpha.data.copy_(threshold)

    @torch.no_grad()
    def refine_pseudo_probs(self, features: Tensor, pseudo_probs: Tensor) -> Tensor:
        if self.n_neighbors <= 0 or self.feature_bank_features.numel() == 0:
            return pseudo_probs

        features = F.normalize(features.float(), dim=-1)
        bank_features = F.normalize(self.feature_bank_features.float(), dim=-1)
        bank_probs = self.feature_bank_probs.to(pseudo_probs.device, dtype=pseudo_probs.dtype)
        n_neighbors = min(self.n_neighbors, len(bank_features))
        refined_probs = []

        for feats in features.split(64):
            similarity = feats @ bank_features.t()
            neighbor_idx = similarity.topk(n_neighbors, dim=-1, largest=True).indices
            refined_probs.append(bank_probs[neighbor_idx].mean(dim=1))

        refined_probs = torch.cat(refined_probs, dim=0)
        return refined_probs / refined_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def update_feature_bank(self, features: Tensor, pseudo_probs: Tensor) -> None:
        if self.feature_bank_size <= 0:
            return

        features = F.normalize(features.detach().float(), dim=-1)
        pseudo_probs = pseudo_probs.detach().float()
        if self.feature_bank_features.device != features.device:
            self.feature_bank_features = self.feature_bank_features.to(features.device)
            self.feature_bank_probs = self.feature_bank_probs.to(features.device)

        self.feature_bank_features = torch.cat((self.feature_bank_features, features), dim=0)
        self.feature_bank_probs = torch.cat((self.feature_bank_probs, pseudo_probs), dim=0)
        if len(self.feature_bank_features) > self.feature_bank_size:
            self.feature_bank_features = self.feature_bank_features[-self.feature_bank_size:]
            self.feature_bank_probs = self.feature_bank_probs[-self.feature_bank_size:]

    @torch.no_grad()
    def reliable_sample_mask(
        self,
        features: Tensor,
        pseudo_probs: Tensor,
        class_prototypes: Tensor,
    ) -> Tensor:
        num_classes = pseudo_probs.shape[-1]
        pseudo_probs = self.sanitize_probs(pseudo_probs)
        entropy_scale = torch.log(torch.tensor(num_classes, device=pseudo_probs.device, dtype=pseudo_probs.dtype))
        u_nc = -(pseudo_probs * pseudo_probs.clamp_min(1e-12).log()).sum(dim=-1) / entropy_scale.clamp_min(1e-12)

        features = F.normalize(features.float(), dim=-1)
        class_prototypes = F.normalize(class_prototypes.float(), dim=-1)
        distances = (1.0 - features @ class_prototypes.t()).clamp_min(0.0)
        nearest = distances.topk(min(2, num_classes), dim=-1, largest=False).values
        if nearest.shape[-1] == 1:
            u_cs = torch.zeros_like(u_nc)
        else:
            u_cs = nearest[:, 0] / nearest[:, 1].clamp_min(1e-12)

        p_nc = torch.nan_to_num(torch.exp(-u_nc), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        p_cs = torch.nan_to_num(torch.exp(-u_cs), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        mask_nc = torch.bernoulli(p_nc).bool()
        mask_cs = torch.bernoulli(p_cs).bool()
        reliable_mask = mask_nc & mask_cs

        self.last_u_nc = u_nc.detach()
        self.last_u_cs = u_cs.detach()
        self.last_p_nc = p_nc.detach()
        self.last_p_cs = p_cs.detach()
        self.last_reliable_mask = reliable_mask.detach()
        return reliable_mask

    def compute_cluster_loss(self) -> Tensor:
        if self.k_unknown == 0 or self.cluster_centers.numel() == 0:
            return self.unknown_prototypes.sum() * 0.0

        valid_indices = self.unknown_center_indices >= 0
        if not valid_indices.any():
            return self.unknown_prototypes.sum() * 0.0

        unknown_prototypes = F.normalize(self.unknown_prototypes[valid_indices], dim=-1)
        target_centers = F.normalize(
            self.cluster_centers[self.unknown_center_indices[valid_indices]].to(unknown_prototypes.dtype),
            dim=-1,
        )
        alignment_loss = 1.0 - (unknown_prototypes * target_centers).sum(dim=-1).mean()

        if len(unknown_prototypes) <= 1:
            diversity_loss = alignment_loss.new_zeros(())
        else:
            similarity = unknown_prototypes @ unknown_prototypes.t()
            off_diag = ~torch.eye(len(unknown_prototypes), device=similarity.device, dtype=torch.bool)
            diversity_loss = similarity[off_diag].pow(2).mean()

        return alignment_loss + diversity_loss

    def compute_negative_learning_loss(
        self,
        logits: Tensor,
        pseudo_probs: Tensor,
    ) -> Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0

        model_probs = F.softmax(self.safe_scaled_logits(logits), dim=-1)
        negative_labels = pseudo_probs.argmin(dim=-1)
        negative_probs = model_probs[torch.arange(len(model_probs), device=logits.device), negative_labels]
        return -(1.0 - negative_probs).clamp_min(1e-12).log().mean()

    def compute_nl_infonce_loss(
        self,
        features: Tensor,
        pseudo_probs: Tensor,
        known_count: int,
    ) -> Tensor:
        """Unknown samples use private-cluster NL-InfoNCE, not unknown text alignment."""
        if features.shape[0] <= 1 or pseudo_probs.shape[-1] <= known_count:
            return features.sum() * 0.0

        private_probs = self.sanitize_probs(pseudo_probs[:, known_count:])
        private_labels = private_probs.argmax(dim=-1)
        temporal_affinity = private_probs.detach() @ private_probs.detach().t()

        num_samples = features.shape[0]
        off_diag = ~torch.eye(num_samples, device=features.device, dtype=torch.bool)
        positive_mask = off_diag & private_labels[:, None].eq(private_labels[None, :])

        # Temporally similar samples may be false negatives. Exclude them from
        # the negative denominator unless they are explicit private positives.
        false_negative_mask = off_diag & (temporal_affinity > 0.5) & ~positive_mask
        denominator_mask = off_diag & ~false_negative_mask

        positive_weights = temporal_affinity * positive_mask
        valid_anchors = positive_weights.sum(dim=-1) > 1e-12
        if not valid_anchors.any():
            return features.sum() * 0.0

        features = F.normalize(features.float(), dim=-1)
        similarity = self.safe_scaled_logits(features @ features.t())
        similarity = similarity.masked_fill(~denominator_mask, -torch.finfo(similarity.dtype).max)
        log_probs = similarity - torch.logsumexp(similarity, dim=-1, keepdim=True)

        positive_weights = positive_weights / positive_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        losses = -(positive_weights * log_probs).sum(dim=-1)
        return losses[valid_anchors].mean()

    def compute_diversity_loss(self, logits: Tensor) -> Tensor:
        """Encourage the batch prediction average to use the extended space."""
        if logits.numel() == 0:
            return logits.sum() * 0.0

        probs = F.softmax(self.safe_scaled_logits(logits), dim=-1)
        mean_probs = probs.mean(dim=0)
        num_classes = mean_probs.shape[0]
        return (
            mean_probs * (mean_probs.clamp_min(1e-12).log() + torch.log(mean_probs.new_tensor(num_classes)))
        ).sum()

    def compute_loss_tta(self, image_features: List[Tensor], update_bank: bool = True) -> Tensor:
        class_prototypes = self.extended_classification_weights
        known_count = len(self.class_prototypes)
        logits = image_features[0] @ class_prototypes.t()
        loss_div = self.compute_diversity_loss(logits)

        # Soft pseudo-labels over known classes and virtual unknown classes,
        # refined by nearest-neighbor soft voting from the feature bank.
        pseudo_probs = F.softmax(self.safe_scaled_logits(logits), dim=-1)
        pseudo_probs = self.sanitize_probs(self.refine_pseudo_probs(image_features[0].detach(), pseudo_probs.detach()))
        pseudo_labels = pseudo_probs.argmax(dim=-1)
        self.last_pseudo_labels = pseudo_labels.detach()
        if update_bank:
            self.update_feature_bank(image_features[0].detach(), pseudo_probs.detach())

        uncertainty_mask = self.reliable_sample_mask(
            image_features[0].detach(),
            pseudo_probs.detach(),
            class_prototypes.detach(),
        )

        known_probs_for_score = F.softmax(self.safe_scaled_logits(logits[:, :known_count]), dim=-1)
        known_mcm_scores = known_probs_for_score.max(dim=-1).values
        known_id_weights = torch.sigmoid(known_mcm_scores - self.alpha.to(known_mcm_scores.device))
        known_reliable_mask = (
            (known_id_weights > 0.5)
            & (pseudo_labels < known_count)
            & uncertainty_mask
        )
        unknown_reliable_mask = (
            ((known_id_weights <= 0.5) | (pseudo_labels >= known_count))
            & uncertainty_mask
        )

        self.last_known_mcm_scores = known_mcm_scores.detach()
        self.last_known_id_weights = known_id_weights.detach()
        self.last_known_reliable_mask = known_reliable_mask.detach()
        self.last_unknown_reliable_mask = unknown_reliable_mask.detach()

        if uncertainty_mask.any():
            loss_nlcls = self.compute_negative_learning_loss(
                logits[uncertainty_mask],
                pseudo_probs[uncertainty_mask].detach(),
            )
        else:
            loss_nlcls = logits.sum() * 0.0

        if unknown_reliable_mask.any():
            loss_nlinfo = self.compute_nl_infonce_loss(
                image_features[0][unknown_reliable_mask],
                pseudo_probs[unknown_reliable_mask].detach(),
                known_count,
            )
        else:
            loss_nlinfo = logits.sum() * 0.0

        if not known_reliable_mask.any():
            return logits.sum() * 0.0, loss_nlcls, loss_nlinfo, loss_div

        selected_features = image_features[0][known_reliable_mask]
        selected_known_probs = self.sanitize_probs(pseudo_probs[known_reliable_mask, :known_count])
        known_text_prototypes = self.class_prototypes.to(class_prototypes.dtype)
        soft_text_features = F.normalize(
            selected_known_probs.to(class_prototypes.dtype) @ known_text_prototypes,
            dim=-1,
        )

        # Batch-wise soft contrastive targets: samples with similar class
        # distributions receive larger positive-pair weights.
        soft_targets = selected_known_probs.detach() @ selected_known_probs.detach().t()
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        logits_per_image = self.safe_scaled_logits(selected_features @ soft_text_features.t())
        logits_per_text = logits_per_image.t()

        # Known-ID reliable samples use CLIPTTA's batch-aware soft
        # image-text contrastive loss instead of per-sample entropy.
        loss_image = -(soft_targets * F.log_softmax(logits_per_image, dim=-1)).sum(dim=-1).mean()
        loss_text = -(soft_targets * F.log_softmax(logits_per_text, dim=-1)).sum(dim=-1).mean()
        loss_s_cont = (loss_image + loss_text) / 2

        return loss_s_cont, loss_nlcls, loss_nlinfo, loss_div

    def _reset_extra(self) -> None:
        if self.use_memory:
            self.memory.reset()
        self.feature_bank_features = self.feature_bank_features[:0]
        self.feature_bank_probs = self.feature_bank_probs[:0]

    def after_adaptation(self, **kwargs: Kwargs) -> None:
        if self.use_scheduler:
            self.scheduler.step()
