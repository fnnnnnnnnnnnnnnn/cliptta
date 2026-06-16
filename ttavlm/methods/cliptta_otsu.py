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
from ttavlm.models.clip import tokenize as clip_tokenize
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
        self.beta_cluster = beta_cluster
        self.beta_nl = beta_nl
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
        loss_s_cont, loss_nl = self.compute_loss_tta(image_features)

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
            loss_s_cont_mem, loss_nl_mem = self.compute_loss_tta(image_features, update_bank=False)
            loss_s_cont += loss_s_cont_mem
            loss_nl += loss_nl_mem

        # Final loss
        loss_cluster = self.compute_cluster_loss()
        loss = (
            loss_s_cont
            + self.beta_ood * loss_ood
            + self.beta_cluster * loss_cluster
            + self.beta_nl * loss_nl
        )
        self.last_losses = {
            "s_cont": loss_s_cont.detach(),
            "oce": loss_ood.detach(),
            "cluster": loss_cluster.detach(),
            "nl": loss_nl.detach(),
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
                if self.update_text:
                    class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder)
                else:
                    class_prototypes = self.class_prototypes
                image_features = self.get_features(images)
                logits = self.get_logits(image_features, class_prototypes)
                scores = self.get_known_confidence(self.get_extended_logits(image_features))
        else:
            logits, scores = None, None

        return logits, scores

    def get_known_confidence(self, extended_logits: List[Tensor]) -> Tensor:
        """Maximum known-class probability in the extended class space."""
        known_count = len(self.class_prototypes)
        probabilities = [
            (self.logit_scale * logits).softmax(dim=-1)
            for logits in extended_logits
        ]
        probabilities = torch.stack(probabilities, dim=0).mean(dim=0)
        return probabilities[:, :known_count].max(dim=-1).values

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
        pseudo_probs = torch.nan_to_num(pseudo_probs.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pseudo_probs = pseudo_probs.clamp_min(0.0)
        pseudo_probs = pseudo_probs / pseudo_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
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

        model_probs = F.softmax(self.logit_scale * logits, dim=-1)
        negative_labels = pseudo_probs.argmin(dim=-1)
        negative_probs = model_probs[torch.arange(len(model_probs), device=logits.device), negative_labels]
        return -(1.0 - negative_probs).clamp_min(1e-12).log().mean()

    def compute_loss_tta(self, image_features: List[Tensor], update_bank: bool = True) -> Tensor:
        class_prototypes = self.extended_classification_weights
        logits = image_features[0] @ class_prototypes.t()

        # Soft pseudo-labels over known classes and virtual unknown classes,
        # refined by nearest-neighbor soft voting from the feature bank.
        pseudo_probs = F.softmax(self.logit_scale * logits, dim=-1)
        pseudo_probs = self.refine_pseudo_probs(image_features[0].detach(), pseudo_probs.detach())
        pseudo_labels = pseudo_probs.argmax(dim=-1)
        self.last_pseudo_labels = pseudo_labels.detach()
        if update_bank:
            self.update_feature_bank(image_features[0].detach(), pseudo_probs.detach())

        reliable_mask = self.reliable_sample_mask(
            image_features[0].detach(),
            pseudo_probs.detach(),
            class_prototypes.detach(),
        )
        if not reliable_mask.any():
            zero_loss = logits.sum() * 0.0
            return zero_loss, zero_loss

        selected_features = image_features[0][reliable_mask]
        selected_probs = pseudo_probs[reliable_mask]
        selected_logits = logits[reliable_mask]
        loss_nl = self.compute_negative_learning_loss(selected_logits, selected_probs.detach())

        soft_text_features = F.normalize(selected_probs.to(class_prototypes.dtype) @ class_prototypes, dim=-1)

        # Batch-wise soft contrastive targets: samples with similar class
        # distributions receive larger positive-pair weights.
        soft_targets = selected_probs.detach() @ selected_probs.detach().t()
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        logits_per_image = self.logit_scale * selected_features @ soft_text_features.t()
        logits_per_text = logits_per_image.t()

        # TTA loss
        if self.use_tent:
            loss_s_cont = lib.softmax_entropy(self.logit_scale * selected_logits).mean(0)
        elif self.use_clipartt:
            known_logits = selected_features @ self.class_prototypes.t()
            _, pred = known_logits.topk(self.K, 1, True, True)
            if self.K == 1:
                text_features = self.class_prototypes[pred[:, 0]]
            else:
                text_prompts = lib.getprompt(self.K, pred.cpu().numpy(), self.class_names, self.template[0])
                pred_inputs = clip_tokenize(text_prompts).to(logits.device)

                # With the new prompts, compute the image-to-image and text-to-text similarities to get targets
                with torch.no_grad():
                    text_features = self.clip_text_encoder(pred_inputs)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

            images_similarity = selected_features @ selected_features.t()
            texts_similarity = text_features @ text_features.t()
            targets = F.softmax(((images_similarity + texts_similarity) / 2) / self.clipartt_temp, dim=-1)

            # Obtain new logits (image v.s. new prompt, i.e. size is B x B)
            predictions = (self.logit_scale * text_features @ selected_features.t()).t()
            loss_s_cont = F.cross_entropy(predictions, targets)
        else:
            if self.use_softmax_entropy:
                loss_s_cont = (lib.softmax_entropy(logits_per_image).mean(0) + lib.softmax_entropy(logits_per_text).mean(0)) / 2
            else:
                loss_image = -(soft_targets * F.log_softmax(logits_per_image, dim=-1)).sum(dim=-1).mean()
                loss_text = -(soft_targets * F.log_softmax(logits_per_text, dim=-1)).sum(dim=-1).mean()
                loss_s_cont = (loss_image + loss_text) / 2

        return loss_s_cont, loss_nl

    def _reset_extra(self) -> None:
        if self.use_memory:
            self.memory.reset()
        self.feature_bank_features = self.feature_bank_features[:0]
        self.feature_bank_probs = self.feature_bank_probs[:0]

    def after_adaptation(self, **kwargs: Kwargs) -> None:
        if self.use_scheduler:
            self.scheduler.step()
