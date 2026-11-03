"""Self-MM unimodal label generation, using attachment-2 train samples only."""
from __future__ import annotations

import math

import torch


MODES = ("fusion", "text", "audio", "vision")


class UnimodalLabelBank:
    """Epoch-level, bounded version of the paper's ULGM and momentum update.

    Features and labels are indexed solely by the train split. Excluding y=0
    from positive/negative centers respects this contest's neutral definition.
    """

    def __init__(self, true_labels: torch.Tensor, hidden: int):
        self.truth = true_labels.detach().float().cpu().clone()
        self.labels = {mode: self.truth.clone() for mode in MODES}
        self.features = {mode: torch.zeros(len(self.truth), hidden) for mode in MODES}
        self.seen = torch.zeros(len(self.truth), dtype=torch.bool)

    def targets(self, index: torch.Tensor, device):
        index = index.detach().cpu().long()
        return {mode: label[index].to(device) for mode, label in self.labels.items()}

    def begin_epoch(self):
        self.seen.zero_()

    def record(self, index: torch.Tensor, features: dict):
        index = index.detach().cpu().long()
        for mode in MODES:
            self.features[mode][index] = features[mode].detach().float().cpu()
        self.seen[index] = True

    @staticmethod
    def _relative_distance(features: torch.Tensor, labels: torch.Tensor):
        positive = labels > 0
        negative = labels < 0
        if not positive.any() or not negative.any():
            return torch.zeros(len(labels))
        center_pos = features[positive].mean(dim=0)
        center_neg = features[negative].mean(dim=0)
        scale = math.sqrt(2 * features.shape[-1])
        d_pos = torch.linalg.vector_norm(features - center_pos, dim=-1) / scale
        d_neg = torch.linalg.vector_norm(features - center_neg, dim=-1) / scale
        return ((d_neg - d_pos) / (d_pos + 1e-6)).clamp(-4.0, 4.0)

    def update(self, epoch: int):
        if not self.seen.all():
            raise RuntimeError("ULGM feature bank missed train samples")
        if epoch == 1:
            return {"updated": False, "mean_shift": 0.0}
        alpha = {mode: self._relative_distance(self.features[mode], self.labels[mode])
                 for mode in MODES}
        denominator = alpha["fusion"]
        safe = torch.where(denominator.abs() < 0.1,
                           torch.where(denominator < 0, -0.1, 0.1), denominator)
        shifts = []
        for mode in MODES[1:]:
            # Equation 8, with a bounded ratio for near-zero fusion distance.
            ratio = (alpha[mode] / safe).clamp(-3.0, 3.0)
            proposal = (0.5 * self.truth * ratio +
                        0.5 * (self.truth + alpha[mode] - alpha["fusion"]))
            proposal = proposal.clamp(-3.0, 3.0)
            old = self.labels[mode]
            updated = ((epoch - 1) / (epoch + 1) * old +
                       2 / (epoch + 1) * proposal).clamp(-3.0, 3.0)
            shifts.append((updated - old).abs().mean().item())
            self.labels[mode] = updated
        return {"updated": True, "mean_shift": float(sum(shifts) / len(shifts))}
