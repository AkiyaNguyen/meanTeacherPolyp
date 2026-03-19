"""
Loss modules for segmentation and consistency training.

Standard interface: pred and target/mask are probabilities in [0, 1]
(same shape). SoftmaxMSELoss is for consistency (input/target can be logits or probs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


## this is just mseloss 
class SoftmaxMSELoss(nn.Module):
    """MSE between logits (e.g. student vs teacher predictions)."""

    def __init__(self):
        super().__init__()

    def forward(self, input_prob: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        num_classes = input_prob.size(1)
        return F.mse_loss(input_prob, target_prob, reduction='mean') / num_classes


class DiceLoss(nn.Module):
    """Dice loss for segmentation (1 - dice score, averaged over batch)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        size = pred.size(0)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score / size


class MaskBCELoss(nn.Module):
    """BCE restricted to confident (high/low) target pixels."""

    def __init__(self, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        mask = ((target > self.threshold) | (target < 1 - self.threshold)).float()
        pred = pred * mask
        target = target * mask
        return self.bce(pred, target)


class BCEDiceLoss(nn.Module):
    """Sum of masked BCE and Dice loss."""

    def __init__(self, bce_threshold: float = 0.95, dice_smooth: float = 1.0):
        super().__init__()
        self.mask_bce = MaskBCELoss(threshold=bce_threshold)
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mask_bce(pred, target) + self.dice(pred, target)


class MinimizeFeatureSimilarityLoss(nn.Module):
    """Cosine-similarity-based loss between two feature maps (GAP then normalize)."""

    def __init__(self):
        super().__init__()

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        fea1 = F.adaptive_avg_pool2d(feat1, 1).flatten(1)
        fea2 = F.adaptive_avg_pool2d(feat2, 1).flatten(1)
        fea1 = F.normalize(fea1, dim=1)
        fea2 = F.normalize(fea2, dim=1)
        sim = F.cosine_similarity(fea1, fea2, dim=1)
        return (1 + sim).mean()
class MaximizeFeatureSimilarityLoss(nn.Module):
    """Cosine-similarity-based loss between two feature maps (GAP then normalize)."""

    def __init__(self):
        super().__init__()

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        fea1 = F.adaptive_avg_pool2d(feat1, 1).flatten(1)
        fea2 = F.adaptive_avg_pool2d(feat2, 1).flatten(1)
        fea1 = F.normalize(fea1, dim=1)
        fea2 = F.normalize(fea2, dim=1)
        sim = F.cosine_similarity(fea1, fea2, dim=1)
        return (1 - sim).mean()

class StructureLoss(nn.Module):
    """Weighted BCE + weighted IoU on structure (edge-weighted by avg_pool distance from mask).
    Interface: pred and mask are both probabilities in [0, 1] (same as other losses).
    """

    def __init__(self, kernel_size: int = 31, padding: int = 15, weight_scale: float = 5.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight_scale = weight_scale

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred, mask: probabilities [0, 1]
        weit = 1 + self.weight_scale * torch.abs(
            F.avg_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=self.padding) - mask
        )
        wbce = F.binary_cross_entropy(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

class L2Loss(nn.Module):
    """L2 loss between two feature maps (GAP then normalize)."""
    def __init__(self):
        super().__init__()

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        return torch.norm(feat1 - feat2, p=2, dim=1).mean()
