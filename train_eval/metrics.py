from typing import Optional, List
import torch

class BaseMetric:
    """Base class for metrics.

    Args:
        threshold (float): Threshold for binarizing predictions (only for binary tasks).
        num_classes (Optional[int]): Number of classes for multi-class tasks. If None, assumes binary task.
        ignore_index (Optional[List[int]]): List of class indices to ignore in calculations.
    """
    def __init__(self, threshold: float = 0.5, num_classes: Optional[int] = None, ignore_index: Optional[List[int]] = None) -> None:
        self.threshold = threshold
        self.num_classes = num_classes
        self.ignore_index = set(ignore_index) if ignore_index is not None else set()

    def __call__(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def compute(self) -> Optional[torch.Tensor]:
        pass

class IoU(BaseMetric):
    """Intersection over Union (IoU) metric for binary and multi-class segmentation.
    
    - For binary segmentation:
        IoU = Intersection / Union
    - For multi-class segmentation:
        Mean IoU over classes, optionally ignoring some classes.
    """
    def __call__(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.num_classes is None:  # Binary
            outputs = (outputs > self.threshold).float()
            intersection = (outputs * masks).sum()
            union = outputs.sum() + masks.sum() - intersection
            return intersection / union if union > 0 else torch.tensor(1.0)
        else:  # Multi-class
            outputs = outputs.argmax(dim=1)
            ious = []
            for class_id in range(self.num_classes):
                if class_id in self.ignore_index:
                    continue
                intersection = ((outputs == class_id) & (masks == class_id)).sum().item()
                union = ((outputs == class_id) | (masks == class_id)).sum().item()
                if union > 0:
                    ious.append(intersection / union)
            return torch.tensor(sum(ious) / len(ious)) if ious else torch.tensor(1.0)

class DiceCoefficient(BaseMetric):
    """Dice Coefficient metric for binary and multi-class segmentation.

    - Binary:
        Dice = 2 * Intersection / (Sum of predictions + Sum of masks)
    - Multi-class:
        Mean Dice over classes, optionally ignoring some classes.
    """
    def __call__(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.num_classes is None:  # Binary
            outputs = (outputs > self.threshold).float()
            intersection = (outputs * masks).sum()
            return (2. * intersection) / (outputs.sum() + masks.sum()) if (outputs.sum() + masks.sum()) > 0 else torch.tensor(1.0)
        else:  # Multi-class
            outputs = outputs.argmax(dim=1)
            dices = []
            for class_id in range(self.num_classes):
                if class_id in self.ignore_index:
                    continue
                intersection = ((outputs == class_id) & (masks == class_id)).sum().item()
                pred_sum = (outputs == class_id).sum().item()
                true_sum = (masks == class_id).sum().item()
                if pred_sum + true_sum > 0:
                    dice = 2 * intersection / (pred_sum + true_sum)
                    dices.append(dice)
            return torch.tensor(sum(dices) / len(dices)) if dices else torch.tensor(1.0)

class PixelAccuracy(BaseMetric):
    """Pixel-wise Accuracy metric for binary and multi-class segmentation.

    - Binary:
        Accuracy = Correct Pixels / Total Pixels
    - Multi-class:
        Same as binary but considers multiple classes.
    """
    def __call__(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.num_classes is None:  # Binary
            outputs = (outputs > self.threshold).float()
        else:  # Multi-class
            outputs = outputs.argmax(dim=1)

        valid_mask = torch.ones_like(masks, dtype=torch.bool)
        for idx in self.ignore_index:
            valid_mask &= masks != idx

        correct = ((outputs == masks) * valid_mask).sum().float()
        total = valid_mask.sum().float()
        return correct / total if total > 0 else torch.tensor(1.0)

class Precision(BaseMetric):
    """Precision metric for binary segmentation.

    Defined as:
        Precision = True Positives / (True Positives + False Positives)
    """
    def __call__(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.num_classes is not None:
            raise ValueError("Precision is defined only for binary segmentation")
        outputs = (outputs > self.threshold).float()
        true_positive = (outputs * masks).sum()
        predicted_positive = outputs.sum()
        return true_positive / predicted_positive if predicted_positive > 0 else torch.tensor(1.0)

class Recall(BaseMetric):
    """Recall metric for binary segmentation.

    Defined as:
        Recall = True Positives / (True Positives + False Negatives)
    """
    def __call__(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.num_classes is not None:
            raise ValueError("Recall is defined only for binary segmentation")
        outputs = (outputs > self.threshold).float()
        true_positive = (outputs * masks).sum()
        actual_positive = masks.sum()
        return true_positive / actual_positive if actual_positive > 0 else torch.tensor(1.0)
