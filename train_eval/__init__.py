from .metrics import BaseMetric, IoU, DiceCoefficient, PixelAccuracy, Precision, Recall
from .preprocessing import (get_landscape_augmentation, get_landscape_validation_augmentation, 
                            get_portrait_augmentation, get_portrait_validation_augmentation)
from .base_segmentation import SegmentationModel
from .binary_segmentation import BinarySegmentationModel
from .multiclass_segmentation import MulticlassSegmentationModel

__metrics = [
    "BaseMetric",
    "IoU",
    "DiceCoefficient",
    "PixelAccuracy",
    "Precision",
    "Recall",
]

__models = [
    "SegmentationModel",
    "BinarySegmentationModel",
    "MulticlassSegmentationModel",
]

__preprocessing = [
    "get_landscape_augmentation", "get_landscape_validation_augmentation", 
    "get_portrait_augmentation", "get_portrait_validation_augmentation"
]

# Combinaison de toutes les listes dans __all__
__all__ = __metrics + __models + __preprocessing