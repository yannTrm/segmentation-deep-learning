from typing import Dict, Optional, Callable

from .base_segmentation import SegmentationModel


class BinarySegmentationModel(SegmentationModel):
    """
    PyTorch Lightning module for binary segmentation tasks.

    Args:
        arch (str): Architecture of the segmentation model (e.g., "Unet").
        encoder_name (str): Name of the encoder backbone (e.g., "resnet34").
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        loss_fn (Optional[Callable]): Loss function to use. Defaults to DiceLoss.
        optimizer (Optional[Callable]): Optimizer to use. Defaults to Adam.
        optimizer_kwargs (Optional[Dict]): Additional arguments for the optimizer.
        lr_scheduler (Optional[Callable]): Learning rate scheduler to use.
        lr_scheduler_kwargs (Optional[Dict]): Additional arguments for the scheduler.
        save_interval (int): Interval for saving figures during training. Defaults to 1.
        **kwargs: Additional arguments for the segmentation model.
    """

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[Callable] = None,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        save_interval: int = 1,
        **kwargs,
    ):
        super().__init__(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            out_classes=1,
            loss_mode="binary",
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            save_interval=save_interval,
            **kwargs,
        )