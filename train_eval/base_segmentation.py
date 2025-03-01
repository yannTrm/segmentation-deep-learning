import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from typing import Dict, Any, Optional, Callable, Tuple, List
import os
import matplotlib.pyplot as plt


class SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for segmentation tasks (binary or multiclass).

    Args:
        arch (str): Architecture of the segmentation model (e.g., "Unet").
        encoder_name (str): Name of the encoder backbone (e.g., "resnet34").
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        out_classes (int): Number of output classes (1 for binary segmentation, >1 for multiclass).
        loss_mode (str): Mode of the loss function ("binary" or "multiclass").
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
        out_classes: int,
        loss_mode: str,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[Callable] = None,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        save_interval: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the segmentation model
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function (default to DiceLoss)
        self.loss_fn = loss_fn or smp.losses.DiceLoss(mode=loss_mode, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self._saved_this_epoch = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output mask tensor of shape (B, out_classes, H, W).
        """
        # Normalize image
        image = (image - self.mean) / self.std
        return self.model(image)

    def save_figures(
        self, image: torch.Tensor, mask: torch.Tensor, pred_mask: torch.Tensor, stage: str
    ) -> None:
        """
        Save figures with the original image, ground truth, and prediction.

        Args:
            image (torch.Tensor): Input image tensor.
            mask (torch.Tensor): Ground truth mask tensor.
            pred_mask (torch.Tensor): Predicted mask tensor.
            stage (str): Stage of the epoch ("train", "val", or "test").
        """
        image = image.cpu().numpy().transpose(1, 2, 0)
        mask = mask.cpu().numpy().squeeze()
        pred_mask = pred_mask.cpu().numpy().squeeze()

        fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        # Original Image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Ground Truth Mask
        axes[1].imshow(mask, cmap="gray" if self.hparams.out_classes == 1 else "nipy_spectral", vmin=0, vmax=self.hparams.out_classes - 1)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        # Prediction Mask
        axes[2].imshow(pred_mask, cmap="gray" if self.hparams.out_classes == 1 else "nipy_spectral", vmin=0, vmax=self.hparams.out_classes - 1)
        axes[2].set_title("Prediction Mask")
        axes[2].axis("off")

        # Overlay Prediction on Original Image
        axes[3].imshow(image)
        axes[3].imshow(pred_mask, cmap="gray" if self.hparams.out_classes == 1 else "nipy_spectral", alpha=0.5, vmin=0, vmax=self.hparams.out_classes - 1)
        axes[3].set_title("Prediction Overlay")
        axes[3].axis("off")

        plt.tight_layout()
        figures_dir = os.path.join(self.trainer.default_root_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(os.path.join(figures_dir, f"{stage}_epoch_{self.current_epoch}.png"))
        plt.close(fig)

    def shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> Dict[str, torch.Tensor]:
        """
        Shared step for training, validation, and testing.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the image and mask tensors.
            stage (str): Stage of the step ("train", "val", or "test").

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing loss and metrics.
        """
        image, mask = batch

        # Ensure input and mask dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]
        assert mask.ndim == 3 if self.hparams.out_classes > 1 else 4  # [batch_size, H, W] for multiclass, [batch_size, 1, H, W] for binary

        # Ensure the mask is a long (index) tensor for multiclass
        if self.hparams.out_classes > 1:
            mask = mask.long()

        # Forward pass
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        # Compute metrics
        if self.hparams.out_classes == 1:
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
        else:
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long() if self.hparams.out_classes == 1 else pred_mask,
            mask.long() if self.hparams.out_classes == 1 else mask,
            mode="binary" if self.hparams.out_classes == 1 else "multiclass",
            num_classes=self.hparams.out_classes,
        )

        # Save figures every save_interval epochs
        if self.current_epoch % self.hparams.save_interval == 0 and not self._saved_this_epoch:
            self.save_figures(image[0], mask[0], pred_mask[0], stage)
            self._saved_this_epoch = True  # Mark that an image has been saved this epoch

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], stage: str) -> None:
        """
        Aggregate metrics at the end of an epoch.

        Args:
            outputs (List[Dict[str, torch.Tensor]]): List of outputs from shared_step.
            stage (str): Stage of the epoch ("train", "val", or "test").
        """
        
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_loss": avg_loss,  
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
            f"{stage}_accuracy": accuracy,
            f"{stage}_f1_score": f1_score,
        }
        self.log_dict(metrics, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        self._saved_this_epoch = False

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the image and mask tensors.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing loss and metrics.
        """
        output = self.shared_step(batch, "train")
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self) -> None:
        """
        Aggregate training metrics at the end of an epoch.
        """
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_start(self) -> None:
        self._saved_this_epoch = False

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the image and mask tensors.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing loss and metrics.
        """
        output = self.shared_step(batch, "val")
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        """
        Aggregate validation metrics at the end of an epoch.
        """
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_start(self) -> None:
        self._saved_this_epoch = False

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the image and mask tensors.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing loss and metrics.
        """
        output = self.shared_step(batch, "test")
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self) -> None:
        """
        Aggregate test metrics at the end of an epoch.
        """
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer and scheduler.
        """
        optimizer = self.hparams.optimizer(self.parameters(), **self.hparams.optimizer_kwargs)
        if self.hparams.lr_scheduler:
            scheduler = self.hparams.lr_scheduler(optimizer, **self.hparams.lr_scheduler_kwargs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

