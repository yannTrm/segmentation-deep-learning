import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from typing import Dict, Any, Optional, Callable, Tuple, List
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

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
        
        
        self.train_metrics = []
        self.val_metrics = []
        
        self.train_csv_path = None
        self.val_csv_path = None
        self._csv_initialized = False

        self._saved_this_epoch = False
        self._image_to_save = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Handles both 3-channel (RGB) and 4-channel (RGB + additional) inputs.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W) where C can be 3 or 4.

        Returns:
            torch.Tensor: Output mask tensor of shape (B, out_classes, H, W) for multiclass
                        or (B, 1, H, W) for binary classification.
        """
        # Check input dimensions
        assert image.dim() == 4, "Input must be 4D tensor (B, C, H, W)"
        num_channels = image.size(1)
        
        # Process RGB channels (normalize only the first 3 channels)
        rgb_channels = image[:, :3, :, :]
        normalized_rgb = (rgb_channels - self.mean) / self.std
        
        # Handle 4-channel case
        if num_channels == 4:
            additional_channel = image[:, 3:, :, :]  # Keep additional channel as-is
            processed_image = torch.cat([normalized_rgb, additional_channel], dim=1)
        else:
            processed_image = normalized_rgb
        
        # Forward pass through model
        logits = self.model(processed_image)
        
        # Ensure proper output dimensions for binary case
        if self.hparams.out_classes == 1 and logits.ndim == 3:
            logits = logits.unsqueeze(1)
            
        return logits

    def save_binary_figures(
        self, image: torch.Tensor, mask: torch.Tensor, pred_mask: torch.Tensor, stage: str
    ) -> None:
        """
        Save figures for binary segmentation (out_classes = 1).
        
        Args:
            image (torch.Tensor): Input image tensor [C, H, W]
            mask (torch.Tensor): Ground truth mask tensor [1, H, W]
            pred_mask (torch.Tensor): Predicted mask tensor [1, H, W]
            stage (str): Stage of the epoch ("train", "val", or "test")
        """
        image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        mask = mask.cpu().numpy().squeeze(0)  # [H, W]
        pred_mask = pred_mask.cpu().numpy().squeeze(0)  # [H, W]

        fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        # Original Image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Ground Truth Mask
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        # Prediction Mask
        axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction Mask")
        axes[2].axis("off")

        # Overlay Prediction on Original Image
        axes[3].imshow(image)
        axes[3].imshow(pred_mask, cmap="gray", alpha=0.5, vmin=0, vmax=1)
        axes[3].set_title("Prediction Overlay")
        axes[3].axis("off")

        plt.tight_layout()
        self._save_figure(fig, stage)

    def save_multiclass_figures(
        self, image: torch.Tensor, mask: torch.Tensor, pred_mask: torch.Tensor, stage: str
    ) -> None:
        """
        Save figures for multiclass segmentation (out_classes > 1).
        
        Args:
            image (torch.Tensor): Input image tensor [C, H, W]
            mask (torch.Tensor): Ground truth mask tensor [1, H, W] (class indices)
            pred_mask (torch.Tensor): Predicted mask tensor [1, H, W] (class indices)
            stage (str): Stage of the epoch ("train", "val", or "test")
        """
        image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        mask = mask.cpu().numpy().squeeze(0)  # [H, W]
        pred_mask = pred_mask.cpu().numpy().squeeze(0)  # [H, W]

        fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        # Original Image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Ground Truth Mask
        axes[1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=self.hparams.out_classes-1)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        # Prediction Mask
        axes[2].imshow(pred_mask, cmap="nipy_spectral", vmin=0, vmax=self.hparams.out_classes-1)
        axes[2].set_title("Prediction Mask")
        axes[2].axis("off")

        # Overlay Prediction on Original Image
        axes[3].imshow(image)
        axes[3].imshow(pred_mask, cmap="nipy_spectral", alpha=0.5, vmin=0, vmax=self.hparams.out_classes-1)
        axes[3].set_title("Prediction Overlay")
        axes[3].axis("off")

        plt.tight_layout()
        self._save_figure(fig, stage)

    def _save_figure(self, fig: plt.Figure, stage: str) -> None:
        """
        Helper function to save figure to disk.
        
        Args:
            fig (plt.Figure): Figure to save
            stage (str): Stage of the epoch ("train", "val", or "test")
        """
        figures_dir = os.path.join(self.trainer.default_root_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(os.path.join(figures_dir, f"{stage}_epoch_{self.current_epoch}.png"))
        plt.close(fig)

    def save_figures(
        self, image: torch.Tensor, mask: torch.Tensor, pred_mask: torch.Tensor, stage: str
    ) -> None:
        """
        Save figures with the original image, ground truth, and prediction.
        Delegates to binary or multiclass version based on out_classes.
        
        Args:
            image (torch.Tensor): Input image tensor [C, H, W]
            mask (torch.Tensor): Ground truth mask tensor [1, H, W] or [H, W]
            pred_mask (torch.Tensor): Predicted mask tensor [1, H, W] or [H, W]
            stage (str): Stage of the epoch ("train", "val", or "test")
        """
        # Ensure mask and pred_mask have 3 dims [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if pred_mask.ndim == 2:
            pred_mask = pred_mask.unsqueeze(0)
        
        if self.hparams.out_classes == 1:
            self.save_binary_figures(image, mask, pred_mask, stage)
        else:
            self.save_multiclass_figures(image, mask, pred_mask, stage)

    def shared_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor], 
            stage: str
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
        if mask.ndim == 3:  # [B, H, W] format
            mask = mask.unsqueeze(1)  # Add channel dimension -> [B, 1, H, W]
        assert mask.ndim == 4  # Now should be [B, C, H, W]

        # Ensure the mask is a long (index) tensor for multiclass
        if self.hparams.out_classes > 1:
            mask = mask.squeeze(1).long()  # Remove channel dimension -> [B, H, W]
        else:
            mask = mask.float()  # Keep as float for binary segmentation

        # Forward pass
        logits_mask = self.forward(image)

        # Align dimensions for binary segmentation
        if self.hparams.out_classes == 1 and logits_mask.ndim == 3:
            logits_mask = logits_mask.unsqueeze(1)  # Add channel dimension -> [B, 1, H, W]

        # Compute loss
        loss = self.loss_fn(logits_mask, mask)

        # Compute metrics
        if self.hparams.out_classes == 1:  # Binary segmentation
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
        else:  # Multiclass segmentation
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long() if self.hparams.out_classes == 1 else pred_mask,
            mask.long() if self.hparams.out_classes == 1 else mask,
            mode="binary" if self.hparams.out_classes == 1 else "multiclass",
            num_classes=self.hparams.out_classes,
        )

        # Save figures logic - separate for train and val
        if self.current_epoch % self.hparams.save_interval == 0:
            # Get the saved flag and image storage for this specific stage
            saved_flag = f"_{stage}_saved_this_epoch"
            image_storage = f"_{stage}_image_to_save"
            
            # If we haven't saved for this stage yet
            if not getattr(self, saved_flag, False):
                # 20% chance to save an image from this batch
                if torch.rand(1).item() < 0.2:
                    random_idx = torch.randint(0, image.size(0), (1,)).item()
                    setattr(self, image_storage, 
                            (image[random_idx], mask[random_idx], pred_mask[random_idx]))
                    setattr(self, saved_flag, True)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }



    def shared_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], stage: str) -> None:
        """
        Aggregate metrics at the end of an epoch and save figures if needed.

        Args:
            outputs (List[Dict[str, torch.Tensor]]): List of outputs from shared_step.
            stage (str): Stage of the epoch ("train", "val", or "test").
        """
        # Save figures if we have one to save for this stage
        if self.current_epoch % self.hparams.save_interval == 0:
            image_storage = f"_{stage}_image_to_save"
            if getattr(self, image_storage, None) is not None:
                image, mask, pred_mask = getattr(self, image_storage)
                self.save_figures(image, mask, pred_mask, stage)
                setattr(self, image_storage, None)  # Reset for next epoch

        # Calculate and log metrics
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
        
        return metrics

    def on_train_epoch_start(self) -> None:
        self._train_saved_this_epoch = False
        self._train_image_to_save = None

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
    
    def on_train_epoch_end(self):
        if not self._csv_initialized:
            self._init_csv_files()
        
        train_metrics = self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()
        
        # Préparer les données pour le CSV
        train_data = {
            "epoch": [self.current_epoch],
            "train_loss": [train_metrics["train_loss"].item()],
            "train_per_image_iou": [train_metrics["train_per_image_iou"].item()],
            "train_dataset_iou": [train_metrics["train_dataset_iou"].item()],
            "train_precision": [train_metrics["train_precision"].item()],
            "train_recall": [train_metrics["train_recall"].item()],
            "train_accuracy": [train_metrics["train_accuracy"].item()],
            "train_f1_score": [train_metrics["train_f1_score"].item()]
        }
        
        # Ajouter au CSV
        pd.DataFrame(train_data).to_csv(
            self.train_csv_path,
            mode='a',
            header=False,
            index=False
        )
        
        self.log_dict(train_metrics, prog_bar=True)
        
        

    def on_validation_epoch_start(self) -> None:
        self._val_saved_this_epoch = False
        self._val_image_to_save = None

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
    
    def on_validation_epoch_end(self):
        if not self._csv_initialized:
            self._init_csv_files()
        
        val_metrics = self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()
        
        # Préparer les données pour le CSV
        val_data = {
            "epoch": [self.current_epoch],
            "val_loss": [val_metrics["val_loss"].item()],
            "val_per_image_iou": [val_metrics["val_per_image_iou"].item()],
            "val_dataset_iou": [val_metrics["val_dataset_iou"].item()],
            "val_precision": [val_metrics["val_precision"].item()],
            "val_recall": [val_metrics["val_recall"].item()],
            "val_accuracy": [val_metrics["val_accuracy"].item()],
            "val_f1_score": [val_metrics["val_f1_score"].item()]
        }
        
        # Ajouter au CSV
        pd.DataFrame(val_data).to_csv(
            self.val_csv_path,
            mode='a',
            header=False,
            index=False
        )
        
        self.log_dict(val_metrics, prog_bar=True)


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
        
    def _init_csv_files(self):
        if self.trainer is None:
            return
            
        # Créer le dossier metrics s'il n'existe pas
        metrics_dir = Path(self.trainer.default_root_dir) / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        # Définir les chemins des fichiers
        self.train_csv_path = metrics_dir / "train_metrics.csv"
        self.val_csv_path = metrics_dir / "val_metrics.csv"
        
        # En-têtes des fichiers CSV
        train_headers = ["epoch", "train_loss", "train_per_image_iou", "train_dataset_iou", 
                        "train_precision", "train_recall", "train_accuracy", "train_f1_score"]
        
        val_headers = ["epoch", "val_loss", "val_per_image_iou", "val_dataset_iou",
                    "val_precision", "val_recall", "val_accuracy", "val_f1_score"]
        
        # Créer les fichiers avec en-têtes si ils n'existent pas
        if not self.train_csv_path.exists():
            pd.DataFrame(columns=train_headers).to_csv(self.train_csv_path, index=False)
        
        if not self.val_csv_path.exists():
            pd.DataFrame(columns=val_headers).to_csv(self.val_csv_path, index=False)
        
        self._csv_initialized = True
        
    def on_train_start(self):
        # Réinitialiser les fichiers CSV au début de l'entraînement
        if self.train_csv_path and self.train_csv_path.exists():
            self.train_csv_path.unlink()
        if self.val_csv_path and self.val_csv_path.exists():
            self.val_csv_path.unlink()
        self._csv_initialized = False

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
