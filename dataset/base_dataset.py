import os
from typing import List, Optional, Dict, Tuple
import cv2
import numpy as np
import albumentations as A
import yaml

class BaseDataset:
    """Base dataset class for segmentation tasks.

    Args:
        images_dir (str): Path to the directory containing images.
        masks_dir (str): Path to the directory containing masks.
        augmentation (Optional[albumentations.Compose]): Data augmentation pipeline.
    """

    CLASSES: List[str] = []
    CLASS_VALUES: Dict[str, List[int]] = {}

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        augmentation: Optional[A.Compose] = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augmentation = augmentation

        self.ids = self._filter_image_files(os.listdir(images_dir))
        
        self.images_fps = []
        self.masks_fps = []
        for image_id in self.ids:
            if image_id.endswith(".jpg"):
                mask_id = image_id.replace(".jpg", "_mask.png")
            elif image_id.endswith(".png"):
                mask_id = image_id.replace(".png", "_mask.png")
            else:
                raise ValueError("Unsupported file format")
            mask_path = os.path.join(masks_dir, mask_id)
            if os.path.exists(mask_path):
                self.images_fps.append(os.path.join(images_dir, image_id))
                self.masks_fps.append(mask_path)
            else:
                print(f"Warning: Mask file not found for image {image_id}. Skipping this image.")

    def __len__(self) -> int:
        return len(self.images_fps)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def _filter_image_files(files: List[str]) -> List[str]:
        """Filter files to keep only image files.

        Args:
            files (List[str]): List of file names.

        Returns:
            List[str]: List of image file names.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        return [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

    @classmethod
    def from_yaml(cls, images_dir: str, masks_dir: str, yaml_path: str, augmentation: Optional[A.Compose] = None):
        """Create a dataset instance using class definitions from a YAML file.

        Args:
            images_dir (str): Path to the directory containing images.
            masks_dir (str): Path to the directory containing masks.
            yaml_path (str): Path to the YAML file containing class definitions.
            augmentation (Optional[albumentations.Compose]): Data augmentation pipeline.

        Returns:
            BaseDataset: An instance of the dataset.
        """
        class_mapping = cls._load_class_mapping(yaml_path)
        cls.CLASSES = list(class_mapping.keys()) 
        cls.CLASS_VALUES = list(class_mapping.values()) 
        return cls(images_dir, masks_dir, augmentation)

    @staticmethod
    def _load_class_mapping(file_path: str) -> Dict[str, int]:
        """Load class mappings from a YAML file.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            Dict[str, int]: Dictionary mapping class names to integer values.
        """
        with open(file_path, "r") as f:
            class_mapping = yaml.safe_load(f)
        return class_mapping