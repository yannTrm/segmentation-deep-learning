from typing import Optional, List, Tuple, Dict
import albumentations as A
import cv2
import numpy as np

from .base_dataset import BaseDataset

import albumentations as A
import cv2
import numpy as np
import yaml


class BinarySegmentationDataset(BaseDataset):
    """Dataset for binary segmentation tasks.

    Args:
        images_dir (str): Path to the directory containing images.
        masks_dir (str): Path to the directory containing masks.
        augmentation (Optional[albumentations.Compose]): Data augmentation pipeline.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        augmentation: Optional[A.Compose] = None,
    ):
        super().__init__(images_dir, masks_dir, augmentation)
        
        self.class_values = self.CLASS_VALUES
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Image and mask tensors.
        """
        image = cv2.imread(self.images_fps[idx])
        if image is None:
            raise ValueError(f"Unable to read image: {self.images_fps[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {self.masks_fps[idx]}")

        binary_mask = np.zeros_like(mask, dtype=np.float32)
        for class_value in self.class_values:
            binary_mask[mask == class_value] = 1  

        binary_mask = np.expand_dims(binary_mask, axis=-1)

        if self.augmentation:
            
            sample = self.augmentation(image=image, mask=binary_mask)
            image, binary_mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), binary_mask.transpose(2, 0, 1)




class MultiLabelSegmentationDataset(BaseDataset):
    """Dataset for multi-label segmentation tasks.

    Args:
        images_dir (str): Path to the directory containing images.
        masks_dir (str): Path to the directory containing masks.
        augmentation (Optional[albumentations.Compose]): Data augmentation pipeline.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        augmentation: Optional[A.Compose] = None,
    ):
        super().__init__(images_dir, masks_dir, augmentation)
        
        self.class_to_value = {cls: value for cls, value in zip(self.CLASSES, self.CLASS_VALUES)}
        self.value_to_class = {value: cls for cls, value in self.class_to_value.items()}

        self.class_map = {gray_value: idx for idx, gray_value in enumerate(self.CLASS_VALUES)}

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Image and mask tensors.
        """
        image = cv2.imread(self.images_fps[idx])
        if image is None:
            raise ValueError(f"Unable to read image: {self.images_fps[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {self.masks_fps[idx]}")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), mask
    
    def get(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Image and mask tensors.
        """
        image = cv2.imread(self.images_fps[idx])
        if image is None:
            raise ValueError(f"Unable to read image: {self.images_fps[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {self.masks_fps[idx]}")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), mask, self.images_fps[idx]

    def get_class_mapping(self) -> Dict[str, int]:
        """Get the mapping from class names to numerical values.

        Returns:
            Dict[str, int]: Mapping from class names to numerical values.
        """
        return self.class_to_value

    def get_inverse_class_mapping(self) -> Dict[int, str]:
        """Get the mapping from numerical values to class names.

        Returns:
            Dict[int, str]: Mapping from numerical values to class names.
        """
        return self.value_to_class

    def get_num_classes(self) -> int:
        """Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(self.class_map)

    def get_class_name(self, class_value: int) -> str:
        """Get the class name corresponding to a numerical value.

        Args:
            class_value (int): Numerical value of the class.

        Returns:
            str: Class name.
        """
        return self.value_to_class.get(class_value, "Unknown")

    def get_class_value(self, class_name: str) -> int:
        """Get the numerical value corresponding to a class name.

        Args:
            class_name (str): Name of the class.

        Returns:
            int: Numerical value of the class.
        """
        return self.class_to_value.get(class_name, -1)