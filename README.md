# Segmentation Project

This project implements image segmentation models using PyTorch Lightning and segmentation_models_pytorch. It supports both binary and multiclass segmentation.

## Project Structure

```
.gitignore
.vscode/
    [`.vscode/settings.json`](.vscode/settings.json )
segmentation/
    dataset/
        [`segmentation/dataset/__init__.py`](segmentation/dataset/__init__.py )
        [`segmentation/dataset/base_dataset.py`](segmentation/dataset/base_dataset.py )
        [`segmentation/dataset/custom_dataset.py`](segmentation/dataset/custom_dataset.py )
    test_notebook/
        [`segmentation/test_notebook/test_dataset.ipynb`](segmentation/test_notebook/test_dataset.ipynb )
        [`segmentation/test_notebook/test_training.ipynb`](segmentation/test_notebook/test_training.ipynb )
    train_eval/
        [`segmentation/dataset/__init__.py`](segmentation/dataset/__init__.py )
        [`segmentation/train_eval/base_segmentation.py`](segmentation/train_eval/base_segmentation.py )
        [`segmentation/train_eval/binary_segmentation.py`](segmentation/train_eval/binary_segmentation.py )
        [`segmentation/train_eval/metrics.py`](segmentation/train_eval/metrics.py )
        [`segmentation/train_eval/multiclass_segmentation.py`](segmentation/train_eval/multiclass_segmentation.py )
        [`segmentation/train_eval/preprocessing.py`](segmentation/train_eval/preprocessing.py )
    utils/
        [`segmentation/dataset/__init__.py`](segmentation/dataset/__init__.py )
        [`segmentation/utils/visualization.py`](segmentation/utils/visualization.py )
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training

To train a binary segmentation model, use the notebook test_training.ipynb.

### Testing

To test the datasets, use the notebook test_dataset.ipynb.

## Modules

### Dataset

- [`BaseDataset`](segmentation/dataset/base_dataset.py ): Base class for segmentation datasets.
- [`BinarySegmentationDataset`](segmentation/dataset/custom_dataset.py ): Dataset for binary segmentation.
- [`MultiLabelSegmentationDataset`](segmentation/dataset/custom_dataset.py ): Dataset for multiclass segmentation.

### Models

- [`SegmentationModel`](segmentation/train_eval/base_segmentation.py ): Base module for segmentation tasks.
- [`BinarySegmentationModel`](segmentation/train_eval/binary_segmentation.py ): Module for binary segmentation.
- [`MulticlassSegmentationModel`](segmentation/train_eval/multiclass_segmentation.py ): Module for multiclass segmentation.

### Preprocessing

- [`get_landscape_augmentation`](segmentation/train_eval/preprocessing.py ): Augmentation for landscape images.
- [`get_portrait_validation_augmentation`](segmentation/train_eval/preprocessing.py ): Augmentation for portrait images.

### Visualization

- [`visualize_binaire`](segmentation/utils/visualization.py ): Visualization for binary segmentation.
- [`visualize_gray`](segmentation/utils/visualization.py ): Visualization for multiclass segmentation.

### Metrics

- [`IoU`](segmentation/train_eval/metrics.py ): Intersection over Union.
- [`DiceCoefficient`](segmentation/train_eval/metrics.py ): Dice Coefficient.
- [`PixelAccuracy`](segmentation/train_eval/metrics.py ): Pixel-wise Accuracy.
- [`Precision`](segmentation/train_eval/metrics.py ): Precision for binary segmentation.
- [`Recall`](segmentation/train_eval/metrics.py ): Recall for binary segmentation.

## Customization

You can customize the models, datasets, and augmentations by modifying the corresponding files in the [`segmentation/dataset`](segmentation/dataset ), [`segmentation/train_eval`](segmentation/train_eval ), and [`segmentation/utils`](segmentation/utils ) directories.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss the changes you want to make.

## License

This project is licensed under the MIT License. See the LICENSE file for details.