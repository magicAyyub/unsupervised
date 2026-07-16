from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_mnist_dataset(data_dir="data", train=True, download=True, shuffle=True, batch_size=64) -> DataLoader:
    """
    Loading the MNIST dataset using torchvision.
    Returns:
        dataloader: PyTorch DataLoader of MNIST images and labels
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def load_shapes_dataset(data_dir="./data/shapes_hard_color", shuffle=True, batch_size=64) -> DataLoader:
    """
    Loading the shapes dataset using torchvision.
    Returns:
        dataloader: PyTorch DataLoader of shapes images
    """
    image_transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create a Dataset
    shapes_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=image_transformation
    )

    # Create a DataLoader from the Dataset
    shapes_dataloader = torch.utils.data.DataLoader(
        dataset=shapes_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return shapes_dataloader


class ShapesArrays(NamedTuple):
    """In-memory view of one split of the colored-shapes dataset."""
    images: np.ndarray        # (N, 32, 32, 3) uint8 RGB images
    labels: np.ndarray        # (N,) int64 shape index in [0, 5]
    factors: np.ndarray       # (N, 7) float32 ground-truth generative factors
    factor_names: np.ndarray  # (7,) names of the factor columns
    class_names: np.ndarray   # (6,) shape names indexed by label


def load_shapes_arrays(
    data_dir: str = "./data/shapes_hard_color", split: str = "train"
) -> ShapesArrays:
    """
    Load one split of the colored-shapes dataset from its packaged .npz archive.

    Args:
        data_dir: directory holding ``shapes_train.npz`` and ``shapes_validation.npz``.
        split: either ``"train"`` or ``"validation"``.

    Returns:
        A ShapesArrays tuple of NumPy arrays.
    """
    if split not in ("train", "validation"):
        raise ValueError("split must be 'train' or 'validation'.")

    path = Path(data_dir) / f"shapes_{split}.npz"
    with np.load(path, allow_pickle=True) as archive:
        return ShapesArrays(
            images=archive["images"],
            labels=archive["labels"],
            factors=archive["factors"],
            factor_names=archive["factor_names"],
            class_names=archive["class_names"],
        )
