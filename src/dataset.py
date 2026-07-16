import os
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


def load_shapes_npz(split="train", data_dir=None, max_samples=None, seed=0):
    """
    Loads the pre-rendered shapes dataset from its .npz archive.

    The archive already stores uniform 32x32 RGB images, which is cleaner and faster than
    decoding the JPEG variant, and it also ships the shape-class names.

    Args:
        split: "train" or "validation".
        data_dir: folder holding shapes_{split}.npz. If None, it is searched both from the
            project root and from the notebooks/ folder so the call works regardless of the CWD.
        max_samples: if set, draw a reproducible random subset of that many images.
        seed: seed controlling the subset selection.

    Returns:
        images: float32 array (N, C, H, W) scaled to [0, 1].
        labels: int64 array (N,) with the shape-class index.
        class_names: list[str] mapping a label index to its shape name.
    """
    if data_dir is None:
        # Le dataset vit sous data/ a la racine ; on couvre les deux CWD possibles
        for base in ("data", "../data"):
            candidate = os.path.join(base, "shapes_hard_color", "shapes_hard_color")
            if os.path.isdir(candidate):
                data_dir = candidate
                break
        if data_dir is None:
            raise FileNotFoundError("shapes_hard_color introuvable sous data/ ou ../data/")

    archive = np.load(os.path.join(data_dir, f"shapes_{split}.npz"), allow_pickle=True)
    images = archive["images"]  # (N, H, W, C) uint8
    labels = archive["labels"]

    if max_samples is not None and max_samples < len(images):
        selection = np.random.default_rng(seed).choice(len(images), size=max_samples, replace=False)
        images, labels = images[selection], labels[selection]

    # Normalisation [0,1] et passage a la convention (N, C, H, W) du projet
    images = (images.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    class_names = [str(name) for name in archive["class_names"]]
    return images, labels.astype(np.int64), class_names
