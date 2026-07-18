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


def load_shapes_npz(
    split: str = "train", 
    data_dir: str | None = None, 
    max_samples: int | None = None, 
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Signature: load_shapes_npz(split: str = "train", data_dir: str | None = None, max_samples: int | None = None, seed: int = 0) -> tuple[np.ndarray de shape (N, C, H, W), np.ndarray de shape (N,), list[str]]
    # Charge le dataset des formes géométriques pré-rendu au format .npz.
    # Algorithme:
    # 1. Recherche du chemin vers le dossier shapes_hard_color.
    # 2. Chargement du fichier .npz correspondant au split ('train' ou 'validation').
    # 3. Sélection aléatoire d'un sous-ensemble d'images si max_samples est défini.
    # 4. Normalisation des images dans [0, 1] et transposition au format standard (N, C, H, W).
    # 5. Conversion des étiquettes en int64 et récupération des noms de classes.
    
    if data_dir is None:
        for base in ("data", "../data"):
            candidate = os.path.join(base, "shapes_hard_color", "shapes_hard_color")
            if os.path.isdir(candidate):
                data_dir = candidate
                break
        if data_dir is None:
            raise FileNotFoundError("Dossier shapes_hard_color introuvable sous data/ ou ../data/")

    archive = np.load(os.path.join(data_dir, f"shapes_{split}.npz"), allow_pickle=True)
    images = archive["images"]  # shape (N, H, W, C), uint8
    labels = archive["labels"]  # shape (N,)

    if max_samples is not None and max_samples < len(images):
        selection = np.random.default_rng(seed).choice(len(images), size=max_samples, replace=False)
        images, labels = images[selection], labels[selection]

    images = (images.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    class_names = [str(name) for name in archive["class_names"]]
    
    return images, labels.astype(np.int64), class_names

