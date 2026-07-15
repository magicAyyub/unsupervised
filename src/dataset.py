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
    