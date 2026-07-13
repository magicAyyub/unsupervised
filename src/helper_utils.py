import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def display_images(dataloader, figsize=(10, 10), nrow=4):
    """
    Fetches a batch of images from a DataLoader, arranges them in a grid, and displays them.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader to fetch images from.
        figsize (tuple, optional): The size of the figure for display. Defaults to (10, 10).
        nrow (int, optional): Number of images to display in each row of the grid. Defaults to 4.
    """
    # Get one batch of images from the dataloader
    images, _ = next(iter(dataloader))

    # Create a grid from the images
    # normalize=True scales the image pixel values to the range [0, 1]
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=True)

    # Display the grid of images
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0)) # Transpose dimensions from (C, H, W) to (H, W, C) for plotting
    plt.axis('off')
    plt.show()

def get_device():
    """
    Detects the best available device (CUDA if available, otherwise MPS, otherwise CPU).

    Returns:
        torch.device: The detected device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    