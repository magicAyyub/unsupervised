import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def display_images(dataloader, figsize=(10, 10), nrow=4, title=None, show=True):
    """
    Fetches a batch of images from a DataLoader, arranges them in a grid, and displays them.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader to fetch images from.
        figsize (tuple, optional): The size of the figure for display. Defaults to (10, 10).
        nrow (int, optional): Number of images to display in each row of the grid. Defaults to 4.
        title (str, optional): Title of the plot.
        show (bool, optional): If True, calls plt.show(). Defaults to True.
    """
    # Get one batch of images from the dataloader
    images, _ = next(iter(dataloader))

    # Create a grid from the images
    # normalize=True scales the image pixel values to the range [0, 1]
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=True)

    # Display the grid of images
    plt.figure(figsize=figsize)
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.imshow(grid.permute(1, 2, 0).cpu()) # Transpose dimensions from (C, H, W) to (H, W, C) for plotting
    plt.axis('off')
    if show:
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

def display_pca_comparison(dataloader, k, num_pairs=4, figsize=(10, 10), title="PCA Comparison", show=True):
    """
    Fetches a batch of images from a DataLoader, applies PCA compression and decompression,
    and displays the original and decompressed images side-by-side in a grid.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader to fetch images from.
        k (int): Number of principal components to retain.
        num_pairs (int, optional): Number of image pairs to display. Defaults to 4.
        figsize (tuple, optional): The size of the figure for display. Defaults to (10, 10).
        title (str, optional): Title of the plot. Defaults to "PCA Comparison".
        show (bool, optional): If True, calls plt.show(). Defaults to True.
    """
    from src.algorithms.pca import compress, decompress

    # Obtenir un lot d'images
    images, _ = next(iter(dataloader))
    
    # Limiter le nombre d'images
    num_pairs = min(len(images), num_pairs)
    images = images[:num_pairs]

    # Compresser et décompresser avec ACP
    compressed_dict = compress(images, k)
    reconstructed = decompress(compressed_dict)

    # Entrelacer les originales et les décompressées
    comparison_list = []
    for i in range(num_pairs):
        comparison_list.append(images[i])
        comparison_list.append(reconstructed[i])

    comparison_tensor = torch.stack(comparison_list)

    # Créer la grille (2 colonnes : Original à gauche, Décompressé à droite)
    grid = vutils.make_grid(comparison_tensor, nrow=2, padding=2, normalize=True)

    # Afficher le résultat
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    if show:
        plt.show()