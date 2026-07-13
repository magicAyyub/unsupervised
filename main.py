from src.dataset import load_mnist_dataset, load_shapes_dataset
from src.helper_utils import display_images, display_pca_comparison
import matplotlib.pyplot as plt

# Create test case dataset and custom dataset
mnist_dataset = load_mnist_dataset(data_dir="data", train=True, download=True, shuffle=True, batch_size=8)
shapes_dataset = load_shapes_dataset(data_dir="./shapes_hard_color", shuffle=True, batch_size=8)

print("\nDemonstrating PCA Compression and Decompression...")

# For MNIST (grayscale, 28x28. Keeping 10 components)
print("Applying PCA to MNIST (k=10)...")
display_pca_comparison(
    mnist_dataset, 
    k=10, 
    num_pairs=4, 
    title="MNIST - Original (Left Column) vs PCA Reconstructed (Right Column) [k=10]", 
    show=False
)

# For Shapes (color, keeping 30 components)
print("Applying PCA to Shapes (k=30)...")
display_pca_comparison(
    shapes_dataset, 
    k=30, 
    num_pairs=4, 
    title="Shapes - Original (Left Column) vs PCA Reconstructed (Right Column) [k=30]", 
    show=False
)

print("Opening both comparison windows simultaneously...")
plt.show()





