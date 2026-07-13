from src.dataset import load_mnist_dataset, load_shapes_dataset
from src.helper_utils import display_images

# Create test case dataset and custom dataset
mnist_dataset = load_mnist_dataset(data_dir="data", train=True, download=True, shuffle=True, batch_size=8)
shapes_dataset = load_shapes_dataset(data_dir="./shapes_hard_color", shuffle=True, batch_size=8)

# display 8 images from each dataset
display_images(mnist_dataset)
display_images(shapes_dataset)



