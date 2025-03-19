import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Define dataset paths
dataset_path = "dataset"
horse_dir = os.path.join(dataset_path, "horse")  # Image folder

# Check if dataset directory exists
if not os.path.exists(horse_dir):
    print(f"❌ ERROR: The directory '{horse_dir}' does not exist!")
    exit()

# Select three images by name
image_names = ["horse004.png", "horse005.png", "horse006.png"]

# Verify if images exist before loading
valid_images = []
for img in image_names:
    img_path = os.path.join(horse_dir, img)
    if not os.path.exists(img_path):
        print(f"❌ ERROR: File '{img}' not found in '{horse_dir}'")
    else:
        valid_images.append(np.array(Image.open(img_path).convert("L")))

# If no valid images found, stop execution
if not valid_images:
    print("❌ ERROR: No valid images found! Please check filenames and dataset location.")
    exit()

# Display original images with filenames
fig, axes = plt.subplots(1, len(valid_images), figsize=(15, 5))
for i, img in enumerate(valid_images):
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Original: {image_names[i]}")
    axes[i].axis("off")
plt.show()

# Manual Convolution Function


def manual_convolution(image, kernel):
    """Perform manual convolution of an image with a given kernel using nested loops."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2  # Padding size

    # Create a padded image (zero-padding)
    padded_image = np.zeros((h + 2 * pad_h, w + 2 * pad_w))
    padded_image[pad_h:-pad_h, pad_w:-pad_w] = image

    # Output image
    output = np.zeros((h, w))

    # Perform convolution manually
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i+kh, j:j+kw]  # Extract region
            output[i, j] = np.sum(region * kernel)  # Apply filter

    return output


# Define edge detection filter kernels
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# Process images
for idx, img in enumerate(valid_images):
    # Apply each filter
    roberts_gx = manual_convolution(img, roberts_x)
    roberts_gy = manual_convolution(img, roberts_y)
    roberts_edges = np.sqrt(roberts_gx**2 + roberts_gy**2)

    prewitt_gx = manual_convolution(img, prewitt_x)
    prewitt_gy = manual_convolution(img, prewitt_y)
    prewitt_edges = np.sqrt(prewitt_gx**2 + prewitt_gy**2)

    sobel_gx = manual_convolution(img, sobel_x)
    sobel_gy = manual_convolution(img, sobel_y)
    sobel_edges = np.sqrt(sobel_gx**2 + sobel_gy**2)

    # Normalize edge images for better visualization
    roberts_edges = (roberts_edges / np.max(roberts_edges)
                     * 255).astype(np.uint8)
    prewitt_edges = (prewitt_edges / np.max(prewitt_edges)
                     * 255).astype(np.uint8)
    sobel_edges = (sobel_edges / np.max(sobel_edges) * 255).astype(np.uint8)

    # Display results with filenames
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Edge Detection Results for {image_names[idx]}", fontsize=16)

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Original: {image_names[idx]}")

    axes[1].imshow(roberts_edges, cmap='gray')
    axes[1].set_title("Roberts Edge Detection")

    axes[2].imshow(prewitt_edges, cmap='gray')
    axes[2].set_title("Prewitt Edge Detection")

    axes[3].imshow(sobel_edges, cmap='gray')
    axes[3].set_title("Sobel Edge Detection")

    for ax in axes:
        ax.axis("off")

    plt.show()
