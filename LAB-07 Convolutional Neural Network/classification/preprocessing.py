from PIL import Image
import numpy as np


# =========================
# Configuration
# =========================

IMAGE_SIZE = (128, 128)


# =========================
# Preprocess Image
# =========================

def preprocess_image(image_path):
    """
    Load and preprocess an image.

    Steps:
    1. Open image
    2. Convert to RGB
    3. Resize to 128x128
    4. Convert to NumPy array
    5. Normalize pixel values to 0-1
    """

    image = Image.open(image_path)

    # Ensure RGB format
    image = image.convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image = np.array(image, dtype=np.float32)

    # Normalize pixel values
    image = image / 255.0

    return image