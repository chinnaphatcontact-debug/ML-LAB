from pathlib import Path

from preprocessing import preprocess_image


# =========================
# Select sample image
# =========================

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

sample_image = next(
    (DATASET_DIR / "Lilly").glob("*.jpg")
)


# =========================
# Preprocess
# =========================

image = preprocess_image(sample_image)


# =========================
# Display information
# =========================

print("=" * 60)
print("PREPROCESSING TEST")
print("=" * 60)

print(f"Image path : {sample_image}")
print(f"Shape      : {image.shape}")
print(f"Data type  : {image.dtype}")
print(f"Min value  : {image.min():.4f}")
print(f"Max value  : {image.max():.4f}")

print("=" * 60)