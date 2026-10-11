from pathlib import Path
import numpy as np

from preprocessing import preprocess_image


# =========================
# Configuration
# =========================

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


# =========================
# Load Dataset
# =========================

def load_dataset():

    class_dirs = sorted(
        [
            folder
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    class_names = [folder.name for folder in class_dirs]

    images = []
    labels = []

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    for label, class_dir in enumerate(class_dirs):

        image_files = sorted(
            [
                file
                for file in class_dir.iterdir()
                if file.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                }
            ]
        )

        print(
            f"{class_dir.name:<12} -> "
            f"{len(image_files)} images "
            f"(label={label})"
        )

        for image_path in image_files:

            try:
                image = preprocess_image(image_path)

                images.append(image)
                labels.append(label)

            except Exception as e:

                print(
                    f"[SKIPPED] {image_path.name} "
                    f"-> {e}"
                )

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print("\n" + "=" * 60)
    print("DATASET LOADED")
    print("=" * 60)

    print(f"Classes : {class_names}")
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")

    return X, y, class_names


# =========================
# Test
# =========================

if __name__ == "__main__":

    X, y, class_names = load_dataset()

    print("\nFirst sample:")
    print(f"Shape : {X[0].shape}")
    print(f"Label : {y[0]}")
    print(f"Class : {class_names[y[0]]}")