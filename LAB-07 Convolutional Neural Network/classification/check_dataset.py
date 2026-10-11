from pathlib import Path
from PIL import Image
from collections import Counter


# =========================
# Configuration
# =========================

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================
# Check Dataset
# =========================

def check_dataset():

    print("=" * 70)
    print("DATASET CHECK")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR.resolve()}")
        return

    class_dirs = sorted(
        [folder for folder in DATASET_DIR.iterdir() if folder.is_dir()]
    )

    if not class_dirs:
        print("No class folders found.")
        return

    total_images = 0
    total_valid = 0
    total_corrupted = 0

    image_sizes = Counter()
    image_modes = Counter()

    for class_dir in class_dirs:

        image_files = [
            file for file in class_dir.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        valid_count = 0
        corrupted_count = 0

        for image_path in image_files:

            try:
                with Image.open(image_path) as img:

                    # Verify image
                    img.verify()

                # Re-open after verify
                with Image.open(image_path) as img:

                    image_sizes[img.size] += 1
                    image_modes[img.mode] += 1

                valid_count += 1

            except Exception:
                corrupted_count += 1
                print(f"[CORRUPTED] {image_path}")

        total_images += len(image_files)
        total_valid += valid_count
        total_corrupted += corrupted_count

        print(f"\nClass: {class_dir.name}")
        print(f"  Total images     : {len(image_files)}")
        print(f"  Valid images     : {valid_count}")
        print(f"  Corrupted images : {corrupted_count}")

    # =========================
    # Summary
    # =========================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Classes          : {len(class_dirs)}")
    print(f"Total images     : {total_images}")
    print(f"Valid images     : {total_valid}")
    print(f"Corrupted images : {total_corrupted}")

    # =========================
    # Image Size
    # =========================

    print("\n" + "=" * 70)
    print("IMAGE SIZE")
    print("=" * 70)

    for size, count in image_sizes.most_common(10):
        print(f"{size[0]} x {size[1]} : {count} images")

    # =========================
    # Image Mode
    # =========================

    print("\n" + "=" * 70)
    print("IMAGE COLOR MODE")
    print("=" * 70)

    for mode, count in image_modes.items():
        print(f"{mode} : {count} images")

    print("=" * 70)


if __name__ == "__main__":
    check_dataset()