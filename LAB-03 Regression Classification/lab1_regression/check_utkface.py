import os
import cv2


# ==========================================
# Dataset Path
# ==========================================

dataset_path = "datasets/regression/UTKFace"


# ==========================================
# Get Image Files
# ==========================================

image_files = [
    file
    for file in os.listdir(dataset_path)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]


# ==========================================
# Display Information
# ==========================================

print("========== UTKFACE DATASET ==========")

print("Dataset path :", dataset_path)
print("Image count  :", len(image_files))


# ==========================================
# Display First 10 Files
# ==========================================

print("\n========== SAMPLE FILES ==========")

for file in image_files[:10]:
    print(file)


# ==========================================
# Read First Image
# ==========================================

if len(image_files) > 0:

    first_image = os.path.join(
        dataset_path,
        image_files[0]
    )

    image = cv2.imread(first_image)

    if image is not None:

        print("\n========== FIRST IMAGE ==========")

        print("Filename :", image_files[0])
        print("Shape    :", image.shape)

        print("\nDataset is ready!")

    else:

        print("\nERROR: Cannot read image.")

else:

    print("\nERROR: No images found.")