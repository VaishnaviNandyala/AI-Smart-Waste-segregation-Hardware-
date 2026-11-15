
from pathlib import Path
import random
import shutil
import hashlib

# ------------------------------------------------
# 1. Dataset paths
# ------------------------------------------------

SOURCE = Path("dataset/processed/all")
OUTPUT = Path("dataset/processed")

# Our four waste categories
classes = ["plastic", "paper", "metal", "organic"]

# Supported image formats
extensions = {".jpg", ".jpeg", ".png", ".webp"}

# Keep the split reproducible
random.seed(42)


# ------------------------------------------------
# 2. Check whether all four classes exist
# ------------------------------------------------

for class_name in classes:

    folder = SOURCE / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing class folder: {folder}"
        )


# ------------------------------------------------
# 3. Remove old split folders
# ------------------------------------------------

for split in ["train", "validation", "test"]:

    folder = OUTPUT / split

    if folder.exists():
        shutil.rmtree(folder)

print("Old split folders cleared.")


# ------------------------------------------------
# 4. Process each waste category
# ------------------------------------------------

for class_name in classes:

    source_folder = SOURCE / class_name

    images = []
    seen_hashes = set()

    # Find images and remove exact duplicates
    for image in source_folder.rglob("*"):

        if image.suffix.lower() not in extensions:
            continue

        # Calculate image hash
        digest = hashlib.sha256(
            image.read_bytes()
        ).hexdigest()

        # Keep only unique images
        if digest not in seen_hashes:

            seen_hashes.add(digest)
            images.append(image)

    # Check minimum image availability
    if len(images) < 3:

        raise ValueError(
            f"Not enough unique images in {class_name}"
        )

    # ------------------------------------------------
    # 5. Shuffle images
    # ------------------------------------------------

    random.shuffle(images)

    total = len(images)

    # Calculate split boundaries
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    # Divide images
    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    # ------------------------------------------------
    # 6. Copy images into split folders
    # ------------------------------------------------

    for split_name, split_images in splits.items():

        destination = (
            OUTPUT / split_name / class_name
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for index, image in enumerate(split_images):

            new_name = (
                f"{class_name}_{index}"
                f"{image.suffix.lower()}"
            )

            shutil.copy2(
                image,
                destination / new_name
            )

        print(
            f"{split_name} | {class_name}: "
            f"{len(split_images)} images"
        )


# ------------------------------------------------
# 7. Completion message
# ------------------------------------------------

print("\nDataset splitting completed successfully!")

print("\nFinal dataset location:")
print(OUTPUT.resolve())
