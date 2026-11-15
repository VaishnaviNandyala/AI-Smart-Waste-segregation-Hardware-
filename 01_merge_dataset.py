
from pathlib import Path
import shutil

# 1. Original dataset location
RAW = Path("dataset/raw")

# 2. Destination for organised images
OUTPUT = Path("dataset/processed/all")

# 3. Map original folders to our four classes
sources = {
    "plastic": ["trashnet/plastic"],
    "paper": ["trashnet/paper", "trashnet/cardboard"],
    "metal": ["trashnet/metal"],
    "organic": ["trashnet/organic"]
}

# 4. Supported image formats
extensions = {".jpg", ".jpeg", ".png", ".webp"}

# 5. Create destination folders and copy images
for class_name, source_folders in sources.items():

    destination = OUTPUT / class_name
    destination.mkdir(parents=True, exist_ok=True)

    count = 0

    for folder_name in source_folders:

        source = RAW / folder_name

        # Organic is optional until we download its dataset
        if not source.exists():
            if class_name == "organic":
                print("Organic dataset not added yet.")
                continue

            raise FileNotFoundError(
                f"Missing folder: {source}"
            )

        for image in source.rglob("*"):

            if image.suffix.lower() not in extensions:
                continue

            # Avoid filename collisions
            new_name = f"{folder_name.replace('/', '_')}_{image.name}"

            shutil.copy2(
                image,
                destination / new_name
            )

            count += 1

    print(f"{class_name}: {count} images copied")

print("\nDataset organisation completed!")


