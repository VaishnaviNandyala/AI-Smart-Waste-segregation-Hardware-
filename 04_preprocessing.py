
from pathlib import Path
from PIL import Image, ImageOps

# -----------------------------------------
# 1. Dataset configuration
# -----------------------------------------

SOURCE = Path("dataset/processed")
OUTPUT = Path("dataset/preprocessed")

classes = ["plastic", "paper", "metal", "organic"]
splits = ["train", "validation", "test"]

extensions = {".jpg", ".jpeg", ".png", ".webp"}

IMAGE_SIZE = (224, 224)


# -----------------------------------------
# 2. Clear old preprocessed dataset
# -----------------------------------------

if OUTPUT.exists():
    import shutil
    shutil.rmtree(OUTPUT)

print("Starting image preprocessing...\n")


# -----------------------------------------
# 3. Process images in each split
# -----------------------------------------

for split in splits:

    for class_name in classes:

        source_folder = SOURCE / split / class_name
        output_folder = OUTPUT / split / class_name

        if not source_folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {source_folder}"
            )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        image_files = [
            file for file in source_folder.rglob("*")
            if file.suffix.lower() in extensions
        ]

        processed_count = 0
        failed_count = 0

        for index, image_path in enumerate(image_files):

            try:
                # Open image
                with Image.open(image_path) as image:

                    # Correct image orientation
                    image = ImageOps.exif_transpose(image)

                    # Convert to RGB (3 colour channels)
                    image = image.convert("RGB")

                    # Resize to 224 x 224
                    image = image.resize(
                        IMAGE_SIZE,
                        Image.Resampling.LANCZOS
                    )

                    # Save processed image
                    output_path = (
                        output_folder /
                        f"{image_path.stem}_{index}.jpg"
                    )

                    image.save(
                        output_path,
                        format="JPEG",
                        quality=95
                    )

                processed_count += 1

            except Exception as error:

                failed_count += 1

                print(
                    f"Could not process {image_path}: {error}"
                )

        print(
            f"{split} | {class_name}: "
            f"{processed_count} processed, "
            f"{failed_count} failed"
        )


# -----------------------------------------
# 4. Completion
# -----------------------------------------

print("\nImage preprocessing completed!")

print("Output location:")
print(OUTPUT.resolve())
