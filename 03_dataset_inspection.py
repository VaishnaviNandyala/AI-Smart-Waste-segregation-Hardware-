
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Dataset configuration
# -----------------------------------------

DATASET = Path("dataset/processed")

classes = ["plastic", "paper", "metal", "organic"]

splits = ["train", "validation", "test"]

extensions = {".jpg", ".jpeg", ".png", ".webp"}


# -----------------------------------------
# 2. Inspect every class in every split
# -----------------------------------------

records = []

for split in splits:

    for class_name in classes:

        folder = DATASET / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

        images = [
            file for file in folder.rglob("*")
            if file.suffix.lower() in extensions
        ]

        valid_count = 0
        invalid_count = 0

        # Check whether each image can be opened
        for image_path in images:

            try:
                with Image.open(image_path) as image:
                    image.verify()

                valid_count += 1

            except Exception:
                invalid_count += 1

        records.append({
            "Split": split,
            "Class": class_name,
            "Valid Images": valid_count,
            "Invalid Images": invalid_count
        })

        print(
            f"{split} | {class_name}: "
            f"{valid_count} valid, "
            f"{invalid_count} invalid"
        )


# -----------------------------------------
# 3. Create inspection report
# -----------------------------------------

df = pd.DataFrame(records)

print("\n========== DATASET REPORT ==========\n")

print(df.to_string(index=False))


# -----------------------------------------
# 4. Save report as CSV
# -----------------------------------------

Path("results").mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    "results/dataset_report.csv",
    index=False
)

print("\nCSV report saved!")


# -----------------------------------------
# 5. Calculate total images per split
# -----------------------------------------

summary = df.groupby("Split")[
    ["Valid Images", "Invalid Images"]
].sum()

print("\n========== TOTAL IMAGES ==========\n")

print(summary)


# -----------------------------------------
# 6. Plot class distribution
# -----------------------------------------

pivot = df.pivot(
    index="Class",
    columns="Split",
    values="Valid Images"
)

pivot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Waste Dataset Distribution")
plt.xlabel("Waste Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "results/dataset_counts.png"
)

plt.show()

print("\nGraph saved in results/dataset_counts.png")

print("\nDataset inspection completed!")
