
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# 1. Paths and settings
DATASET = Path("dataset/preprocessed/test")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASSES = ["plastic", "paper", "metal", "organic"]


# 2. Prepare test data for CNN
cnn_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

cnn_test_data = cnn_test_datagen.flow_from_directory(
    DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=False
)


# 3. Prepare test data for MobileNetV2
mobile_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

mobile_test_data = mobile_test_datagen.flow_from_directory(
    DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=False
)


# 4. Load all three trained models
models_to_evaluate = {
    "CNN Baseline": MODEL_DIR / "cnn_baseline.keras",
    "MobileNetV2 Initial": MODEL_DIR / "mobilenetv2_stage1.keras",
    "MobileNetV2 Fine-tuned": MODEL_DIR / "mobilenetv2_finetuned.keras"
}


# 5. Evaluate each model
comparison = {}
all_reports = {}

for model_name, model_path in models_to_evaluate.items():

    print("\n" + "=" * 55)
    print("Evaluating:", model_name)
    print("=" * 55)

    model = tf.keras.models.load_model(model_path)

    # Select the correct test preprocessing
    if model_name == "CNN Baseline":
        test_data = cnn_test_data
    else:
        test_data = mobile_test_data

    test_data.reset()

    # Generate predictions
    predictions = model.predict(test_data, verbose=1)

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    comparison[model_name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    # Class-wise classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    all_reports[model_name] = report

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    plt.savefig(
        RESULTS_DIR / f"{filename}_confusion_matrix.png"
    )
    plt.close()


# 6. Save model comparison
comparison_df = pd.DataFrame(comparison).T

print("\n\n========== MODEL COMPARISON ==========")
print(comparison_df.round(4))

comparison_df.to_csv(
    RESULTS_DIR / "model_comparison.csv"
)


# 7. Save detailed classification reports
with open(
    RESULTS_DIR / "classification_reports.json",
    "w"
) as file:
    json.dump(all_reports, file, indent=4)


# 8. Save comparison graph
metrics = ["accuracy", "precision", "recall", "f1_score"]

comparison_df[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("CNN vs MobileNetV2 Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "model_comparison.png"
)
plt.close()


print("\nModel evaluation completed!")
print("All reports and graphs saved in:", RESULTS_DIR)
