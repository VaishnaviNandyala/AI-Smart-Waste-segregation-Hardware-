
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input
)
from sklearn.utils.class_weight import compute_class_weight


# 1. Project paths and settings
DATASET = Path("dataset/preprocessed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

CLASSES = ["plastic", "paper", "metal", "organic"]

tf.keras.utils.set_random_seed(42)


# 2. Image augmentation and preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# 3. Load training images
train_data = train_datagen.flow_from_directory(
    DATASET / "train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

validation_data = validation_datagen.flow_from_directory(
    DATASET / "validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=False
)


# 4. Calculate class weights
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=train_data.classes
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights_array)
}

print("\nClass weights:")
print(class_weights)


# 5. Load pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False


# 6. Add our own four-class classification head
inputs = layers.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = models.Model(inputs, outputs)


# 7. Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# 8. Training callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "mobilenetv2_stage1.keras",
        monitor="val_loss",
        save_best_only=True
    )
]


# 9. Train the classification head
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# 10. Save trained model
model.save(MODEL_DIR / "mobilenetv2_stage1.keras")


# 11. Save training history
history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(RESULTS_DIR / "mobilenetv2_history.json", "w") as file:
    json.dump(history_data, file, indent=4)


# 12. Save accuracy plot
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(RESULTS_DIR / "mobilenetv2_accuracy.png")
plt.close()


# 13. Save loss plot
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("MobileNetV2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(RESULTS_DIR / "mobilenetv2_loss.png")
plt.close()

print("\nMobileNetV2 initial training completed!")
print("Model saved in:", MODEL_DIR)
print("Results saved in:", RESULTS_DIR)
