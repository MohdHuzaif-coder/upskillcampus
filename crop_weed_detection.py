"""
Crop and Weed Detection using Deep Learning (CNN)
=====================================================
An end-to-end Computer Vision project that classifies/detects images as
"Crop" or "Weed" using a Convolutional Neural Network (CNN) built with
TensorFlow/Keras. Includes training, evaluation, and a prediction function
for single-image inference.

Dataset:
--------
Works with any image-classification-style dataset organized like:

    dataset/
        train/
            crop/
                img1.jpg
                img2.jpg
                ...
            weed/
                img1.jpg
                ...
        val/
            crop/
            weed/
        test/
            crop/
            weed/


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ======================================================================
# 1. CONFIGURATION
# ======================================================================

DATASET_DIR = "dataset"                  # <-- change to your dataset root path
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2   # crop, weed
CLASS_NAMES = ["crop", "weed"]


# ======================================================================
# 2. DATA LOADING & AUGMENTATION
# ======================================================================

def build_data_generators():
    """Create train/val/test data generators with augmentation for training."""
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", classes=CLASS_NAMES
    )

    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", classes=CLASS_NAMES
    )

    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", classes=CLASS_NAMES, shuffle=False
    )

    print(f"Train samples: {train_gen.samples}, Val samples: {val_gen.samples}, "
          f"Test samples: {test_gen.samples}")
    return train_gen, val_gen, test_gen


def visualize_sample_images(generator, n=9):
    """Plot a grid of sample training images with labels."""
    images, labels = next(generator)
    plt.figure(figsize=(8, 8))
    for i in range(min(n, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        label = CLASS_NAMES[int(labels[i])]
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.close()
    print("Sample images saved as 'sample_images.png'")


# ======================================================================
# 3. MODEL 1 - CUSTOM CNN (built from scratch)
# ======================================================================

def build_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid")   # binary: crop (0) vs weed (1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ======================================================================
# 4. MODEL 2 - TRANSFER LEARNING (MobileNetV2, recommended)
# ======================================================================

def build_transfer_model(fine_tune_at=100):
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False   # freeze initially

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model


def fine_tune_model(model, base_model, fine_tune_at=100):
    """Unfreeze top layers of the base model and fine-tune at a low LR."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ======================================================================
# 5. TRAINING
# ======================================================================

def get_callbacks(checkpoint_path="best_crop_weed_model.keras"):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True
        )
    ]


def plot_training_history(history, filename="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Training history plot saved as '{filename}'")


# ======================================================================
# 6. EVALUATION
# ======================================================================

def evaluate_on_test(model, test_gen):
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    preds = model.predict(test_gen)
    pred_labels = (preds > 0.5).astype(int).flatten()
    true_labels = test_gen.classes

    print("\n--- Classification Report ---")
    print(classification_report(true_labels, pred_labels, target_names=CLASS_NAMES))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

    return test_acc, test_loss


# ======================================================================
# 7. SINGLE IMAGE PREDICTION
# ======================================================================

def predict_image(model, image_path):
    """Predict whether a single image is 'crop' or 'weed'."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = CLASS_NAMES[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Image: {image_path}")
    print(f"Prediction: {label} (confidence: {confidence:.2%})")
    return label, confidence


# ======================================================================
# 8. MAIN PIPELINE
# ======================================================================

def main(use_transfer_learning=True):
    # Step 1: Data generators
    train_gen, val_gen, test_gen = build_data_generators()
    visualize_sample_images(train_gen)

    # Step 2: Build model
    if use_transfer_learning:
        print("\nUsing Transfer Learning (MobileNetV2)...")
        model, base_model = build_transfer_model()
    else:
        print("\nUsing Custom CNN built from scratch...")
        model = build_custom_cnn()
        base_model = None

    model.summary()

    # Step 3: Train (frozen base, if transfer learning)
    callbacks = get_callbacks()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    plot_training_history(history)

    # Step 4: Fine-tune (only relevant for transfer learning)
    if use_transfer_learning:
        print("\nFine-tuning top layers of MobileNetV2...")
        model = fine_tune_model(model, base_model, fine_tune_at=100)
        fine_tune_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks
        )
        plot_training_history(fine_tune_history, filename="fine_tune_history.png")

    # Step 5: Evaluate on test set
    evaluate_on_test(model, test_gen)

    # Step 6: Save final model
    model.save("crop_weed_final_model.keras")
    print("\nFinal model saved as 'crop_weed_final_model.keras'")

    # Step 7: Example single-image prediction
    # predict_image(model, "dataset/test/weed/sample1.jpg")


if __name__ == "__main__":
    main(use_transfer_learning=True)


# ======================================================================
# OPTIONAL: OBJECT DETECTION VERSION USING YOLOv8 (bounding boxes)
# ======================================================================
"""
If your dataset has bounding-box annotations (YOLO format) instead of
folder-per-class images, you can train a real-time crop/weed *detector*
(not just classifier) very quickly using Ultralytics YOLOv8:

    pip install ultralytics

    from ultralytics import YOLO

    # data.yaml should define:
    #   train: path/to/train/images
    #   val: path/to/val/images
    #   names: ['crop', 'weed']

    model = YOLO("yolov8n.pt")               # start from pretrained nano model
    model.train(data="data.yaml", epochs=50, imgsz=640, batch=16)

    metrics = model.val()                    # evaluate on val set
    results = model.predict("test_image.jpg", save=True, conf=0.4)
