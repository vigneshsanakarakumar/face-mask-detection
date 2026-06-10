"""
train.py
--------
Train the face mask detection CNN model.
Run: python src/train.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import build_model, compile_model
from src.utils import plot_training_history

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "dataset_dir": "dataset",          # Path to dataset with with_mask/ and without_mask/
    "model_save_path": "models/mask_detector.h5",
    "img_size": (224, 224),
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-4,
    "validation_split": 0.2,
    "test_split": 0.1,
    "seed": 42,
}


def load_dataset(dataset_dir, img_size):
    """
    Load images from dataset directory.

    Expected structure:
        dataset/
        ├── with_mask/
        └── without_mask/

    Returns:
        X (np.ndarray): Image arrays normalized to [0, 1]
        y (np.ndarray): Binary labels (1 = mask, 0 = no mask)
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    X, y = [], []
    classes = {"with_mask": 1, "without_mask": 0}

    for class_name, label in classes.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  Directory not found: {class_dir}")
            continue

        images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  📂 {class_name}: {len(images)} images")

        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = load_img(img_path, target_size=img_size)
                arr = img_to_array(img) / 255.0
                X.append(arr)
                y.append(label)
            except Exception as e:
                print(f"  ⚠️  Skipped {img_file}: {e}")

    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def create_data_generators():
    """
    Create augmented data generators for training robustness.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator()  # No augmentation for validation

    return train_datagen, val_datagen


def main():
    print("=" * 60)
    print("  😷 Face Mask Detection — Model Training")
    print("=" * 60)

    # ── Load Dataset ──────────────────────────────────────────
    print("\n📦 Loading dataset...")
    X, y = load_dataset(CONFIG["dataset_dir"], CONFIG["img_size"])
    print(f"  ✅ Total images: {len(X)}")
    print(f"  ✅ With mask: {int(y.sum())} | Without mask: {int((y == 0).sum())}")

    # ── Train / Val / Test Split ───────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(CONFIG["validation_split"] + CONFIG["test_split"]),
        random_state=CONFIG["seed"],
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=CONFIG["test_split"] / (CONFIG["validation_split"] + CONFIG["test_split"]),
        random_state=CONFIG["seed"],
        stratify=y_temp
    )

    print(f"\n📊 Data Split:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Augmentation ──────────────────────────────────────────
    train_datagen, val_datagen = create_data_generators()

    # ── Build Model ───────────────────────────────────────────
    print("\n🏗️  Building model...")
    model = build_model(input_shape=(*CONFIG["img_size"], 3))
    model = compile_model(model, learning_rate=CONFIG["learning_rate"])
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            CONFIG["model_save_path"],
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # ── Train ─────────────────────────────────────────────────
    print(f"\n🚀 Starting training for up to {CONFIG['epochs']} epochs...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=CONFIG["batch_size"]),
        validation_data=val_datagen.flow(X_val, y_val, batch_size=CONFIG["batch_size"]),
        epochs=CONFIG["epochs"],
        steps_per_epoch=len(X_train) // CONFIG["batch_size"],
        validation_steps=len(X_val) // CONFIG["batch_size"],
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate on Test Set ──────────────────────────────────
    print("\n📊 Evaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy : {test_results[1]:.4f}")
    print(f"  Test Loss     : {test_results[0]:.4f}")

    # ── Plot Results ──────────────────────────────────────────
    print("\n📈 Generating training plots...")
    os.makedirs("assets", exist_ok=True)
    plot_training_history(history, save_path="assets/training_history.png")

    print(f"\n✅ Model saved to: {CONFIG['model_save_path']}")
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
