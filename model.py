"""
model.py
--------
Defines the CNN architecture for face mask detection.
Uses MobileNetV2 as backbone with transfer learning.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_model(input_shape=(224, 224, 3), fine_tune_layers=20):
    """
    Build face mask detection model using MobileNetV2 transfer learning.

    Args:
        input_shape (tuple): Input image shape (H, W, C)
        fine_tune_layers (int): Number of top layers of base model to unfreeze

    Returns:
        tf.keras.Model: Compiled model
    """

    # ── Base Model (pretrained on ImageNet) ──────────────────
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,        # Remove final classification layer
        weights="imagenet"
    )

    # Freeze all layers initially
    base_model.trainable = False

    # Unfreeze the top N layers for fine-tuning
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # ── Custom Classification Head ────────────────────────────
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocessing: scale pixels to [-1, 1] (MobileNetV2 expects this)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Pass through base
    x = base_model(x, training=False)

    # Global average pooling (replaces Flatten, fewer params)
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout for regularization
    x = layers.Dropout(0.3)(x)

    # Dense layer
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # Output: sigmoid for binary classification (mask vs no mask)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="MaskDetector_MobileNetV2")

    return model


def compile_model(model, learning_rate=1e-4):
    """
    Compile model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile
        learning_rate (float): Adam optimizer learning rate

    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model = compile_model(model)
    model.summary()
