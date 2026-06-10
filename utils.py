"""
utils.py
--------
Helper functions for preprocessing, visualization, and evaluation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MASK_COLOR = (0, 255, 0)      # Green for mask detected
NO_MASK_COLOR = (0, 0, 255)   # Red for no mask
LABELS = {0: "No Mask", 1: "Mask"}


def preprocess_face(face_img):
    """
    Preprocess a face region for model inference.

    Args:
        face_img (np.ndarray): BGR face image from OpenCV

    Returns:
        np.ndarray: Preprocessed image ready for model (1, 224, 224, 3)
    """
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    return face_expanded


def draw_prediction(frame, x, y, w, h, label, confidence):
    """
    Draw bounding box and label on the frame.

    Args:
        frame (np.ndarray): Video frame
        x, y, w, h (int): Bounding box coordinates
        label (str): "Mask" or "No Mask"
        confidence (float): Prediction confidence (0-1)
    """
    color = MASK_COLOR if label == "Mask" else NO_MASK_COLOR
    icon = "✓" if label == "Mask" else "✗"

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Background for label
    label_text = f"{icon} {label}: {confidence:.1%}"
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)

    # Label text (white)
    cv2.putText(frame, label_text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def plot_training_history(history, save_path=None):
    """
    Plot training & validation accuracy and loss curves.

    Args:
        history: Keras training history object
        save_path (str): Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", color="#1A56A4", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", color="#E05C2A", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", color="#1A56A4", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", color="#E05C2A", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Training plot saved to: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix for model evaluation.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path (str): Optional path to save
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Mask", "Mask"],
        yticklabels=["No Mask", "Mask"]
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Mask", "Mask"]))


def load_face_cascade():
    """Load OpenCV Haar Cascade for face detection."""
    import os
    cascade_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "haarcascade_frontalface_default.xml"
    )
    if not os.path.exists(cascade_path):
        # Fallback to OpenCV built-in
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError("❌ Haar Cascade XML not found. See README for setup instructions.")
    return cascade
