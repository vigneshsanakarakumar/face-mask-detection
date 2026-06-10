"""
inference_image.py
------------------
Run face mask detection on a static image.
Run: python src/inference_image.py --image path/to/image.jpg
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import preprocess_face, draw_prediction, load_face_cascade

MODEL_PATH = "models/mask_detector.h5"


def run_image_detection(image_path, output_path=None, show=True):
    """
    Detect face masks in a static image.

    Args:
        image_path (str): Path to input image
        output_path (str): Optional path to save annotated output
        show (bool): Whether to display the result
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at '{MODEL_PATH}'. Run train.py first.")
        sys.exit(1)

    # Load model and cascade
    print("🔄 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    face_cascade = load_face_cascade()

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not read image: {image_path}")
        sys.exit(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    print(f"🔍 Found {len(faces)} face(s)")

    results = []

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_input = preprocess_face(face_roi)
        prediction = model.predict(face_input, verbose=0)[0][0]

        label = "Mask" if prediction >= 0.5 else "No Mask"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        frame = draw_prediction(frame, x, y, w, h, label, confidence)
        results.append({"bbox": (x, y, w, h), "label": label, "confidence": float(confidence)})
        print(f"  Face at ({x},{y}): {label} ({confidence:.2%})")

    # Save output
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"✅ Saved annotated image to: {output_path}")

    # Display
    if show:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 7))
        plt.imshow(rgb_frame)
        plt.title("Face Mask Detection Result", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection on Static Image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default=None, help="Path to save annotated output image")
    parser.add_argument("--no-display", action="store_true", help="Don't display the result window")
    args = parser.parse_args()

    run_image_detection(
        image_path=args.image,
        output_path=args.output,
        show=not args.no_display
    )
