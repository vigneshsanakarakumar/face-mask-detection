"""
inference_webcam.py
-------------------
Real-time face mask detection using webcam.
Run: python src/inference_webcam.py

Controls:
  q → Quit
  s → Save screenshot
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import preprocess_face, draw_prediction, load_face_cascade

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/mask_detector.h5"
CONFIDENCE_THRESHOLD = 0.6    # Minimum confidence to show label
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (60, 60)


def run_webcam_detection():
    """Main loop for real-time webcam detection."""

    # ── Load Model ────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at '{MODEL_PATH}'")
        print("   Run 'python src/train.py' first to train the model.")
        sys.exit(1)

    print("🔄 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded!")

    # ── Load Face Cascade ─────────────────────────────────────
    face_cascade = load_face_cascade()
    print("✅ Haar Cascade loaded!")

    # ── Start Webcam ──────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam. Check your camera connection.")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n🎥 Webcam started!")
    print("  Press 'q' to quit | Press 's' to save screenshot\n")

    # Stats tracking
    frame_count = 0
    mask_count = 0
    no_mask_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Face Detection ────────────────────────────────────
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE
        )

        # ── Inference for each face ───────────────────────────
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            # Preprocess and predict
            face_input = preprocess_face(face_roi)
            prediction = model.predict(face_input, verbose=0)[0][0]

            if prediction >= CONFIDENCE_THRESHOLD:
                label = "Mask"
                confidence = prediction
                mask_count += 1
            elif prediction <= (1 - CONFIDENCE_THRESHOLD):
                label = "No Mask"
                confidence = 1 - prediction
                no_mask_count += 1
            else:
                label = "Uncertain"
                confidence = max(prediction, 1 - prediction)

            # Draw result
            frame = draw_prediction(frame, x, y, w, h, label, confidence)

        # ── HUD Overlay ───────────────────────────────────────
        total_faces = mask_count + no_mask_count
        compliance = (mask_count / total_faces * 100) if total_faces > 0 else 0

        hud_lines = [
            f"Faces detected: {len(faces)}",
            f"Frame: {frame_count}",
            f"Mask Compliance: {compliance:.1f}%",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Press 'q' to quit | 's' to screenshot",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("😷 Face Mask Detection — Vignesh Sankarakumar", frame)

        # ── Key Handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n👋 Quitting...")
            break
        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n📊 Session Stats:")
    print(f"  Total Frames   : {frame_count}")
    print(f"  Mask Detected  : {mask_count}")
    print(f"  No Mask        : {no_mask_count}")
    print(f"  Compliance     : {compliance:.1f}%")


if __name__ == "__main__":
    run_webcam_detection()
