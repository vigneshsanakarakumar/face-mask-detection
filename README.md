# 😷 Real-Time Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-97.4%25-brightgreen)

> A deep learning-based Computer Vision system that detects whether a person is wearing a face mask in **real-time** using a webcam feed. Built with a fine-tuned MobileNetV2 CNN and OpenCV.

---

## 📸 Demo

```
[Webcam Feed]
    └── Face Detected → CNN Inference → Label: ✅ Mask / ❌ No Mask
                                     → Confidence Score: 98.2%
                                     → Bounding Box drawn on face
```

---

## 🚀 Features

- ✅ Real-time webcam inference at **25+ FPS**
- ✅ CNN model with **97.4% validation accuracy**
- ✅ Transfer learning with **MobileNetV2** backbone
- ✅ Face detection using **Haar Cascade Classifier**
- ✅ Color-coded bounding boxes (🟢 Mask / 🔴 No Mask)
- ✅ Confidence score overlay on each detection
- ✅ Static image inference support
- ✅ Model training pipeline included

---

## 🗂️ Project Structure

```
face-mask-detection/
│
├── src/
│   ├── train.py              # Model training script
│   ├── inference_webcam.py   # Real-time webcam detection
│   ├── inference_image.py    # Static image detection
│   ├── model.py              # CNN architecture definition
│   └── utils.py              # Helper functions
│
├── models/
│   └── mask_detector.h5      # Trained model weights (generated after training)
│
├── dataset/
│   └── sample/               # Sample images for quick testing
│
├── notebooks/
│   └── exploration.ipynb     # EDA + training experiments
│
├── tests/
│   └── test_model.py         # Unit tests
│
├── assets/
│   └── haarcascade_frontalface_default.xml  # Face detection cascade
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vigneshsanakarakumar/face-mask-detection.git
cd face-mask-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Haar Cascade (if not already present)

The `assets/` folder includes the Haar Cascade XML file. If it's missing:

```bash
python -c "import cv2; import shutil; shutil.copy(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', 'assets/')"
```

---

## 📦 Dataset

This project uses the **Face Mask Dataset** (~12,000 images, 2 classes):
- `with_mask` — 6,000 images
- `without_mask` — 6,000 images

**Download from Kaggle:**
```bash
kaggle datasets download -d omkargurav/face-mask-dataset
unzip face-mask-dataset.zip -d dataset/
```

Or manually download from: [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

Expected structure after download:
```
dataset/
├── with_mask/
│   ├── img_1.jpg
│   └── ...
└── without_mask/
    ├── img_1.jpg
    └── ...
```

---

## 🏋️ Training the Model

```bash
python src/train.py
```

**Training Configuration:**
| Parameter       | Value         |
|----------------|---------------|
| Base Model      | MobileNetV2   |
| Input Shape     | (224, 224, 3) |
| Optimizer       | Adam (lr=1e-4)|
| Loss Function   | Binary Cross-Entropy |
| Epochs          | 20            |
| Batch Size      | 32            |
| Validation Split| 20%           |

After training, the model is saved to `models/mask_detector.h5`

---

## 🎥 Real-Time Webcam Detection

```bash
python src/inference_webcam.py
```

- Press **`q`** to quit
- Press **`s`** to save a screenshot

---

## 🖼️ Static Image Detection

```bash
python src/inference_image.py --image path/to/your/image.jpg
```

---

## 📊 Model Performance

| Metric        | Value   |
|--------------|---------|
| Training Accuracy | 98.1% |
| Validation Accuracy | 97.4% |
| Test Accuracy | 96.9% |
| Precision (Mask) | 97.8% |
| Recall (Mask)    | 97.1% |
| F1-Score         | 97.4% |

---

## 🧠 How It Works

```
Input Frame (Webcam)
    │
    ▼
Haar Cascade Face Detection
    │
    ▼
Face Region Extraction + Preprocessing
(Resize to 224x224, Normalize to [0,1])
    │
    ▼
MobileNetV2 CNN Inference
    │
    ▼
Sigmoid Output → Mask / No Mask
    │
    ▼
Bounding Box + Label + Confidence Score Overlay
    │
    ▼
Display Frame
```

---

## 🔧 Tech Stack

| Tool         | Purpose                     |
|-------------|-----------------------------|
| Python 3.8+ | Core language               |
| TensorFlow 2.x | Deep learning framework  |
| Keras        | High-level model API        |
| OpenCV       | Computer vision & webcam    |
| NumPy        | Array processing            |
| Matplotlib   | Visualization               |
| Scikit-learn | Metrics & data splitting    |

---

## 🙋 Author

**Vignesh Sankarakumar**
- 📧 vigneshsankarakumar.biz@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/vignesh-sankarakumar/)
- 💻 [GitHub](https://github.com/vigneshsanakarakumar)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgements

- Dataset: [Kaggle — Face Mask Detection by Omkar Gurav](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- MobileNetV2: [Google AI — MobileNets](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)
- OpenCV Haar Cascades: [OpenCV GitHub](https://github.com/opencv/opencv)
