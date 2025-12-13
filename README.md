# DeepFake Detection Using CNN-LSTM

This repository contains a **DeepFake video detection system** built using **Deep Learning (CNN + LSTM)**. The entire training pipeline is handled in **one main file**, and a **separate testing file** is provided to check whether a video is **Real or Fake**.

---

## 🚀 Project Overview

The goal of this project is to detect DeepFake videos by learning:

* **Spatial features** from individual video frames (CNN)
* **Temporal inconsistencies** across frames (LSTM)

The system is simple, modular, and suitable for academic / final-year projects.

---

## 🧠 Model Architecture

* **CNN** → Extracts facial spatial features
* **LSTM** → Captures temporal frame dependencies
* **Output Layer** → Binary classification

  * `0 = Real`
  * `1 = Fake`

---

## 📁 Complete File Structure (Single Run File)

```bash
DeepFake-web/
│
├── dataset/
│       ├── real/            # Original videos
│       ├── fake/            # DeepFake videos
│
├── app.ipynb                # ✅ MAIN FILE (training + saving model)
├── TEST_MODEL.ipynb         # ✅ Test a video (Real/Fake)
│
├── model_real/
│   └── model.h5        # Saved trained model
│   └── model.keras
│   └── model.weights.h5
│
└── README.md
```

---

## 🔄 Project Workflow

### 🔹 1. Training Phase (`app.ipynb`)

This is the **only file required to train the model**.

Steps performed:

1. Load CelebDF-v2 dataset (real & fake videos)
2. Extract fixed-length clips (e.g., 16 frames per video)
3. Detect & crop faces from each frame
4. Resize frames to `224 × 224` and normalize
5. Train CNN-LSTM model
6. Save best model as `models/best_model.h5`

---

### 🔹 2. Testing Phase (`TEST_MODEL.ipynb`)

Used **only for prediction**, not training.

Steps performed:

1. Load trained model (`model.keras`)
2. Input a single video file
3. Extract frames and faces
4. Predict video class

Output:

```text
REAL  → Authentic video
FAKE  → Manipulated video
```

---

## 📊 Dataset Information (CelebDF-v2)

### 🔗 Official Dataset Link

CelebDF-v2 (Celebrities DeepFake Dataset):
[https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)

> Dataset is for **research and educational use only**.

---

## 🗂 Dataset Division

### ✅ Real Videos

```bash
dataset/CelebDF-v2/real/
```

* Original, unmodified celebrity videos

### ❌ Fake Videos

```bash
dataset/CelebDF-v2/fake/
```

* DeepFake-generated face swap videos

---

## 📈 Dataset Statistics (Approx.)

| Type        | Count  |
| ----------- | ------ |
| Real Videos |    890 |
| Fake Videos |  5,645 |
| Total       |  6,535 |

> Dataset is **highly imbalanced** (handled during training).

---

## ⚙️ Training Configuration

* Optimizer: **Adam**
* Learning Rate: `1e-4`
* Epochs: `15`
* Batch Size: `8–16`
* Loss: `Binary Cross-Entropy`
* Early Stopping Enabled

---

## ▶️ How to Run

### 🔹 Train the Model

```bash
jupyter notebook app.ipynb
```

Run all cells to train and save the model.

---

### 🔹 Test a Video

```bash
python test_model.py --video path_to_video.mp4
```

Sample Output:

```text
Prediction: FAKE
Confidence: ~0.88
```

---

## 📌 Results

* Achieved ~**88% accuracy** on CelebDF-v2
* Stable training with early stopping
* Effective detection of temporal artifacts

---

## 🔮 Future Enhancements

* Transformer-based models
* Audio + video DeepFake detection
* Real-time webcam inference

---

## 👤 Author

**Shankar Kumar**
Final Year Project | Deep Learning & Computer Vision

---

## 📜 License

For **educational and research purposes only**. Dataset usage must follow CelebDF-v2 license terms.
