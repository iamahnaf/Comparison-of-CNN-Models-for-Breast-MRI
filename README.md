# 🩺 Breast MRI Tumor Classification
### Comparing Custom CNN · VGG16 · ResNet50

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-red?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20Accuracy-94.81%25-brightgreen?style=for-the-badge"/>
</p>

---

## 📌 Project Overview

This project compares three deep learning architectures for **binary classification of Breast MRI images** into:
- 🟢 **Benign** — non-cancerous tumor
- 🔴 **Malignant** — cancerous tumor

The models benchmarked are:
1. **Custom CNN** — built from scratch with 3 convolutional blocks
2. **VGG16** — pretrained on ImageNet (transfer learning, frozen base)
3. **ResNet50** — pretrained on ImageNet (transfer learning, frozen base)

A **Grad-CAM** explainability layer is included to visualize which regions in the MRI influenced each prediction.

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Breast MRI Tumor Classification Dataset](https://www.kaggle.com/datasets/abenjelloun/breast-mri-tumor-classification-dataset) |
| **Format** | JPEG images, RGB, resized to 224×224 |
| **Classes** | Benign · Malignant (balanced) |

### Dataset Split

| Split | Benign | Malignant | Total |
|-------|--------|-----------|-------|
| **Train** | 8,413 | 8,413 | **16,826** |
| **Val** | 1,797 | 1,797 | **3,594** |
| **Test** | 1,802 | 1,802 | **3,604** |

> The dataset is **perfectly balanced** — equal Benign and Malignant samples in every split.

---

## 🏗️ Model Architectures

### 1. 🔷 Custom CNN (Built from Scratch)

```
Input (224×224×3)
  → Rescaling (÷255) + Data Augmentation
  → Conv2D(32) → MaxPool2D
  → Conv2D(64) → MaxPool2D
  → Conv2D(128) → MaxPool2D
  → Flatten
  → Dense(512, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)
```

### 2. 🟣 VGG16 (Transfer Learning)

```
Input (224×224×3)
  → Data Augmentation
  → VGG16 Preprocess
  → VGG16 Base (frozen, ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(128, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)
```

### 3. 🟠 ResNet50 (Transfer Learning)

```
Input (224×224×3)
  → Data Augmentation
  → ResNet50 Preprocess
  → ResNet50 Base (frozen, ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(128, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)
```

---

## ⚙️ Training Configuration

| Setting | Value |
|---------|-------|
| **Image Size** | 224 × 224 |
| **Batch Size** | 32 |
| **Optimizer** | Adam (lr = 0.001) |
| **Loss** | Binary Cross-Entropy |
| **Metrics** | Accuracy, AUC |
| **Max Epochs** | 10 |
| **Early Stopping** | patience = 5 (Custom CNN) / 3 (VGG16, ResNet50) |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5) |

### 🔄 Data Augmentation
- Random Rotation (±10°)
- Random Zoom (10%)
- Random Translation (5%)
- Horizontal Flip

---

## 📊 Results

### ✅ Test Set Performance

| Model | Test Accuracy | Test AUC | Precision | Recall | F1-Score |
|-------|:------------:|:--------:|:---------:|:------:|:--------:|
| 🔷 **Custom CNN** | **94.81%** | — | 0.95 | 0.95 | 0.95 |
| 🟣 **VGG16** | 92.12% | **0.9816** | — | — | — |
| 🟠 **ResNet50** | 86.90% | 0.9521 | — | — | — |

> 🏆 **Custom CNN wins on accuracy** (94.81%) despite having no pretrained weights!  
> 🩺 **VGG16 achieves the highest AUC** (0.9816) — most reliable for clinical use since AUC measures discrimination ability regardless of threshold.

---

### 📈 Training Curves

#### Custom CNN — Accuracy, Loss & AUC per Epoch

| Epoch | Train Acc | Val Acc | Train AUC | Val AUC | Val Loss |
|:-----:|:---------:|:-------:|:---------:|:-------:|:--------:|
| 1 | 83.76% | 87.06% | 0.9218 | 0.9533 | 0.2822 |
| 2 | 85.91% | 90.98% | 0.9410 | 0.9769 | 0.1974 |
| 3 | 88.05% | 90.57% | 0.9563 | 0.9744 | 0.2109 |
| 4 | 89.37% | 92.74% | 0.9653 | 0.9832 | 0.1687 |
| 5 | 90.47% | **95.46%** | 0.9705 | **0.9920** | **0.1202** |

#### Classification Report — Custom CNN on Test Set

```
              precision    recall  f1-score   support

      Benign       0.95      0.95      0.95      1802
   Malignant       0.95      0.95      0.95      1802

    accuracy                           0.95      3604
   macro avg       0.95      0.95      0.95      3604
weighted avg       0.95      0.95      0.95      3604
```

---

## 🔥 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the **regions of the MRI that most influenced the model's decision**. This is critical in medical AI — doctors need to understand *why* a model made a prediction.

The implementation includes:
- 🗺️ Raw heatmap overlay
- 🎯 Top 15% activation regions highlighted
- 🔵 Contour boundary of important areas

---

## 🗂️ Project Structure

```
Breast_mri/
│
├── Breast_mri.ipynb        # Main notebook (EDA + training + evaluation)
├── .gitignore              # Ignores model weights, checkpoints, datasets
└── README.md               # You are here
```

---

## 🚀 How to Run

### Option 1: Run on Kaggle (Recommended)
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Import `Breast_mri.ipynb`
3. Enable **GPU accelerator** (P100 or T4)
4. Run All Cells

### Option 2: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/iamahnaf/Breast.git
cd Breast

# 2. Install dependencies
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn pillow kagglehub

# 3. Set up Kaggle API credentials
#    Place your kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<user>\.kaggle\ (Windows)

# 4. Launch Jupyter
jupyter notebook Breast_mri.ipynb
```

---

## 📦 Requirements

```
tensorflow>=2.13
scikit-learn
numpy
pandas
matplotlib
seaborn
pillow
kagglehub
```

---

## 🔑 Key Findings

- ✅ The **Custom CNN outperformed both pretrained models** on accuracy — likely because the MRI image distribution differs significantly from natural ImageNet images, reducing the advantage of pretrained weights without fine-tuning.
- 📊 **VGG16 achieved the best AUC (0.9816)**, making it the most reliable model for minimizing missed malignancies (false negatives) — which is the priority in cancer screening.
- ⚡ **ResNet50 performed worst** at 86.90% — its deeper residual connections may need fine-tuning to adapt well to this medical domain.
- 🧠 **Grad-CAM** confirms the Custom CNN focuses on anatomically relevant regions of the MRI for its predictions.

---

## 🔭 Future Work

- [ ] Fine-tune VGG16 and ResNet50 (unfreeze last few layers, train at lr = 1e-5)
- [ ] Add ROC curve comparison plot
- [ ] Experiment with EfficientNetB0 / DenseNet
- [ ] Add cross-validation for more robust evaluation
- [ ] Deploy as a web app using Gradio or Streamlit

---

## 👨‍💻 Author

**Ahnaf** — [@iamahnaf](https://github.com/iamahnaf)  
*Deep Learning · Computer Vision · Medical AI*

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <i>⭐ Star this repo if you found it helpful!</i>
</p>
