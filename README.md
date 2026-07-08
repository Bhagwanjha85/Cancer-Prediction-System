# OncoVision: Deep Learning Cancer Detection Portal

OncoVision is an advanced, production-ready AI-powered web portal designed to assist in screening for **Oral Cancer** and **Skin Cancer**. The system uses PyTorch Convolutional Neural Networks (CNNs) with transfer learning, integrates robust pre-validation check (to reject non-human images), and provides explainable AI insights via Grad-CAM heatmaps.

---

## 🌟 Key Features
- **Dual Cancer Detection:** Categorizes images into four classes: `Oral_Cancer`, `Oral_Normal`, `Skin_Cancer`, and `Skin_Normal`.
- **Pre-validation Image Check:** Integrates a pre-trained ImageNet classifier (MobileNetV2) to verify if an uploaded image is a human body part, preventing erroneous predictions on objects, pets, or outdoor scenes.
- **Model Sweeps & Evaluation:** Trains and compares 7 transfer learning backbones (`DenseNet121`, `ResNet50`, `EfficientNetB0`, `EfficientNetB3`, `MobileNetV3`, `Xception`, `VGG16`) and automatically selects the highest-performing option.
- **Explainable AI (Grad-CAM):** Computes and renders gradient-weighted activation maps to highlight exactly where the model is looking when making a prediction.
- **Clinical Report Export:** Generates clinical-grade PDF reports including metadata, predictions, Grad-CAM overlays, disease details, and medical recommendations.
- **Prediction History Logging:** Logs screening events locally to a CSV database for subsequent analysis or export.
- **GPU Acceleration Support:** Autodetects compatible CUDA GPUs for accelerated model training with fallback to CPU.

---

## 📁 Folder Structure

```text
CancerDetection/
│
├── .streamlit/
│   └── config.toml          # Streamlit theme & file upload configuration
│
├── assets/
│   └── css/
│       └── style.css        # Custom CSS styles (glassmorphism cards, fonts, buttons)
│
├── models/                  # Holds final trained artifacts
│   ├── best_model.pth       # Automatically chosen best model (PyTorch package)
│   ├── weights.pth          # Best model weight weights
│   ├── label_encoder.pkl    # Serialized class labels mapping
│   └── history.pkl          # Training history and evaluation metrics
│
├── pages/                   # Multi-page layout files
│   ├── 1_Prediction.py      # Diagnostic Prediction Portal
│   ├── 2_Performance.py     # Architecture Performance & Evaluation
│   ├── 3_Dataset.py         # Dataset Statistics & Distributions
│   ├── 4_About.py           # Deep Learning Concepts & Project Workflow
│   └── 5_Contact.py         # Developer Contact Profile
│
├── dataset/                 # Dataset folder (automatically mocked if empty)
│   ├── oral/
│   │   ├── train/           # Oral train directories (Cancer/Normal)
│   │   ├── validation/      # Oral validation directories (Cancer/Normal)
│   │   └── test/            # Oral test directories (Cancer/Normal)
│   └── skin/
│       ├── train/           # Skin train directories (Cancer/Normal)
│       ├── validation/      # Skin validation directories (Cancer/Normal)
│       └── test/            # Skin test directories (Cancer/Normal)
│
├── app.py                   # Main landing page (Home Page)
├── config.py                # Hyperparameters, directories, and global configs
├── dataset_loader.py        # PyTorch Dataset pipeline & mock data generator
├── preprocessing.py         # Bilateral noise filters, scaling, and augmentations
├── body_detector.py         # ImageNet validation classifier (non-human filter)
├── train.py                 # Multi-model training and evaluation script (PyTorch)
├── predict.py               # Unified prediction orchestration class (PyTorch)
├── gradcam.py               # Grad-CAM heatmap engine (PyTorch hooks)
├── performance.py           # Evaluation history parser
├── plots.py                 # Matplotlib & Seaborn styling
├── medical_info.py          # Clinical advice, symptoms, and disclaimer database
├── utils.py                 # GPU diagnostics, CSV loggers, and ReportLab PDF compiler
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone & Initialize Workspace
Open your terminal (PowerShell/Command Prompt) in the directory:
```bash
cd d:\Resume_Projects\CancerDetection
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Training & Evaluation

The training pipeline automatically scans the `dataset/` folder. If no dataset is found, it will **automatically generate a mock synthetic dataset** containing healthy/lesion patterns in skin/oral colors to ensure the entire portal is instantly runnable!

To begin training and benchmark all 7 models:
```bash
python train.py
```

*During training, the PyTorch training pipeline will:*
1. Initialize PyTorch transforms and DataLoaders.
2. Cycle through and train all 7 transfer learning networks (from the `timm` library).
3. Apply optimizer learning rate drops, track validation losses, and checkpoint weights.
4. Compare validation accuracies, select the best model, and save `best_model.pth`, `weights.pth`, `label_encoder.pkl`, and `history.pkl`.

---

## 🚀 Running the Streamlit Web Application

To launch the portal:
```bash
streamlit run app.py
```
This launches a browser session pointing to `http://localhost:8501`. 

### Web App Navigation:
- **🧬 Home Page (`app.py`):** Visual diagnostic dashboard, system diagnostic check, and workflow flowchart.
- **🔮 Prediction Portal (`pages/1_Prediction.py`):** File drag-and-drop, ImageNet pre-validation check, Grad-CAM visual highlight, clinical descriptions, and PDF/CSV report exports.
- **📊 Model Performance (`pages/2_Performance.py`):** Interactive model comparison metrics (Confusion Matrix, ROC curves, accuracy/loss graphs).
- **📁 Dataset Information (`pages/3_Dataset.py`):** Splits representation, pie charts, and grids displaying actual dataset samples.
- **📖 About Page (`pages/4_About.py`):** Core explanation of Transfer Learning, CNN layers, and Grad-CAM math.
- **📧 Contact (`pages/5_Contact.py`):** Developer information and email submit form.

---

## 🔮 Explainable AI (Grad-CAM) Workflow
Grad-CAM calculates the visual focus weights using the following steps:
1. Calculates the forward pass of the preprocessed image in PyTorch.
2. Registers a hook to extract activations and gradients of the final 2D convolutional layer of the backbone model.
3. Computes the gradient of the highest class probability output with respect to the extracted conv feature maps.
4. Pools the gradients across spatial dimensions to calculate channel importance weights.
5. Computes a weighted combination of activation channels and applies a ReLU filter to extract positive activation maps.
6. Resizes the resulting activation maps and overlays them on the input image using a jet color map.

---

## 🏥 Critical Medical Disclaimer
> [!IMPORTANT]  
> **OncoVision is not a medical device.** The AI models incorporated are intended only for research, education, and screening awareness. They should never be used as a substitute for professional clinical advice, biopsy verification, or treatment planning. Always consult a qualified physician or dermatologist.

---

## 🔮 Future Improvements
- **Interactive Segmentation:** Implement SAM (Segment Anything Model) or U-Net to draw exact contours around cancer lesions.
- **Clinical Metadata Integration:** Combine image features with patient medical history (age, smoking habits, family history) for multi-modal prediction.
- **DICOM File Support:** Add medical standard format loaders for direct clinic compatibility.
