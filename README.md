# 🛰️ LULC Classification System (EuroSAT)

### *A Production-Ready Deep Learning Pipeline for Geospatial Satellite Imagery*

This project provides a professional, modular framework for Land Use and Land Cover (LULC) classification. It represents a significant architectural transition from a research-heavy Google Colab workflow into a scalable, high-performance local system. The pipeline supports both standard 3-channel **RGB** imagery and high-fidelity **13-band Multispectral** (Sentinel-2) data using the **EuroSAT** dataset.

---

## 👥 Project Team

* **Eng. Nourhan Magdy (Team Leader)**
  * Lead the project and prepared all the satellite data for the training pipeline.
* **Eng. Abdelrahamn Ashraf**
  * Performed model training, fine-tuning, and accuracy optimization.
* **Eng. Eslam Abdin**
  * Built the Streamlit application and organized the code for GitHub.
* **Eng. Heba Adel**
* **Eng. Ahmed Abdelkhalek**
* **Eng. Neamat Gamal**


---
## 🏗️ Project Architecture

| File | Purpose | Key Features |
|:--- |:--- |:--- |
| **config.py** | Centralized Configuration | Path management, hyperparameters, and band normalization stats. |
| **data_loader.py** | Unified Data Pipeline | Handles RGB/MS loading, 80/10/10 split, and adaptive augmentation. |
| **model.py** | Dynamic ResNet50 | Auto-adapts input layer for 3 or 13 channels; frozen backbone logic. |
| **engine.py** | Training Engine | Implemented with AMP, Smart Scheduling, and Early Stopping. |
| **train.py** | Main Entry Point | CLI for training RGB or MS models and saving metrics/plots. |
| **app.py** | Streamlit Dashboard | Dual visualization, False Color Composites, and Top-3 predictions. |
| **requirements.txt**| Dependencies | Locked versions for reproducibility (PyTorch, Rasterio, Streamlit). |

---

## 📊 Project Structure
```text

├── config.py                # Hyperparameters & path constants
├── data_loader.py           # Dataset classes and DataLoader factory
├── model.py                 # ResNet50 model definition
├── engine.py                # Trainer class (AMP & logic)
├── train.py                 # CLI for training models
├── app.py                   # Streamlit web application
├── requirements.txt         # Required libraries
├── models/                  # Best model weights (.pth) are saved here
├── outputs/                 # Metrics (JSON) and training plots (PNG)
│   └── plots/               # Confusion matrices and loss curves
├── EuroSAT_RGB/             # Local Data: 27K RGB JPGs (Excluded from Git)
└── EuroSAT_MS/              # Local Data: 27K MS TIFFs (Excluded from Git)
```

--- 

## 📋 Prerequisites & Data Setup

To maintain a lightweight repository, the dataset files are excluded. To run this project locally, you must provide the following:

1.  **Dataset**: Download the [EuroSAT Dataset](https://github.com/phelber/EuroSAT).
    - Ensure you have the `EuroSAT_RGB` (JPG files) and/or `EuroSAT_MS` (TIFF files) folders.
    - Place these in your local project directory.
      
2.  **Configure Paths**: 
    - Open `config.py` and update `DATA_RGB` and `DATA_MS` to match your local paths (e.g., `G:/download/depi/EuroSAT_RGB/`).
      
3.  **Model Weights**: 
    - The Streamlit app requires trained weights (e.g., `rgb_best.pth`) in the `models/` folder. You must run the training script first to generate these.

---

## 🚀 How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
The system is designed to handle different hardware and data modalities. You can train specific models using the CLI:

```bash
# Train the high-accuracy 13-band Multispectral model
python train.py --modality MS --epochs 50

# Train the 3-channel RGB model for faster testing
python train.py --modality RGB --epochs 50

# Train both modalities sequentially
python train.py --modality both --epochs 50
```
### 3. Launch the Dashboard
Once you have trained models (saved in the `models/` directory), start the interactive web application:

```bash
streamlit run app.py
```
* **Real-time Inference**: Upload any `.jpg` (RGB) or `.tif` (Multispectral) file to see predictions.
* **Geospatial Analysis**: Toggle between standard RGB views and **False Color Composites** to visualize vegetation health.
* **Confidence Scores**: View the Top-3 predicted classes with their probability percentages.

---

## 🔑 Key Improvements Over Notebooks

| Feature | Research Notebook | This Production System |
|:--- |:--- |:--- |
| **Organization** | Hardcoded variables in cells | Centralized, cross-platform `config.py` |
| **Efficiency** | Standard FP32 (Slow) | **AMP (Automatic Mixed Precision)** ~2x speedup |
| **Robustness** | Minimal error handling | Try-except blocks for corrupted 16-bit TIFFs |
| **Optimization** | Static Learning Rate | `ReduceLROnPlateau` + Early Stopping |
| **Deployment** | Static code blocks | Interactive **Streamlit Dashboard** |

---

## 📈 Advanced Features Implemented

1.  **Automatic Mixed Precision (AMP):** Optimized for significantly faster training and lower memory usage on NVIDIA GPUs.
2.  **Smart Learning Rate Scheduling:** Uses a dual-patience system (`ReduceLROnPlateau` for LR reduction and early stopping) to find the absolute optimal model state.
3.  **Robust TIFF Handling:** Implements specialized scaling (2nd and 98th percentile) to ensure multispectral data is properly processed, avoiding the "black image" pitfall common with 16-bit satellite sensors.
4.  **Geospatial Insights:** The dashboard provides a "False Color Composite" (NIR-Red-Green) toggle to highlight vegetation health, mimicking professional-grade GIS software.

---

## 🔮 Future Work

- **Live API Integration**: Transitioning from manual file uploads to a direct connection with the **Sentinel-2 API** or **Google Earth Engine** to classify real-time coordinates.
- **Temporal Change Detection**: Adding a time-series component to monitor how land use changes over multiple years (e.g., tracking urban sprawl or deforestation in a specific region).
- **Region-Specific Fine-Tuning**: Adapting the model specifically for Egyptian landscapes and urban features to maximize local accuracy for DEPI objectives.
- **Cloud Deployment**: Containerizing the application using **Docker** for deployment on cloud platforms like Azure or AWS.

---

## 🎓 DEPI Graduation Project
This system was engineered as a final graduation project for the **Digital Egypt Pioneers Initiative (DEPI)**. It demonstrates a complete Machine Learning engineering lifecycle—from raw geospatial data processing to a professional, deployable analytical application.
## 
