# Medical AI Assistant: Segmentation & Diagnosis

> **End-to-end Machine Learning project that performs structural segmentation using PyTorch U-Net and generates detailed medical diagnoses using Google Gemini.**

---

## Overview

This project has been upgraded into a fully functional interactive web application. It takes medical scans (X-Rays, MRIs, CTs) and processes them through a two-step AI pipeline:

1. **U-Net Structural Segmentation:** A custom PyTorch U-Net model highlights and segments structural regions or potential anomalies inside the scan. 
2. **Generative Diagnosis:** Google's `gemini-3.6-flash` Vision model analyzes the raw scan alongside the structural context to generate a highly plausible diagnosis and a detailed treatment plan.

No deep ML expertise is required to run the final app. Everything is bundled into an interactive **Streamlit** web interface!

---

## Architecture

```
User Uploads Scan (X-Ray / MRI)
         |
         v
[OpenCV Preprocessing]
   - Resize to 224x224
   - ImageNet Normalization
         |
         v
[PyTorch U-Net Model]
   - Evaluates spatial features
   - Outputs color-coded segmentation mask overlay
         |
         v
[Google Gemini 3.6 Flash]
   - Receives the raw uploaded image
   - Analyzes anomalies
   - Outputs Diagnosis & Treatment Plan
         |
         v
[Streamlit Web UI]
   - Displays Side-by-Side Segmentation
   - Renders Diagnosis in Markdown
```

---

## Project Structure

```
ML majaor/
├── medical_coseg/
│   ├── unet.py               # PyTorch U-Net Architecture
│   ├── dataset.py            # Custom PyTorch Dataset
│   ├── postprocessing.py     # Morphology and Color mapping
│   └── ...                   # Clustering modules for pseudo-labels
├── app.py                    # Streamlit Web Application (Main Entrypoint)
├── train.py                  # Supervised Training Loop for U-Net
├── infer.py                  # Command-line Inference script
├── demo.py                   # Generates synthetic data and clusters
├── requirements.txt          # Python dependencies
└── outputs/                  
    └── unet_weights.pth      # Saved PyTorch Model weights
```

---

## Setup & Installation

**1. Install Python Dependencies**
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

*(This includes PyTorch, Streamlit, OpenCV, and the Google GenAI SDK).*

**2. API Keys**
The application uses the Google Gemini API to generate the medical diagnosis. The API key is currently hardcoded into `app.py`. If you wish to use a different key, simply update the `GEMINI_API_KEY` variable at the top of `app.py`.

---

## Usage

### 1. Run the Web Application (Recommended)
To launch the interactive web interface, run:
```bash
streamlit run app.py
```
Then, open your web browser to `http://localhost:8501`. 
Upload any image, wait a few seconds, and the AI will display the segmentation and treatment plan.

### 2. Retrain the U-Net Model
If you want to train the U-Net model from scratch (using synthetic generated data):
```bash
python train.py
```
This will run the zero-shot clustering pipeline to generate pseudo-labels, train the U-Net for 5 epochs, and save the new weights to `outputs/unet_weights.pth`.

### 3. Command Line Inference
If you just want to test the model without the web UI, you can run the inference script which generates predictions on new synthetic images:
```bash
python infer.py
```
Results will be saved to `outputs/unet_predictions.png`.
