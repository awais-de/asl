# 🖐️ English-to-Sign Language (ASL) Translation System  
*A Semester-Long NLP + Computer Vision Project*

---

## 📜 Overview  
This project implements a **multi-stage training and evaluation pipeline** for translating English sentences into **American Sign Language (ASL)** gloss and mapping them to corresponding sign videos.

Unlike basic rule-based approaches, this system integrates:  
- **Natural Language Processing (NLP)** for accurate English → ASL gloss translation.  
- **Dataset preprocessing and tokenization** for machine learning training.  
- **Model training & evaluation** with checkpointing.  
- **Potential extensions** for video retrieval or avatar-based rendering.

---

## 🎯 Project Goals  
1. **Translate** natural English text into **ASL gloss**.  
2. **Align gloss tokens** with real-world ASL sign videos.  
3. **Train & evaluate** a text-to-gloss translation model.  
4. **Enable future integration** with video rendering or sign language avatars.

---

## 📂 Directory Structure  
asl_project/
├── data/ # Dataset storage and processing outputs
│ ├── raw/ # Original datasets (ASLG_PC12, WLASL2000)
│ ├── processed/ # Cleaned, tokenized datasets & mappings
│ └── poses/ # (Optional) Pose keypoints from sign videos
│
├── src/ # Source code
│ ├── data_loading/ # Scripts for loading datasets
│ │ ├── load_videos_dataset.py
│ │ └── prepare_dataloader.py
│ ├── preprocessing/ # Dataset cleaning & tokenization
│ │ ├── data_preprocessing.py
│ │ └── tokenize_aslg_pc12.py
│ ├── verification/ # Pre-training checks
│ │ └── check_dataset_and_model.py
│ ├── models/ # Training & evaluation scripts
│ │ ├── train.py
│ │ └── evaluate.py
│ ├── utils/ # Utility scripts
│ └── app.py # CLI or API entry point
│
├── notebooks/ # Prototyping & experiments
├── logs/ # Training and run logs
├── models/ # Saved checkpoints & trained models
├── requirements.txt # Dependencies
└── README.md # This file

---

## 🔄 Pipeline Overview  

The workflow is divided into **7 main steps**:

| **Step** | **Description** | **Main Script** | **Artifacts Required** | **Artifacts Generated** |
|----------|-----------------|-----------------|------------------------|-------------------------|
| **0. Configuration Setup** | Ensure `conf.json` exists with dataset URLs, paths, and parameters. | — | `conf.json` | `Run_info.json` |
| **1. Load Datasets** | Download ASLG_PC12 & WLASL2000 datasets from URLs in `conf.json`. | `src/data_loading/load_videos_dataset.py` | `conf.json` | Raw dataset files in `data/raw/` |
| **2. Preprocess Datasets** | Clean datasets, generate JSON mappings for gloss-to-video IDs. | `src/preprocessing/data_preprocessing.py` | Raw datasets | `data/processed/aslg_pc12_clean.jsonl`, `data/processed/gloss_to_videoid_map.json` |
| **3. Tokenization** | Tokenize gloss data and save as `.pt` file for PyTorch. | `src/preprocessing/tokenize_aslg_pc12.py` | `data/processed/aslg_pc12_clean.jsonl` | `data/processed/aslg_pc12_tokenized.pt` |
| **4. DataLoader Preparation** | Create PyTorch DataLoaders for training & evaluation. | `src/data_loading/prepare_dataloader.py` | Tokenized `.pt` file | — |
| **5. Pre-Training Verification** | Validate environment setup and dataset readiness. | `src/verification/check_dataset_and_model.py` | Tokenized `.pt` file | — |
| **6. Training** | Train the text-to-gloss model, save checkpoints. | `src/models/train.py` | — | Model checkpoints in `models/checkpoints/` |
| **7. Evaluation** | Evaluate trained model and generate predictions. | `src/models/evaluate.py` | Tokenized `.pt` file, model checkpoints | `Predictions.csv` |

---

## 📊 Datasets Used  
- **ASLG_PC12** — Primary gloss-text dataset.  
- **WLASL2000** — Gloss-to-video mapping dataset.  

*(Future integration planned for How2Sign, SignBank, etc.)*

---

## 💻 Installation  
```bash
git clone https://github.com/yourusername/asl_project.git
cd asl_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage  

### 1️⃣ Configuration Setup  
Edit `conf.json` to include dataset URLs, file paths, and training parameters.

---

### 2️⃣ Run the Pipeline  

```bash
# Step 1: Load datasets
python src/data_loading/load_videos_dataset.py

# Step 2: Preprocess datasets
python src/preprocessing/data_preprocessing.py

# Step 3: Tokenize dataset
python src/preprocessing/tokenize_aslg_pc12.py

# Step 4: Prepare DataLoader
python src/data_loading/prepare_dataloader.py

# Step 5: Verify pre-training setup
python src/verification/check_dataset_and_model.py

# Step 6: Train the model
python src/models/train.py

# Step 7: Evaluate the model
python src/models/evaluate.py
