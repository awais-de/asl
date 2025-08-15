# 🖐️ English-to-Sign Language (ASL) Translation System  
*A Semester-Long NLP + Computer Vision Project*

---

## 📜 Overview  
This project implements a **multi-stage training and evaluation pipeline** for translating English sentences into **American Sign Language (ASL)** gloss and mapping them to corresponding sign videos.

Unlike basic rule-based approaches, this system integrates:  
- **Natural Language Processing (NLP)** for accurate English → ASL gloss translation.  
- **Dataset preprocessing and tokenization** for machine learning training.  
- **Text-to-gloss model training & evaluation** using PyTorch + Hugging Face.  
- **Optional video mapping** via the WLASL2000 dataset.  

You can run everything either via:  
- 📓 **`notebooks/TextToSignLanguage.ipynb`** (end-to-end workflow in a single notebook).  
- 🖥️ **Python scripts in `src/`** (modular execution).  

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
├── src/ # Source code (modular scripts)
│ ├── data_loading/
│ │ ├── load_videos_dataset.py
│ │ └── prepare_dataloader.py
│ ├── preprocessing/
│ │ ├── data_preprocessing.py
│ │ └── tokenize_aslg_pc12.py
│ ├── verification/
│ │ └── check_dataset_and_model.py
│ ├── models/
│ │ ├── train.py
│ │ └── evaluate.py
│ └── app.py
│
├── notebooks/ # Prototyping & experiments
│ └── TextToSignLanguage.ipynb # Main notebook
│
├── logs/ # Training and run logs
├── models/ # Saved checkpoints & trained models
├── requirements.txt # Dependencies
└── README.md # This file


---

## 🔄 Pipeline Overview  

The workflow can be followed either inside the **notebook** or step-by-step via **scripts**:

| **Step** | **Notebook Section / Script** | **Description** |
|----------|-------------------------------|-----------------|
| **0. Config Setup** | Notebook: *Setup & Imports* | Initialize paths, configs, and dependencies |
| **1. Load Datasets** | `src/data_loading/load_videos_dataset.py` | Download ASLG_PC12 & WLASL2000 |
| **2. Preprocess Datasets** | Notebook: *Data Preprocessing* / `src/preprocessing/data_preprocessing.py` | Clean datasets, build gloss-to-video mapping |
| **3. Tokenization** | Notebook: *Tokenization* / `src/preprocessing/tokenize_aslg_pc12.py` | Tokenize dataset for training |
| **4. DataLoader Prep** | Notebook: *DataLoader* / `src/data_loading/prepare_dataloader.py` | Build PyTorch DataLoaders |
| **5. Verification** | Notebook: *Sanity Checks* / `src/verification/check_dataset_and_model.py` | Validate setup |
| **6. Training** | Notebook: *Model Training* / `src/models/train.py` | Train text-to-gloss translation model |
| **7. Evaluation** | Notebook: *Evaluation* / `src/models/evaluate.py` | Generate predictions and evaluate model |
| **8. Gloss-to-Video Mapping** | Notebook: *Video Retrieval* | Map predicted gloss tokens to videos and stitch into final output |

---

## 📊 Datasets Used  
- **ASLG_PC12** — English ↔ ASL gloss parallel dataset.  
- **WLASL2000** — Gloss-to-video mapping dataset.  

*(Future integration planned for How2Sign, SignBank, etc.)*

---

## 🚀 Usage
1. Run LoadAndTrainDataset.ipynb
2. Run TextToSignLanguage.ipynb
