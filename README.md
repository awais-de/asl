# 🖐️ English-to-Sign Language (ASL) Translation System  
*A Semester-Long NLP + Computer Vision Project*

## 📜 Overview  
This project implements a **complex, multi-stage pipeline** that translates English sentences into **American Sign Language (ASL)**, then renders the translation as **sign videos** or **3D avatar animations**.

Unlike simple rule-based or one-step implementations, this system integrates:  
- **Natural Language Processing (NLP)** for grammar-aware English → ASL gloss translation.  
- **Computer Vision (CV)** for sign video retrieval and pose-based rendering.  
- **Dataset fusion** from multiple large-scale ASL datasets.

## 🎯 Project Goals  
1. **Translate** natural English text into **ASL gloss** (a written representation of sign language).  
2. **Align gloss tokens** with real-world ASL sign videos or pose sequences.  
3. **Render** an animated or video-based output for end-users.  
4. **Enable scalability** for more sign languages and larger vocabularies.

## 📂 Directory Structure  
asl_project/
├── data/ # Datasets and preprocessing outputs
│ ├── raw/ # Original datasets (ASLG-PC12, WLASL2000, How2Sign)
│ ├── processed/ # Processed / preprocessed dataset files & mappings
│ └── poses/ # Extracted pose keypoints from sign videos
│
├── src/ # Source code
│ ├── data_loading/ # Scripts to download and load datasets
│ │ └── load_videos_dataset.py
│ ├── preprocessing/ # Dataset cleaning, tokenization, alignment
│ │ └── preprocess_videos.py
│ ├── text_to_gloss/ # Text-to-gloss model and related scripts
│ │ └── text_to_gloss_model.py
│ ├── retrieval/ # Sign video retrieval / pose matching scripts
│ ├── rendering/ # Video player or avatar rendering code
│ ├── evaluation/ # Model evaluation metrics
│ ├── utils/ # Utility scripts for normalization, mapping, IO
│ └── app.py # CLI or web app interface
│
├── notebooks/ # Jupyter notebooks for prototyping and experiments
│
├── logs/ # Logs for training and experiments
├── models/ # Saved ML models and checkpoints
├── requirements.txt # Python dependencies
└── README.md # This file

## 🔄 Pipeline Overview  
1. **Data Preparation**: Download & preprocess ASLG-PC12, How2Sign, WLASL datasets using scripts in `src/data_loading/` and `src/preprocessing/`.  
2. **NLP Model**: Fine-tune transformer (T5/mBART) for English → ASL gloss inside `src/text_to_gloss/`.  
3. **Gloss Mapping**: Map gloss tokens to sign videos or generate fingerspelling.  
4. **Pose Rendering**: Render 3D avatar or 2D pose keypoints.  
5. **Output**: Play sign video sequence or avatar animation.

## 📊 Datasets  
- ASLG-PC12  
- WLASL2000  
- How2Sign  
- SignBank

## 💻 Installation  
```bash
git clone https://github.com/yourusername/asl_project.git
cd asl_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
