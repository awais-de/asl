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
```
asl_project/
├── data/                  # Datasets and preprocessing outputs
│   ├── raw/               # Original datasets (ASLG-PC12, WLASL, How2Sign)
│   ├── processed/         # Tokenized and cleaned datasets
│   ├── poses/             # Extracted pose keypoints from sign videos
├── src/                   # Source code
│   ├── config.py          # Paths, constants, and hyperparameters
│   ├── preprocess.py      # Dataset cleaning, tokenization, alignment
│   ├── translation.py     # English → ASL gloss transformer model
│   ├── retrieval.py       # Sign video retrieval / pose matching
│   ├── render.py          # Video player or avatar rendering
│   ├── evaluate.py        # Model evaluation metrics
│   ├── app.py             # CLI or web app interface
├── notebooks/             # Jupyter notebooks for experiments
├── models/                # Saved ML models and checkpoints
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── LICENSE
```

## 🔄 Pipeline Overview
1. **Data Preparation**: Download & preprocess ASLG-PC12, How2Sign, WLASL datasets.
2. **NLP Model**: Fine-tune transformer (T5/mBART) for English → ASL gloss.
3. **Gloss Mapping**: Map gloss tokens to sign videos or generate fingerspelling.
4. **Pose Rendering**: Render 3D avatar or 2D pose keypoints.
5. **Output**: Play sign video sequence or avatar animation.

## 📊 Datasets
- ASLG-PC12
- WLASL
- How2Sign
- SignBank

## 💻 Installation
```
git clone https://github.com/yourusername/asl_project.git
cd asl_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage
```
python src/app.py --mode video
```

## 📈 Evaluation Metrics
- BLEU / ROUGE for translation
- Retrieval accuracy
- User comprehension studies

## 📅 Timeline
- Weeks 1-2: Literature review & dataset collection
- Weeks 3-4: Preprocessing
- Weeks 5-6: Model training
- Weeks 7-8: Retrieval
- Weeks 9-10: Rendering
- Weeks 11-12: Evaluation
- Weeks 13-14: Report & presentation

## 📜 License
MIT License
