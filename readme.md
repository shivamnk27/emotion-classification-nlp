# Emotion Classification with DistilBERT and Model Ensemble

## 📖 Overview
This repository provides two complementary Python (<3.12) scripts for multi-class emotion recognition from text:

1. **`main.py`** – trains three lightweight base learners (TF‑IDF + Logistic Regression, Bi‑LSTM, and Sentence‑BERT + k‑Nearest Centroids) and combines their predictions through soft‑voting.
2. **`DistillBERT.py`** – fine‑tunes 🤗 Transformers **DistilBERT** on the *track‑a.csv* dataset.  

The script expect the same input CSV and export ready‑to‑use models plus a `predict()` helper.

## 🗂️ Repository Structure
```text
├── track-a.csv # raw dataset (text,label)
│              
├── main.py
├── DistillBERT.py
├── requirements.txt
└── README.md  (this file)
```
## 📊 Power BI Dashboard Extension
To demonstrate the model's practical application, I connected its CSV output to Power BI to build an interactive dashboard. This dashboard allows non-technical users to visually analyze and filter emotion-based patterns in the text data, as shown in the preview image below, which is filtered for the "anger" emotion. This new report provides a user-friendly interface to explore the model's classifications in real-time. The complete Power BI project file (NLP_Emotion_Analysis_Dashboard.pbix) is also available in this repository for a detailed review of the data transformations and model.

!(dashboard-preview.png)

## ⚙️ Installation
```bash
# 1. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## 🗄️ Data Format
`track-a.csv` **must** contain at least:

```csv
id,text,anger,fear,joy,sadness,surprise
eng_train_track_a_00001,"Colorado, middle of nowhere.",0,1,0,0,1
eng_train_track_a_00002,This involved swimming a pretty large lake that was over my head.,0,1,0,0,0
...
```
Labels are case‑insensitive strings and are automatically label‑encoded.

## 🚀 Quick Start

### python main.py {your test csv file}


## 🩹 Troubleshooting
* **Training stalls at 100 % CPU** – ensure TensorFlow sees your GPU (`tf.debugging.set_log_device_placement(True)`).  
* **Low F1‑score** – double‑check class distribution; consider stratified split and more epochs.


