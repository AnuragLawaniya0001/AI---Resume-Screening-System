# Model Training Pipeline

This module handles the **training of machine learning / deep learning models** for text-based tasks such as:

- Resume ↔ Job Description (JD) matching
- Text similarity scoring
- Candidate shortlisting models

It is designed to take **raw text data**, process it, and train a model that outputs a **match score or prediction**.

---

## Overall Pipeline

The training workflow follows this structure:

Dataset → Preprocessing → Encoding → Model Training → Evaluation → Model Saving

---

## Project Structure


model-training/
│── model training.ipynb # Main training notebook
│── model testing.ipynb # (optional) script version
│── dataset.csv # Training dataset
│── models/
│ └── saved_model # Trained model output
│── requirements.txt
│── README.md


---

## Dataset Format

The model expects structured input like:

| resume_text | jd_text | label |
|-------------|--------|-------|
| "..."       | "..."  | 0.85  |

- **resume_text** → Candidate resume content  
- **jd_text** → Job description  
- **label** → Matching score (0–1 or 0–100)

---

## Steps in Training

### 1. Data Loading
- Load dataset using pandas
- Handle missing/null values

### 2. Text Preprocessing
- Lowercasing
- Removing special characters
- Cleaning extra spaces

### 3. Text Encoding
- Convert text into embeddings using:
  - BERT
  - Sentence Transformers (SBERT)

### 4. Model Training
- Train using:
  - Cosine similarity (embedding-based)
  - OR regression/classification model

### 5. Evaluation
- Metrics:
  - Accuracy (for classification)
  - MSE / MAE (for regression)
  - Cosine similarity score

### 6. Model Saving
- Save trained model for later inference

---

## How to Run

### Option 1: Jupyter Notebook
jupyter notebook "model training.ipynb"
### Option 2: Python Script
python train.py

