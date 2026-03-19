# Threads vs Twitter Reviews Analysis (NLP)

## Overview
This project performs a comparative Natural Language Processing (NLP) analysis of user reviews from Threads and Twitter.

The goal is to understand differences in user sentiment, platform perception, and key discussion topics using both classical and modern NLP techniques.

---

## Authors
Haeyeon Jeong | Sai Rachana Kandikattu | Snehitha Tadapaneni  

---

## Source
Original dataset: 
- Threads Reviews Dataset: [Kaggle - Threads Reviews](https://www.kaggle.com/datasets/saloni1712/threads-an-instagram-app-reviews/data)
- Twitter Reviews Dataset: [Kaggle - Twitter Reviews](https://www.kaggle.com/datasets/bwandowando/2-million-formerly-twitter-google-reviews)

---

## Key Features

- Data preprocessing and cleaning pipeline  
- Sentiment analysis using VADER + KNN threshold tuning  
- Classification models:
  - TF-IDF + Logistic Regression (baseline)
  - Fine-tuned DistilBERT  
- Topic modeling:
  - LDA
  - NMF
  - BERTopic (best performing)  
- Comparative analysis between Threads and Twitter  

---

## Methodology

The pipeline includes:

1. Data Cleaning  
   - Duplicate removal  
   - Text normalization  
   - Stopword removal and lemmatization  
   - Language filtering  

2. Sentiment Labeling  
   - VADER compound scores  
   - Threshold tuning using KNN  

3. Model Training  
   - Classical ML (TF-IDF + Logistic Regression)  
   - Deep Learning (DistilBERT)  

4. Topic Modeling  
   - BERTopic using Sentence-BERT + UMAP + HDBSCAN  

BERTopic provided the most coherent topics and DistilBERT outperformed baseline models.

---

## Installation & Requirements

### Install required packages:

```bash
pip install pandas numpy nltk emoji langdetect vaderSentiment scikit-learn gensim wordcloud seaborn matplotlib torch transformers sentence-transformers umap-learn hdbscan bertopic
```

### Environment Requirements:
This project was developed and tested with

- Python: 3.11
- Transformers: 4.33.3
- Datasets: 2.14.6
- HuggingFace Hub: 0.19.4
- Accelerate: 0.23.0
- BERTopic: 0.15.0
- Sentence-Transformers: 2.2.2
- NumPy: 1.26.4
- PyArrow: 12.0.1
- Scikit-Learn: 1.3.2

---

## How to Run

Run the main analysis script:

```bash
python comparative_analysis_threads_twitter.py
```

Make sure the dataset files are placed in the Data/ folder:

- threads_reviews.csv
- threads_reviews_labelled.csv
- twitter_reviews.csv
- twitter_reviews_labelled.csv

---

## Report

The full project report is available here:

- `report/Threads and Twitter Reviews.pdf`

The report includes:
- Background and research motivation  
- Dataset description and preprocessing pipeline  
- Sentiment analysis methodology and results  
- Model performance comparison (TF-IDF vs DistilBERT)  
- Topic modeling approach (LDA, NMF, BERTopic)  
- Final insights and conclusions  
