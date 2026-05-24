# Spam Detector: A Neural Network Approach for Text Classification

An end-to-end deep learning-based SMS spam detection system that classifies text messages as Spam or Ham (Not Spam) using an LSTM neural network.

##  Project Overview

SMS spam remains a major issue in mobile communication, often involving:

- Fake promotions
- Fraudulent prize notifications
- Phishing links
- Scam messages

This project implements a deep learning solution for automatically detecting spam messages using Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) networks.


## Objectives

- Build an automated SMS spam classification model
- Apply text preprocessing techniques
- Train an LSTM-based neural network
- Evaluate performance using multiple classification metrics
- Deploy the model for real-time predictions


##  Dataset

This project uses the SMS Spam Collection Dataset v1

### Dataset Statistics

- **Total Messages: 5,574
- **Ham Messages: 4,827
- **Spam Messages: 747

The dataset contains labeled SMS messages categorized as:

- Ham → Legitimate messages
- Spam → Promotional, fraudulent, or phishing messages

---

##  Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- NLTK


##  Model Architecture

The spam detection model uses a Sequential Neural Network with:

### 1. Embedding Layer
- Vocabulary Size: 5000
- Output Dimension: 64

### 2. LSTM Layer
- Units: 64

### 3. Dense Output Layer
- Activation: Sigmoid

---

## 🔄 Data Preprocessing

The following preprocessing steps were applied:

- Lowercasing text
- Removing special characters
- Removing stopwords
- Tokenization
- Sequence padding


##  Training Configuration

| Parameter | Value |
|----------|-------|
| Epochs | 5 |
| Batch Size | 32 |
| Train-Test Split | 80:20 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Evaluation Metric | Accuracy |


##  Results

### Model Performance

| Metric | Score |
|-------|------|
| Test Accuracy | 79.55% |
| Evaluation Accuracy | 99.01% |
| AUC Score | 0.9885 |
| Precision | 0.1742 |
| Recall | 0.1620 |
| F1 Score | 0.1679 |

---

##  Confusion Matrix

| Prediction | Count |
|-----------|------|
| True Negatives | 969 |
| False Positives | 2 |
| False Negatives | 14 |
| True Positives | 130 |


##  Features

✔ SMS text classification  
✔ Spam probability prediction  
✔ Deep learning-based sequence modeling  
✔ Performance visualization  
✔ End-to-end workflow from preprocessing to deployment


##  Limitations

The project has several limitations:

- Class imbalance affects spam detection
- Low precision and recall for minority spam class
- Inconsistent preprocessing during inference
- Limited validation monitoring
- Lack of reproducibility controls


##  Future Improvements

Potential improvements include:

- Oversampling / Undersampling
- Cost-sensitive learning
- Hyperparameter tuning
- Threshold optimization
- Transformer-based architectures (BERT/XLM-R)
- Better feature engineering

