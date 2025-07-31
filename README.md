# Virtual Internship: Predicting Team Performance from Chatlogs  

## 📌 Overview  
This project analyzes chatlogs from the *Nephrotex* virtual internship program, where students collaborated to design biomedical devices for kidney failure patients. Using **machine learning** and **natural language processing (NLP)**, we predicted team performance scores (0–8) based on communication behaviors.  

**Team**: Gue Zhen Xue, Andres Xue, Zohaib Javed, Denisha Fam Wen Hsiu  
**Supervisor**: Dr. Simon Clarke  
**Date**: June 5, 2025  

---

## 🎯 Objectives  
1. **Aggregate** chat data into team-level statistics.  
2. **Build predictive models** to forecast final report scores.  
3. **Interpret** how communication features influence performance.  

---

## 🔍 Methodology  

### 📂 Data Preprocessing  
- **Text Normalization**: Handled slang, abbreviations, and misspellings (e.g., "lo!" → "laugh out loud").  
- **Feature Engineering**:  
  - **TF-IDF**: Extracted 4,400+ term importance features.  
  - **Sentiment Analysis**: Quantified emotional tone (scores from -1 to 1).  
  - **Structural Features**: Calculated team size, mentor ratios, and engagement metrics.  
- **Data Aggregation**: Used **Batch Gradient Descent** (lowest MSE: 1.75) to merge individual data into team-level stats.  

### 🤖 Models Tested  
| Model               | Best Accuracy (Test) | Best F1-Score |  
|---------------------|----------------------|---------------|  
| **Random Forest** (Tuned) | 0.759               | 0.758         |  
| SVM (Linear Kernel) | 0.713               | 0.710         |  
| Decision Tree       | 0.702               | 0.690         |  
| **Neural Networks** (LSTM/CNN) | *See below*       | *See below*   |  

### 🧠 Neural Network Architectures  
We experimented with **LSTM** and **CNN** to capture sequential and spatial patterns in text:  

#### **LSTM**  
- **Layers**: Embedding → LSTM (128 units) → Dropout (0.5) → Dense (softmax).  
- **Use Case**: Leveraged for sequential chatlog analysis.  

#### **CNN**  
- **Layers**: Embedding → Conv1D (64 filters, kernel=3) → MaxPooling → Flatten → Dense (softmax).  
- **Use Case**: Detected local phrase-level patterns.  

*Note: Neural networks showed promise but were limited by dataset size and computational constraints.*  

---

## 📊 Key Findings  
- **Top Features**: Words like *"hit"* and *"please"* had the highest importance (though all features contributed weakly).  
- **Surprise**: Mentor interactions had **minimal impact** on outcomes (contrary to our hypothesis).  
- **Best Model**: **Tuned Random Forest** (accuracy: 75.9%, F1: 75.8%).  

---

## 🚧 Limitations & Future Work  
- **Text Normalization**: NLTK’s lemmatization missed variations (e.g., "describe" vs. "describes").  
  - *Fix*: Use **spaCy** for better NLP pipelines.  
- **Class Imbalance**: Resampling reduced bias but didn’t eliminate it.  
  - *Fix*: Apply SMOTE **only to training data** to avoid leakage.  
- **Neural Networks**: Needed more data for optimal performance.  
  - *Fix*: Explore **transfer learning** (e.g., BERT embeddings).  

---

## 🛠️ Tools Used  
- **Python Libraries**: Scikit-learn, NLTK, TensorFlow/Keras (for LSTM/CNN), Pandas.  
- **Aggregation**: Batch Gradient Descent.  
- **Visualization**: Matplotlib, Seaborn.  

---
