# 📈 Social Media Post Popularity Classifier

This project predicts whether a social media post will become *popular* based on features like text content, hashtags, and metadata.   
It's a practical application of supervised machine learning for digital marketing and content optimization.

---

## 🔍 Problem Statement

Content creators and marketers want to understand which posts will gain traction.  
This project builds a binary classifier that predicts post popularity using historical data, helping users optimize for reach and engagement.

---

## 🧠 Approach

- **Data Cleaning & Exploration**  
  Cleaned and explored a synthetic dataset mimicking real social media metrics (likes, shares, hashtags, post time, etc.).

- **Feature Engineering**  
  - Text vectorization using **TF-IDF**
  - Extracted numerical features (length, time posted, hashtag count)
  - Encoded categorical variables

- **Model Training**  
  Tested multiple models including:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting

- **Evaluation Metrics**  
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix

---

## 🛠️ Tools & Libraries

- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- Jupyter Notebook

---

## 📊 Results

- Achieved over 80% accuracy on test set.
- Identified key factors influencing popularity (e.g., time posted, keyword strength).
- Provided insights on content structure and engagement drivers.

---

## 📁 File Structure

