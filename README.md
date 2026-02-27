# Fake News Detection using NLP and Machine Learning

## Project Overview
Fake news has become a major concern in the digital era due to the rapid spread of misinformation through online platforms. This project aims to automatically classify news articles as **Fake** or **Real** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The system analyzes the textual content of news articles, extracts meaningful features, and applies a supervised learning model to make predictions. The project is implemented in **Python** using a notebook-based approach.

---

## Objectives
- To understand and apply NLP techniques for text classification  
- To build a supervised machine learning model for fake news detection  
- To evaluate the model using standard classification metrics  
- To predict whether unseen news articles are fake or real  

---

## Technologies Used
- Python  
- Jupyter Notebook  
- Pandas  
- NumPy  
- Scikit-learn  
- Natural Language Processing (TF-IDF)  

---

## Dataset Information
The model was trained using a labeled dataset consisting of **real** and **fake** news articles.

Due to GitHub’s file size limitations (25MB per file), the **full dataset is not included** in this repository.

### Sample Dataset (Included in Repository)
This repository contains small sample versions of the dataset for demonstration purposes:
- `sample_true.csv` – 100 real news articles  
- `sample_fake.csv` – 100 fake news articles  

These samples are provided to showcase the dataset structure and format.

> The full dataset was used during model training and evaluation but is excluded from this repository due to size constraints.

---

## Project Workflow
1. Data Loading  
2. Data Cleaning and Text Preprocessing  
3. Feature Extraction using TF-IDF Vectorization  
4. Train-Test Split  
5. Model Training using Logistic Regression  
6. Model Evaluation  
7. Prediction on New News Articles  

---

## Machine Learning Model
- **Algorithm Used:** Logistic Regression  
- **Type:** Supervised Learning  
- **Feature Extraction:** TF-IDF (Term Frequency–Inverse Document Frequency)

Logistic Regression was chosen due to its effectiveness and efficiency in text classification problems.

---

## Model Evaluation
The model performance was evaluated using:
- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)

The model demonstrated strong performance on the test dataset.

---

## Sample Prediction
The system can take a custom news article as input and classify it as:
- **Fake News**
- **Real News**

This functionality demonstrates how the trained model can be used for real-world prediction scenarios.

---

## Conclusion
This project successfully demonstrates the application of NLP and Machine Learning techniques for fake news detection. While the model performs well on test data, occasional misclassification may occur due to the probabilistic nature of machine learning models and ambiguity in language.

---

## Future Scope
- Improve accuracy using advanced deep learning models (LSTM, BERT)  
- Use larger and more diverse datasets  
- Extend the system to multilingual fake news detection  
- Deploy the model as a web or mobile application  

---

## References
- Python Documentation: https://docs.python.org/3/  
- Pandas Documentation: https://pandas.pydata.org/docs/  
- NumPy Documentation: https://numpy.org/doc/  
- Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html  
- TF-IDF Vectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html  
- Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  

---

## Author
**Anisha Shaw**  
B.Tech – Computer Science & Engineering  

---

## 📝 Note
This project was developed as part of a virtual internship and is intended for educational and learning purposes.
