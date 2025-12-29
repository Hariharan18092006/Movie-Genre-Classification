🎬 Movie Genre Classification using Machine Learning
📌 Project Overview

Movie genre classification is an important task in the entertainment industry for content recommendation, search optimization, and catalog organization. This project focuses on building a machine learning model that predicts the genre of a movie based on its plot summary or textual description using Natural Language Processing (NLP) techniques.

🎯 Objectives

Automatically classify movies into genres using text data

Apply NLP techniques for feature extraction

Compare different machine learning classifiers

Improve prediction accuracy for multi-class text classification

📂 Dataset Description

The dataset consists of:

Movie plot summaries or descriptions

Corresponding movie genres (e.g., Action, Drama, Comedy, Thriller, etc.)

Each record includes raw textual information and its associated genre label.

🧠 Techniques & Models Used
🔹 Text Preprocessing

Lowercasing text

Removing punctuation and special characters

Stopword removal

Tokenization

🔹 Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency)

(Optional) Word embeddings for capturing semantic meaning

🔹 Machine Learning Models

Naive Bayes – Efficient baseline classifier for text data

Logistic Regression – Linear model for multi-class classification

Support Vector Machine (SVM) – High-performance classifier for sparse textual features

⚙️ Methodology

Data loading and cleaning

Text preprocessing and normalization

Feature extraction using TF-IDF

Train-test split

Model training and evaluation

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

🚀 Results

The trained models successfully classify movie genres based on plot descriptions. SVM and Logistic Regression demonstrated strong performance, making them suitable for genre prediction tasks involving large textual datasets.

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

NLTK / SpaCy

📌 Conclusion

This project highlights how NLP and machine learning can be used to extract meaningful insights from unstructured text data. The approach can be extended using deep learning models or deployed as a recommendation and tagging system for movie platforms.# Movie-Genre-Classification
