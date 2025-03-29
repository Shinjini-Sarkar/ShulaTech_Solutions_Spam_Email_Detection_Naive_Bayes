📧 Spam Email Detection using Naïve Bayes

📌 Overview

This project implements a Spam Email Classifier using Naïve Bayes and TF-IDF Vectorization. The model can distinguish between spam and non-spam (ham) emails by analyzing their text content. The dataset consists of spam and ham emails stored in separate folders.

🚀 Features

Natural Language Processing (NLP): Preprocessing email text (lowercasing, removing URLs, punctuation, stopwords, and tokenization).

TF-IDF Vectorization: Converts email text into numerical features.

Naïve Bayes Classifier: Trains a machine learning model to classify emails.

Spam Prediction: Predicts whether a given email is spam or not.

📂 Dataset

The dataset consists of two folders:

Spam Folder: Contains spam emails labeled as 1.

Ham Folder: Contains non-spam emails labeled as 0.

🛠️ Installation & Setup

1️⃣ Install Python & VS Code

Ensure you have Python installed. You can download it from Python.org.

Also, install VS Code from VS Code.

2️⃣ Install Dependencies

Open VS Code Terminal (Ctrl + ~) and run:

pip install pandas scikit-learn nltk matplotlib

3️⃣ Clone the Repository

git clone https://github.com/yourusername/spam-email-detection.git
cd spam-email-detection

4️⃣ Run the Project

Modify the folder paths in spam_detection.py and execute:

python spam_detection.py

📜 Code Explanation

1️⃣ Load Dataset

Reads spam and ham emails from their respective folders.

Stores them in a Pandas DataFrame with labels (0 for spam, 1 for ham).

2️⃣ Preprocessing Emails

Converts text to lowercase.

Removes URLs, HTML tags, numbers, and punctuation.

Tokenizes text and removes stopwords.

3️⃣ Feature Extraction

Uses TF-IDF Vectorization to convert text into numerical features.

4️⃣ Train the Model

Splits data into training and testing sets.

Trains a Multinomial Naïve Bayes classifier.

5️⃣ Evaluate Performance

Prints accuracy, classification report, and confusion matrix.

6️⃣ Test New Emails

Allows users to input text and predict if it’s spam or not.

🔍 Example Predictions

Email 1: "Congratulations! You've won a lottery of $1,000,000. Click here to claim now."
Prediction: Spam

Email 2: "Hey, are we still on for the meeting tomorrow?"
Prediction: Not Spam

📈 Model Performance

Accuracy: ~98%

Precision & Recall: Evaluated using the confusion matrix.

🛠️ Future Improvements

Integrate Deep Learning (LSTM/RNN) for better accuracy.

Deploy as a web app using Flask or Streamlit.

🌟 Acknowledgments

Kaggle Datasets

Scikit-Learn & NLTK Documentation

