import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = r"D:\Ahmed Elrouby\DEPI\DEBI-ONL4_AIS2_S2\projects ML\NLP_project\Collage\restaurant_reviews.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}")

df = pd.read_csv(file_path)

print("Sample Data:")
print(df.head())

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemm.lemmatize(w) for w in words]
    return " ".join(words)

df["cleaned"] = df["sentence"].apply(clean_text)

print("\nAfter Cleaning:")
print(df.head())

X = df["cleaned"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("\nAccuracy =", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

while True:
    user_review = input("\nEnter a review to test (or 'q' to quit): ")
    if user_review.lower() == 'q':
        break
    cleaned = clean_text(user_review)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    print("Prediction:", prediction)
