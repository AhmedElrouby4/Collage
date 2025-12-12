# ============================================
# STEP 1 — Load Dataset
# ============================================

import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv(r"D:\Ahmed Elrouby\DEPI\DEBI-ONL4_AIS2_S2\projects ML\NLP_project\Collage\restaurant_reviews.csv")

# Rename columns if needed
df = df.rename(columns={"sentence": "Review", "label": "Liked"})

# Check missing values
print(df.isnull().sum())

# Sentiment distribution
print(df['Liked'].value_counts())
print("-------------------------------------------")

# ============================================
# STEP 2 — Text Cleaning
# ============================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

df["cleaned_review"] = df["Review"].apply(clean_text)

# ============================================
# STEP 3 — Train-Test Split
# ============================================

X = df["cleaned_review"]
y = df["Liked"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# ============================================
# STEP 4 — TF-IDF
# ============================================

tfidf = TfidfVectorizer(max_features=3000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ============================================
# STEP 5 — Train Models
# ============================================

log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train_vec, y_train)
log_pred = log_reg.predict(X_test_vec)

svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_pred = svm.predict(X_test_vec)

# ============================================
# STEP 6 — Evaluation
# ============================================

print("\n===== Logistic Regression =====\n")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

print("\n===== SVM =====\n")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# ============================================
# STEP 7 — Add Predictions
# ============================================

df["predicted_sentiment"] = svm.predict(tfidf.transform(df["cleaned_review"]))
df["correct"] = (df["predicted_sentiment"] == df["Liked"])

# ============================================
# STEP 8 — Save Final Dataset
# ============================================

df.to_csv("final_processed_dataset.csv", index=False)
print("Saved final_processed_dataset.csv")
# ============================================
# STEP 10 — Manual User Input Prediction
# ============================================

while True:
    user_review = input("\nEnter a review (or type 'exit' to stop): ")

    if user_review.lower() == "exit":
        print("Exiting program...")
        break

    # Clean review same as training process
    cleaned = clean_text(user_review)

    # Vectorize
    vector = tfidf.transform([cleaned])

    # Predict using SVM
    pred = svm.predict(vector)[0]

    sentiment = "Positive 👍" if pred == 1 else "Negative 👎"
    print(f"Prediction: {sentiment}")
