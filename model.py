# ==========================
# Fake News Detection Training Script
# ==========================

import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load dataset (Excel, row 2 = header row)
df = pd.read_excel("Philippine Fake News 2.xlsx", header=1)

# Step 2: Keep only needed columns
df = df[['title', 'content', 'Label']]
print("\nFirst 5 rows of cleaned data:")
print(df.head())

# Step 3: Preprocessing
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # keep only letters/numbers
    return text

df['content'] = df['content'].apply(preprocess_text)

# Step 4: Split into features and labels
X = df['content']
y = df['Label']  # use uppercase L

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train Logistic Regression
model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

# Step 7: Evaluate model
y_pred = model.predict(X_test_tfidf)
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 8: Save model + vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n✅ Model and vectorizer saved successfully!")
