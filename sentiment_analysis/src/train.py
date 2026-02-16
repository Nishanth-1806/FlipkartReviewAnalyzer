import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_data, preprocess_dataframe

# Define paths
# Using data from sibling directory
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'reviews_badminton', 'data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')

def train():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print("Preprocessing data...")
    df = preprocess_dataframe(df)
    
    X = df['cleaned_text']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Pipeline: TF-IDF -> Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print("Done!")

if __name__ == "__main__":
    train()
