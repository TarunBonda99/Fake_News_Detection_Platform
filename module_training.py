import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
from preprocess import preprocess_text
import os

def train_model():
    df = pd.read_csv('fake_news.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('saved_models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully!")
