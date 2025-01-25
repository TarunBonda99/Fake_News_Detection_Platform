import os
from combine_csv import combine_datasets
from module_training import train_model
from app1 import start_flask_app

def main():
    if not os.path.exists('fake_news.csv'):
        print("Combining datasets...")
        combine_datasets()
    else:
        print("fake_news.csv already exists. Skipping dataset combination.")

    if not os.path.exists('saved_models/model.pkl') or not os.path.exists('saved_models/vectorizer.pkl'):
        print("Training model...")
        train_model()
    else:
        print("Model already trained. Skipping training.")

    print("Starting Flask API...")
    start_flask_app()

if __name__ == "__main__":
    main()
