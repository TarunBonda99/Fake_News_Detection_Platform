import pandas as pd

def combine_datasets():
    true_news = pd.read_csv('True.csv')
    fake_news = pd.read_csv('Fake.csv')

    true_news['label'] = 0
    fake_news['label'] = 1

    combined = pd.concat([true_news, fake_news])
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv('fake_news.csv', index=False)
    print("fake_news.csv has been created successfully!")
