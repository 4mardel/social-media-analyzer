import pandas as pd
import os

def label_data(input_path="data/processed/clean_data.csv", output_path="data/processed/labeled_data.csv"):
    df = pd.read_csv(input_path)

    threshold = df["upvotes"].median()
    df["popularity_target"] = (df["upvotes"] >= threshold).astype(int)

    df["topic"] = df["subreddit"].str.lower().fillna("unknown")

    df.to_csv(output_path, index=False)
    print(f"Labeled data saved to {output_path}")
    print(f"Popularity classes: {df['popularity_target'].value_counts().to_dict()}")
    print(f"Topic classes: {df['topic'].value_counts().to_dict()}")

if __name__ == "__main__":
    label_data()