import os
import json
import pandas as pd
import re
import string
import nltk
from tqdm import tqdm
from PIL import Image


nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()


def is_valid_image(path):
    full_path = os.path.join("data/raw", path)
    try:
        img = Image.open(full_path)
        img.verify()
        return True
    except:
        return False


def resize_image(path, size=(640, 640)):
    full_path = os.path.join("data/raw", path)
    try:
        img = Image.open(full_path)
        img = img.resize(size)
        img.save(full_path)
    except:
        pass


def load_data(json_dir="data/raw"):
    all_posts = []
    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    for file in files:
        with open(os.path.join(json_dir, file), encoding="utf-8") as f:
            posts = json.load(f)
            all_posts.extend(posts)

    df = pd.DataFrame(all_posts)
    print(f"Loaded {len(df)} posts from {len(files)} files")
    return df


def preprocess():
    df = load_data()

    
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["clean_text"] = df["full_text"].apply(clean_text)

    
    df.drop_duplicates(subset="id", inplace=True)
    df = df[df["clean_text"].str.strip() != ""]
    df = df[df["image_path"].notna()]

    
    df["valid_image"] = df["image_path"].apply(is_valid_image)
    df = df[df["valid_image"] == True]

    
    for path in tqdm(df["image_path"]):
        resize_image(path)

    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_data.csv", index=False)
    print(f"Final dataset saved: {len(df)} records in clean_data.csv")

if __name__ == "__main__":
    preprocess()