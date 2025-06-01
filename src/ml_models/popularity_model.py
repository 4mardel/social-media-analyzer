import pandas as pd
import re
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def extract_features(df):
    df["text_length"] = df["clean_text"].str.len()
    df["num_words"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["has_emoji"] = df["title"].apply(lambda x: bool(re.search(r"[^\w\s,.!?]", str(x))))
    df["has_question"] = df["title"].str.contains(r"\?", na=False).astype(int)
    df["created_hour"] = pd.to_datetime(df["created_utc"]).dt.hour

    
    le = LabelEncoder()
    df["subreddit_encoded"] = le.fit_transform(df["subreddit"].astype(str))

    return df

def train_model(input_path="data/processed/labeled_data.csv", model_path="saved_models/popularity_model.pkl"):
    df = pd.read_csv(input_path)
    df = extract_features(df)

    features = ["text_length", "num_words", "has_emoji", "has_question", "created_hour", "subreddit_encoded"]
    X = df[features]
    y = df["popularity_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()