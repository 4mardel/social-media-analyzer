import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lightgbm as lgb
import joblib

from sentence_transformers import SentenceTransformer


df = pd.read_csv("data/processed/labeled_with_objects.csv")
df = df.dropna(subset=["clean_text", "topic"])

texts = df["clean_text"].astype(str).tolist()
labels = df["topic"].astype(str).tolist()


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)



embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = embedder.encode(texts, convert_to_numpy=True)


X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)


model = lgb.LGBMClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nClassification report for Sentence Transformers:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


joblib.dump(model, "saved_models/topic_classifier_lgbm.pkl")
joblib.dump(label_encoder, "saved_models/topic_label_encoder.pkl")

print("\nLightGBM classifier and label encoder saved.")