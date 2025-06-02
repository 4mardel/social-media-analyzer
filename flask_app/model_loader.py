import re
import datetime
import joblib
from sentence_transformers import SentenceTransformer


topic_model = joblib.load("saved_models/topic_classifier_lgbm.pkl")
topic_encoder = joblib.load("saved_models/topic_label_encoder.pkl")
popularity_model = joblib.load("saved_models/popularity_model.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_features(text, created_time, has_image=True):
    text_length = len(text)
    word_count = len(text.split())
    caps_count = sum(1 for word in text.split() if word.isupper())
    emoji_count = len(re.findall(r'[^\w\s,]', text))
    image_flag = int(has_image)
    post_hour = created_time.hour if isinstance(created_time, datetime.datetime) else 12
    return [text_length, word_count, caps_count, emoji_count, image_flag, post_hour]