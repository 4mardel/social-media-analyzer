import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import os

def load_data(csv_path="data/processed/labeled_with_objects.csv"):
    df = pd.read_csv(csv_path)
    df = df[["clean_text", "topic"]].dropna()
    return df

def preprocess_text(df, max_words=5000, max_len=100):
    texts = df["clean_text"].astype(str).tolist()
    labels = df["topic"].astype(str).tolist()

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)

    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(labels)
    y = to_categorical(y_int)

    return X, y, tokenizer, label_encoder

def build_model(vocab_size=5000, max_len=100, num_classes=3):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model():
    df = load_data()
    X, y, tokenizer, label_encoder = preprocess_text(df)
    num_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(vocab_size=5000, max_len=100, num_classes=num_classes)

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}")

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/topic_classifier_lstm.h5")

    import joblib
    joblib.dump(tokenizer, "saved_models/tokenizer.pkl")
    joblib.dump(label_encoder, "saved_models/label_encoder.pkl")

    print("Model and tokenizer saved.")

if __name__ == "__main__":
    train_model()