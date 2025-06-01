# Intelligent Social Media Analyzer

🎯 This project aims to analyze social media posts (text + images) using modern ML & DL techniques and deploy them via a Flask web app.

## 🔧 Project Structure

- `src/crawler/` – data collection from Reddit
- `src/preprocessing/` – cleaning text and images
- `src/ml_models/` – classic ML for popularity prediction
- `src/nlp_models/` – text classification using LSTM and embeddings
- `src/detection/` – object detection using YOLO
- `flask_app/` – Flask-based web app with API

## 📊 Tools & Libraries

- Python, Pandas, Scikit-learn, LightGBM
- PyTorch, TensorFlow/Keras
- YOLOv8
- Flask + Gunicorn