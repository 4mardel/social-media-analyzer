# 📊 Social Media Post Analyzer

A Flask-based web application that analyzes social media posts using machine learning. It predicts the **topic**, determines **popularity potential**, and performs **object detection** on uploaded images.

---

## 🔍 Features

- 📝 **Text Classification** (Art, Cats, Pics) using Sentence Transformers + LightGBM  
- 🌟 **Popularity Prediction** using Random Forest and engineered post features  
- 🖼️ **Image Object Detection** using YOLOv8  
- 🌐 **Flask Web Interface** for interactive user input  
- 🐳 **Docker deployment**

---

## 📁 Project Structure

project/
├── data/
│ ├── raw/ # Reddit JSON data
│ └── processed/ # Cleaned data (CSV)
├── flask_app/
│ ├── app.py # Flask application
│ ├── model_loader.py # Loads models and utilities
│ ├── templates/index.html # HTML interface
│ ├── static/uploads/ # Uploaded images
│ ├── requirements.txt # Dependencies
│ ├── yolo_detection/ # YOLOv8 object detection
│ └── Dockerfile # Docker setup
├── saved_models/ # Trained model files
│ ├── topic_classifier_lgbm.pkl
│ ├── topic_classifier_lstm.h5
│ ├── topic_label_encoder.pkl
│ ├── tokenizer.pkl
│ ├── tfidf_vectorizer.pkl
│ ├── label_encoder.pkl
│ └── popularity_model.pkl
├── src/
│ ├── crawler/ # Reddit data scraper
│ ├── detection/ # YOLOv8 object detection
│ ├── ml_models/ # Training scripts for ml models
│ ├── nlp_models/ # Training scripts for nlp models
│ └── preprocessing/ # Preprocessing utilities
├── notebooks/
│ └── analysis.ipynb # Experiment notebook

---

## ⚙️ Setup & Installation

### 1. Clone the repository

bash
git clone https://github.com/your-username/social-media-analyzer.git
cd social-media-analyzer

### 2. Install dependencies
Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install required packages:

cd flask_app
pip install -r requirements.txt

### 3. Run the application
python app.py
Visit the web app at: http://127.0.0.1:5000

## 🤖 Models & Technologies
Text - Embedding, sentence-transformers (MiniLM)
Topic Classification - LightGBM + LabelEncoder
Popularity Prediction - RandomForestClassifier
Object Detection - YOLOv8 (via ultralytics)
Web App - Flask, HTML, Bootstrap
Model I/O - Joblib, Keras, H5

## 🧠 How It Works
User submits post text and/or image.

Text is embedded and passed to the topic classifier.

Engineered features (e.g., post length, time, image presence) are passed to the popularity model.

If an image is uploaded, YOLO detects objects and displays them.

The web interface shows:

Predicted topic

Popularity status

Detected image objects (if any)

## 🐳 Docker
### Build Docker image
docker build -t social-media-app .

### Run the container
docker run -p 5000:5000 social-media-app
Then access the app via http://localhost:5000

## 👨‍💻 Author
Developed by Damir Niyazbek
GitHub: @4mardel