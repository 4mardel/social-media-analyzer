import os
import datetime
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from yolo_detection import detect_objects
from model_loader import topic_model, topic_encoder, popularity_model, embedder, extract_features


app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    try:
        if request.method == "POST":
            text = request.form.get("text", "").strip()
            file = request.files.get("image")

            if not text and not file:
                return render_template("index.html", error="Please provide text or image.")

            
            embedding = embedder.encode([text])
            topic_pred = topic_model.predict(embedding)
            topic_label = topic_encoder.inverse_transform(topic_pred)[0]

            
            image_url = None
            objects = []

            if file and file.filename:
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                print(f"[DEBUG] Saving image to: {image_path}")
                file.save(image_path)

                image_url = f"/static/uploads/{filename}"
                objects = detect_objects(image_path)

            
            now = datetime.datetime.now()
            has_image = bool(file)
            features = extract_features(text, now, has_image)
            popularity_label = int(popularity_model.predict([features])[0])

            return render_template(
                "index.html",
                text=text,
                topic=topic_label,
                is_popular=popularity_label,
                image_url=image_url,
                objects=objects
            )

    except Exception as e:
        print(f"[ERROR] {e}")
        return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
