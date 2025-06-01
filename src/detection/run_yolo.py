import os
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

def detect_objects(df, model_path='yolov8n.pt'):
    model = YOLO(model_path)

    results = []

    for path in tqdm(df["image_path"]):
        full_path = os.path.join("data/raw", path)

        try:
            detection = model(full_path, verbose=False)[0]
            names = detection.names
            labels = [names[int(cls)] for cls in detection.boxes.cls]

            label_counts = pd.Series(labels).value_counts().to_dict()

            results.append({
                "num_objects": len(labels),
                "has_person": int("person" in labels),
                "has_cat": int("cat" in labels),
                "has_dog": int("dog" in labels),
                "has_car": int("car" in labels),
                "has_food": int(any(x in labels for x in ["banana", "cake", "pizza", "hot dog", "sandwich"]))
            })
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            results.append({
                "num_objects": 0,
                "has_person": 0,
                "has_cat": 0,
                "has_dog": 0,
                "has_car": 0,
                "has_food": 0
            })

    object_df = pd.DataFrame(results)
    df = df.reset_index(drop=True).join(object_df)
    return df

def main():
    df = pd.read_csv("data/processed/labeled_data.csv")
    df = detect_objects(df)

    os.makedirs("data/processed/", exist_ok=True)
    df.to_csv("data/processed/labeled_with_objects.csv", index=False)
    print(f"Object features added and saved to labeled_with_objects.csv")
if __name__ == "__main__":
    main()