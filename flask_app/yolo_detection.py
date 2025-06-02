from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    results = model(image_path)
    result = results[0]
    if result.boxes is None:
        return []
    class_ids = result.boxes.cls.tolist()
    class_names = [result.names[int(cls_id)] for cls_id in class_ids]
    return class_names
