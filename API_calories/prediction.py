import cv2
from ultralytics import YOLO
import numpy as np

def prediction(image_path):
    model = YOLO('best.pt')
    results = model.predict(image_path)

    BOX_COLOR = (255, 0, 0) # Red
    TEXT_COLOR = (255, 255, 255) # White

    def visualize_bbox(image, bbox, class_name, prob, color=BOX_COLOR, thickness=2):
        """Visualizes a single bounding box on the image"""
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(image, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            image,
            text=class_name + str(prob),
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
        return image

    def visualize(image, bboxes, category_ids, category_id_to_name, prob):
        for bbox, category_id, prob in zip(bboxes, category_ids, prob):
            class_name = category_id_to_name[category_id]
            image = visualize_bbox(image, bbox, class_name, prob)
        return image

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    bboxes = []
    category_ids = []
    probs = []

    for box in results[0].boxes:
        bboxes.append(box.xyxy[0].tolist())
        category_ids.append(box.cls[0].item())
        probs.append(round(box.conf[0].item(), 2))

    # You may need to update the category_id_to_name dictionary according to your model's classes
    category_id_to_name = {0: 'apple 52cal/100grams', 1: 'coin 0cal', 2: 'banana 89cal/100grams', 3: 'bread 315cal/100grams',
                           4: 'bun 2.23cal/100grams', 5: 'doughnut 4.34cal/100grams', 6: 'egg 1.43cal/100grams',
                           7: 'fired_dough_twist 2416cal/100grams', 8: 'grape 69cal/100grams', 9: 'lemon 29cal/100grams',
                           10: 'litchi 66cal/100grams', 11: 'mango 60cal/100grams', 12: 'orange 63cal/100grams',
                           13: 'qiwi 61cal/100grams', 14: 'tomato 27cal/100grams', 15: 'pear 39cal/100grams',
                           16: 'mooncake 1883cal/100grams', 17: 'peach 57cal/100grams', 18: 'plum 46cal/100grams', 19: 'sachima 2145cal/100grams'}

    processed_image = visualize(image, bboxes, category_ids, category_id_to_name, probs)

    total_calories = 0
    calories = {'apple': 52, 'coin': 0, 'banana': 89, 'bread': 315, 'bun': 223,
                'doughnut': 434, 'egg': 143, 'fired_dough_twist': 2416,
                'grape': 69, 'lemon': 29, 'litchi': 66, 'mango': 60,
                'orange': 63, 'qiwi': 61, 'tomato': 27, 'pear': 39,
                'mooncake': 1883, 'peach': 57, 'plum': 46, 'sachima': 2145}

    for box in results[0].boxes:
        class_id = results[0].names[box.cls[0].item()]
        total_calories += calories[class_id]

    return processed_image, total_calories