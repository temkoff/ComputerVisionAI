#!pip install ultralytics

import cv2
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# ===================== ПАРАМЕТРЫ =====================
video_path = "pigs.mp4"
model = YOLO("yolov8n-seg.pt")

REF_LENGTH_REAL = 1.2
REF_WIDTH_REAL = 0.45

results = []
frame_id = 0

# история всех кадров (не ограниченная)
all_frames = []

# ограничение, чтобы не перегрузить Colab
MAX_TOTAL_FRAMES = 30

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    if frame_id % 10 != 0:
        continue

    res = model(frame)[0]

    if res.masks is None:
        continue

    annotated = res.plot()

    # сохраняем кадр
    all_frames.append(annotated)

    # ограничение
    if len(all_frames) > MAX_TOTAL_FRAMES:
        break

    print(f"Обработан кадр: {frame_id}")

    # показываем текущий кадр
    cv2_imshow(annotated)

    # ===================== ПРИЗНАКИ =====================
    for mask in res.masks.data:

        mask = mask.cpu().numpy()
        mask_area = np.sum(mask)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        width_px = xs.max() - xs.min()
        height_px = ys.max() - ys.min()

        length_m = (height_px / frame.shape[0]) * REF_LENGTH_REAL
        width_m = (width_px / frame.shape[1]) * REF_WIDTH_REAL

        volume = length_m * width_m * 0.5
        weight = volume * 1000

        results.append({
            "frame_id": frame_id,
            "body_length": length_m * 100,
            "withers_height": width_m * 100,
            "mask_area": int(mask_area),
            "camera_height": 1.8,
            "curvature_angle": 0,
            "weight": weight
        })

cap.release()

# ===================== СОХРАНЕНИЕ =====================
df = pd.DataFrame(results)
df.to_csv("generated_dataset.csv", index=False)

print("Готово: generated_dataset.csv")
