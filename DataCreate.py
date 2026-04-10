import os
import subprocess
from pathlib import Path
import time

# ---------- Настройки ----------
CLIP_LENGTH_SEC = 300
MIN_LENGTH_SEC = 360
CHECK_INTERVAL = 60

# ---------- ML БЛОК ----------
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np

from google.colab.patches import cv2_imshow
from IPython.display import clear_output
import matplotlib.pyplot as plt

# --- Параметры ---
video_path = "pigs.mp4"
model = YOLO("yolov8n-seg.pt")

REF_LENGTH_REAL = 1.2
REF_WIDTH_REAL = 0.45

results = []
frame_id = 0

# История кадров
history_frames = []
MAX_HISTORY = 6  # сколько кадров хранить

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

    # Отрисовка
    annotated = res.plot()

    # Сохраняем в историю
    history_frames.append(annotated)
    if len(history_frames) > MAX_HISTORY:
        history_frames.pop(0)

    # Очистка экрана
    clear_output(wait=True)

    # ТЕКУЩИЙ КАДР
    print(f"Текущий кадр: {frame_id}")
    cv2_imshow(annotated)

    # ИСТОРИЯ
    print("История кадров:")

    fig, axes = plt.subplots(1, len(history_frames), figsize=(15, 5))

    if len(history_frames) == 1:
        axes = [axes]

    for i, img in enumerate(history_frames):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{i+1}")
        axes[i].axis('off')

    plt.show()

    time.sleep(0.1)

    # Расчёт признаков
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

# ---------- СОХРАНЕНИЕ ----------
df = pd.DataFrame(results)
df.to_csv("generated_dataset.csv", index=False)

print("Готово: generated_dataset.csv")
