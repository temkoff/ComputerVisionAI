# --- Установка библиотек (если ещё не установлены) ---
!pip install ultralytics opencv-python-headless numpy scipy pandas

# --- Импорт ---
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import time

# --- Параметры ---
video_path = "pigs.mp4"
model = YOLO("yolov8n-seg.pt")  # предобученная сегментационная модель YOLOv8

REF_LENGTH_REAL = 1.2  # м
REF_WIDTH_REAL = 0.45  # м
results = []
frame_id = 0

# функция площади треугольника для FEM
def triangle_area(a, b, c):
    return abs(
        a[0]*(b[1]-c[1]) +
        b[0]*(c[1]-a[1]) +
        c[0]*(a[1]-b[1])
    ) / 2

# функция объединения масок (удаление сильно пересекающихся)
def merge_masks(masks_list, iou_threshold=0.3):
    merged = []
    for mask in masks_list:
        keep = True
        for m in merged:
            # проверка пересечения
            intersection = np.logical_and(mask, m).sum()
            union = np.logical_or(mask, m).sum()
            if union == 0:
                continue
            iou = intersection / union
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            merged.append(mask)
    return merged

# --- Открываем видео ---
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    overlay = frame.copy()

    # --- Два прогона YOLO с разными порогами ---
    results1 = model.predict(frame, conf=0.4, verbose=False)
    results2 = model.predict(frame, conf=0.2, verbose=False)

    masks_list = []

    # --- собираем все маски с прогонов ---
    for res in [results1, results2]:
        if res is None or len(res) == 0:
            continue
        res0 = res[0]
        if not hasattr(res0, "masks") or res0.masks is None:
            continue
        for poly in res0.masks.xy:
            mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask_img, [pts], 255)
            masks_list.append(mask_img)

    # --- объединяем маски с удалением дубликатов ---
    masks_merged = merge_masks(masks_list)

    # --- обработка каждой отдельной маски ---
    for pig_id, mask_img in enumerate(masks_merged):
        # разделяем контуры на случай нескольких объектов в одной маске
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt_id, contour in enumerate(contours):
            mask_obj = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_obj, [contour], -1, 255, -1)

            # bounding box
            x, y, w, h = cv2.boundingRect(contour)
            scale_L = REF_LENGTH_REAL / w
            scale_W = REF_WIDTH_REAL / h
            scale = (scale_L + scale_W) / 2
            length_real = w * scale
            width_real = h * scale

            # FEM
            ys, xs = np.where(mask_obj == 255)
            points = np.column_stack((xs, ys))
            if len(points) < 3:
                continue
            if len(points) > 2000:
                idx = np.random.choice(len(points), 2000, replace=False)
                points = points[idx]
            tri = Delaunay(points)
            fem_area_pixels = sum(
                triangle_area(points[a], points[b], points[c])
                for a,b,c in tri.simplices
            )\\\\\\
            fem_area_real = fem_area_pixels * scale * scale

            # Hu-моменты
            contour2, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            m = cv2.moments(contour2[0])
            hu = cv2.HuMoments(m).flatten()

            results.append({
                "frame": frame_id,
                "pig_id": f"{pig_id+1}_{cnt_id+1}",
                "length_m": length_real,
                "width_m": width_real,
                "area_m2": fem_area_real,
                "hu1": hu[0], "hu2": hu[1], "hu3": hu[2],
                "hu4": hu[3], "hu5": hu[4], "hu6": hu[5], "hu7": hu[6]
            })

            # --- визуализация ---
            cv2.polylines(overlay, [contour], isClosed=True, color=(0,255,0), thickness=2)
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            cv2.putText(overlay, f"Pig {pig_id+1}_{cnt_id+1}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # --- вывод кадра с найденными свиньями ---
    if len(masks_merged) > 0:
        cv2_imshow(overlay)
        time.sleep(0.1)

cap.release()

# --- сохраняем CSV ---
df = pd.DataFrame(results)
df.to_csv("pig_measurements_seg_double.csv", index=False)
print("Готово! Результаты сохранены в pig_measurements_seg_double.csv")
