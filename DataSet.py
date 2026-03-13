import cv2
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

video_path = "pigs.mp4"

# эталонная йоркширская свинья
REF_LENGTH_REAL = 1.2
REF_WIDTH_REAL = 0.45

cap = cv2.VideoCapture(video_path)

results = []
frame_id = 0


def triangle_area(a, b, c):
    return abs(
        a[0]*(b[1]-c[1]) +
        b[0]*(c[1]-a[1]) +
        c[0]*(a[1]-b[1])
    ) / 2


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # морфология (убирает шум)
    kernel = np.ones((7,7),np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    pig_id = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 5000:   # фильтр мусора
            continue

        pig_id += 1

        rect = cv2.minAreaRect(contour)
        (cx,cy),(w,h),angle = rect

        length_pixels = max(w,h)
        width_pixels = min(w,h)

        scale_L = REF_LENGTH_REAL / length_pixels
        scale_W = REF_WIDTH_REAL / width_pixels

        scale = (scale_L + scale_W)/2

        length_real = length_pixels * scale
        width_real = width_pixels * scale

        # создаём маску свиньи
        mask = np.zeros(gray.shape, dtype=np.uint8)

        cv2.drawContours(mask,[contour],-1,255,-1)

        ys, xs = np.where(mask==255)

        points = np.column_stack((xs,ys))

        # уменьшаем число точек
        if len(points) > 2000:
            idx = np.random.choice(len(points),2000,replace=False)
            points = points[idx]

        if len(points) < 3:
            continue

        tri = Delaunay(points)

        fem_area_pixels = 0

        for simplex in tri.simplices:

            a = points[simplex[0]]
            b = points[simplex[1]]
            c = points[simplex[2]]

            fem_area_pixels += triangle_area(a,b,c)

        fem_area_real = fem_area_pixels * scale * scale

        moments = cv2.moments(contour)

        hu = cv2.HuMoments(moments).flatten()

        results.append({

            "frame": frame_id,
            "pig": pig_id,

            "length_m": length_real,
            "width_m": width_real,

            "area_m2": fem_area_real,

            "hu1": hu[0],
            "hu2": hu[1],
            "hu3": hu[2],
            "hu4": hu[3],
            "hu5": hu[4],
            "hu6": hu[5],
            "hu7": hu[6]

        })

cap.release()

df = pd.DataFrame(results)

df.to_csv("pig_measurements.csv",index=False)

print("Done")
