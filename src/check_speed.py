import time

import cv2
import torch

from detector import MaskDetector

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())

d = MaskDetector()
print("device info:", d.device_info)

img = cv2.imread("data/images/maksssksksss0.png")
for i in range(10):
    start = time.perf_counter()
    d.predict(img)
    ms = (time.perf_counter() - start) * 1000
    print(f"run {i}: total {ms:.1f} ms | stages {d.last_speed}")