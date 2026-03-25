import cv2
import os
import numpy as np
SOURCE_DIR = "Dataset"
OUTPUT_DIR = "Training"
STRIDE = 3
def get_robust_motion_mask(prev_frame, curr_frame):
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.GaussianBlur(gray_curr, (5, 5), 0)
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask
for sign_name in os.listdir(SOURCE_DIR):
    sign_path = os.path.join(SOURCE_DIR, sign_name)
    if not os.path.isdir(sign_path):
        continue
    output_sign_path = os.path.join(OUTPUT_DIR, sign_name)
    os.makedirs(output_sign_path, exist_ok=True)
    frames = sorted([f for f in os.listdir(sign_path) if f.endswith(".jpg")])
    for i in range(STRIDE, len(frames)):
        prev_img = cv2.imread(os.path.join(sign_path, frames[i - STRIDE]))
        curr_img = cv2.imread(os.path.join(sign_path, frames[i]))
        masked_frame = get_robust_motion_mask(prev_img, curr_img)
        save_name = f"mask_{i:05d}.jpg"
        cv2.imwrite(os.path.join(output_sign_path, save_name), masked_frame)
print("Preprocessing complete.")