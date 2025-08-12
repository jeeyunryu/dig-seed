import cv2
import numpy as np
import os

def estimate_contrast_std(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return np.std(img)  # 표준편차로 대비 추정

def average_contrast_std_in_folder(folder_path):
    std_list = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, file_name)
            contrast_std = estimate_contrast_std(image_path)
            std_list.append(contrast_std)
    return np.mean(std_list)

folder_path = "/home/jyryu/workspace/DiG/customUtils/outliers_2dim"
avg_contrast_proxy = average_contrast_std_in_folder(folder_path)
print(f"데이터셋의 평균 대비 (표준편차 기반): {avg_contrast_proxy:.2f}")
