import os
import csv
import re

# 대상 폴더 경로 지정 (예: 현재 폴더일 경우 '.')
folder_path = "/home/jyryu/workspace/DiG/output/mpsc/train/250722_2143/eval_unfiltered/rslts"

pattern = re.compile(r"image-(\d+)_")

numbers = []

for file_name in os.listdir(folder_path):
    match = pattern.search(file_name)
    if match:
        numbers.append(int(match.group(1)))

numbers.sort()

# # 하위 파일 목록 가져오기 (파일만, 하위 폴더 제외)
# file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# CSV 파일로 저장
csv_path = "file_list.csv"
with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["File Name"])  # 헤더
    for number in numbers:
        writer.writerow([number])

print(f"CSV 파일이 저장되었습니다: {csv_path}")