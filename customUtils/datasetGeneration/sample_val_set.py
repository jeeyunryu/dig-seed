import random
from tqdm import tqdm

# 파일 경로 설정
input_file = "mpsc_train_lmdb.txt"     # 원본 목록이 들어 있는 텍스트 파일
output_file = "mpsc_val_lmdb.txt" # 결과 저장할 텍스트 파일
remaining_file = "mpsc_train_remain_lmdb.txt"

# 1. 파일 읽기
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 2. 1000개 랜덤 샘플링 (줄 수가 1000개 이상이어야 함)
sampled = random.sample(lines, 1000)

sampled_set = set(sampled)
remaining = [line for line in tqdm(lines, desc="남은 데이터 추리는 중...") if line not in sampled_set]

# 3. 저장
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(sampled)

with open(remaining_file, "w", encoding="utf-8") as f:
    f.writelines(remaining)

print("1000개 샘플링 완료:", output_file)
print(f"나머지 {len(remaining)}개 저장 완료 → {remaining_file}")