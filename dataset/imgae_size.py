import os
import shutil

# 이미지가 원래 있던 최상위 폴더 경로
base_folder = "/home/jyryu/workspace/DiG/dataset/Reann_MPSC/jyryu/image/test"  # 예: "C:/Users/username/Pictures"

# 최상위 폴더 안의 모든 폴더 탐색
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    
    # 폴더인지 확인 (사이즈 폴더)
    if os.path.isdir(folder_path):
        # 폴더 안의 모든 파일 이동
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            shutil.move(file_path, os.path.join(base_folder, filename))
        
        # 폴더가 비워지면 삭제
        os.rmdir(folder_path)

print("모든 이미지가 원래 폴더로 되돌아왔습니다.")
