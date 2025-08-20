import os
from PIL import Image, ImageOps

def fit_with_aspect(img, target_size=(640, 360), bg_color=(0, 0, 0)):
    """이미지를 target_size 안에 비율 유지로 맞추고 패딩.
       단, 작은 이미지는 업샘플링하지 않고 좌상단에 배치."""
    
    # target보다 큰 경우 → 비율 유지 축소
    if img.width > target_size[0] or img.height > target_size[1]:
        img.thumbnail(target_size, Image.LANCZOS)
    
    # 배경 캔버스 생성
    background = Image.new("RGBA" if img.mode == "RGBA" else "RGB", target_size, bg_color)
    
    # 업샘플링하지 않고 좌상단(0,0)에 붙여넣기
    background.paste(img, (0, 0))
    return background

def process_all_images(input_dir, output_dir, target_size=(640, 360), bg_color=(0, 0, 0)):
    """input_dir 안의 모든 이미지를 처리 후 output_dir에 저장."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                img = Image.open(input_path).convert("RGBA" if len(bg_color) == 4 else "RGB")
                processed = fit_with_aspect(img, target_size, bg_color)
                processed.save(output_path)
            except Exception as e:
                print(f"❌ {filename} 처리 실패: {e}")

# 사용 예시
input_folder = "dataset/Reann_MPSC/jyryu/image/train"
output_folder = "dataset/Reann_MPSC/jyryu/image_pad_sqr/train_pad_sqr"

process_all_images(input_folder, output_folder, target_size=(128, 128), bg_color=(0, 0, 0))
print('finished!')
