import os
from PIL import Image, ImageOps

def fit_with_aspect(img, target_size=(640, 360), bg_color=(0, 0, 0)):
    """이미지를 target_size 안에 비율 유지로 맞추고 중앙 패딩."""
    img.thumbnail(target_size, Image.LANCZOS)  # 비율 유지 축소
    return ImageOps.pad(
        img,
        target_size,
        method=Image.LANCZOS,
        color=bg_color,
        centering=(0.5, 0.5)
    )

def process_all_images(input_dir, output_dir, target_size=(640, 360), bg_color=(0, 0, 0)):
    """input_dir 안의 모든 이미지를 처리 후 output_dir에 저장."""
    os.makedirs(output_dir, exist_ok=True)  # 저장 폴더 생성

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                img = Image.open(input_path).convert("RGBA" if len(bg_color) == 4 else "RGB")
                processed = fit_with_aspect(img, target_size, bg_color)
                processed.save(output_path)
                # print(f"✅ {filename} → 저장 완료")
            except Exception as e:
                print(f"❌ {filename} 처리 실패: {e}")

# 사용 예시
input_folder = "dataset/Reann_MPSC/jyryu/image/test"       # 원본 이미지 폴더
output_folder = "dataset/Reann_MPSC/jyryu/image_pad_L/test_pad_L"     # 저장할 폴더
process_all_images(input_folder, output_folder, target_size=(224, 128), bg_color=(0, 0, 0))
print('finished!')
