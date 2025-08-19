from PIL import Image, ImageOps

def fit_with_aspect(image_path, target_size=(128, 32), bg_color=(0,0,0)):
    img = Image.open(image_path)
    img.thumbnail(target_size, Image.LANCZOS)  # 비율 유지 축소
    # 패딩 추가
    padded_img = ImageOps.pad(img, target_size, method=Image.LANCZOS, color=bg_color, centering=(0.5, 0.5))
    return padded_img

# 사용 예시
result = fit_with_aspect("dataset/Reann_MPSC/jyryu/image/test/MPSC_img_0_0.jpg")
result.save("output.jpg")
