import os
import random
import re
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from tqdm import tqdm


# ----------- 경로 설정 -----------
img_root = '/home/jyryu/workspace/DiG/dataset/Reann_MPSC/jyryu/image/test'
text_file_path = 'output/mpsc/train/250722_2143/eval_unfiltered/wrong_predictions.txt'
output_dir = 'output/mpsc/train/250722_2143/eval_unfiltered/rslts'
os.makedirs(output_dir, exist_ok=True)

# ----------- 랜덤 시드 -----------
random.seed(42)

def draw_text_with_outline(draw, pos, text, font, fill, outline='black'):
    x, y = pos
    # 테두리
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            draw.text((x+dx, y+dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)

# ----------- 폰트 설정 -----------
# try:
#     font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
# except:
font = ImageFont.load_default()

# ----------- 숫자 기준 정렬 함수 -----------
def extract_sort_key(filename):
    nums = re.findall(r'\d+', filename)
    return tuple(map(int, nums)) if nums else (0, 0)

# ----------- 이미지 정렬 -----------
image_files = sorted([
    f for f in os.listdir(img_root)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
], key=extract_sort_key)

print(f"[총 이미지 수]: {len(image_files)}")

# ----------- 결과 파일 파싱 (인덱스 기반) -----------
results = []
with open(text_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or '|' not in line:
            continue
        try:
            match = re.match(r'(image-\d+)\s*\|\s*GT:\s*(.+?)\s*\|\s*Pred:\s*(.+)', line)
            if match:
                image_tag, gt, pred = match.groups()
            # parts = line.split('|')
            # image_tag = parts[0].strip()  # ex: image-000000016
            
            index = int(image_tag.replace('image-', ''))
            
            # gt = parts[1].strip().replace('GT: ', '')
            # pred = parts[2].strip().replace('Pred: ', '')

            if 0 <= index < len(image_files)+1:
                filename = image_files[index-1]
                results.append((filename, gt, pred, index))
            else:
                print(f"[경고] 인덱스 {index}가 이미지 리스트 범위를 벗어남")
        except Exception as e:
            print(f"[에러] 라인 파싱 실패: {line} → {e}")

print(f"[유효 결과 수]: {len(results)}")

# ----------- 랜덤 100개 선택 -----------
sampled = random.sample(results, min(100, len(results)))

# ----------- 시각화 -----------
for idx, (filename, gt_text, pred_text, index) in tqdm(enumerate(sampled)):
    try:
        img_path = os.path.join(img_root, filename)
        image = Image.open(img_path).convert('RGB')
        # draw = ImageDraw.Draw(image)


        orig_w, orig_h = image.size

        padding_top = 50
        padding_bottom = 20
        new_w = max(300, orig_w + 20)
        new_h = orig_h + padding_top + padding_bottom

        canvas = Image.new('RGB', (new_w, new_h), (255, 255, 255))

        canvas.paste(image, ((new_w - orig_w) // 2, padding_top))


        draw = ImageDraw.Draw(canvas)

    
        draw.text((10, 10), f'GT: {gt_text}', fill=(0, 255, 0), font=font)
        draw.text((10, 30), f'Pred: {pred_text}', fill=(255, 0, 0), font=font)

        # draw_text_with_outline(draw, (10, 10), f'GT: {gt_text}', font, fill=(0, 128, 0))
        # draw_text_with_outline(draw, (10, 40), f'Pred: {pred_text}', font, fill=(200, 0, 0))


        save_path = os.path.join(output_dir, f'image-{index}_vis_{filename}')
        canvas.save(save_path)
        # print(f"[{idx+1}] 저장 완료: {save_path}")
    except Exception as e:
        print(f"[에러] {filename} 처리 중 오류: {e}")
