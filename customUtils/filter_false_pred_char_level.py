import re
import json
import difflib

# GT/PRED 문자열 정규화 함수 (예시: 대문자로 통일)
def normalize(text):
    return text.strip()

# GT vs Pred 비교
def compare_strings(gt, pred):
    matcher = difflib.SequenceMatcher(None, gt, pred)
    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            for k in range(max(i2 - i1, j2 - j1)):
                gt_char = gt[i1 + k] if i1 + k < i2 else ''
                pred_char = pred[j1 + k] if j1 + k < j2 else ''
                changes.append({
                    'type': '변경',
                    '위치_GT': i1 + k,
                    'GT_문자': gt_char,
                    'Pred_문자': pred_char
                })
        elif tag == 'delete':
            for k in range(i1, i2):
                changes.append({
                    'type': '삭제',
                    '위치_GT': k,
                    'GT_문자': gt[k]
                })
        elif tag == 'insert':
            for k in range(j1, j2):
                changes.append({
                    'type': '추가',
                    '위치_Pred': k,
                    'Pred_문자': pred[k]
                })

    return changes

# 메인 처리 함수
def process_file_to_json(filepath, output_file='comparison_results.json'):
    results = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'(image-\d+)\s*\|\s*GT:\s*(.+?)\s*\|\s*Pred:\s*(.+)', line)
            if match:
                image_id, gt_raw, pred_raw = match.groups()
                gt = normalize(gt_raw)
                pred = normalize(pred_raw)
                diffs = compare_strings(gt, pred)

                results.append({
                    'image_id': image_id,
                    'GT': gt,
                    'Pred': pred,
                    'changes': diffs
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 비교 결과가 '{output_file}'에 저장되었습니다.")

# 실행 예시
if __name__ == '__main__':
    process_file_to_json('output/mpsc/train/250809_1700/eval/wrong_predictions.txt')