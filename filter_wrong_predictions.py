import os

def extract_wrong_predictions(input_txt_path, output_txt_path):
    wrong_lines = []

    with open(input_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                # 예: image-000000001 | GT: 45mm | Pred: 45mm/
                parts = line.split('|')
                if len(parts) < 3:
                    continue

                gt = parts[1].strip().replace('GT: ', '')
                pred = parts[2].strip().replace('Pred: ', '')

                if gt.lower() != pred.lower():
                    wrong_lines.append(line)
            except Exception as e:
                print(f"[Warning] Failed to parse line: {line}")
                continue

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for l in wrong_lines:
            f.write(l + '\n')

    print(f"Done. {len(wrong_lines)} wrong predictions saved to:")
    print(f"   {output_txt_path}")


if __name__ == '__main__':
    # 기본 경로 설정 (필요시 argparse로 대체 가능)
    input_file = '/home/jyryu/workspace/DiG/output/mpsc/eval_250716/eval_predictions.txt'
    output_file = 'logs/wrong_predictions_3.txt'

    extract_wrong_predictions(input_file, output_file)