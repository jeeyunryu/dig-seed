import os
import re



def get_imgkeys(input_txt_path, isTrue):
    # wrong_lines = []
    imgkeys = []

    with open(input_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                if isTrue:
                    match = re.match(r'(image-\d+)\s*\|\s*GT:\s*(.+?)\s*\|\s*Pred:\s*(.+)', line)
                    image_id, _, _ = match.groups()
                else:
                    match = re.match(r'^image-\d{9}$', line)
                    image_id  = match.group(0)
                if match:
                    # image_id, gt, pred = match.groups()
                    # if gt == ' ':
                    #     import pdb;pdb.set_trace()
                   
                    imgkeys.append(image_id)
                    # if gt != pred:
                    #     wrong_lines.append(line)
                    
                # # 예: image-000000001 | GT: 45mm | Pred: 45mm/
                # parts = line.split('|')
                # if len(parts) != 3:
                #     continue


                # gt = parts[1].strip().replace('GT: ', '')
                # pred = parts[2].strip().replace('Pred: ', '')

                # if gt != pred:
                #     wrong_lines.append(line)
            except Exception as e:
                print(f"[Warning] Failed to parse line: {line}")
                continue
    return imgkeys
    # os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    # with open(output_txt_path, 'w', encoding='utf-8') as f:
    #     for l in wrong_lines:
    #         f.write(l + '\n')

    # print(f"Done. {len(wrong_lines)} wrong predictions saved to:")
    # print(f"   {output_txt_path}")


if __name__ == '__main__':
    # 기본 경로 설정 (필요시 argparse로 대체 가능)
    input_file = 'output/mpsc/train/250809_1700/eval/wrong_predictions.txt'
    out_imgkeys_file = 'port_keys.txt'
    # output_file = 'output/mpsc/train/250722_2143/eval_unfiltered/wrong_predictions.txt'

    wrong_pred = get_imgkeys(input_file, True)
    outliers = get_imgkeys(out_imgkeys_file, False)
    intersection_count = len(set(wrong_pred) & set(outliers))
    print(intersection_count)
