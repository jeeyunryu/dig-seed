import re

def normalize(s):
    return s.strip().lower()

def parse_file(filepath):
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'(image-\d+)\s*\|\s*GT:\s*(.+?)\s*\|\s*Pred:\s*(.+)', line)
            if match:
                image_id, gt, pred = match.groups()
                results[image_id] = {
                    'GT': normalize(gt),
                    'Pred': normalize(pred)
                }
    return results

def compare_results(results_a, results_b):
    case1 = []  # A: O, B: X
    case2 = []  # A: X, B: X
    case3 = []  # A: X, B: O

    all_keys = set(results_a.keys()) & set(results_b.keys())

    for key in sorted(all_keys):
        gt = results_a[key]['GT']
        pred_a = results_a[key]['Pred']
        pred_b = results_b[key]['Pred']

        is_correct_a = pred_a == gt
        is_correct_b = pred_b == gt

        if is_correct_a and not is_correct_b:
            case1.append((key, gt, pred_a, pred_b))
        elif not is_correct_a and not is_correct_b:
            case2.append((key, gt, pred_a, pred_b))
        elif not is_correct_a and is_correct_b:
            case3.append((key, gt, pred_a, pred_b))

    return case1, case2, case3

def save_cases(case_list, filename, title):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n")
        f.write("Image ID       | GT        | A Pred    | B Pred\n")
        f.write("-" * 50 + "\n")
        for img_id, gt, pred_a, pred_b in case_list:
            f.write(f"{img_id:<14}| {gt:<10}| {pred_a:<10}| {pred_b:<10}\n")

# 파일 경로 설정
file_a = "output/mpsc/train/250722_2143/eval_unfiltered/eval_predictions.txt" # 시멘틱 모듈 추가 전
file_b = "output/mpsc/train/250725_1631/eval_unfiltered/eval_predictions.txt" # 시멘틱 모듈 추가 후

# 비교 수행
results_a = parse_file(file_a)
results_b = parse_file(file_b)
case1, case2, case3 = compare_results(results_a, results_b)

# 결과 저장
save_cases(case1, "case1_A_O_B_X.txt", "Case 1: A는 맞고 B는 틀림")
save_cases(case2, "case2_A_X_B_X.txt", "Case 2: A도 B도 틀림")
save_cases(case3, "case3_A_X_B_O.txt", "Case 3: A는 틀리고 B는 맞음")
