import argparse
import os
import re
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import lmdb


# lmdb_path = "dataset/Reann_MPSC/jyryu/mpsc.lmdb"

def extract_sort_key(filename):
    nums = re.findall(r'\d+', filename)
    return tuple(map(int, nums)) if nums else (0, 0)


def main(image_folder, anno_folder, out_folder):
    image_input_path = image_folder
    label_input = anno_folder
    image_out_path = out_folder
    # label_out_path = args.label_out
    os.makedirs(image_out_path, exist_ok=True)

    files = os.listdir(image_input_path)
    files = sorted(files, key=extract_sort_key)


    # env = lmdb.open(lmdb_path, map_size=1 << 30) # 1 gigabytes
    # txn = env.begin(write=True) # transaction objet

    total_num = 0
    final_num = 0

    for filename in tqdm(files):
        index = re.findall(r'\d+', filename)[0]
        anno_filename = f"gt_img_{index}.txt"
        anno_file_path = os.path.join(label_input, anno_filename)
        image_path = os.path.join(image_input_path, filename)
        image = Image.open(image_path)

        with open(anno_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
            for i, line in enumerate(lines):
               total_num += 1
               if line == '':
         # maybe text file with extra new line in the end? -> yes confirmed!
                continue
               parts = line.split(',')
               coords = parts[:8]
               text = parts[-1]
               if text == '###':
                  continue           

               int_list = list(map(int, coords))
               points = [(int_list[i], int_list[i + 1]) for i in range(0, len(int_list), 2)]
               xs, ys = zip(*points)
               bbox = (min(xs), min(ys), max(xs), max(ys))
               cropped = image.crop(bbox)
               img_file_name = f'MPSC_img_{index}_{i}.jpg'
            #    label_file_name = f'gt_img_{index}_{i}.txt'
            #    img_out_path = os.path.join(image_out_path, img_file_name)
            #    label_out_path = os.path.join(image_out_path, label_file_name)
               full_path = os.path.join(image_out_path, img_file_name)

               # cropped.save(full_path)

               with open('mpsc_test_lmdb_temp.txt', 'a') as f: # 여러번 호출 시 이전 것에 append 함
                  f.write(full_path + ' ' + text + '\n')
            
               final_num += 1

               


        # if os.path.isfile(file_path):
           
            # with open(file_path, 'r', encoding='utf-8') as f:
            #     content = f.read()
    print(f'total lines: {total_num}')
    print(f'without invalid lines: {final_num}')
                

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image_in', type=str, required=True, help='이미지 파일 경로')
#     parser.add_argument('--image_out', type=str, required=True, help='이미지 출력 파일 경로')
#     parser.add_argument('--label_in', type=str, required=True, help='라벨 파일 경로')
#     parser.add_argument('--label_out', type=str, required=True, help='라벨 출력 파일 경로')
#     args = parser.parse_args()
#     main(args)

image_folder = 'dataset/Reann_MPSC/MPSC/image/test'
anno_folder = 'dataset/Reann_MPSC/MPSC/annotation/test'
out_folder = 'dataset/Reann_MPSC/jyryu/image/test2'
main(image_folder, anno_folder, out_folder)