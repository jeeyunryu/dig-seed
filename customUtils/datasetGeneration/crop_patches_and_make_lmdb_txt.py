import argparse
import os
import re
from PIL import Image, ImageOps
from tqdm import tqdm
from pathlib import Path
import lmdb
from collections import defaultdict


# lmdb_path = "dataset/Reann_MPSC/jyryu/mpsc.lmdb"

def extract_sort_key(filename):
    nums = re.findall(r'\d+', filename)
    return tuple(map(int, nums)) if nums else (0, 0)


def main(image_folder, anno_folder, out_folder, gen_type):
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
    orientations = defaultdict(dict)
    max_text = 0
    no_6 = 0

    for filename in tqdm(files):
        index = re.findall(r'\d+', filename)[0]
        anno_filename = f"gt_img_{index}.txt"
        anno_file_path = os.path.join(label_input, anno_filename)
        image_path = os.path.join(image_input_path, filename)
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)

        # exif = image._getexif()
        # if exif is not None:
        #     orientation = exif.get(274)  # 274번 태그가 Orientation
        #     orientations[orientation] = image_path
        #     # print("Orientation:", orientation)
        #     # if orientation == 6:
        #     #    no_6 += 1
        #     #    print(image_path)
        # else:
        #     pass
        #     # print("EXIF 정보 없음")
        
        with open(anno_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
            for i, line in enumerate(lines):
               total_num += 1
               if line == '': # maybe text file with extra new line in the end? -> yes confirmed!
                continue
               if line.split(',')[-1] == '###':
                continue
               parts = line.split(',', maxsplit=8) ###!!!!!
               coords = parts[:8]
               text = parts[-1]
               if text == '###':
                #   if len(parts) != 9:
                #      import pdb;pdb.set_trace()
                  continue
               
               len_of_text = len(text)

               if len_of_text > max_text:
                max_text = len_of_text
               
                

               int_list = list(map(int, coords))
               points = [(int_list[i], int_list[i + 1]) for i in range(0, len(int_list), 2)]
               xs, ys = zip(*points)
               bbox = (min(xs), min(ys), max(xs), max(ys))
               cropped = image.crop(bbox)
               img_file_name = f'MPSC_img_{index}_{i}.jpg'
               full_path = os.path.join(image_out_path, img_file_name)

            #    cropped.save(full_path)

               with open(f'mpsc_{gen_type}_exif.txt', 'a') as f: # 여러번 호출 시 이전 것에 append 함
                  f.write(full_path + ' ' + text + '\n')
            
               final_num += 1

               


        # if os.path.isfile(file_path):
           
            # with open(file_path, 'r', encoding='utf-8') as f:
            #     content = f.read()
    print(f'total lines: {total_num}')
    print(f'without invalid lines: {final_num}')
   #  print(f'max length: {max_text}')

    # # unique_orientations = set(orientations)
    # print("\n=== Orientation 값 요약 ===")
    # # print(f"총 이미지 수: {len(orientations)}")
    # # print(f"발견된 Orientation 값: {unique_orientations}")
    # # print(orientations)
    # for key, value in orientations.items():
    #     print(key, value)
    # # print(f'6: {no_6}')
                
gen_type = 'test'
image_folder = f'dataset/Reann_MPSC/MPSC/image/{gen_type}'
anno_folder = f'dataset/Reann_MPSC/MPSC/annotation/{gen_type}'
out_folder = f'/home/jyryu/workspace/DiG/dataset/Reann_MPSC/jyryu/image/{gen_type}'
main(image_folder, anno_folder, out_folder, gen_type)