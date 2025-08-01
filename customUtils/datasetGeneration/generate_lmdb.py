import argparse
import os
import re
from PIL import Image
from tqdm import tqdm
import lmdb

txt_path = 'mpsc_test_lmdb_temp.txt' #$
lmdb_path = "dataset/Reann_MPSC/jyryu/lmdb/test2/mpsc.lmdb.test" #$

os.makedirs(lmdb_path, exist_ok=True)
env = lmdb.open(lmdb_path, map_size=1 << 30) # 1 gigabytes
txn = env.begin(write=True) # transaction objet
num = 0

with open(txt_path, 'r') as f:
    lines = f.readlines()

for i, line in tqdm(enumerate(lines),  total=len(lines)):
    line = line.strip() # get rid of trailing new line char
   
    
    parts = line.split(' ')

    img_path = parts[0]
    label = parts[-1]
    image = Image.open(img_path)
    with open(img_path, "rb") as f:
        img_bin = f.read()
    key_img =  b'image-%09d' % (i+1) # b'label-000000042'
    
    txn.put(key_img, img_bin)
    key_label = b'label-%09d' % (i+1)
    txn.put(key_label, label.encode('utf-8'))
    num += 1
print(f'finished generating files in \'{lmdb_path}\'')
print(f'number of samples: {num}')
txn.put(b'num-samples', str(num).encode())
txn.commit()
env.close()