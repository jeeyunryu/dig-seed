import lmdb
import os
import cv2
import numpy as np
from PIL import Image
import six

# Path to LMDB and output folder
lmdb_path = "dataset/Reann_MPSC/jyryu/lmdb/train/mpsc.lmdb.train"
output_dir = "lmdb_images"

os.makedirs(output_dir, exist_ok=True)

# Open LMDB environment
env = lmdb.open(lmdb_path, readonly=True, lock=False)
num = 13092
with env.begin() as txn:
    img_key = b'image-%09d' % num
    imgbuf = txn.get(img_key)
    label_key = b'label-%09d' % num
    label = txn.get(label_key)
    print(label)
    label_num_sam = b'num-samples'
    num_samples = int(txn.get(label_num_sam))
    print(num_samples)

    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      img = Image.open(buf).convert('RGB')
      img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
      output_path = os.path.join(output_dir, f"mpsc_train_{num}.jpg")
      cv2.imwrite(output_path, img_cv)
    except IOError:
      print('Corrupted image')
    
    print('finished loading image from file')

    # cursor = txn.cursor()
    # for idx, (key, value) in enumerate(cursor):
    #     # Decode image from bytes
    #     img_array = np.frombuffer(value, dtype=np.uint8)
    #     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    #     if img is not None:
    #         out_path = os.path.join(output_dir, f"{key.decode('utf-8')}.jpg")
    #         cv2.imwrite(out_path, img)
    #         print(f"Saved {out_path}")
    #     else:
    #         print(f"Failed to decode image for key: {key.decode('utf-8')}")

env.close()
