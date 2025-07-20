# import lmdb
# import os

# # Path where your LMDB file will be saved
# lmdb_path = "example.lmdb"

# # Data to write (key-value pairs)
# data = {
#     "item1": b"dataset/Reann_MPSC/jyryu/image/train/MPSC_img_0_0.jpg",
#     "item2": b"dataset/Reann_MPSC/jyryu/image/train/MPSC_img_0_1.jpg",
# }

# # Create an LMDB environment
# env = lmdb.open(lmdb_path, map_size=1 << 30)  # 1GB max size

# with env.begin(write=True) as txn:
#     for key, value in data.items():
#         txn.put(key.encode(), value)

# env.close()

import lmdb
import os

# Path where your LMDB file will be saved
lmdb_path = "example.lmdb.img"

# Original data: key -> image file path
data = {
    "item1": "dataset/Reann_MPSC/jyryu/image/train/MPSC_img_0_0.jpg",
    "item2": "dataset/Reann_MPSC/jyryu/image/train/MPSC_img_0_1.jpg",
}

# Create LMDB environment
env = lmdb.open(lmdb_path, map_size=1 << 30)  # 1GB max size

with env.begin(write=True) as txn:
    for key, img_path in data.items():
        # Read the image as binary
        with open(img_path, "rb") as f:
            img_bin = f.read()
        # Store the binary image data with the key
        txn.put(key.encode(), img_bin)

env.close()
