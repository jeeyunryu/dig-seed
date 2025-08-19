import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import re
# import fasttext
import json
from tqdm import tqdm
import pickle
from dotenv import load_dotenv
import os

load_dotenv() # loads variable from .env file into environment

# fasttext_model = fasttext.load_model('../../fastText/cc.en.300.bin')
# response = client.embeddings.create(
#         input=char,
#         model="text-embedding-3-small"
#     )
# response.data[0].embedding

def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  imageBuf = np.fromstring(imageBin, dtype=np.uint8)
  try:
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
  except Exception as e:
      print(e)
      return False
  if imgH * imgW == 0:
    return False
  return True

def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      txn.put(k.encode(), v)

def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
  """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
  assert(len(imagePathList) == len(labelList))
  nSamples = len(imagePathList)
  env = lmdb.open(outputPath, map_size=1099511627776)
  cache = {}
  cnt = 1
  for i in tqdm(range(nSamples)):
    # if (cnt == 6):
    #   break
    imagePath = imagePathList[i]
    label = labelList[i]
    if len(label) == 0:
      import pdb;pdb.set_trace()
      continue
    if not os.path.exists(imagePath):
      print('%s does not exist' % imagePath)
      continue
    with open(imagePath, 'rb') as f: # 'rb': read binary
      imageBin = f.read() # already a bytes type
    if checkValid:
      if not checkImageIsValid(imageBin):
        print('%s is not a valid image' % imagePath)
        continue
    
    
    
    
  
    
    imageKey = 'image-%09d' % cnt
    labelKey = 'label-%09d' % cnt
    
    cache[imageKey] = imageBin
    cache[labelKey] = label.encode()
    
    if lexiconList:
      lexiconKey = 'lexicon-%09d' % cnt
      cache[lexiconKey] = ' '.join(lexiconList[i])
    if cnt % 1000 == 0:
      writeCache(env, cache)
      cache = {}
      print('Written %d / %d' % (cnt, nSamples))
    cnt += 1
  nSamples = cnt-1
  cache['num-samples'] = str(nSamples).encode()
  writeCache(env, cache)
  print('Created dataset with %d samples' % nSamples)

if __name__ == "__main__":
  """
  gt_file: the annotation of the dataset with the format:
  image_path_1 label_1
  image_path_2 label_2
  ...
  
  data_dir: the root dir of the images. i.e. data_dir + image_path_1 is the path of the image_1
  lmdb_output_path: the path of the generated LMDB file
  """
  data_dir = '/home/jyryu/workspace/DiG'
  lmdb_output_path = '/home/jyryu/workspace/DiG/dataset/Reann_MPSC/jyryu/lmdb/lmdb.train.exif.pad.L'
  # lmdb_output_path = 'trash'
  gt_file = os.path.join(data_dir, 'mpsc_train_exif_pad_L.txt')

  with open(gt_file, 'r') as f:
      lines = [line.strip('\n') for line in f.readlines()]

  imagePathList, labelList, embedList = [], [], []
  for i, line in enumerate(lines):
      splits = line.split(' ')
      image_name = splits[0]
      gt_text = splits[1]
      imagePathList.append(os.path.join(data_dir, image_name))
      labelList.append(gt_text)
  createDataset(lmdb_output_path, imagePathList, labelList)

  