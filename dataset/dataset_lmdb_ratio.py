from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import io
import math
import lmdb
import string
import six
import numpy as np
from PIL import Image, ImageFile
import cv2
import pickle
from torchvision import transforms as T
from torchvision.transforms import functional as F

from transforms import CVColorJitter, CVDeterioration, CVGeometry

from preprocess import transform

from imgaug import augmenters as iaa
ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(0) # cv2's multiprocess will impact the dataloader's workers.

class ImageLmdb(Dataset):
  def __init__(self, root, voc_type, max_len, num_samples, transform,
               use_aug=False, use_abi_aug=False, use_color_aug=False):
    super(ImageLmdb, self).__init__()

    self.max_text_len = max_len



    self.ds_width = True # 이게 뭐지?
    min_ratio = 1
    max_ratio = 4
    # transforms_config = [{'DecodeImagePIL': {'img_mode': 'RGB'}}, {'PARSeqAugPIL': None}, {'CTCLabelEncode': {'character_dict_path': './tools/utils/EN_symbol_dict.txt', 'use_space_char': False, 'max_text_length': 25}}, {'KeepKeys': {'keep_keys': ['image', 'label', 'length']}}]
    self.padding = False
    self.padding_rand = False
    data_dir_list = root
    # self.seed = epoch

    ratio_list = 1.0
    ratio_list = [float(ratio_list)]
    self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir_list, ratio_list)
    self.data_idx_order_list = self.dataset_traversal()
    
    wh_ratio = np.around(np.array(self.get_wh_ratio())) # 정수로 반올림
    self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
    self.wh_ratio_sort = np.argsort(self.wh_ratio)
    # for i in range(max_ratio + 1):
    #   logger.info((1 * (self.wh_ratio == i)).sum())
    # self.ops = create_operators(transforms_config,
    #                                 global_config)
    self.base_shape = [[64, 64], [96, 48], [112, 40], [128, 32]]
    self.base_h = 32
    self.interpolation = T.InterpolationMode.BICUBIC
    mean = std = 0.5
    transforms = []
    transforms.extend([
            # T.ToTensor(),
            # T.Normalize(0.5, 0.5),
            CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
            CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
            CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
            # T.Resize((128, 224), interpolation=3),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    self.transforms = T.Compose(transforms)



    # self.env = lmdb.open(root, max_readers=32, readonly=True) # 데이터베이스 열어줌
    # # try:
    # #     self.env = lmdb.open(root, readonly=True, lock=False)
    # # except lmdb.Error as e:
    # #     print(f"LMDB Error: {e}")
    # #     if "No such file or directory" in str(e):
    # #         import pdb;pdb.set_trace()  # Start interactive debugger
    # # self.dig_mode = dig_mode
    # self.txn = self.env.begin()
    # self.nSamples = int(self.txn.get(b"num-samples"))

    # num_samples = num_samples if num_samples > 1 else int(self.nSamples * num_samples)
    # self.nSamples = int(min(self.nSamples, num_samples))

    # self.root = root
    # self.max_len = max_len
    # self.transform = transform
    # self.use_aug = use_aug
    # self.use_abi_aug = use_abi_aug
    # self.use_color_aug = use_color_aug
    # if use_aug:
    #   if use_abi_aug:
    #     mean = std = 0.5
    #     self.augment_abi = T.Compose([
    #           CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
    #           CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
    #           CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
    #           transforms.Resize((128, 224), interpolation=3),
    #           transforms.ToTensor(),
    #           transforms.Normalize(
    #               mean=torch.tensor(mean),
    #               std=torch.tensor(std))
    #       ])
    #   else:
    #     # augmentation following seqCLR
    #     if use_color_aug:
    #       self.augmentor = self.color_aug()
    #     else:
    #       self.augmentor = self.sequential_aug()
    #     mean = std = 0.5
    #     self.aug_transformer = transforms.Compose([
    #           transforms.Resize((128, 224), interpolation=3),
    #           transforms.RandomApply([
    #               transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    #           ], p=0.8),
    #           transforms.RandomGrayscale(p=0.2),
    #           transforms.ToTensor(),
    #           transforms.Normalize(
    #               mean=torch.tensor(mean),
    #               std=torch.tensor(std))
    #       ])

    # Generate vocabulary
    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.classes = self._find_classes(voc_type)
    self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
    self.idx_to_class = dict(zip(range(len(self.classes)), self.classes))
    self.use_lowercase = (voc_type == 'LOWERCASE')

  def load_hierarchical_lmdb_dataset(self, data_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, ratio in zip(data_dir_list, ratio_list):
            env = lmdb.open(dirpath, max_readers=32, readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            txn = env.begin(write=False)
            num_samples = int(txn.get('num-samples'.encode()))
            lmdb_sets[dataset_idx] = {
                'dirpath': dirpath,
                'env': env,
                'txn': txn,
                'num_samples': num_samples,
                'ratio_num_samples': int(ratio * num_samples)
            }
            dataset_idx += 1
        return lmdb_sets

  def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['ratio_num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['ratio_num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno # index
            data_idx_order_list[beg_idx:end_idx, 1] = list(
                random.sample(range(1, self.lmdb_sets[lno]['num_samples'] + 1),
                              self.lmdb_sets[lno]['ratio_num_samples']))
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list
  
  def get_wh_ratio(self):
        wh_ratio = []
        for idx in range(self.data_idx_order_list.shape[0]):
            lmdb_idx, file_idx = self.data_idx_order_list[idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            wh_key = 'wh-%09d'.encode() % file_idx
            wh = self.lmdb_sets[lmdb_idx]['txn'].get(wh_key)
            if wh is None:
                img_key = f'image-{file_idx:09d}'.encode()
                img = self.lmdb_sets[lmdb_idx]['txn'].get(img_key)
                buf = io.BytesIO(img)
                w, h = Image.open(buf).size
            else:
                wh = wh.decode('utf-8')
                w, h = wh.split('_')
            wh_ratio.append(float(w) / float(h))
        return wh_ratio
  
  def _find_classes(self, voc_type, EOS='EOS',
                    PADDING='PADDING', UNKNOWN='UNKNOWN'):
    '''
    voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
    '''
    voc = None
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    if voc_type == 'LOWERCASE':
      # voc = list(string.digits + string.ascii_lowercase)
      voc = list('0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    elif voc_type == 'ALLCASES':
      voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
      voc = list(string.printable[:-6])
    else:
      raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')
    

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    return voc

  def __len__(self):
    # return self.nSamples
    return self.data_idx_order_list.shape[0]

  def sequential_aug(self):
    aug_transform = transforms.Compose([
      iaa.Sequential(
        [
          iaa.SomeOf((2, 5),
          [
            iaa.LinearContrast((0.5, 1.0)),
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Crop(percent=((0, 0.3),
                              (0, 0.0),
                              (0, 0.3),
                              (0, 0.0)),
                              keep_size=True),
            iaa.Crop(percent=((0, 0.0),
                              (0, 0.1),
                              (0, 0.0),
                              (0, 0.1)),
                              keep_size=True),
            iaa.Sharpen(alpha=(0.0, 0.5),
                        lightness=(0.0, 0.5)),
            # iaa.AdditiveGaussianNoise(scale=(0, 0.15*255), per_channel=True),
            iaa.Rotate((-10, 10)),
            # iaa.Cutout(nb_iterations=1, size=(0.15, 0.25), squared=True),
            iaa.PiecewiseAffine(scale=(0.03, 0.04), mode='edge'),
            iaa.PerspectiveTransform(scale=(0.05, 0.1)),
            iaa.Solarize(1, threshold=(32, 128), invert_above_threshold=0.5, per_channel=False),
            iaa.Grayscale(alpha=(0.0, 1.0)),
          ],
          random_order=True)
        ]
      ).augment_image,
    ])
    return aug_transform
  
  def color_aug(self):
    aug_transform = transforms.Compose([
      iaa.Sequential(
        [
          iaa.SomeOf((2, 5),
          [
            iaa.LinearContrast((0.5, 1.0)),
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Sharpen(alpha=(0.0, 0.5),
                        lightness=(0.0, 0.5)),
            iaa.Solarize(1, threshold=(32, 128), invert_above_threshold=0.5, per_channel=False),
            iaa.Grayscale(alpha=(0.0, 1.0)),
          ],
          random_order=True)
        ]
      ).augment_image,
    ])
    return aug_transform

  def open_lmdb(self):
    self.env = lmdb.open(self.root, readonly=True, create=False)
    # self.txn = self.env.begin(buffers=True)
    self.txn = self.env.begin()

  '''
  def __getitem__(self, index):
    if not hasattr(self, 'txn'):
      self.open_lmdb()

    # Load image
    assert index <= len(self), 'index range error'
    index += 1
    img_key = b'image-%09d' % index
    img_key_str = 'image-%09d' % index
    imgbuf = self.txn.get(img_key)

    buf = six.BytesIO()
    if imgbuf is None:
      import pdb;pdb.set_trace()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      img = Image.open(buf).convert('RGB')
    except IOError:
      print('Corrupted image for %d' % index)
      # return self[index + 1]
      raise ValueError('Corrupted image')
    
    # Load label
    label_key = b'label-%09d' % index
    word = self.txn.get(label_key).decode()
    if self.use_lowercase:
      word = word.lower()
    if len(word) + 1 >= self.max_len:
      # print('%s is too long.' % word)
      # return self[index + 1]
      raise ValueError('word is too long')
    ## fill with the padding token
    label = np.full((self.max_len,), self.class_to_idx['PADDING'], dtype=int)
    label_list = []
    for char in word:
      if char in self.class_to_idx:
        label_list.append(self.class_to_idx[char])
      else:
        label_list.append(self.class_to_idx['UNKNOWN'])
    ## add a stop token
    label_list = label_list + [self.class_to_idx['EOS']]
    assert len(label_list) <= self.max_len
    label[:len(label_list)] = np.array(label_list)
    if len(label) <= 0:
      # return self[index + 1]
      raise ValueError('empty label')
    
    # Label length
    label_len = len(label_list)

    # added 
    # if self.dig_mode == 'dig-seed':

    #   embed_key = b'embed-%09d' % index
    #   embed_vec = self.txn.get(embed_key)
    #   if embed_vec is not None:
    #     embed_vec = embed_vec.decode()
    #   else:
    #     embed_vec = ' '.join(['0']*300)
    #   # make string vector to numpy ndarray
    #   embed_vec = np.array(embed_vec.split()).astype(np.float32)
    #   if embed_vec.shape[0] != 300:
    #     # return self[index + 1]
    #     raise ValueError('vector dim not 300')
    
    # if self.dig_mode == 'dig-seed-char':
    #   embed_key = b'embed-%09d' % index
    #   embed_vec = self.txn.get(embed_key)
    #   if embed_vec is not None:
    #     embed_vec = pickle.loads(embed_vec)
        
    #     embed_vec = np.array(embed_vec)
    #   else:
    #     raise ValueError('couldn\'t get embeddings')
    
    # augmentation
    if self.use_aug:
      if self.use_abi_aug:
        aug_img = self.augment_abi(img)
      else:
        # augmentation
        aug_img = self.augmentor(np.asarray(img))
        aug_img = Image.fromarray(np.uint8(aug_img))
        aug_img = self.aug_transformer(aug_img)
      # return aug_img, label, label_len, embed_vec
      # if self.dig_mode == 'dig':
      #   return aug_img, label, label_len, img_key_str
          
      # else:
      #   return aug_img, label, label_len, embed_vec, img_key_str # 시각화 시 
      return aug_img, label, label_len, img_key_str
        
      
    else:
      assert self.transform is not None
      img = self.transform(img)
      # if self.dig_mode == 'dig':
      #   return img, label, label_len, img_key_str
        
      # else:

      #   return img, label, label_len, embed_vec, img_key_str
      return img, label, label_len, img_key_str
      '''
    
  def resize_norm_img(self, data, gen_ratio, padding=True):
    
    img = data['image']
    w, h = img.size
    if self.padding_rand and random.random() < 0.5:
        padding = not padding
    imgW, imgH = self.base_shape[gen_ratio - 1] if gen_ratio <= 4 else [
        self.base_h * gen_ratio, self.base_h
    ]
    use_ratio = imgW // imgH
    if use_ratio >= (w // h) + 2:
        self.error += 1
        return None
    if not padding:
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(
                math.ceil(imgH * ratio * (random.random() + 0.5)))
            resized_w = min(imgW, resized_w)
    resized_image = F.resize(img, (imgH, resized_w),
                              interpolation=self.interpolation)
    img = self.transforms(resized_image)
    def _get_hw(x):
        # 텐서(C,H,W) 또는 PIL 모두 지원
        if torch.is_tensor(x):
            return int(x.shape[-2]), int(x.shape[-1])
        else:  # PIL
            return int(x.size[1]), int(x.size[0])

    curH, curW = _get_hw(img)
    if (curH != imgH) or (curW != imgW):
        img = F.resize(img, (int(imgH), int(resized_w)),
                       interpolation=self.interpolation)
    if resized_w < imgW and padding:
        # img = F.pad(img, [0, 0, imgW-resized_w, 0], fill=0.)
        if self.padding_doub and random.random() < 0.5:
            img = F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
        else:
            img = F.pad(img, [imgW - resized_w, 0, 0, 0], fill=0.)
    # valid_ratio = min(1.0, float(resized_w / imgW))
    data['image'] = img
    # data['valid_ratio'] = valid_ratio
    return data
    
  def get_lmdb_sample_info(self, txn, index):
    label_key = 'label-%09d'.encode() % index
    label = txn.get(label_key)
    if label is None:
        return None
    label = label.decode('utf-8')
    img_key = 'image-%09d'.encode() % index
    imgbuf = txn.get(img_key)
    return imgbuf, label

  def __getitem__(self, properties):
        
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        img_key_str = 'image-%09d' % file_idx
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        img, word = sample_info

        buf = six.BytesIO()
        if img is None:
          import pdb;pdb.set_trace()
        buf.write(img)
        buf.seek(0)
        try:
          img = Image.open(buf).convert('RGB')
        except IOError:
          # print('Corrupted image for %d' % index)
          raise ValueError('Corrupted image')
        
        
        if self.use_lowercase:
          word = word.lower()
        
        if len(word) == 0 or len(word) + 1 >= self.max_text_len:
          ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
          ids = random.sample(ratio_ids, 1)
          return self.__getitem__([img_width, img_height, ids[0], ratio])
        
        ## fill with the padding token
        label = np.full((self.max_text_len,), self.class_to_idx['PADDING'], dtype=int)
        label_list = []
        for char in word:
          if char in self.class_to_idx:
            label_list.append(self.class_to_idx[char])
          else:
            label_list.append(self.class_to_idx['UNKNOWN'])
        ## add a stop token
        label_list = label_list + [self.class_to_idx['EOS']]
        # if len(label_list) > self.max_text_len:
        #   import pdb; pdb.set_trace()  # pdb에서 중단
        #   raise AssertionError(f"label_list 길이가 너무 깁니다: {len(label_list)} > {self.max_text_len}")
        assert len(label_list) <= self.max_text_len
        label[:len(label_list)] = np.array(label_list)
        
        
        # Label length
        label_len = len(label_list)
        
        data = {'image': img, 'label': label, 'length': label_len, 'imgkey': img_key_str}
        outs = data
        # outs = transform(data, self.ops[:-1]) # decode image
        if outs is not None:
            outs = self.resize_norm_img(outs, ratio, padding=self.padding)
            if outs is None:
                ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
                ids = random.sample(ratio_ids, 1)
                return self.__getitem__([img_width, img_height, ids[0], ratio])
            # outs = transform(outs, self.ops[-1:])
            outs = outs
        # if outs is None: # text length = 0 or > max_length
        #     ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
        #     ids = random.sample(ratio_ids, 1)
        #     return self.__getitem__([img_width, img_height, ids[0], ratio])
        return outs
      
