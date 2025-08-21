from torch.utils.data import Sampler
# from torch.utils.data import DistributedSampler
import torch.distributed as dist
import random
import numpy as np
import os
import torch

class RatioSampler(Sampler):
    def __init__(self, data_source, num_replicas=None, rank=None, shuffle=True, custom_arg=None, is_training=True, bs=64):
        # super().__init__(data_source, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.custom_arg = custom_arg

        self.data_source = data_source
        self.ds_width = data_source.ds_width
        # self.seed = data_source.seed
        self.base_batch_size = bs

        scales = [128, 32]
        self.base_im_h = scales[1]
        self.base_im_w = scales[0]
        self.epoch = 0
        


        self.n_data_samples = len(self.data_source)
        img_indices = [idx for idx in range(self.n_data_samples)]

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        # rank = (int(os.environ['LOCAL_RANK'])
        #         if 'LOCAL_RANK' in os.environ else 0)
        # self.rank = rank
        # num_replicas = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # self.num_replicas = num_replicas

        if is_training:
            self.shuffle = True

        self.img_indices = img_indices
        

        if is_training:
            indices_rank_i = self.img_indices[
                self.rank:len(self.img_indices):self.num_replicas]
        else:
            indices_rank_i = self.img_indices

        if self.ds_width:
            self.wh_ratio = data_source.wh_ratio
            self.wh_ratio_sort = data_source.wh_ratio_sort

        if is_training:
            self.shuffle = True
        else:
            self.shuffle = False

        self.is_training = is_training
        self.indices_rank_i_ori = np.array(self.wh_ratio_sort[indices_rank_i])
        self.indices_rank_i_ratio = self.wh_ratio[self.indices_rank_i_ori]
        indices_rank_i_ratio_unique = np.unique(self.indices_rank_i_ratio)
        self.indices_rank_i_ratio_unique = indices_rank_i_ratio_unique.tolist()
        
        self.batch_list = self.create_batch()
        self.length = len(self.batch_list)
        self.batchs_in_one_epoch_id = [i for i in range(self.length)]

    def create_batch(self):
        batch_list = []
        for ratio in self.indices_rank_i_ratio_unique:
            ratio_ids = np.where(self.indices_rank_i_ratio == ratio)[0]
            ratio_ids = self.indices_rank_i_ori[ratio_ids]
            if self.shuffle:
                random.shuffle(ratio_ids)
            num_ratio = ratio_ids.shape[0]
            if ratio < 5:
                batch_size_ratio = self.base_batch_size # 256
                # batch_size_ratio = 50
            else:
                batch_size_ratio = min(
                    self.max_bs,
                    int(
                        max(1, (self.base_elements /
                                (self.base_im_h * ratio * self.base_im_h)))))
            if num_ratio > batch_size_ratio:
                batch_num_ratio = num_ratio // batch_size_ratio
                print(self.rank, num_ratio, ratio * self.base_im_h,
                      batch_num_ratio, batch_size_ratio)
                ratio_ids_full = ratio_ids[:batch_num_ratio *
                                           batch_size_ratio].reshape(
                                               batch_num_ratio,
                                               batch_size_ratio, 1) # (17, 256, 1)
                w = np.full_like(ratio_ids_full, ratio * self.base_im_h) # ratio * 32 # 우선 같은 배치 내에서 ration 통일
                h = np.full_like(ratio_ids_full, self.base_im_h) # 32
                ra_wh = np.full_like(ratio_ids_full, ratio)
                ratio_ids_full = np.concatenate([w, h, ratio_ids_full, ra_wh],
                                                axis=-1) # (17, 256, 4)
                batch_ratio = ratio_ids_full.tolist()

                if batch_num_ratio * batch_size_ratio < num_ratio:
                    drop = ratio_ids[batch_num_ratio * batch_size_ratio:] # 예) 225
                    if self.is_training:
                        drop_full = ratio_ids[:batch_size_ratio - (
                            num_ratio - batch_num_ratio * batch_size_ratio)]
                        drop = np.append(drop_full, drop)
                    drop = drop.reshape(-1, 1)
                    w = np.full_like(drop, ratio * self.base_im_h)
                    h = np.full_like(drop, self.base_im_h)
                    ra_wh = np.full_like(drop, ratio)

                    drop = np.concatenate([w, h, drop, ra_wh], axis=-1)

                    batch_ratio.append(drop.tolist())
                    batch_list += batch_ratio
            else:
                print(self.rank, num_ratio, ratio * self.base_im_h,
                      batch_size_ratio)
                ratio_ids = ratio_ids.reshape(-1, 1)
                w = np.full_like(ratio_ids, ratio * self.base_im_h)
                h = np.full_like(ratio_ids, self.base_im_h)
                ra_wh = np.full_like(ratio_ids, ratio)

                ratio_ids = np.concatenate([w, h, ratio_ids, ra_wh], axis=-1)
                batch_list.append(ratio_ids.tolist())
        return batch_list

    def __iter__(self):
        
        if self.shuffle or self.is_training:
            random.seed(self.epoch)
            self.epoch += 1
            self.batch_list = self.create_batch()
            if len(self.batch_list) != self.length:
                raise RuntimeError(
                    f"[RatioSampler] __len__({self.length})와 __iter__에서 생성된 배치 수({len(new_batch_list)})가 다릅니다. "
                    "에포크마다 batch_list를 다시 만들면 length/batchs_in_one_epoch_id도 함께 갱신해야 합니다."
                )

            random.shuffle(self.batchs_in_one_epoch_id)
        
        for batch_tuple_id in self.batchs_in_one_epoch_id:
            yield self.batch_list[batch_tuple_id] # 배치 하나! 64개 샘플
            # for sample in self.batch_list[batch_tuple_id]:  # 개별 sample로 풀어서
            #     yield sample


    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):

        return self.length