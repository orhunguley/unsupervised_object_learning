import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from common import *
from PIL import Image, ImageFile
import glob
import itertools
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True
import time


video_id_embs = {str(k):torch.randn(10) for k in top_down_list}

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if isinstance(polygons['counts'], list):
            rles = coco_mask.frPyObjects(polygons, height, width)
        
        else:
            rles = [polygons]

        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        #mask = torch.as_tensor(mask, dtype=torch.uint8)
        #mask = mask.any(dim=2)
        masks.append(mask)
    # if masks:
    #     masks = torch.stack(masks, dim=0)
    # else:
    #     masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

    
class MOTSynth(Dataset):
    '''Class for training SCALOR with original MOTSynth sequences'''
    def __init__(self, data_dir, train=False):

        self.data_dir = data_dir
        self.skip_freq = 5
        self.phase_train = train
        #         self.video_dirs = [os.path.join(data_dir, video_dir) for video_dir in os.listdir(data_dir)]

        self.video_dirs = [
            os.path.join(data_dir, video_dir) for video_dir in top_down_list
        ]

        random.shuffle(self.video_dirs)
        
        # if self.phase_train:
        #     self.video_dirs = self.video_dirs[: -(len(self.video_dirs) // 10)]
        # else:
        #     self.video_dirs = self.video_dirs[-(len(self.video_dirs) // 10) :]
        print(f"DataSet Used: MOTSynth", flush=True)
        print(f"Suitable Videos: {self.video_dirs}", flush=True)
        print(f"Suitable Videos Length: {len(self.video_dirs)}", flush=True)
        print("Video Dirs constructed.", flush=True)
        self.frame_dirs = [
            sorted(glob.glob(video_dir + "/rgb/*")) for video_dir in self.video_dirs
        ]
        self.frame_dirs = sorted(list(itertools.chain(*self.frame_dirs)))
        print("Frame Dirs constructed.", flush=True)
        #         self.frame_dirs = []
        self.video_id = 0
        self.idx = 0
        print("Dataset initialized.", flush=True)

    def get_frames(self, index, seq_len=10, skip_freq=5):
        b = np.random.randint(0, 5, (1))[0]
        starting_idx = index * (seq_len * skip_freq) + np.random.randint(0, 5, (1))[0]
        return np.arange(starting_idx, starting_idx + (seq_len * skip_freq), 5)

    def __getitem__(self, index):
        #         print(index)
        index_list = self.get_frames(index, seq_len, self.skip_freq)
        image_list = []
        #         print("-------------------------", flush=True)
        k = int(np.random.rand(1)[0] * 840)
        k = 0
        for idx in index_list:
            f_n = self.frame_dirs[idx]
            #             print(f_n, flush=True)
            im = Image.open(f_n)

            im = im.crop(box=(k, 0, 1920 - (840 - k), 1080))
            #             print(im.size)
            im_array = np.array(im)

            #             im = im.crop(box=(left_edge, upper_edge, left_edge + self.args.train_station_cropping_origin,
            #                               upper_edge + self.args.train_station_cropping_origin))
            im = im.resize((img_h, img_w), resample=Image.BILINEAR)
            im_tensor = torch.from_numpy(np.array(im) / 255).permute(2, 0, 1)
            image_list.append(im_tensor)
        #         print("-------------------------", flush=True)

        current_video_id = f_n.split("/")[-3]
        img = torch.stack(image_list, dim=0)
        #         print(img.shape)
        #         self.idx +=1
        #         print(f"Status: {self.idx}/{int(len(self.frame_dirs) / seq_len)} done.", end='\r', flush=True)
        return img.float(), video_id_embs[current_video_id]

    def __len__(self):
        return int(len(self.frame_dirs) / (seq_len * self.skip_freq))


class MOTSynthV2(Dataset):
    '''This class is for singleVideo batch training'''
    def __init__(self, data_dir, train=False):

        self.data_dir = data_dir
        self.skip_freq = 5
        self.phase_train = train
        self.current_video = None
        #         self.video_dirs = [os.path.join(data_dir, video_dir) for video_dir in os.listdir(data_dir)]
        
        print(f"Dataset Length: {len(top_down_list)}")
        self.video_dirs = [
            os.path.join(data_dir, video_dir) for video_dir in top_down_list
        ]

        random.shuffle(self.video_dirs)
        
        # if self.phase_train:
        #     self.video_dirs = self.video_dirs[: -(len(self.video_dirs) // 10)]
        # else:
        #     self.video_dirs = self.video_dirs[-(len(self.video_dirs) // 10) :]
        print(f"DataSet Used: MOTSynth", flush=True)
        print(f"Suitable Videos: {self.video_dirs}", flush=True)
        print(f"Suitable Videos Length: {len(self.video_dirs)}", flush=True)
        print("Video Dirs constructed.", flush=True)
        self.frame_dirs = [
            sorted(glob.glob(video_dir + "/rgb/*")) for video_dir in self.video_dirs
        ]
        self.frame_dirs = sorted(list(itertools.chain(*self.frame_dirs)))
        print("Frame Dirs constructed.", flush=True)
        #         self.frame_dirs = []
        self.video_id = 0
        self.idx = 0
        print("Dataset initialized.", flush=True)

    def get_frames(self, index, seq_len=10, skip_freq=5):
        b = np.random.randint(0, 5, (1))[0]
        starting_idx = index * (seq_len * skip_freq) + np.random.randint(0, 5, (1))[0]
        return np.arange(starting_idx, starting_idx + (seq_len * skip_freq), 5)

    def __getitem__(self, index):

        current_video_frame_dirs = [f for f in self.frame_dirs if f"frames/{self.current_video}" in f]
#         print(len(current_video_frame_dirs))
        real_index = index % 36
        index_list = self.get_frames(real_index, seq_len, self.skip_freq)
        image_list = []
        k = int(np.random.rand(1)[0] * 840)
        for idx in index_list:
#             f_n = self.frame_dirs[idx]
            f_n = current_video_frame_dirs[idx]
            im = Image.open(f_n)

            im = im.crop(box=(k, 0, 1920 - (840 - k), 1080))
            im_array = np.array(im)

            im = im.resize((img_h, img_w), resample=Image.BILINEAR)
            im_tensor = torch.from_numpy(np.array(im) / 255).permute(2, 0, 1)
            image_list.append(im_tensor)


        img = torch.stack(image_list, dim=0)

        return img.float(), torch.zeros(1)

    def __len__(self):
        return int(len(self.frame_dirs) / (seq_len * self.skip_freq))



class MOTSynthBlackBG(Dataset):
    '''This class is for BG extracted dataset'''
    def __init__(self, data_dir, train=True):

        self.data_dir = data_dir
        self.skip_freq = 5
        self.phase_train = train

        self.frame_dirs = sorted(
            [os.path.join(data_dir, a) for a in os.listdir(data_dir)]
        )

        if len(self.frame_dirs) != 34200:
            raise Exception("Something is not correct.")

        #         if self.phase_train:
        #             self.frame_dirs = self.frame_dirs[:-(len(self.frame_dirs) // 10)]
        #         else:
        #             self.frame_dirs = self.frame_dirs[-(len(self.frame_dirs) // 10):]

        print("Frame Dirs constructed.", flush=True)
        self.video_id = 0
        self.idx = 0
        print("Dataset initialized.", flush=True)

    def get_frames(self, index, seq_len=10, skip_freq=5):
        b = np.random.randint(0, 5, (1))[0]
        if not self.phase_train:
            b = 0 #THIS IS FOR EVAL
#         starting_idx = index * (seq_len * skip_freq) + np.random.randint(0, 5, (1))[0]
        starting_idx = index * (seq_len * skip_freq) + b
        return np.arange(starting_idx, starting_idx + (seq_len * skip_freq), 5)

    def __getitem__(self, index):
        index_list = self.get_frames(index, seq_len, self.skip_freq)
        image_list = []
        k = int(np.random.rand(1)[0] * 840)

        if not self.phase_train:
            k = 0 #THIS IS FOR EVAL
        # k = 0
        for idx in index_list:
            f_n = self.frame_dirs[idx]
            im = Image.open(f_n)

            im = im.crop(box=(k, 0, 1920 - (840 - k), 1080))
            im_array = np.array(im)

            im = im.resize((img_h, img_w), resample=Image.BILINEAR)
            im_tensor = torch.from_numpy(np.array(im) / 255).permute(2, 0, 1)
            image_list.append(im_tensor)

        img = torch.stack(image_list, dim=0)

        return img.float(), torch.zeros(1)

    def __len__(self):
        return int(len(self.frame_dirs) / (seq_len * self.skip_freq))


class RandomMaskDataset(Dataset):
    def __init__(self, data_dir="/usr/stud/gueley/Git/SCALOR/random_masks_dataset/", train=False):

        self.mask_f_names = os.listdir(data_dir)

        self.mask_paths = [
            os.path.join(data_dir, f_name) for f_name in self.mask_f_names
        ]


    def __getitem__(self, index):
        file_path = self.mask_paths[index]
        with open(file_path) as f:
            annotation = json.load(f)
        
        mask = convert_coco_poly_to_mask([annotation['segmentation']], 1920, 1080)
        
        k = int(np.random.rand(1)[0] * 840)
        
        
        return mask[0].squeeze(2)


    def __len__(self):
        return len(self.mask_f_names)