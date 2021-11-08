import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
from data import TrainStation
from motsynth import MOTSynth, MOTSynthBlackBG
from log_utils import log_summary
from utils import save_ckpt, load_ckpt, print_scalor
from common import *
import parse
from utils import spatial_transform, visualize
from tensorboardX import SummaryWriter
from scalor import SCALOR
from pycocotools import mask as coco_mask


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


def calculate_IoU(pred, target):
    '''
    Calculates the Intersection over Union(Intersection over Union).
    '''
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


import json
with open('predictions_model_perceptual_gan_v2_7x7_003.json') as f:
    preds = json.load(f)
    
# with open('predictions_toy.json') as f:
#     preds = json.load(f)
annotation_indices = [28, 33, 42, 59]


threshold = 0.1
import copy

last_preds_list = []
counter = 0
for ann_idx in annotation_indices:
    video_objects = [p for p in preds if p["video_id"] == ann_idx]
    video_objects_dict = {i:obj for i, obj in enumerate(video_objects)}
    
    for i in range(9, 350, 10):

    #     candidate_objs = [(idx, obj) for (idx, obj) in enumerate(video_objects) if obj["segmentations"][i] != None]
        candidate_objs = [(idx, obj) for idx, obj in video_objects_dict.items() if obj["segmentations"][i] != None]
    #     objs_to_concetenate = [(idx, obj) for (idx, obj) in enumerate(video_objects) if obj["segmentations"][i+1] != None]
        objs_to_concetenate = [(idx, obj) for idx, obj in video_objects_dict.items() if obj["segmentations"][i+1] != None]

        keys_to_remove_list = []
        for cand_idx, cand in candidate_objs:
            iou_calcs = []

            for next_seq_obj_idx, next_seq_obj in objs_to_concetenate:
                a = convert_coco_poly_to_mask([cand["segmentations"][i]], 1080, 1080)[0]
                b = convert_coco_poly_to_mask([next_seq_obj["segmentations"][i+1]], 1080, 1080)[0]
                iou_calc = calculate_IoU(a, b)
                iou_calcs.append(iou_calc)

            if max(iou_calcs) > threshold:
                counter += 1
                max_index = np.argmax(iou_calcs)

                cand["segmentations"][i+1:i+11] = copy.deepcopy(objs_to_concetenate[max_index][1]["segmentations"][i+1:i+11])
                video_objects_dict[cand_idx]["segmentations"] = copy.deepcopy(cand["segmentations"])
                keys_to_remove_list.append(objs_to_concetenate[max_index][0])
    #             print(f"cand_idx:{cand_idx} concetanated with next_seq_obj_idx: {i+1+max_index}, with iou: {max(iou_calcs)}")

            else:
                max_index = np.argmax(iou_calcs)
    #             print(f"cand_idx:{cand_idx} CANNOT BE next_seq_obj_idx: {i+1+max_index}, max iou: {max(iou_calcs)}")

        keys_to_remove_list = list(set(keys_to_remove_list))
        for key in keys_to_remove_list:
            video_objects_dict.pop(key)

    last_preds_list.append(copy.deepcopy(video_objects_dict))
    print(f"{ann_idx} done.")
    
    
a = [list(vid.values()) for vid in last_preds_list]

import itertools
all_preds = list(itertools.chain.from_iterable(a))

import json
with open('evaluation/predictions_model_perceptual_gan_v2_7x7_003_last_v.json', 'w') as f:
    json.dump(all_preds, f)