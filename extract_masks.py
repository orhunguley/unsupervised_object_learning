from torch.utils.data import Dataset, DataLoader
from motsynth import MOTSynth
import os
import json
import numpy as np
import cv2
import glob
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if isinstance(polygons["counts"], list):
            rles = coco_mask.frPyObjects(polygons, height, width)

        else:
            rles = [polygons]

        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # mask = torch.as_tensor(mask, dtype=torch.uint8)
        # mask = mask.any(dim=2)
        masks.append(mask)
    # if masks:
    #     masks = torch.stack(masks, dim=0)
    # else:
    #     masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def extract_masks(video_id, global_idx, error_list):
    with open(f"/storage/user/brasoand/motsyn2/annotations/{video_id}.json") as fp:
        annotation = json.load(fp)
    print("Annotation File loaded.")
    frame_src = f"/storage/user/brasoand/motsyn2/frames/{video_id}/rgb/"
    segmentations = [an.get("segmentation") for an in annotation.get("annotations")]
    ped_ids = [an.get("ped_id") for an in annotation.get("annotations")]
    #     frame_indices = [an.get('image_id') for an in annotation.get('annotations')]
    frame_indices = np.array(
        [an.get("image_id") for an in annotation.get("annotations")]
    )
    changed_indices = np.where(frame_indices[:-1] != frame_indices[1:])[0]
    constant = frame_indices[0]
    prev_change = 0
    total_masks_list = []
    black_bgg_img_list = []
    for k, idx in enumerate(changed_indices):
        try:
            frame_id = frame_indices[prev_change] - constant
            f_segmentations = segmentations[prev_change : (idx + 1)]
            prev_change = idx + 1
            masks = [
                convert_coco_poly_to_mask([s], 1920, 1080) for s in f_segmentations
            ]
            save_dst = f"/usr/stud/gueley/Git/SCALOR/rand_static_bg_frames_td_gradient_masks/{str(global_idx).zfill(6)}.npy"
            np.save(save_dst, masks, allow_pickle=True)
        except Exception as e:
            raise e
            error_list.append(
                (
                    f"/storage/user/brasoand/motsyn2/frames/{video_id}/rgb/{str(frame_id).zfill(4)}.jpg",
                    e,
                )
            )
        global_idx += 1

    try:
        frame_id = frame_indices[prev_change] - constant
        f_segmentations = segmentations[prev_change:]
        prev_change = idx + 1
        masks = [convert_coco_poly_to_mask([s], 1920, 1080) for s in f_segmentations]

        save_dst = f"/usr/stud/gueley/Git/SCALOR/rand_static_bg_frames_td_gradient_masks/{str(global_idx).zfill(6)}.npy"
        np.save(save_dst, masks, allow_pickle=True)

    except Exception as e:
        error_list.append(
            (
                f"/storage/user/brasoand/motsyn2/frames/{video_id}/rgb/{str(frame_id).zfill(4)}.jpg",
                e,
            )
        )
        print(e)
    global_idx += 1

    return global_idx, error_list

def extract_masks_v2(video_id, global_idx, error_list):
    with open(f"/storage/user/brasoand/motsyn2/annotations/{video_id}.json") as fp:
        annotation = json.load(fp)
    print("Annotation File loaded.")
    frame_src = f"/storage/user/brasoand/motsyn2/frames/{video_id}/rgb/"
    segmentations = [an.get("segmentation") for an in annotation.get("annotations")]
    ped_ids = [an.get("ped_id") for an in annotation.get("annotations")]
    frame_nos = [an.get("frame_n") for an in annotation.get("annotations")]

    #     frame_indices = [an.get('image_id') for an in annotation.get('annotations')]
    frame_indices = np.array(
        [an.get("image_id") for an in annotation.get("annotations")]
    )
    changed_indices = np.where(frame_indices[:-1] != frame_indices[1:])[0]
    constant = frame_indices[0]
    prev_change = 0
    total_masks_list = []
    black_bgg_img_list = []
    for k, idx in enumerate(changed_indices):
        try:
            frame_id = frame_indices[prev_change] - constant
            f_segmentations = segmentations[prev_change : (idx + 1)]
            f_frame_nos = frame_nos[prev_change : (idx + 1)]
            f_ped_ids = ped_ids[prev_change : (idx + 1)]
            prev_change = idx + 1
            # masks = [
            #     convert_coco_poly_to_mask([s], 1920, 1080) for s in f_segmentations
            # ]
            save_dst = f"/usr/stud/gueley/Git/SCALOR/rand_static_bg_frames_td_gradient_masks/{str(global_idx).zfill(6)}.npy"
            # mask_lists = np.load(save_dst, allow_pickle=True)

            frame_ann = {"ped_ids": f_ped_ids,
                         "frame_nos": f_frame_nos,
                         "segmentations": f_segmentations}

            np.save(save_dst, frame_ann, allow_pickle=True)
        except Exception as e:
            raise e
            error_list.append(
                (
                    f"/storage/user/brasoand/motsyn2/frames/{video_id}/rgb/{str(frame_id).zfill(4)}.jpg",
                    e,
                )
            )
        global_idx += 1

    try:
        frame_id = frame_indices[prev_change] - constant
        f_segmentations = segmentations[prev_change:]
        f_frame_nos = frame_nos[prev_change : (idx + 1)]
        f_ped_ids = ped_ids[prev_change : (idx + 1)]
        prev_change = idx + 1
        # masks = [convert_coco_poly_to_mask([s], 1920, 1080) for s in f_segmentations]

        save_dst = f"/usr/stud/gueley/Git/SCALOR/rand_static_bg_frames_td_gradient_masks/{str(global_idx).zfill(6)}.npy"
        # mask_lists = np.load(save_dst, allow_pickle=True)

        frame_ann = {"ped_ids": f_ped_ids,
                     "frame_nos": f_frame_nos,
                     "segmentations": f_segmentations}

        np.save(save_dst, frame_ann, allow_pickle=True)

    except Exception as e:
        error_list.append(
            (
                f"/storage/user/brasoand/motsyn2/frames/{video_id}/rgb/{str(frame_id).zfill(4)}.jpg",
                e,
            )
        )
        print(e)
    global_idx += 1

    return global_idx, error_list

suitable_video_list_only_from_top = [
    2,
    10,
    19,
    28,
    33,
    34,
    42,
    43,
    50,
    52,
    59,
    60,
    62,
    68,
    69,
    73,
    77,
    78,
    84,
    86,
    87,
    90,
    94,
    96,
    103,
    104,
    112,
    113,
    121,
    122,
    130,
    138,
    147,
    156,
    161,
    162,
    170,
    171,
    178,
    180,
    187,
    188,
    196,
    197,
]


top_down_list = [
    19,
    28,
    33,
    42,
    59,
    60,
    68,
    73,
    86,
    94,
    103,
    112,
    121,
    130,
    147,
    161,
    170,
    178,
    187,
]

error_list = []
total_frames = len(top_down_list) * 1800
global_idx = 0


# img = cv2.imread("gradient.jpg")
# random_bg = np.uint8(cv2.resize(img, dsize=(1920, 1080), interpolation=cv2.INTER_CUBIC))
# random_bg = np.transpose(random_bg, (2, 0, 1))

# random_bg = np.random.randint(0, 256, (3, 1080, 1920))

# suitable_video_list_only_from_top = [suitable_video_list_only_from_top[27]]

for video_id in top_down_list:
    global_idx, error_list = extract_masks_v2(
        str(video_id).zfill(3), global_idx, error_list
    )
    print(
        f"********************{global_idx}/{total_frames} done.***********************",
        flush=True,
    )

print(error_list, flush=True)
