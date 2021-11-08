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
from utils import spatial_transform, visualize
from common import *
import parse
import pickle
import json
import skimage.transform as st
from pycocotools import mask as coco_mask
from tensorboardX import SummaryWriter
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
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from tensorboardX import SummaryWriter
from scalor import SCALOR
import pickle
import torch




def get_log_disc_dict(log_disc_list, j = 0, cs = 8, prefix="train", bs = 2):
    
    log_disc = {
        'z_what': log_disc_list[j]['z_what'].view(-1, cs * cs, z_what_dim),
        'z_where_scale':
            log_disc_list[j]['z_where'].view(-1, cs * cs, z_where_scale_dim + z_where_shift_dim)[:, :,
            :z_where_scale_dim],
        'z_where_shift':
            log_disc_list[j]['z_where'].view(-1, cs * cs, z_where_scale_dim + z_where_shift_dim)[:, :,
            z_where_scale_dim:],
        'z_pres': log_disc_list[j]['z_pres'].permute(0, 2, 3, 1),
        'z_pres_probs': torch.sigmoid(log_disc_list[j]['z_pres_logits']).permute(0, 2, 3, 1),
        'z_what_std': log_disc_list[j]['z_what_std'].view(-1, cs * cs, z_what_dim),
        'z_what_mean': log_disc_list[j]['z_what_mean'].view(-1, cs * cs, z_what_dim),
        'z_where_scale_std':
            log_disc_list[j]['z_where_std'].permute(0, 2, 3, 1)[:, :, :z_where_scale_dim],
        'z_where_scale_mean':
            log_disc_list[j]['z_where_mean'].permute(0, 2, 3, 1)[:, :, :z_where_scale_dim],
        'z_where_shift_std':
            log_disc_list[j]['z_where_std'].permute(0, 2, 3, 1)[:, :, z_where_scale_dim:],
        'z_where_shift_mean':
            log_disc_list[j]['z_where_mean'].permute(0, 2, 3, 1)[:, :, z_where_scale_dim:],
        'glimpse': log_disc_list[j]['x_att'].view(-1, cs * cs, 3, glimpse_size, glimpse_size) \
            if prefix != 'generate' else None,
        'glimpse_recon': log_disc_list[j]['y_att'].view(-1, cs * cs, 3, glimpse_size, glimpse_size),
        'prior_z_pres_prob': log_disc_list[j]['prior_z_pres_prob'].unsqueeze(0),
        'o_each_cell': spatial_transform(log_disc_list[j]['o_att'], log_disc_list[j]['z_where'],
                                         (cs * cs * bs, 3, img_h, img_w),
                                         inverse=True).view(-1, cs * cs, 3, img_h, img_w),
        'alpha_hat_each_cell': spatial_transform(log_disc_list[j]['alpha_att_hat'],
                                                 log_disc_list[j]['z_where'],
                                                 (cs * cs * bs, 1, img_h, img_w),
                                                 inverse=True).view(-1, cs * cs, 1, img_h, img_w),
        'alpha_each_cell': spatial_transform(log_disc_list[j]['alpha_att'], log_disc_list[j]['z_where'],
                                             (cs * cs * bs, 1, img_h, img_w),
                                             inverse=True).view(-1, cs * cs, 1, img_h, img_w),
        'y_each_cell': (log_disc_list[j]['y_each_cell'] * log_disc_list[j]['z_pres'].
                        view(-1, 1, 1, 1)).view(-1, cs * cs, 3, img_h, img_w),
        'z_depth': log_disc_list[j]['z_depth'].view(-1, cs * cs, z_depth_dim),
        'z_depth_std': log_disc_list[j]['z_depth_std'].view(-1, cs * cs, z_depth_dim),
        'z_depth_mean': log_disc_list[j]['z_depth_mean'].view(-1, cs * cs, z_depth_dim),
        'z_pres_logits': log_disc_list[j]['z_pres_logits'].permute(0, 2, 3, 1),
        'z_pres_y': log_disc_list[j]['z_pres_y'].permute(0, 2, 3, 1)
    }
    return log_disc

def get_log_prop_dict(log_prop_list, j = 0, cs = 8, prefix="train", bs = 2):
    if log_prop_list[j]:
        log_prop = {
            'z_what': log_prop_list[j]['z_what'].view(bs, -1, z_what_dim),
            'z_where_scale':
                log_prop_list[j]['z_where'].view(bs, -1, z_where_scale_dim + z_where_shift_dim)[:, :,
                :z_where_scale_dim],
            'z_where_shift':
                log_prop_list[j]['z_where'].view(bs, -1, z_where_scale_dim + z_where_shift_dim)[:, :,
                z_where_scale_dim:],
            'z_pres': log_prop_list[j]['z_pres'],
            'z_what_std': log_prop_list[j]['z_what_std'].view(bs, -1, z_what_dim),
            'z_what_mean': log_prop_list[j]['z_what_mean'].view(bs, -1, z_what_dim),
            'z_where_bias_scale_std':
                log_prop_list[j]['z_where_bias_std'][:, :, :z_where_scale_dim],
            'z_where_bias_scale_mean':
                log_prop_list[j]['z_where_bias_mean'][:, :, :z_where_scale_dim],
            'z_where_bias_shift_std':
                log_prop_list[j]['z_where_bias_std'][:, :, z_where_scale_dim:],
            'z_where_bias_shift_mean':
                log_prop_list[j]['z_where_bias_mean'][:, :, z_where_scale_dim:],
            'z_pres_probs': torch.sigmoid(log_prop_list[j]['z_pres_logits']),
            'glimpse': log_prop_list[j]['glimpse'],
            'glimpse_recon': log_prop_list[j]['glimpse_recon'],
            'prior_z_pres_prob': log_prop_list[j]['prior_z_pres_prob'],
            'prior_where_bias_scale_std':
                log_prop_list[j]['prior_where_bias_std'][:, :, :z_where_scale_dim],
            'prior_where_bias_scale_mean':
                log_prop_list[j]['prior_where_bias_mean'][:, :, :z_where_scale_dim],
            'prior_where_bias_shift_std':
                log_prop_list[j]['prior_where_bias_std'][:, :, z_where_scale_dim:],
            'prior_where_bias_shift_mean':
                log_prop_list[j]['prior_where_bias_mean'][:, :, z_where_scale_dim:],

            'lengths': log_prop_list[j]['lengths'],
            'z_depth': log_prop_list[j]['z_depth'],
            'z_depth_std': log_prop_list[j]['z_depth_std'],
            'z_depth_mean': log_prop_list[j]['z_depth_mean'],

            'y_each_obj': log_prop_list[j]['y_each_obj'],
            'alpha_hat_each_obj': log_prop_list[j]['alpha_map'],

            'z_pres_logits': log_prop_list[j]['z_pres_logits'],
            'z_pres_y': log_prop_list[j]['z_pres_y'],
            'o_each_obj':
                spatial_transform(log_prop_list[j]['o_att'].view(-1, 3, glimpse_size, glimpse_size),
                                  log_prop_list[j]['z_where'].view(-1, (z_where_scale_dim +
                                                                        z_where_shift_dim)),
                                  (log_prop_list[j]['o_att'].size(1) * bs, 3, img_h, img_w),
                                  inverse=True).view(bs, -1, 3, img_h, img_w),
            'z_where_bias_scale':
                log_prop_list[j]['z_where_bias'].view(bs, -1, z_where_scale_dim + z_where_shift_dim)
                [:, :, :z_where_scale_dim],
            'z_where_bias_shift':
                log_prop_list[j]['z_where_bias'].view(bs, -1, z_where_scale_dim + z_where_shift_dim)
                [:, :, z_where_scale_dim:],
        }
    return log_prop


def save(imgs, dest):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(60,60))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.savefig(dest)

    
import pathlib

model_name_prefix = "model_perceptual_gan_v2_7x7"
for i in range(5):
    
    file_name = f"{model_name_prefix}_ex_{i+1}"
    directory = f"visuals/{file_name}"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    with open(f'example_viz/{file_name}.pickle', 'rb') as handle:
        preds = pickle.load(handle)

    imgs, y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
    kl_z_pres, kl_z_bg, log_imp, counting, \
    log_disc_list, log_prop_list, scalor_log_list = preds
    j = 0
    cs = 7
    bs = 1
    log_disc = get_log_disc_dict(log_disc_list, j = 0, cs = cs, prefix="train", bs = 1)

    bbox = visualize(imgs[0, j].cpu().unsqueeze(0),
                log_disc['z_pres'][0].unsqueeze(0).cpu().detach(),
                log_disc['z_where_scale'][0].unsqueeze(0).cpu().detach(),
                log_disc['z_where_shift'][0].unsqueeze(0).cpu().detach())

    grid = make_grid(bbox, 8, normalize=True, pad_value=1)
    save(grid, dest=os.path.join(directory, "bbox.png"))
    
    grid = make_grid(imgs[0, 0, :, :, :].cpu(), 1)
    save(grid, dest=os.path.join(directory, "original_img.png"))
    
    grid = make_grid(y_seq[0, 0, :, :, :].cpu(), 1)
    save(grid, dest=os.path.join(directory, "recons_img.png"))
    
    grid = make_grid(log_disc["alpha_each_cell"][0].cpu(), cs, normalize=True, pad_value=1)
    save(grid, dest=os.path.join(directory, "alpha_map.png"))
    
    grid = make_grid(log_disc["glimpse"][0].cpu(), cs, normalize=True, pad_value=1)
    save(grid, dest=os.path.join(directory, "glimpses.png"))