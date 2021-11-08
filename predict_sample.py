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
from skimage import img_as_bool
from skimage.transform import resize
from scalor import SCALOR


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

def calculate_IoU(pred, targed):
    '''
    Calculates the Intersection over Union(Intersection over Union).
    '''
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def predict(log_disc_list, log_prop_list, video_id, indice):
    
    prediction ={
                "video_id" : video_id, 
                "category_id" : 1, 
                "segmentations" : [None for i in range(360)], 
                "score" : [], 
            }
    import copy

    obj_idx_dict = {}
    start_idx = indice * 10
    cs = 6
    for j in range(10):
        
        log_disc = get_log_disc_dict(log_disc_list=log_disc_list, j = j, cs = cs, prefix="train", bs = 1)
        discovery_ids= log_disc_list[j]["ids"][0].view(-1)
        discovery_pres_scores = log_disc_list[j]["z_pres"][0].view(-1)
        discovered_object_indices = np.where(discovery_pres_scores > 0.7)[0]
        discovered_object_masks = log_disc['alpha_hat_each_cell'][0].squeeze(1)[discovered_object_indices]
        
        for k, discovered_obj_indice in enumerate(discovered_object_indices):
            discovered_obj_id = int(discovery_ids[discovered_obj_indice].item())
            discovery_z_pres = discovery_pres_scores[discovered_obj_indice].item()
            
            if discovered_obj_id in obj_idx_dict:
                
                mask = copy.deepcopy(discovered_object_masks[k].numpy())
#                 mask[mask > 0.005] = 1
#                 mask = st.resize(mask, (1080, 1080))
#                 mask[mask>0.00005] = 1
                mask = mask >= 0.03
                mask = img_as_bool(resize(mask, (1080, 1080)))
                seg = coco_mask.encode(np.asfortranarray(np.uint8(mask)))
                seg["counts"] = seg["counts"].decode("utf-8")
                
                obj_idx_dict[discovered_obj_id]["segmentations"][start_idx + j] = copy.deepcopy(seg)
                
                obj_idx_dict[discovered_obj_id]["score"].append(discovery_z_pres)
            else:
                obj_idx_dict[discovered_obj_id] = copy.deepcopy(prediction)
                mask = copy.deepcopy(discovered_object_masks[k].numpy())
#                 mask[mask > 0.005] = 1
#                 mask = st.resize(mask, (1080, 1080))
#                 mask[mask>0.00005] = 1
                mask = mask >= 0.03
                mask = img_as_bool(resize(mask, (1080, 1080)))
                seg = coco_mask.encode(np.asfortranarray(np.uint8(mask)))
                seg["counts"] = seg["counts"].decode("utf-8")
                
                obj_idx_dict[discovered_obj_id]["segmentations"][start_idx + j] = copy.deepcopy(seg)
                obj_idx_dict[discovered_obj_id]["score"].append(discovery_z_pres)
                
        if log_prop_list[j]:
            log_prop = get_log_prop_dict(log_prop_list=log_prop_list, j = j, cs = cs, prefix="train", bs = 1)
            propagated_ids = log_prop_list[j]["ids"][0].view(-1)
            propagation_pres_scores = log_prop_list[j]["z_pres"][0].view(-1)
            propagated_object_indices = np.where(propagation_pres_scores > 0.7)[0]
            # print(propagated_ids[propagated_object_indices])
            propagated_object_masks = log_prop['alpha_hat_each_obj'][0].squeeze(1)[propagated_object_indices]
            
            for k, propagated_obj_indice in enumerate(propagated_object_indices):
                propagated_obj_id = int(propagated_ids[propagated_obj_indice].item())
                propagated_z_pres = propagation_pres_scores[propagated_obj_indice].item()
                if not propagated_obj_id == 0:
                    if discovered_obj_id in obj_idx_dict:
                        
                        mask = copy.deepcopy(propagated_object_masks[k].numpy())
#                         mask[mask > 0.005] = 1
#                         mask = st.resize(mask, (1080, 1080))
#                         mask[mask>0.00005] = 1
                        mask = mask >= 0.03
                        mask = img_as_bool(resize(mask, (1080, 1080)))
                        seg = coco_mask.encode(np.asfortranarray(np.uint8(mask)))
                        seg["counts"] = seg["counts"].decode("utf-8")

                        obj_idx_dict[propagated_obj_id]["segmentations"][start_idx + j] = copy.deepcopy(seg)
                        obj_idx_dict[propagated_obj_id]["score"].append(propagated_z_pres)
                    else:
                        raise Exception("ahahahah")
                        obj_idx_dict[propagated_obj_id] = copy.deepcopy(prediction)
                        obj_idx_dict[propagated_obj_id]["segmentations"][start_idx + j] = propagated_object_masks[k].numpy()
                        obj_idx_dict[propagated_obj_id]["score"].append(propagated_z_pres)
    
    

    return obj_idx_dict





parser = argparse.ArgumentParser(description='SCALOR')
# args = parser.parse_args()
parser.add_argument('-f')# # common.cfg overrides
args = parse.parse(parser)
args.batch_size = 1

device = torch.device("cuda" if not args.nocuda and torch.cuda.is_available() else "cpu")

# data_dir = "images_bw_5"
data_dir = "images_heavily_blurred_bw"
train_data = MOTSynthBlackBG(data_dir, train=False)

train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

num_train = len(train_data)

model = SCALOR(args)
model = model.to(device)
model.eval()

# args.last_ckpt = './model_gradient_2/ckpt_epoch_11200.pth'
args.last_ckpt = './model_perceptual_gan_v2_6x6/ckpt_epoch_8000.pth'

optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
# global_step = 0

print(f"Last checkpoint: {args.last_ckpt}")
if args.last_ckpt:
    global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)
    
    
args.global_step = global_step

log_tau_gamma = np.log(args.tau_end) / args.tau_ep
annotation_indices = [28, 33, 42, 59]
video_indice = -1
seq_id = 0
predictions_list = []
for i in range(len(annotation_indices) * 36):
    sample, counting_gt = train_loader.dataset.__getitem__(i+36)
    sample = sample.unsqueeze(0)
    if i % 36 == 0:
        video_indice += 1
        video_id = annotation_indices[video_indice]
        seq_id = 0


    model.eval()

    tau = np.exp(global_step * log_tau_gamma)
    tau = max(tau, args.tau_end)
    args.tau = tau

    global_step += 1

    log_phase = True
    args.global_step = global_step
    args.log_phase = log_phase

    imgs = sample.to(device)

    print(f"imgs shape: {imgs.shape}", flush=True)


    preds = model(imgs)

    y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
    kl_z_pres, kl_z_bg, log_imp, counting, \
    log_disc_list, log_prop_list, scalor_log_list = preds

    id_dict = predict(log_disc_list, log_prop_list, video_id, seq_id)

    preds = list(id_dict.values())
    for pr_id, pr in enumerate(preds):
        preds[pr_id]["score"] = sum(preds[pr_id]["score"]) / len(preds[pr_id]["score"]) 

    predictions_list.extend(list(id_dict.values()))
    seq_id += 1

with open('predictions_model_perceptual_gan_6x6_003.json', 'w') as handle:
    json.dump(predictions_list, handle)
