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
import pickle

from tensorboardX import SummaryWriter

from scalor import SCALOR

parser = argparse.ArgumentParser(description='SCALOR')
# args = parser.parse_args()
parser.add_argument('-f')# # common.cfg overrides
args = parse.parse(parser)
args.batch_size = 1

device = torch.device("cuda" if not args.nocuda and torch.cuda.is_available() else "cpu")





# data_dir = "images_heavily_blurred_bw"
# train_data = MOTSynthBlackBG(data_dir, train=True)


data_dir = "/storage/user/brasoand/motsyn2/frames/"

# train_data = MOTSynthBlackBG(data_dir, train=True)

train_data = MOTSynth(data_dir, train=True)



train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

num_train = len(train_data)

model = SCALOR(args)
model = model.to(device)
model.eval()

args.last_ckpt = './model_perceptual_gan_real_images/ckpt_epoch_8800.pth'

optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
# global_step = 0

print(f"Last checkpoint: {args.last_ckpt}")
if args.last_ckpt:
    global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)
    
    
args.global_step = global_step

log_tau_gamma = np.log(args.tau_end) / args.tau_ep
generator = iter(train_loader)

model.eval()
sample, counting_gt = next(generator)
sample, counting_gt = next(generator)
sample, counting_gt = next(generator)
sample, counting_gt = next(generator)
sample, counting_gt = next(generator)
sample, counting_gt = next(generator)


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

print(log_disc_list)
print("-----------------------------")
print(log_prop_list)
print("-----------------------------")
print(scalor_log_list)
print("-----------------------------")



with open('example_viz/model_perceptual_gan_real_images_ex_5.pickle', 'wb') as handle:
    pickle.dump((imgs,)+preds, handle)
