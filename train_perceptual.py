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
from torch.distributions import Normal, kl_divergence

from tensorboardX import SummaryWriter

from scalor import SCALOR

import numpy as np
from torchvision import transforms
import torchvision.models as models
from LossNetwork import LossNetwork

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


def recover_image(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)


def main(args):
    print(torch.cuda.is_available(), flush=True)
    args.color_t = torch.rand(700, 3)

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    device = torch.device(
       "cuda" if not args.nocuda and torch.cuda.is_available() else "cpu")

    # train_data = TrainStation(args=args, train=True)
    # data_dir = "/storage/user/brasoand/motsyn2/frames/"
    # data_dir = "images_blurred_birdview_rbg"
    data_dir = "images_heavily_blurred_bw"
    # data_dir = "/storage/user/brasoand/motsyn2/frames/019/rgb/"
    train_data = MOTSynthBlackBG(data_dir, train=True)
    # train_data = MOTSynth(data_dir, train=True)
    # train_data = MOTSynthSingleVideo(data_dir, train=True)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    num_train = len(train_data)

    print(f"Before Scalor", flush=True)
    model = SCALOR(args)
    print("Scalor initialized", flush=True)
    model.to(device)
    print(f"Scalor to {device}", flush=True)
    model.train()
    print(f"After models are initialized", flush=True)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    global_step = 0
    print(f"Last checkpoint: {args.last_ckpt}")
    if args.last_ckpt:
        global_step, args.start_epoch = \
            load_ckpt(model, optimizer, args.last_ckpt, device)

    writer = SummaryWriter(args.summary_dir)

    args.global_step = global_step

    log_tau_gamma = np.log(args.tau_end) / args.tau_ep
#     print("Right before epoch", flush=True)
    mse_loss = torch.nn.MSELoss()
    print("before vgg model", flush=True)
    vgg_model = models.vgg16(pretrained=True)
    vgg_model.to(device)
    print(f"vgg added to {device}", flush=True)
    loss_network = LossNetwork(vgg_model).to(device)
    loss_network.eval()
    
    print(f"Before for loop", flush=True)
    for epoch in range(int(args.start_epoch), args.epochs):
        local_count = 0
        last_count = 0
        end_time = time.time()
        
#         print("Right before new batch is generated.", flush=True)
        generator = iter(train_loader)
#         for batch_idx, (sample, counting_gt) in enumerate(train_loader):
        for batch_idx in range(int(num_train/args.batch_size)):
            
            try:
                sample, counting_gt = next(generator)
                # print(f"Sample initialized.", flush=True)
                tau = np.exp(global_step * log_tau_gamma)
                tau = max(tau, args.tau_end)
                args.tau = tau

                global_step += 1

                log_phase = global_step % args.print_freq == 0 or global_step == 1
                args.global_step = global_step
                args.log_phase = log_phase
                # print(f"before imgs to device: {device}", flush=True)
                imgs = sample.to(device)
                # print(f"imgs to device: {device}", flush=True)
    #             print(f"imgs shape: {imgs.shape}", flush=True)
                y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                kl_z_pres, kl_z_bg, log_imp, counting, \
                log_disc_list, log_prop_list, scalor_log_list = model(imgs)
                # print(f"after model(imgs)", flush=True)

                log_like = log_like.mean(dim=0)
                kl_z_what = kl_z_what.mean(dim=0)
                kl_z_where = kl_z_where.mean(dim=0)
                kl_z_depth = kl_z_depth.mean(dim=0)
                kl_z_pres = kl_z_pres.mean(dim=0)
                kl_z_bg = kl_z_bg.mean(0)
                
                content_loss = 0
                reg_loss = 0
                # print("Before perceptual")
                perc_log_like_list = []
                for b_id in range(imgs.shape[0]):
#                     x_org = imgs[b_id].detach()
                    y_frame = y_seq[b_id]
                    with torch.no_grad():
                        xc = imgs[b_id].detach()


                    features_y = loss_network(y_frame)
                    features_xc = loss_network(xc)
                    
                    
                    for fxc_idx, feature_rep in enumerate(features_xc):
                        # print(f"fxc_idx: {fxc_idx}", flush=True)
                        with torch.no_grad():
                            f_xc_c = features_xc[fxc_idx].detach()

                        p_perc = Normal(features_y[fxc_idx].flatten(1), 0.1)
                        perc_log_like = p_perc.log_prob(f_xc_c.expand_as(features_y[fxc_idx]).flatten(1)).sum()  # sum image dims (C, H, W)
                        # print(f"perc_log_like: {perc_log_like}", flush=True)
                        perc_log_like_list.append(perc_log_like.reshape(1))
                    
                    # print(f"perc_log_like_list: {perc_log_like_list}", flush=True)
                        # content_loss += mse_loss(features_y[fxc_idx], f_xc_c)
                
                content_loss = -torch.cat(perc_log_like_list).mean()
                content_loss *= 0.01
                # content_loss *= 0.01 # model_perceptual_gan_v2
                    # reg_loss += 0.05 * (
                    #             torch.sum(torch.abs(y_frame[:, :, :, :-1] - y_frame[:, :, :, 1:])) + 
                    #             torch.sum(torch.abs(y_frame[:, :, :-1, :] - y_frame[:, :, 1:, :])))
                

                total_loss = - (log_like - kl_z_what - kl_z_where - kl_z_depth - kl_z_pres - kl_z_bg)
                total_loss = total_loss +  content_loss
                optimizer.zero_grad()
                total_loss.backward()

                clip_grad_norm_(model.parameters(), args.cp)
                optimizer.step()

                local_count += imgs.data.shape[0]

                if log_phase:

                    time_inter = time.time() - end_time
                    end_time = time.time()

                    count_inter = local_count - last_count

                    print_scalor(global_step, epoch, local_count, count_inter,
                                   num_train, total_loss, log_like, kl_z_what, kl_z_where,
                                   kl_z_pres, kl_z_depth, time_inter)

                    print(f"Content Loss: {content_loss}  | Reg Loss: {reg_loss}")
                    print(f"----------------------------------------------------/n")

                    writer.add_scalar('train/total_loss', total_loss.item(), global_step=global_step)
                    writer.add_scalar('train/log_like', log_like.item(), global_step=global_step)
                    writer.add_scalar('train/What_KL', kl_z_what.item(), global_step=global_step)
                    writer.add_scalar('train/Where_KL', kl_z_where.item(), global_step=global_step)
                    writer.add_scalar('train/Pres_KL', kl_z_pres.item(), global_step=global_step)
                    writer.add_scalar('train/Depth_KL', kl_z_depth.item(), global_step=global_step)
                    writer.add_scalar('train/Bg_KL', kl_z_bg.item(), global_step=global_step)
                    # writer.add_scalar('train/Bg_alpha_KL', kl_z_bg_mask.item(), global_step=global_step)
                    writer.add_scalar('train/tau', tau, global_step=global_step)

                    log_summary(args, writer, imgs, y_seq, global_step, log_disc_list,
                                log_prop_list, scalor_log_list, prefix='train')

                    last_count = local_count

                if global_step % args.generate_freq == 0:
                    ####################################### do generation ####################################
                    model.eval()
                    with torch.no_grad():
                        args.phase_generate = True
                        y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                        kl_z_pres, kl_z_bg, log_imp, counting, \
                        log_disc_list, log_prop_list, scalor_log_list = model(imgs)
                        args.phase_generate = False
                        log_summary(args, writer, imgs, y_seq, global_step, log_disc_list,
                                    log_prop_list, scalor_log_list, prefix='generate')
                    model.train()
                    ####################################### end generation ####################################

                if global_step % args.save_epoch_freq == 0 or global_step == 1:
                    save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                              local_count, args.batch_size, num_train)
                
            except Exception as e:
                print("----------------ERROR OCCURED------------")
                raise e
                print(e, flush=True)
                generator = iter(train_loader)
                print("-----------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCALOR')
    args = parse.parse(parser)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)

