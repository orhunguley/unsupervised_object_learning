import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
from data import TrainStation
from motsynth import MOTSynth, MOTSynthBlackBG, MOTSynthV2
from log_utils import log_summary
from utils import save_ckpt, load_ckpt, print_scalor
from common import *
from gan import Discriminator
import parse
import itertools
from torch.autograd import Variable
import random
from tensorboardX import SummaryWriter

from scalor import SCALOR, SCALORV2
from torchvision import transforms
import torchvision.models as models
from LossNetwork import LossNetwork
from torch.distributions import Normal, kl_divergence

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
#     data_dir = "/storage/user/brasoand/motsyn2/frames/"
    # data_dir = "static_bg_birdview_images"
    # data_dir = "rand_static_bg_frames_td_gradient"

    # data_dir = "images_heavily_blurred_bw"
    data_dir = "images_blurred_birdview_rbg"

    # data_dir = "/storage/user/brasoand/motsyn2/frames/"

    # data_dir = "/storage/user/brasoand/motsyn2/frames/019/rgb/"
    train_data = MOTSynthBlackBG(data_dir, train=True)

    # train_data = MOTSynth(data_dir, train=True)
    # train_data = MOTSynthV2(data_dir, train=True)

    # train_data = MOTSynthSingleVideo(data_dir, train=True)

    # train_loader = DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    # disc_loader = DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    print(f"Batch Size {args.batch_size}", flush=True)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    disc_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    l1_loss = torch.nn.L1Loss(reduction='sum')

    num_train = len(train_data)
    
    
    model = SCALOR(args)
    model.to(device)
    model.train()

    discriminator = Discriminator()
    discriminator.to(device)
    discriminator.train()

    optimizer_scalor = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

    # optimizer_generator = torch.optim.RMSprop(
    # itertools.chain(model.parameters(), discriminator.parameters()), lr=args.lr)

    # optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

    adversarial_loss = torch.nn.BCELoss()


    print("before vgg model", flush=True)
    vgg_model = models.vgg16(pretrained=True)
    vgg_model.to(device)
    print(f"vgg added to {device}", flush=True)
    loss_network = LossNetwork(vgg_model).to(device)
    loss_network.eval()



    global_step = 0
    print(f"Last checkpoint: {args.last_ckpt}")
    if args.last_ckpt:
        global_step, args.start_epoch = \
            load_ckpt(model, optimizer_scalor, args.last_ckpt, device)

    print(f"Model Loaded")
    writer = SummaryWriter(args.summary_dir)

    args.global_step = global_step

    log_tau_gamma = np.log(args.tau_end) / args.tau_ep
#     print("Right before epoch", flush=True)
    print("before epoch")
    for epoch in range(int(args.start_epoch), args.epochs):
        local_count = 0
        last_count = 0
        end_time = time.time()

#         print("Right before new batch is generated.", flush=True)
        generator = iter(train_loader)
        disc_generator = iter(disc_loader)
#         random_video_list = random.sample(top_down_list, len(top_down_list))
#         for batch_idx, (sample, counting_gt) in enumerate(train_loader):
        for batch_idx in range(int(num_train/args.batch_size)):
            
            try:
                # #This is for single video batch training
                # video_idx = batch_idx % len(random_video_list)
                # train_data.current_video = random_video_list[video_idx]

                sample, video_emb = next(generator)

                tau = np.exp(global_step * log_tau_gamma)
                tau = max(tau, args.tau_end)
                args.tau = tau

                global_step += 1

                log_phase = global_step % args.print_freq == 0 or global_step == 1
                gan_phase = global_step % 10 == 0 or global_step == 1

                args.global_step = global_step
                args.log_phase = log_phase

                imgs = sample.to(device)


                ############## TRAIN SCALOR VAE ##############

    #             print(f"imgs shape: {imgs.shape}", flush=True)
                # y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                # kl_z_pres, kl_z_bg, log_imp, counting, \
                # log_disc_list, log_prop_list, scalor_log_list, \
                # alpha_map_disc_list, alpha_map_prop_list = model(imgs)
                
                # y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                # kl_z_pres, kl_z_bg, log_imp, counting, \
                # log_disc_list, log_prop_list, scalor_log_list = model(imgs, video_embs=video_emb)

                y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                kl_z_pres, kl_z_bg, log_imp, counting, \
                log_disc_list, log_prop_list, scalor_log_list = model(imgs)

                # print(f"alpha_map_disc_list len: {len(alpha_map_disc_list)}")
                # print(f"alpha_map_prop_list len: {len(alpha_map_prop_list)}")

                # alpha_masks_all = torch.cat((alpha_map_disc_list + alpha_map_prop_list), 0)
                # alpha_masks_gt = (alpha_masks_all > 0.01).float()
                # mask_loss = l1_loss(alpha_masks_all, alpha_masks_gt) * 5
                # print(f"mask_loss: {mask_loss}")

                # print(f"alpha_map_disc shape: {alpha_map_disc_list[0].shape}")
                # print(f"alpha_map_prop shape: {alpha_map_prop_list[0].shape}")
                log_like = log_like.mean(dim=0)
                kl_z_what = kl_z_what.mean(dim=0)
                kl_z_where = kl_z_where.mean(dim=0)
                kl_z_depth = kl_z_depth.mean(dim=0)
                kl_z_pres = kl_z_pres.mean(dim=0)
                kl_z_bg = kl_z_bg.mean(0)


                # loss_l1 = l1_loss(y_seq, imgs)
                # total_loss = - (- kl_z_what - kl_z_where - kl_z_depth - kl_z_pres - kl_z_bg) + loss_l1

                total_loss = - (log_like - kl_z_what - kl_z_where - kl_z_depth - kl_z_pres - kl_z_bg)
                # total_loss += mask_loss

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
                        # perc_l1 = l1_loss(features_y[fxc_idx], f_xc_c)
                        # print(f"perc_log_like: {perc_log_like}", flush=True)
                        # perc_log_like_list.append(perc_l1.reshape(1))
                        perc_log_like_list.append(perc_log_like.reshape(1))

                    # print(f"perc_log_like_list: {perc_log_like_list}", flush=True)
                        # content_loss += mse_loss(features_y[fxc_idx], f_xc_c)
                

                # content_loss = -torch.cat(perc_log_like_list).mean()
                content_loss = -torch.cat(perc_log_like_list).mean()
                content_loss *= 0.01 # model_perceptual_gan_v2
                    # reg_loss += 0.05 * (
                    #             torch.sum(torch.abs(y_frame[:, :, :, :-1] - y_frame[:, :, :, 1:])) + 
                    #             torch.sum(torch.abs(y_frame[:, :, :-1, :] - y_frame[:, :, 1:, :])))
                


                # loss_l1 = l1_loss(y_seq, imgs)
                # total_loss = - (- kl_z_what - kl_z_where - kl_z_depth - kl_z_pres - kl_z_bg) + loss_l1
                
                total_loss = - (log_like - kl_z_what - kl_z_where - kl_z_depth - kl_z_pres - kl_z_bg)
                total_loss = total_loss +  content_loss
                # total_loss = total_loss +  content_loss + mask_loss


                # Adversarial ground truths
                valid = Variable(torch.cuda.FloatTensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.cuda.FloatTensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

                loss_adversarial = 0

                if gan_phase:
                    for seq_id in range(10):
                        
                        y_frame = y_seq[:, seq_id]
                        # real_y_frame = imgs[:, seq_id]
                        
                        pred_d = discriminator(y_frame)
                        loss_adversarial += adversarial_loss(pred_d, valid)

                total_loss += (1000 * loss_adversarial)
                # print(f"loss_adversarial: {loss_adversarial}", flush=True)

                optimizer_scalor.zero_grad()
                total_loss.backward()

                clip_grad_norm_(model.parameters(), args.cp)
                optimizer_scalor.step()


                # ############## TRAIN GENERATOR ##############

                # # Adversarial ground truths
                # valid = Variable(FloatTensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
                # fake = Variable(FloatTensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

                # loss_adversarial = 0
                # for seq_id in range(10):
                    
                #     y_frame = y_seq[:, seq_id]
                #     # real_y_frame = imgs[:, seq_id]

                #     pred_d = discriminator(y_frame)
                #     loss_adversarial += adversarial_loss(pred_d, valid)


                # optimizer_D.zero_grad()
                # real_pred = discriminator(real_imgs)



                ############## TRAIN DISCRIMINATOR ##############
                if gan_phase:
                    optimizer_discriminator.zero_grad()

                    real_sample, counting_gt = next(disc_generator)
                    real_imgs = sample.to(device)

                    disc_loss = 0
                    for seq_id in range(10):
                        
                        y_fake_frame = y_seq[:, seq_id].detach()
                        y_real_frame = real_imgs[:, seq_id]
                        # real_y_frame = imgs[:, seq_id]
                        
                        pred_d = discriminator(y_fake_frame)

                        fake_loss = adversarial_loss(discriminator(y_fake_frame), fake)
                        real_loss = adversarial_loss(discriminator(y_real_frame), valid)
                        disc_loss += 0.5 * (real_loss + fake_loss)


                    
                    disc_loss.backward()
                    optimizer_discriminator.step()


                local_count += imgs.data.shape[0]

                if log_phase:
                    print(f"log_like: {log_like} || generator training loss_adversarial: {loss_adversarial}  || discriminator_loss: {disc_loss} || content_loss: {content_loss}")
                    time_inter = time.time() - end_time
                    end_time = time.time()

                    count_inter = local_count - last_count

                    print_scalor(global_step, epoch, local_count, count_inter,
                                   num_train, total_loss, log_like, kl_z_what, kl_z_where,
                                   kl_z_pres, kl_z_depth, time_inter)

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

                        # y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                        # kl_z_pres, kl_z_bg, log_imp, counting, \
                        # log_disc_list, log_prop_list, scalor_log_list = model(imgs, video_embs=video_emb)

                        # y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                        # kl_z_pres, kl_z_bg, log_imp, counting, \
                        # log_disc_list, log_prop_list, scalor_log_list, _, _ = model(imgs)
                        args.phase_generate = False
                        log_summary(args, writer, imgs, y_seq, global_step, log_disc_list,
                                    log_prop_list, scalor_log_list, prefix='generate')
                    model.train()
                    ####################################### end generation ####################################

                if global_step % args.save_epoch_freq == 0 or global_step == 1:
                    save_ckpt(args.ckpt_dir, model, optimizer_scalor, global_step, epoch,
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

