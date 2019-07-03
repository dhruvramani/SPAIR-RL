import argparse
import sys
import os
import time
import torch
import math
import numpy as np

from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn
from torchvision import datasets
from torch.optim.lr_scheduler import LambdaLR
from data_atari import ATARI
from utils import save_ckpt, load_ckpt, linear_annealing, visualize, bbox_in_one,\
    calc_count_acc, calc_count_more_num, print_spair_clevr, spatial_transform
from common import *
# from eval import evaluation

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from spair import Spair


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SPAIR')
    parser.add_argument('--data-dir', default='./dataset/pong/', metavar='DIR',
                        help='train.pt file')
    parser.add_argument('--nocuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('--epochs', default=1600, type=int, metavar='N',
                        help='number of total epochs to run (default: 1600)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--cp', '--clip-gradient', default=1.0, type=float,
                        metavar='CP', help='rate of gradient clipping')
    parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print batch frequency (default: 100)')
    parser.add_argument('--save-epoch-freq', '-s', default=400, type=int,
                        metavar='N', help='save epoch frequency (default: 400)')
    parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                        help='decay rate of learning rate (default: 0.8)')
    parser.add_argument('--lr-epoch-per-decay', default=1000, type=int,
                        help='epoch of per decay of learning rate (default: 1000)')
    parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                        help='path to save checkpoints')
    parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                        help='path to save summary')
    parser.add_argument('--tau-start', default=5, type=float, metavar='T',
                        help='initial temperature for gumbel')
    parser.add_argument('--tau-end', default=0.5, type=float, metavar='T',
                        help='final temperature for gumbel')
    parser.add_argument('--tau-ep', default=200, type=float, metavar='E',
                        help='exponential decay factor for tau')
    # parser.add_argument('--epochs-per-eval', default=100, type=int,
    #                     metavar='N', help='Interval of epochs for a testing data evaluation.')
    parser.add_argument('--seed', default=1, type=int,
                        help='Fixed random seed.')
    parser.add_argument('--sigma', default=0.01, type=float, metavar='S',
                        help='Sigma for log likelihood.')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    #torch.cuda.set_device(1)
    device = torch.device(
        "cuda:1" if not args.nocuda and torch.cuda.is_available() else "cpu")
    #if not args.nocuda:
    #    torch.cuda.empty_cache()
    # torch.manual_seed(args.seed)

    train_data = ATARI(root=args.data_dir, phase_train=True)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)

    num_train = len(train_data)

    model = Spair(sigma=args.sigma)
    model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[1,0,2,3])
    model.train()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.last_ckpt:
        global_step, args.start_epoch = \
            load_ckpt(model, optimizer, args.last_ckpt, device)

    writer = SummaryWriter(args.summary_dir)

    global_step = 0

    #log_tau_gamma = math.log(args.tau_end) / args.tau_ep
    delta_tau = (args.tau_end - args.tau_start) / args.tau_ep

    for epoch in range(int(args.start_epoch), args.epochs + 1):
        local_count = 0
        last_count = 0
        end_time = time.time()
        tau = max(args.tau_start + epoch * delta_tau, args.tau_end)
        if (not args.last_ckpt and epoch != 0) or (args.last_ckpt and epoch != args.start_epoch):
            # if args.epochs_per_eval:
            #     if epoch % args.epochs_per_eval == 0:
            #         model.eval()
            #         evaluation(model, test_loader, epoch, args.workers,
            #                    args.batch_size, device, writer)
            #         model.train()
            if epoch % args.save_epoch_freq == 0:
                save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                          local_count, args.batch_size, num_train)
        for batch_idx, sample in enumerate(train_loader):

            imgs = sample[0].to(device)
            target_count = sample[1]

            recon_x, log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth, kl_bg_what, log = \
                model(imgs, global_step, tau)

            log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth, kl_bg_what = \
                log_like.mean(), kl_z_what.mean(), kl_z_where.mean(), \
                kl_z_pres.mean(), kl_z_depth.mean(), kl_bg_what.mean()

            #if mode_img is not None:
            #    kl_bg_what = torch.tensor(0)

            #total_loss = - (log_like - kl_z_what - kl_z_where - kl_z_pres - kl_z_depth - kl_bg_what)
            log_like_alpha_map = log['log_like_alpha'].mean()
            total_loss = - (log_like - kl_z_what - kl_z_where - kl_z_pres - kl_z_depth - kl_bg_what + log_like_alpha_map)

            optimizer.zero_grad()
            total_loss.backward()

            if DEBUG:
                for name, param in model.named_parameters():
                    if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                        breakpoint()

            clip_grad_norm_(model.parameters(), args.cp)
            optimizer.step()

            local_count += imgs.data.shape[0]
            global_step += 1
            if global_step % args.print_freq == 0 or global_step == 1:

                bs = imgs.size(0)

                log = {
                    'bg': log['bg'].view(-1, 3, img_h, img_w),
                    'z_where_scale':
                        log['z_where'].view(-1, 8 * 8, z_where_scale_dim + z_where_shift_dim)[:, :, :z_where_scale_dim],
                    'z_where_shift':
                        log['z_where'].view(-1, 8 * 8, z_where_scale_dim + z_where_shift_dim)[:, :, z_where_scale_dim:],
                    'z_pres': log['z_pres'].permute(0, 2, 3, 1),
                    'o_each_cell': spatial_transform(log['o_att'], log['z_where'], (8 * 8 * bs, 3, img_h, img_w),
                                                     inverse=True).view(-1, 8 * 8, 3, img_h, img_w),
                    'alpha_hat_each_cell': spatial_transform(log['alpha_att_hat'], log['z_where'],
                                                             (8 * 8 * bs, 1, img_h, img_w),
                                                             inverse=True).view(-1, 8 * 8, 1, img_h, img_w),
                    'alpha_each_cell': spatial_transform(log['alpha_att'], log['z_where'],
                                                         (8 * 8 * bs, 1, img_h, img_w),
                                                         inverse=True).view(-1, 8 * 8, 1, img_h, img_w),
                    'y_each_cell': (log['y_each_cell'] * log['z_pres'].
                                    view(-1, 1, 1, 1)).view(-1, 8 * 8, 3, img_h, img_w),
                    'importance_map_full_res_norm':
                        log['importance_map_full_res_norm'].view(-1, 8 * 8, 1, img_h, img_w),
                    'fg': log['fg'].view(-1, 3, img_h, img_w),
                }

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_spair_clevr(global_step, epoch, local_count, count_inter,
                                  num_train, total_loss, log_like, kl_z_what,
                                  kl_z_where, kl_z_pres, kl_z_depth, kl_bg_what, time_inter)
                end_time = time.time()

                for name, param in model.named_parameters():
                    writer.add_histogram(
                        name, param.cpu().detach().numpy(), global_step)
                    if param.grad is not None:
                        writer.add_histogram(
                            'grad/' + name, param.grad.cpu().detach(), global_step)
                        # writer.add_scalar(
                        #     'grad_std/' + name + '.grad', param.grad.cpu().detach().std().item(), global_step)
                        # writer.add_scalar(
                        #     'grad_mean/' + name + '.grad', param.grad.cpu().detach().mean().item(), global_step)

                for key, value in log.items():
                    if value is None:
                        continue

                    #if key == 'importance_map_full_res_norm' or key == 'alpha_hat_each_cell' or key == 'alpha_each_cell':
                    #    writer.add_histogram('inside_value/' + key, value[value > 0].cpu().detach().numpy(),
                    #                         global_step)
                    #else:
                    #    writer.add_histogram('inside_value/' + key, value.cpu().detach().numpy(),
                    #                         global_step)

                bg = log['bg'].cpu().detach()[:10].view(-1, 1, 3, img_h, img_w)
                fg = log['fg'].cpu().detach()[:10]
                fg = bbox_in_one(fg, log['z_pres'][:10].cpu().detach(),
                                 log['z_where_scale'][:10].cpu().detach(),
                                 log['z_where_shift'][:10].cpu().detach(), num_obj=8*8)\
                        .view(-1, 1, 3, img_h, img_w)
                imgs_10 = imgs.cpu().detach()[:10].view(-1, 1, 3, img_h, img_w)
                separates = torch.cat((imgs_10, fg, bg), 1).view(-1, 3, img_h, img_w)
                grid_image = make_grid(separates, 3, normalize=False, pad_value=1)
                writer.add_image('train/#0-separations', grid_image, global_step)

                grid_image = make_grid(imgs.cpu().detach()[:10].view(-1, 3, img_h, img_w),
                                       5, normalize=False, pad_value=1)
                writer.add_image('train/1-image', grid_image, global_step)

                grid_image = make_grid(recon_x.cpu().detach()[:10].view(-1, 3, img_h, img_w).clamp(0., 1.),
                                       5, normalize=False, pad_value=1)
                writer.add_image('train/2-reconstruction_overall', grid_image, global_step)

                grid_image = make_grid(log['bg'].cpu().detach()[:10].view(-1, 3, img_h, img_w),
                                       5, normalize=False, pad_value=1)
                writer.add_image('train/3-background', grid_image, global_step)

                bbox = visualize(imgs[:num_img_summary].cpu(), log['z_pres'][:num_img_summary].cpu().detach(),
                                 log['z_where_scale'][:num_img_summary].cpu().detach(),
                                 log['z_where_shift'][:num_img_summary].cpu().detach(), num_obj=8*8)

                y_each_cell = log['y_each_cell'].view(-1, 3, img_h, img_w)[:num_img_summary * (8*8)].cpu().detach()
                o_each_cell = log['o_each_cell'].view(-1, 3, img_h, img_w)[:num_img_summary * (8*8)].cpu().detach()
                alpha_each_cell = log['alpha_hat_each_cell'].view(-1, 1, img_h, img_w)[
                                  :num_img_summary * (8*8)].cpu().detach()
                importance_each_cell = \
                    log['importance_map_full_res_norm'].view(-1, 1, img_h, img_w)[:num_img_summary * (8*8)].cpu().detach()

                for i in range(num_img_summary):
                    grid_image = make_grid(bbox[i * (8*8):(i + 1) * (8*8)], 8, normalize=True, pad_value=1)
                    writer.add_image('train/8-bbox_{}'.format(i), grid_image, global_step)

                    grid_image = make_grid(y_each_cell[i * (8*8):(i + 1) * (8*8)], 8, normalize=True, pad_value=1)
                    writer.add_image('train/5-y_each_cell_{}'.format(i), grid_image, global_step)

                    grid_image = make_grid(o_each_cell[i * (8*8):(i + 1) * (8*8)], 8, normalize=True, pad_value=1)
                    writer.add_image('train/6-o_each_cell_{}'.format(i), grid_image, global_step)

                    grid_image = make_grid(alpha_each_cell[i * (8*8):(i + 1) * (8*8)], 8, normalize=True, pad_value=1)
                    writer.add_image('train/7-alpha_hat_each_cell_{}'.format(i), grid_image, global_step)

                    grid_image = make_grid(importance_each_cell[i * (8*8):(i + 1) * (8*8)], 8, normalize=True, pad_value=1)
                    writer.add_image('train/8-importance_each_cell_{}'.format(i), grid_image, global_step)


                writer.add_scalar('train/total_loss', total_loss.item(), global_step=global_step)
                writer.add_scalar('train/log_like', log_like.item(), global_step=global_step)
                writer.add_scalar('train/log_like_alpha', log_like_alpha_map.item(), global_step=global_step)
                writer.add_scalar('train/What_KL', kl_z_what.item(), global_step=global_step)
                writer.add_scalar('train/Where_KL', kl_z_where.item(), global_step=global_step)
                writer.add_scalar('train/Pres_KL', kl_z_pres.item(), global_step=global_step)
                writer.add_scalar('train/Depth_KL', kl_z_depth.item(), global_step=global_step)
                writer.add_scalar('train/tau', tau, global_step=global_step)
                writer.add_scalar('train/count_acc', calc_count_acc(log['z_pres'].cpu().detach(), target_count),
                                  global_step=global_step)
                writer.add_scalar('train/count_more', calc_count_more_num(log['z_pres'].cpu().detach(), target_count),
                                  global_step=global_step)
                writer.add_scalar('train/Bg_KL', kl_bg_what.item(), global_step=global_step)
                # writer.add_scalar('train/Bg_Beta', kg_kl_beta.item(), global_step=global_step)

                last_count = local_count


if __name__ == '__main__':
    main()


