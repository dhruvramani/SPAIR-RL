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
from utils import save_ckpt, load_ckpt, linear_annealing, visualize, \
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
    parser.add_argument('--epochs', default=800, type=int, metavar='N',
                        help='number of total epochs to run (default: 400)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--cp', '--clip-gradient', default=1.0, type=float,
                        metavar='CP', help='rate of gradient clipping')
    parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print batch frequency (default: 100)')
    parser.add_argument('--save-epoch-freq', '-s', default=160, type=int,
                        metavar='N', help='save epoch frequency (default: 160)')
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
    parser.add_argument('--sigma', default=0.05, type=float, metavar='S',
                        help='Sigma for log likelihood.')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    device = torch.device(
        "cuda" if not args.nocuda and torch.cuda.is_available() else "cpu")
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
        model = nn.DataParallel(model)
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

    for epoch in range(int(args.start_epoch), args.epochs):
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

            total_loss = - (log_like - kl_z_what - kl_z_where - kl_z_pres - kl_z_depth - kl_bg_what)

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
                    'bg_what': log['bg_what'].view(-1, bg_what_dim),
                    'bg_what_std': log['bg_what_std'].view(-1, bg_what_dim),
                    'bg_what_mean': log['bg_what_mean'].view(-1, bg_what_dim),
                    'bg': log['bg'].view(-1, 3, img_h, img_w),
                    'z_what': log['z_what'].view(-1, 4 * 4, z_what_dim),
                    'z_where_scale':
                        log['z_where'].view(-1, 4 * 4, z_where_scale_dim + z_where_shift_dim)[:, :, :z_where_scale_dim],
                    'z_where_shift':
                        log['z_where'].view(-1, 4 * 4, z_where_scale_dim + z_where_shift_dim)[:, :, z_where_scale_dim:],
                    'z_pres': log['z_pres'].permute(0, 2, 3, 1),
                    'z_pres_probs': torch.sigmoid(log['z_pres_logits']).permute(0, 2, 3, 1),
                    'z_what_std': log['z_what_std'].view(-1, 4 * 4, z_what_dim),
                    'z_what_mean': log['z_what_mean'].view(-1, 4 * 4, z_what_dim),
                    'z_where_scale_std':
                        log['z_where_std'].permute(0, 2, 3, 1)[:, :, :z_where_scale_dim],
                    'z_where_scale_mean':
                        log['z_where_mean'].permute(0, 2, 3, 1)[:, :, :z_where_scale_dim],
                    'z_where_shift_std':
                        log['z_where_std'].permute(0, 2, 3, 1)[:, :, z_where_scale_dim:],
                    'z_where_shift_mean':
                        log['z_where_mean'].permute(0, 2, 3, 1)[:, :, z_where_scale_dim:],
                    'glimpse': log['x_att'].view(-1, 4 * 4, 3, glimpse_size, glimpse_size),
                    'glimpse_recon': log['y_att'].view(-1, 4 * 4, 3, glimpse_size, glimpse_size),
                    'prior_z_pres_prob': log['prior_z_pres_prob'].unsqueeze(0),
                    'o_each_cell': spatial_transform(log['o_att'], log['z_where'], (4 * 4 * bs, 3, img_h, img_w),
                                                     inverse=True).view(-1, 4 * 4, 3, img_h, img_w),
                    'alpha_hat_each_cell': spatial_transform(log['alpha_att_hat'], log['z_where'],
                                                             (4 * 4 * bs, 1, img_h, img_w),
                                                             inverse=True).view(-1, 4 * 4, 1, img_h, img_w),
                    'alpha_each_cell': spatial_transform(log['alpha_att'], log['z_where'],
                                                         (4 * 4 * bs, 1, img_h, img_w),
                                                         inverse=True).view(-1, 4 * 4, 1, img_h, img_w),
                    'y_each_cell': (log['y_each_cell'] * log['z_pres'].
                                    view(-1, 1, 1, 1)).view(-1, 4 * 4, 3, img_h, img_w),
                    'z_depth': log['z_depth'].view(-1, 4 * 4, z_depth_dim),
                    'z_depth_std': log['z_depth_std'].view(-1, 4 * 4, z_depth_dim),
                    'z_depth_mean': log['z_depth_mean'].view(-1, 4 * 4, z_depth_dim),
                    'importance_map_full_res_norm':
                        log['importance_map_full_res_norm'].view(-1, 4 * 4, 1, img_h, img_w),
                    'z_pres_logits': log['z_pres_logits'].permute(0, 2, 3, 1),
                    'z_pres_y': log['z_pres_y'].permute(0, 2, 3, 1),
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

                    if key == 'importance_map_full_res_norm' or key == 'alpha_hat_each_cell' or key == 'alpha_each_cell':
                        writer.add_histogram('inside_value/' + key, value[value > 0].cpu().detach().numpy(),
                                             global_step)
                    else:
                        writer.add_histogram('inside_value/' + key, value.cpu().detach().numpy(),
                                             global_step)

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
                                 log['z_where_shift'][:num_img_summary].cpu().detach())

                y_each_cell = log['y_each_cell'].view(-1, 3, img_h, img_w)[:num_img_summary * 16].cpu().detach()
                o_each_cell = log['o_each_cell'].view(-1, 3, img_h, img_w)[:num_img_summary * 16].cpu().detach()
                alpha_each_cell = log['alpha_hat_each_cell'].view(-1, 1, img_h, img_w)[
                                  :num_img_summary * 16].cpu().detach()
                importance_each_cell = \
                    log['importance_map_full_res_norm'].view(-1, 1, img_h, img_w)[:num_img_summary * 16].cpu().detach()

                for i in range(num_img_summary):
                    grid_image = make_grid(bbox[i * 16:(i + 1) * 16], 4, normalize=True, pad_value=1)
                    writer.add_image('train/4-bbox_{}'.format(i), grid_image, global_step)

                    grid_image = make_grid(y_each_cell[i * 16:(i + 1) * 16], 4, normalize=True, pad_value=1)
                    writer.add_image('train/5-y_each_cell_{}'.format(i), grid_image, global_step)

                    grid_image = make_grid(o_each_cell[i * 16:(i + 1) * 16], 4, normalize=True, pad_value=1)
                    writer.add_image('train/6-o_each_cell_{}'.format(i), grid_image, global_step)

                    grid_image = make_grid(alpha_each_cell[i * 16:(i + 1) * 16], 4, normalize=True, pad_value=1)
                    writer.add_image('train/7-alpha_hat_each_cell_{}'.format(i), grid_image, global_step)

                    grid_image = make_grid(importance_each_cell[i * 16:(i + 1) * 16], 4, normalize=True, pad_value=1)
                    writer.add_image('train/8-importance_each_cell_{}'.format(i), grid_image, global_step)


                writer.add_scalar('train/total_loss', total_loss.item(), global_step=global_step)
                writer.add_scalar('train/log_like', log_like.item(), global_step=global_step)
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

