import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, RelaxedBernoulli
from utils import linear_annealing, spatial_transform, calc_kl_z_pres_bernoulli
from module import NumericalRelaxedBernoulli
from common import *


class ImgEncoder(nn.Module):

    def __init__(self):
        super(ImgEncoder, self).__init__()

        # self.z_pres_bias = +2

        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, img_encode_dim, 1),
            nn.CELU(),
            nn.GroupNorm(16, img_encode_dim)
            # nn.Conv2d(img_encode_dim, img_encode_dim, 1),
            # nn.CELU(),
            # nn.GroupNorm(16, 128)
        )

        self.enc_lat = nn.Sequential(
            nn.Conv2d(img_encode_dim, img_encode_dim, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(img_encode_dim, img_encode_dim, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128)
        )

        self.enc_cat = nn.Sequential(
            nn.Conv2d(img_encode_dim * 2, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64)
        )

        self.z_where_net = nn.Conv2d(64, (z_where_shift_dim + z_where_scale_dim) * 2, 1)

        self.z_pres_net = nn.Conv2d(64, z_pres_dim, 1)

        self.z_depth_net = nn.Conv2d(64, z_depth_dim * 2, 1)

        offset_y, offset_x = torch.meshgrid([torch.arange(4.), torch.arange(4.)])

        # since first dimension of z_where_shift is x, as in spatial transform matrix
        self.register_buffer('offset', torch.stack((offset_x, offset_y), dim=0))

    def forward(self, x, tau):
        img_enc = self.enc(x)

        lateral_enc = self.enc_lat(img_enc)

        cat_enc = self.enc_cat(torch.cat((img_enc, lateral_enc), dim=1))

        # (bs, 1, 4, 4)
        z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(cat_enc))

        # z_pres = q_z_pres.rsample()
        q_z_pres = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=tau)

        z_pres_y = q_z_pres.rsample()

        z_pres = torch.sigmoid(z_pres_y)

        # (bs, dim, 4, 4)
        z_depth_mean, z_depth_std = self.z_depth_net(cat_enc).chunk(2, 1)
        z_depth_std = F.softplus(z_depth_std)
        q_z_depth = Normal(z_depth_mean, z_depth_std)

        z_depth = q_z_depth.rsample()

        # (bs, 4 + 4, 4, 4)
        z_where_mean, z_where_std = self.z_where_net(cat_enc).chunk(2, 1)
        z_where_std = F.softplus(z_where_std)

        q_z_where = Normal(z_where_mean, z_where_std)

        z_where = q_z_where.rsample()
        # make it global
        z_where[:, :2] = (-(scale_bias + z_where[:, :2].tanh())).exp()
        z_where[:, 2:] = 0.5 * (self.offset + 0.5 + z_where[:, 2:].tanh()) - 1

        z_where = z_where.permute(0, 2, 3, 1).reshape(-1, 4)

        return z_where, z_pres, z_depth, q_z_where, \
               q_z_depth, z_pres_logits, z_pres_y


class ZWhatEnc(nn.Module):

    def __init__(self):
        super(ZWhatEnc, self).__init__()

        self.enc_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
        )

        self.enc_what = nn.Linear(128, z_what_dim * 2)

        # self.enc_depth = nn.Linear(128, z_depth_dim * 2)

    def forward(self, x):
        x = self.enc_cnn(x)

        z_what_mean, z_what_std = self.enc_what(x.flatten(start_dim=1)).chunk(2, -1)
        z_what_std = F.softplus(z_what_std)
        q_z_what = Normal(z_what_mean, z_what_std)

        z_what = q_z_what.rsample()

        return z_what, q_z_what


class GlimpseDec(nn.Module):

    def __init__(self):
        super(GlimpseDec, self).__init__()

        # self.alpha_bias = +2

        self.dec = nn.Sequential(
            nn.Conv2d(z_what_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 128 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 32 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),

            nn.Conv2d(32, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
        )


        self.dec_o = nn.Conv2d(16, 3, 3, 1, 1)

        self.dec_alpha = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        x = self.dec(x.view(x.size(0), -1, 1, 1))

        o = torch.sigmoid(self.dec_o(x))
        alpha = torch.sigmoid(self.dec_alpha(x))

        return o, alpha

class BgDecoder(nn.Module):

    def __init__(self):
        super(BgDecoder, self).__init__()

        # self.background_bias = +2

        self.dec = nn.Sequential(
            nn.Conv2d(bg_what_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 128 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

    def forward(self, x):
        bg = torch.sigmoid(self.dec(x))
        return bg

class BgEncoder(nn.Module):

    def __init__(self):
        super(BgEncoder, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(img_h * img_w * 3, 256),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Linear(256, 128),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Linear(128, bg_what_dim * 2),
        )

    def forward(self, x):
        bs = x.size(0)
        bg_what_mean, bg_what_std = self.enc(x.view(x.size(0), -1)).chunk(2, -1)
        bg_what_std = F.softplus(bg_what_std)
        q_bg_what = Normal(bg_what_mean.view(bs, bg_what_dim, 1, 1), bg_what_std.view(bs, bg_what_dim, 1, 1))

        bg_what = q_bg_what.rsample()

        return bg_what, q_bg_what


class Spair(nn.Module):

    def __init__(self, sigma=0.3):
        super(Spair, self).__init__()

        self.z_pres_anneal_start_step = 0000
        self.z_pres_anneal_end_step = 1000
        # self.z_pres_anneal_start_value = 0.6
        self.z_pres_anneal_start_value = 1e-1
        self.z_pres_anneal_end_value = 1e-5
        self.likelihood_sigma = sigma

        self.img_encoder = ImgEncoder()
        self.z_what_net = ZWhatEnc()
        self.glimpse_dec = GlimpseDec()
        self.bg_encoder = BgEncoder()
        self.bg_decoder = BgDecoder()

        self.register_buffer('prior_what_mean', torch.zeros(1))
        self.register_buffer('prior_what_std', torch.ones(1))
        self.register_buffer('prior_bg_mean', torch.zeros(1))
        self.register_buffer('prior_bg_std', torch.ones(1))
        self.register_buffer('prior_depth_mean', torch.zeros(1))
        self.register_buffer('prior_depth_std', torch.ones(1))
        self.register_buffer('prior_where_mean',
                             torch.tensor([0., 0., 0., 0.]).view((z_where_scale_dim + z_where_shift_dim), 1, 1))
        self.register_buffer('prior_where_std',
                             torch.tensor([1., 1., 1., 1.]).view((z_where_scale_dim + z_where_shift_dim), 1, 1))
        self.register_buffer('prior_z_pres_prob', torch.tensor(
            self.z_pres_anneal_start_value))

    @property
    def p_bg_what(self):
        return Normal(self.prior_bg_mean, self.prior_bg_std)

    @property
    def p_z_what(self):
        return Normal(self.prior_what_mean, self.prior_what_std)

    @property
    def p_z_depth(self):
        return Normal(self.prior_depth_mean, self.prior_depth_std)

    @property
    def p_z_where(self):
        return Normal(self.prior_where_mean, self.prior_where_std)

    def forward(self, x, global_step, tau, eps=1e-15):
        bs = x.size(0)
        self.prior_z_pres_prob = linear_annealing(self.prior_z_pres_prob, global_step,
                                                  self.z_pres_anneal_start_step, self.z_pres_anneal_end_step,
                                                  self.z_pres_anneal_start_value, self.z_pres_anneal_end_value)

        # (bs, bg_what_dim)
        bg_what, q_bg_what = self.bg_encoder(x)
        # (bs, 3, img_h, img_w)
        bg = self.bg_decoder(bg_what)

        # z_where: (4*4*bs, 4)
        # z_pres, z_depth, z_pres_logits: (bs, dim, 4, 4)
        z_where, z_pres, z_depth, q_z_where, \
        q_z_depth, z_pres_logits, z_pres_y = self.img_encoder(x, tau)

        # (4 * 4 * bs, 3, glimpse_size, glimpse_size)
        x_att = spatial_transform(torch.stack(4 * 4 * (x,), dim=1).view(-1, 3, img_h, img_w), z_where,
                                  (4 * 4 * bs, 3, glimpse_size, glimpse_size), inverse=False)

        # (4 * 4 * bs, dim)
        z_what, q_z_what = self.z_what_net(x_att)

        # (4 * 4 * bs, dim, glimpse_size, glimpse_size)
        o_att, alpha_att = self.glimpse_dec(z_what)
        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)
        y_att = alpha_att_hat * o_att

        # (4 * 4 * bs, 3, img_h, img_w)
        y_each_cell = spatial_transform(y_att, z_where, (4 * 4 * bs, 3, img_h, img_w),
                                        inverse=True)

        # (4 * 4 * bs, 1, glimpse_size, glimpse_size)
        importance_map = alpha_att_hat * 4.4 * torch.sigmoid(-z_depth).view(-1, 1, 1, 1)
        # (4 * 4 * bs, 1, img_h, img_w)
        importance_map_full_res = spatial_transform(importance_map, z_where, (4 * 4 * bs, 1, img_h, img_w),
                                                    inverse=True)
        # # (bs, 4 * 4, 1, img_h, img_w)
        importance_map_full_res = importance_map_full_res.view(-1, 4 * 4, 1, img_h, img_w)
        importance_map_full_res_norm = importance_map_full_res / \
                                       (importance_map_full_res.sum(dim=1, keepdim=True) + eps)

        # (bs, 4 * 4, 1, img_h, img_w)
        alpha_map = spatial_transform(alpha_att_hat, z_where, (4 * 4 * bs, 1, img_h, img_w),
                                      inverse=True).view(-1, 4 * 4, 1, img_h, img_w).sum(dim=1)
        # (bs, 1, img_h, img_w)
        alpha_map = alpha_map + (alpha_map.clamp(eps, 1 - eps) - alpha_map).detach()

        # (bs, 3, img_h, img_w)
        y_nobg = (y_each_cell.view(-1, 4 * 4, 3, img_h, img_w) * importance_map_full_res_norm).sum(dim=1)
        y = y_nobg + (1 - alpha_map) * bg

        # (bs, bg_what_dim)
        kl_bg_what = kl_divergence(q_bg_what, self.p_bg_what)
        kl_bg_what = kl_bg_what.view(bs, bg_what_dim)
        # (4 * 4 * bs, z_what_dim)
        kl_z_what = kl_divergence(q_z_what, self.p_z_what) * z_pres.view(-1, 1)
        # (bs, 4 * 4, z_what_dim)
        kl_z_what = kl_z_what.view(-1, 4 * 4, z_what_dim)
        # (4 * 4 * bs, z_depth_dim)
        kl_z_depth = kl_divergence(q_z_depth, self.p_z_depth) * z_pres
        # (bs, 4 * 4, z_depth_dim)
        kl_z_depth = kl_z_depth.view(-1, 4 * 4, z_depth_dim)
        # (bs, dim, 4, 4)
        kl_z_where = kl_divergence(q_z_where, self.p_z_where) * z_pres
        kl_z_pres = calc_kl_z_pres_bernoulli(z_pres_logits, self.prior_z_pres_prob)
        if DEBUG:
            if torch.any(torch.isnan(kl_z_pres)):
                breakpoint()
        p_x_given_z = Normal(y.flatten(start_dim=1), self.likelihood_sigma)
        log_like = p_x_given_z.log_prob(x.expand_as(y).flatten(start_dim=1))

        self.log = {
            'bg_what': bg_what,
            'bg_what_std': q_bg_what.stddev,
            'bg_what_mean': q_bg_what.mean,
            'bg': bg,
            'z_what': z_what,
            'z_where': z_where,
            'z_pres': z_pres,
            'z_pres_logits': z_pres_logits,
            'z_what_std': q_z_what.stddev,
            'z_what_mean': q_z_what.mean,
            'z_where_std': q_z_where.stddev,
            'z_where_mean': q_z_where.mean,
            'x_att': x_att,
            'y_att': y_att,
            'prior_z_pres_prob': self.prior_z_pres_prob.unsqueeze(0),
            'o_att': o_att,
            'alpha_att_hat': alpha_att_hat,
            'alpha_att': alpha_att,
            'y_each_cell': y_each_cell,
            'z_depth': z_depth,
            'z_depth_std': q_z_depth.stddev,
            'z_depth_mean': q_z_depth.mean,
            'importance_map_full_res_norm': importance_map_full_res_norm,
            'z_pres_y': z_pres_y,
        }

        return y, log_like.flatten(start_dim=1).sum(dim=1), \
               kl_z_what.flatten(start_dim=1).sum(dim=1), \
               kl_z_where.flatten(start_dim=1).sum(dim=1), \
               kl_z_pres.flatten(start_dim=1).sum(dim=1), \
               kl_z_depth.flatten(start_dim=1).sum(dim=1), \
               kl_bg_what.flatten(start_dim=1).sum(dim=1), self.log
