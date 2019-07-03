import torch
from torch.distributions import RelaxedBernoulli, Normal
from torch.distributions.utils import lazy_property, broadcast_all
from common import *


# def clamp_probs(probs, eps=1e-20):
#     return probs.clamp(min=eps, max=1 - eps)


class NumericalRelaxedBernoulli(RelaxedBernoulli):

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    # def rsample(self, sample_shape=torch.Size()):
    #     shape = self._extended_shape(sample_shape)
    #     probs = clamp_probs(self.probs.expand(shape))
    #     uniforms = clamp_probs(torch.rand(shape, dtype=probs.dtype, device=probs.device))
    #
    #     return (uniforms.log() - (-uniforms).log1p() + probs.log() - (
    #             -probs).log1p()) / self.temperature

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)

        out = self.temperature.log() + diff - 2 * diff.exp().log1p()

        if DEBUG:
            if torch.isinf(out).any():
                breakpoint()

        return out



def mixture_kl_divergence(
    z, q_z_x, p_c, p_z_where_clusters, num_cluster, num_cell, d_z=4):

    # z: n * 4 * cell * cell
    #z = z.view(-1, num_cell, num_cell, d_z).permute(0, 3, 1, 2)
    #z = z.reshape(-1, num_cell, num_cell, d_z).permute(0, 3, 1, 2)
    log_q_z_given_x = q_z_x.log_prob(z)
    #print("q_z_x", q_z_x)
    #print("z", z.shape)

    #    m.probs.
    prob_cs = p_c.probs.view(1, num_cluster, 1, 1, 1)

    log_p_z_c_up = p_z_where_clusters.log_prob(
        z.repeat(num_cluster, 1, 1, 1)\
            .view(-1, num_cluster, d_z, num_cell, num_cell))
    p_z_c_up = log_p_z_c_up.exp() + 1e-15
    #print("p_z_where_clusters", p_z_where_clusters)
    #print("p_z_c_up", p_z_c_up)

    weighted_p_z_c_up = p_z_c_up * prob_cs
    #print("weighted_p_z_c_up", weighted_p_z_c_up)
    sum_weighted_p_z_c_up = weighted_p_z_c_up.sum(dim=1).view(
        -1, 1, d_z, num_cell, num_cell)
    q_c_x = weighted_p_z_c_up / sum_weighted_p_z_c_up
    #print("q_c_x", q_c_x)
    #print("q_c_x_sum", q_c_x.sum(dim=1))

    E__q_c_x__logp_z_c = (log_p_z_c_up * q_c_x).sum(dim=1)\
        .view(-1, d_z, num_cell, num_cell)
    E__q_c_x__logq_c_x = ((q_c_x+1e-15).log() * q_c_x).sum(dim=1)\
        .view(-1, d_z, num_cell, num_cell)
    E__q_c_x__logp_c = (
            (prob_cs.repeat(z.shape[0], 1, 1, 1, 1)+1e-15).log() * q_c_x
        ).sum(dim=1).view(-1, d_z, num_cell, num_cell)
    #print("E__q_c_x__logp_z_c", E__q_c_x__logp_z_c.shape, E__q_c_x__logp_z_c.flatten(start_dim=1).sum(dim=1).mean())
    #print("E__q_c_x__logp_c", E__q_c_x__logp_c.shape, E__q_c_x__logp_c.flatten(start_dim=1).sum(dim=1).mean())
    #print("log_q_z_given_x", log_q_z_given_x.flatten(start_dim=1).sum(dim=1).mean())
    #print("E__q_c_x__logq_c_x", E__q_c_x__logq_c_x.shape, E__q_c_x__logq_c_x.flatten(start_dim=1).sum(dim=1).mean())

    return -E__q_c_x__logp_z_c - E__q_c_x__logp_c \
        + log_q_z_given_x + E__q_c_x__logq_c_x


if __name__ == "__main__":
    from torch.distributions import Normal, kl_divergence, RelaxedBernoulli, Categorical
    nor = Normal(
        torch.tensor([
            -0.7,
            -1.6,
            -2.7,
        ]).view(3, 1, 1, 1),
        torch.tensor([
            .05,
            .05,
            .05,
        ]).view(3, 1, 1, 1)
    )
    cat = Categorical(torch.tensor([0.5, 0.2, 0.3]))

    '''
    q_z_where = Normal(
        torch.tensor([-0.7, -0.7, -0.7, -0.7]).view(1, 1, 2, 2),
        torch.tensor([0.1, 0.1, 0.1, 0.1]).view(1, 1, 2, 2))
    z = q_z_where.rsample().view(1, 1, 2, 2)
    z = torch.tensor([-0.1, -0.5, -1, -1.6]).view(1, 1, 2, 2)
    '''
    q_z_where = Normal(
        torch.tensor([-2.7]).view(1, 1, 1, 1),
        torch.tensor([0.05]).view(1, 1, 1, 1))
    z = q_z_where.rsample().view(1, 1, 1, 1)
    z = torch.tensor([2.7]).view(1, 1, 1, 1)
    print(z)
    print(mixture_kl_divergence(
        z, q_z_where, cat, nor, 3, 1, 1
    ))
