import torch
from torch.distributions import RelaxedBernoulli
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
