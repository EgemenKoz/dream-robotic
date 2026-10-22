"""
SAC policy whose twin Q-networks put LayerNorm after every hidden layer.

Why: in `multi_v4` the stock critic diverged with nothing wrong in the data.
Every reward in its buffer lay in [-50.5, +0.35] and the policy never reached
the goal, so the true Q could not be much above zero; the critic still climbed
to Q ≈ +40 by 20k steps, +530 by 40k and +7000 by 300k, the actor saturated its
actions against that error and the entropy coefficient ran from 0.03 to 8.7.
That is bootstrapping feeding on its own extrapolation, not a reward bug, and
it struck one run of an unchanged setup while the previous run stayed healthy.

LayerNorm bounds the magnitude of the last hidden layer, so Q can only grow
through the output weights rather than through every activation at once; it
is the standard fix for exactly this failure (Ball et al., RLPD, ICML'23;
Nauman et al., "Overestimation, Overfitting, and Plasticity in Actor-Critic",
ICML'24). Turned on by RLModelConfig.CRITIC_LAYER_NORM.

Models saved with it record this class, so loading them needs this module to
be importable; models saved before it load as plain MultiInputPolicy.
"""

from typing import List, Optional, Type

import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import MultiInputPolicy


def layer_norm_mlp(in_dim: int, net_arch: List[int],
                   activation_fn: Type[nn.Module]) -> nn.Sequential:
    """Linear → LayerNorm → activation per hidden layer, then a scalar head."""
    layers: List[nn.Module] = []
    for width in net_arch:
        layers += [nn.Linear(in_dim, width), nn.LayerNorm(width), activation_fn()]
        in_dim = width
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


class LayerNormContinuousCritic(ContinuousCritic):
    """ContinuousCritic with LayerNorm Q-networks; otherwise identical."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space, action_space, net_arch, features_extractor,
            features_dim, activation_fn, normalize_images, n_critics,
            share_features_extractor,
        )
        in_dim = features_dim + get_action_dim(self.action_space)
        for idx in range(n_critics):
            q_net = layer_norm_mlp(in_dim, net_arch, activation_fn)
            self.add_module(f"qf{idx}", q_net)   # replaces the stock network
            self.q_networks[idx] = q_net


class LayerNormSACPolicy(MultiInputPolicy):
    """MultiInputPolicy (Dict observations) with LayerNorm twin critics."""

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None
                    ) -> LayerNormContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)
