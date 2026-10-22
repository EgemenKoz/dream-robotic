"""
Two-branch feature extractor for the Dict observation of BarnEnv.

Architecture (from the LiDAR DRL-navigation literature — Fan & Long et al.
ICRA'18 / IJRR'20, DRL-VO Xie & Dames T-RO'23):

    obs["scan"]  (K, N)  ──► 1D-CNN over sectors (frames = channels) ──► FC ──►┐
                                                                               ├─► concat ─► SAC heads
    obs["state"] (5,)    ──► FC ──────────────────────────────────────────────►┘

Why not a plain MLP on the flat vector: an MLP has no notion that scan
sector i and i+1 are spatially adjacent, so it must *learn* locality from
samples. The 1D conv bakes it in, and treating the K stacked frames as input
channels lets the kernels see how each sector evolves over time (approach
speed), exactly like frame-stacked Atari CNNs.

Used via SB3:

    policy_kwargs = dict(
        features_extractor_class=ScanStateExtractor,
        net_arch=RLModelConfig.NET_ARCH,
    )
    model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, ...)
"""

import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from configs.rl_config import RLModelConfig


class ScanStateExtractor(BaseFeaturesExtractor):
    """Encodes {"scan": (K, N), "state": (S,)} into a flat feature vector."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        scan_features: int = RLModelConfig.SCAN_FEATURES,
        state_features: int = RLModelConfig.STATE_FEATURES,
    ):
        super().__init__(observation_space, features_dim=scan_features + state_features)

        k, n = observation_space["scan"].shape
        state_dim = observation_space["state"].shape[0]

        conv_layers = []
        in_ch = k
        for out_ch, kernel, stride in RLModelConfig.CONV_LAYERS:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                          padding=kernel // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        conv_layers.append(nn.Flatten())
        self.scan_cnn = nn.Sequential(*conv_layers)

        with th.no_grad():
            flat_dim = self.scan_cnn(th.zeros(1, k, n)).shape[1]

        self.scan_fc = nn.Sequential(nn.Linear(flat_dim, scan_features), nn.ReLU())
        self.state_fc = nn.Sequential(nn.Linear(state_dim, state_features), nn.ReLU())

    def forward(self, observations: dict) -> th.Tensor:
        scan_feat = self.scan_fc(self.scan_cnn(observations["scan"]))
        state_feat = self.state_fc(observations["state"])
        return th.cat([scan_feat, state_feat], dim=1)
