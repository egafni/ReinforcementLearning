import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18


class DQN_resnet18(nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
    ):
        super().__init__()

        self.net = resnet18()
        self.net.conv1 = nn.Conv2d(
            input_shape[0],
            64,
            kernel_size=(7, 7),
            stride=(
                2,
                2,
            ),
            padding=(3, 3),
            bias=False,
        )
        self.net.fc = nn.Linear(512, n_actions)

    def forward(self, x):
        return self.net(x)


class DQN_Conv2d(nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc(x)


class DQN_Mlp(nn.Module):
    def __init__(self, input_shape, n_actions, width=64):
        super().__init__()

        input_shape = np.prod(input_shape)

        self.seq = nn.Sequential(
            nn.Linear(input_shape, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, n_actions),
        )

    def forward(self, x):
        return self.seq(x.flatten(1))
