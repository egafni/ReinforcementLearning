import numpy
import numpy as np
import torch
from torch import nn

from oracle.utils.torch_utils import net_to_device


class PolicyNet(nn.Module):
    def select_action(self, state):
        device = net_to_device(self)

        state = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0)
        probs = self(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return int(action.item()), m.log_prob(action)


class PolicyMlp(PolicyNet):
    def __init__(self, input_shape, n_actions, width=64):
        super().__init__()

        input_shape = numpy.prod(input_shape)

        self.seq = nn.Sequential(
            nn.Linear(input_shape, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.seq(x.flatten(1)), dim=1)


class PolicyConv(PolicyNet):
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
        return torch.softmax(self.fc(x), dim=1)
