"""
refactored
https://github.com/pytorch/examples/tree/master/reinforcement_learning

So far, can only solve CartPole with this one
"""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

import gym
import numpy
import numpy as np
import torch
import torch.optim as optim
from pytorch_lightning import seed_everything

from oracle.common import make_env_oracle
from oracle.rl.reinforce.models import PolicyConv, PolicyMlp, PolicyNet
from oracle.rl.rl_utils import RewardTracker, get_returns  # type: ignore
from oracle.utils.misc_utils import DataClassMixin


@dataclass
class ReinforceParams(DataClassMixin):
    env_name: str = "CartPole-v1"  # gym environment name
    env_kwargs: dict = field(default_factory=dict)

    early_stopping_patience: int = 200  # converge if this number of episodes has passed with no improvement

    gamma: float = 0.99  # discount rate
    batch_size: int = 32  # batch size
    cpu: bool = False  # train on the gpu
    seed: Optional[int] = None
    log_interval: int = 10

    max_episodes: int = 10000

    model_classes = {"PolicyMlp": PolicyMlp, "PolicyConv": PolicyConv}
    model_name: str = field(default="PolicyMlp", metadata=dict(choices=model_classes.keys()))

    @property
    def model_class(self):
        return self.model_classes[self.model_name]


@dataclass
class Episode:
    actions: numpy.ndarray
    action_log_probs: list  # need to keep a list of torch tensors to backprop
    rewards: numpy.ndarray

    @classmethod
    def generate(cls, env: gym.Env, net: PolicyNet):
        """
        Generates one episode
        """
        state = env.reset()

        rewards = []
        actions = []
        action_log_probs = []

        # generate 1 episode
        while True:
            action, action_log_prob = net.select_action(state)
            state, reward, done, _ = env.step(action)
            actions.append(action)
            action_log_probs.append(action_log_prob)
            rewards.append(reward)
            if done:
                break

        return cls(numpy.array(actions), action_log_probs, numpy.array(rewards))

    @property
    def length(self):
        return len(self.actions)

    def learn(self, gamma, optimizer):
        """
        E is the expected trajectory given Pi_theta

        J = E[Q_pi(s,a) * Pi_theta(a|s)]
        """

        eps = np.finfo(np.float32).eps.item()

        returns = torch.tensor(get_returns(self.rewards, gamma))
        # normalize the returns
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # FIXME vectorize this
        # self.action_log_probs is a list of tensor scalars which follows the episode trajectory
        # one solution is to store the states, and re-process the episode's states
        # through the network to create a batch, then use torch.gather() on the actions chosen
        # (that's what we do in dqn.py)
        policy_loss = []

        for log_prob, G in zip(self.action_log_probs, returns):
            policy_loss.append(-log_prob * G)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()


def train(params: ReinforceParams):
    seed_everything(params.seed)
    env = make_env_oracle(params.env_name, params.env_kwargs)
    env.seed(params.seed)
    device = torch.device("cpu" if params.cpu else "cuda")
    net = params.model_class(env.observation_space.shape, env.action_space.n).to(device)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    reward_tracker = RewardTracker()

    for i_episode in count(1):
        episode = Episode.generate(env, net)
        reward_tracker.append(episode.rewards.sum())
        episode.learn(params.gamma, optimizer)

        if i_episode % params.log_interval == 0:
            print(
                f"Episode {i_episode}\t"
                f"Last reward: {episode.rewards.sum():.2f}"
                f"\tMean last 10 reward: {reward_tracker.mean_last_n_rewards(n=10):.2f}"
            )

        if reward_tracker.should_early_stop(params.early_stopping_patience):
            print(
                f"Early Stopped! max reward at ep {reward_tracker.max_reward_index}, the last episode runs to {episode.length} time steps!"
            )
            break
        if i_episode >= params.max_episodes:
            print(f"Unsolved! reached {params.max_episodes} episodes")
            break

    return net, reward_tracker.mean_last_n_rewards(n=10)


if __name__ == "__main__":
    p = ArgumentParser()
    # Add all the parameters from the Params class to argparse
    ReinforceParams.add_to_argparser(p)
    args = p.parse_args()
    train(ReinforceParams.from_parsed_args(args))
