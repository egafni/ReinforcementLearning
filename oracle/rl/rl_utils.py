# type: ignore

from dataclasses import dataclass
from typing import Optional, Union

import gym
import numpy
import torch
from pyvirtualdisplay import Display


def get_returns(rewards, gamma):
    """
    Returns a list of discounted returns
    G_t = r_t + gamma * G_t+1

    :param rewards: iterable of rewards
    :param gamma: discount rate

    >>> get_returns([1,1,1], 1)
    array([3, 2, 1])
    >>> get_returns([1,1,1], .99)
    array([2.9701, 1.99  , 1.    ])
    >>> get_returns([1,1,1],.5)
    array([1.75, 1.5 , 1.  ])
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + G * gamma
        returns.insert(0, G)
    return numpy.array(returns)


@dataclass
class Runner:
    observations: torch.FloatTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    renders: Optional[torch.FloatTensor] = None
    last_obs: torch.LongTensor = None

    @staticmethod
    def play(
        env: gym.Env,
        net: torch.nn.Module = None,
        net_policy="stochastic",
        render=False,
        max_steps=None,
        return_torch=True,
    ):
        """Play one episode of the environment
        :param net: if set, use net to produce actions, otherwise choose a random action
        :param net_policy: if 'stochastic' action = net(obs).softmax().sample(), if 'deterministic', action = net(obs).argmax()
        :param render: render the environment
        """
        observations = []
        rewards = []
        actions = []
        renders = []
        is_done = False

        obs = env.reset()

        with Display():
            while not is_done:
                observations.append(obs)

                if net is None:
                    # Randomly sample an action
                    action = env.action_space.sample()
                elif hasattr(net, "predict"):
                    # for stable-baseline3 models
                    action, _states = net.predict(obs, deterministic=net_policy == "deterministic")  # type: ignore
                else:
                    # standard pytorch net
                    device = "cuda" if next(net.parameters()).is_cuda else "cpu"
                    action_logits = net(torch.FloatTensor(obs).unsqueeze(0).to(device)).to("cpu").squeeze(0)

                    if net_policy == "stochastic":
                        action_probs = torch.softmax(action_logits, dim=0).detach().numpy()
                        action = numpy.random.choice(range(len((action_probs))), 1, p=action_probs)[0]
                    elif net_policy == "deterministic":
                        action = torch.argmax(action_logits)
                    else:
                        raise ValueError(f"{net_policy} must be stochastic or deterministic")

                actions.append(action)

                if render:
                    renders.append(env.render("rgb_array"))

                obs, reward, is_done, _ = env.step(action)

                rewards.append(reward)

                if max_steps is not None and len(observations) >= max_steps:
                    break

        last_obs = obs

        if return_torch:
            f = torch.tensor
        else:
            f = numpy.array

        return Runner(f(observations), f(actions), f(rewards), f(renders), f(last_obs))


@dataclass
class RewardTracker:
    """
    Tracks rewards and provides helpers such as when the best reward was,
    and if we should early stop.

    See tests for example usage.
    """

    rewards: list
    max_reward: Union[None, int]  # max reward seen yet
    max_reward_index: Union[None, int]  # index of the max_reward

    def __init__(self):
        self.rewards = []
        self.max_reward = None
        self.max_reward_index = None

    def append(self, reward):
        """Add a reward"""
        self.rewards.append(reward)

        if self.max_reward is None or reward > self.max_reward:
            self.max_reward = reward
            self.max_reward_index = self.current_index

    @property
    def current_index(self):
        """The current index, or the length of the rewards"""
        if len(self.rewards) == 0:
            return None
        else:
            return len(self.rewards) - 1

    def should_early_stop(self, patience) -> bool:
        """True if our max reward was more than patience steps ago """
        if self.max_reward_index is None:
            return False
        else:
            return self.current_index - self.max_reward_index > patience

    def last_reward_was_best(self):
        """True if the last reward was the best reward"""
        return self.current_index == self.max_reward_index

    def mean_last_n_rewards(self, n):
        """Average of the last n rewards"""
        return numpy.mean(self.rewards[-n:])
