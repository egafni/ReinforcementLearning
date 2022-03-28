"""
Cartpole:
python algos/algos/dqn/sample.py --env_name CartPole-v1 --model_name DQN_Mlp --eps_decay_last_frame 10000 --replay_start_size 1000
By episode 390 we achieve a mean reward of 260

Can solve CartPole and Pong, and get up to 8 points on BreakOut
"""
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import gym
import numpy as np
import torch
from numpy.random import RandomState
from pytorch_lightning.utilities.seed import seed_everything
from simple_parsing import ArgumentParser
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from rl.common import make_env
from rl.algos.dqn.models import DQN_Conv2d, DQN_Mlp, DQN_resnet18
from rl.algos.rl_utils import RewardTracker  # type:ignore
from rl.utils.misc_utils import DataClassMixin


@dataclass
class DQNParams(DataClassMixin):
    env_name: str = "PongNoFrameskip-v4"  # gym environment name
    env_kwargs: dict = field(default_factory=dict)
    early_stopping_patience: int = 500  # converge if this number of episodes has passed with no improvement

    gamma: float = 0.99  # discount rate
    batch_size: int = 32  # batch size
    replay_size: int = 10000  # how big the replay buffer is
    lr: float = 1e-4  # learning rate
    sync_target_net_frames: int = 1000  # sync target network every N frames
    replay_start_size: int = 10000  # how many experiences to gain before training starts

    eps_decay_last_frame: int = 10 ** 5  # frame at which epsilon becomes final epsilon
    eps_start: float = 0.8  # starting epsilon
    eps_final: float = 0.02  # final epsilon

    cpu: bool = False  # trani on the gpu
    seed: Optional[int] = None

    max_frames: float = float("inf")  # maximum number of frames to iterate over

    model_classes = {"DQN_Mlp": DQN_Mlp, "DQN_Conv2d": DQN_Conv2d, "DQN_resnet18": DQN_resnet18}

    model_name: str = field(default="DQN_resnet18", metadata=dict(choices=model_classes.keys()))

    @property
    def model_class(self):
        return self.model_classes[self.model_name]


def compute_loss(net, tgt_net, gamma, batch, device) -> torch.Tensor:
    """
    loss = mse(Q(s,a) - (r + gamma * max_a__Q^(s',a')))
    Q^ is the target network
    Q(s',a') is 0 if in a terminal state
    """
    # Covert batch to torch and unpack
    #   shapes: states, next_states = (batch, 4, height, width)
    #   shapes: actions, rewards = (batch,)
    # TODO this line is slow.  Consider storing experiences as Torch tensors, maybe with pinned memory?
    states, actions, rewards, dones, next_states = (torch.tensor(arr).to(device) for arr in batch)
    # q := Q(s, a)
    q = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    # q_next := Q^(s',a')
    q_next = tgt_net(next_states).max(1).values
    # set q_next to 0 for terminal states
    q_next[dones] = 0.0
    next_state_values = q_next.detach()

    # q_improved := (r + gamma * max_a__Q^(s',a')
    q_improved = rewards + next_state_values * gamma

    loss: torch.Tensor = nn.MSELoss()(q, q_improved)
    return loss


@dataclass
class Experience(DataClassMixin):
    state: np.ndarray
    action: int
    reward: float
    done: bool
    new_state: np.ndarray


class ExperienceBuffer(deque):
    # TODO a deque of experience dataclasses is a bit slow
    #   consider keeping all experiences in an array and tracking a cyclic current_index
    def sample(self, batch_size, seed=None):
        indices = RandomState(seed=seed).choice(len(self), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self[idx] for idx in indices])

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            np.array(next_states, dtype=np.float32),
        )


@dataclass
class Agent:
    env: gym.Env
    state: np.ndarray
    experience_buffer: ExperienceBuffer
    total_reward: float = 0.0

    def reset(self):
        self.total_reward = 0.0
        self.state = self.env.reset()

    def step(self, net, epsilon, device):
        """Take a step in the environment"""

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # unsequeeze to add a batch dim
            pi = net(torch.Tensor(self.state).unsqueeze(0).to(device))
            action = int(torch.argmax(pi))

        # environment step
        new_state, reward, done, _ = self.env.step(action)
        self.experience_buffer.append(Experience(self.state, action, reward, done, new_state))

        # update agent state
        self.total_reward += reward
        self.state = new_state

        return done


def train(params: DQNParams):
    seed_everything(params.seed)
    env = make_env(params.env_name, params.env_kwargs)

    device = torch.device("cpu" if params.cpu else "cuda")

    env.seed(params.seed)
    env.action_space.seed(params.seed)
    state = env.reset()
    writer = SummaryWriter(comment=f"- {params.env_name}")
    model_fname = params.env_name + "__" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".dat"  # model filename

    net = params.model_class(env.observation_space.shape, env.action_space.n).to(device)
    print(net)
    # tgt_net used to estimate value of the next state, its weights are updated only once every params.sync_target_frames
    tgt_net = params.model_class(env.observation_space.shape, env.action_space.n).to(device)
    agent = Agent(env, state, ExperienceBuffer(maxlen=params.replay_size))

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    loss = torch.tensor(np.nan)  # loss
    frame = 0  # current observation frame number
    episode_start = time.time()  # time the episode started
    episode = 1  # episode number
    episode_start_frame = 1  # the frame number this episode started at
    reward_tracker = RewardTracker()
    while frame < params.max_frames:
        frame += 1

        if frame < params.replay_start_size:
            epsilon = 1.0
        else:
            # decay epsilon over time
            d = (frame - params.replay_start_size) / (params.eps_decay_last_frame - params.replay_start_size)
            epsilon = max(params.eps_final, params.eps_start - d)

        done = agent.step(net, epsilon, device)

        if done:
            reward_tracker.append(agent.total_reward)
            # Log metrics
            writer.add_scalar("reward", agent.total_reward, frame)
            writer.add_scalar("mean_reward_running_avg", reward_tracker.mean_last_n_rewards(100), frame)
            writer.add_scalar("epsilon", epsilon, frame)
            writer.add_scalar("loss", loss, frame)
            writer.add_scalar("episode", episode, frame)

            print(
                f"[e{episode:03d}] mean_reward {reward_tracker.mean_last_n_rewards(100):.3}  "
                f"frame: {frame}  "
                f"frames/s: {int((frame - episode_start_frame) / (time.time() - episode_start))}  "
                f"eps: {epsilon:.2}  "
                f"loss: {float(loss):.2}"
            )

            # next episode
            agent.reset()  # also resets the env
            episode_start = time.time()
            episode_start_frame = frame
            episode += 1

        # if initializing, gain at least replay_start_size experiences first
        if len(agent.experience_buffer) < params.replay_start_size:
            continue

        # sync target network every sync_target_frames (target net is required for optimization stability)
        if frame % params.sync_target_net_frames == 0:
            tgt_net.load_state_dict(OrderedDict(net.state_dict()))

        # train on a random batch of experience
        batch = agent.experience_buffer.sample(params.batch_size)
        optimizer.zero_grad()
        loss = compute_loss(net, tgt_net, params.gamma, batch, device)
        loss.backward()
        optimizer.step()

        # save if the best model
        if reward_tracker.last_reward_was_best() and episode_start_frame == frame:
            print(f"New best_mean_reward {reward_tracker.mean_last_n_rewards(100):.2}, model saved to {model_fname}")

        # exit if we're done
        if reward_tracker.should_early_stop(params.early_stopping_patience):
            print(f"Done! mean_reward={reward_tracker.mean_last_n_rewards(100)}")
            break

    return net, reward_tracker.mean_last_n_rewards(100)


if __name__ == "__main__":
    if __name__ == "__main__":
        p = ArgumentParser()
        # Add all the parameters from the Params class to argparse
        DQNParams.add_to_argparser(p)
        args = p.parse_args()
        train(DQNParams.from_parsed_args(args))
