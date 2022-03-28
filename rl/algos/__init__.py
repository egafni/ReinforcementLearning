# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="ContinuousCartPole-v0", entry_point="algos.algos.envs.continuous_cartpole:ContinuousCartPole")