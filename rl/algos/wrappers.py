import gym
import numpy as np


class BufferWrapper2(gym.ObservationWrapper):
    """
    adds a channel to the 0th axis.  This generalizes the atari_Wrappers.BufferWrapper to work for any shape
    """

    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            np.repeat(np.expand_dims(env.observation_space.low, 0), n_steps, axis=0),
            np.repeat(np.expand_dims(env.observation_space.high, 0), n_steps, axis=0),
            shape=[n_steps] + list(old_space.shape),
            dtype=dtype,
        )

    def reset(self, *args, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset(*args, **kwargs))

    def observation(self, observation):
        """remove first observation in buffer, move rest in queue up one, add new one to back"""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer.copy()
