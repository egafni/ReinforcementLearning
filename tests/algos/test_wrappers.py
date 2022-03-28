import gym

from rl.algos.wrappers import BufferWrapper2


def test_buffer_wrapper2():
    env = gym.make("Sin-v0", do_offset=True, n_cycles=4, n_steps=50)
    env.seed(40294064)
    assert env.observation_space.shape == (4,)

    o, r, d, i = env.step(env.action_space.sample())
    assert o.shape == (4,)

    env2 = BufferWrapper2(env, n_steps=10)
    assert env2.observation_space.shape == (10, 4), "first dimension should be equal to n_steps"
    env2.reset()
    o, r, d, i = env2.step(env2.action_space.sample())
    assert o.shape == (10, 4)
