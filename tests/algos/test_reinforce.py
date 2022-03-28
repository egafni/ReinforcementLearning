from rl.algos.reinforce.train import ReinforceParams, train


def test_reinforce():
    params = ReinforceParams(max_episodes=3, seed=1, cpu=True)
    net, running_reward = train(params)
    assert running_reward == 14.666666666666666
