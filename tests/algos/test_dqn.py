from pytorch_lightning import seed_everything

from rl.algos.dqn import train
from rl.algos.dqn.train import DQNParams


def test_dqn():
    seed_everything(1)
    net, reward_avg = train.train(
        DQNParams(
            env_name="CartPole-v1",
            model_name="DQN_Mlp",
            replay_start_size=1,
            sync_target_net_frames=2,
            batch_size=1,
            seed=1,
            max_frames=100,
            cpu=True,
        )
    )
    assert reward_avg == 19.8
