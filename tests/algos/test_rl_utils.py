from rl.algos.rl_utils import RewardTracker  # type:ignore


def test_reward_tracker():
    rt = RewardTracker()

    rewards = [1, 2, 10, 0, 3, 4]
    for i, r in enumerate(rewards):
        rt.append(r)
        assert rt.current_index == i, rt.current_index
    assert rt.max_reward == 10
    assert rt.max_reward_index == 2, rt.max_reward_index
    assert rt.current_index == 5
    assert not rt.should_early_stop(3), "patience = 3 is satisfied, since max reward was 3 steps ago"
    assert rt.should_early_stop(2), "max reward was > 2 steps ago"
