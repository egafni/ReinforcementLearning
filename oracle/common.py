
def make_env_oracle(env_name: str, env_kwargs: dict):
    """
    The single interface we use for creating environments to be used by our custom training scripts
    """
    if isinstance(env_name, gym.Env):
        return env_name
    elif env_name in ["CartPole-v1", "HalfCheetahBulletEnv-v0"]:
        return gym.make(env_name, **env_kwargs)
    else:
        return atari_wrappers.make_env(env_name, **env_kwargs)  # type: ignore
