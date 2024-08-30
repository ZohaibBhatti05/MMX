import retro #type: ignore

from MMX.wrappers import MegamanXObservationWrapper, MegamanXActionSpaceWrapper

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

def make_single_env(render_mode = "human"):
    env = retro.make(
        game='MegamanX-Snes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=render_mode
    )
    return env

@staticmethod
def make_recording_env():
    env = retro.make(
        game='MegamanX-Snes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode="human",
        record = '.'
    )
    env = MegamanXActionSpaceWrapper(env)
    env = MegamanXObservationWrapper(env)
    return env

def make_env_wrapped_func(render_mode):
    def func():
        env = make_single_env(render_mode)
        env = MegamanXActionSpaceWrapper(env)
        env = MegamanXObservationWrapper(env)
        return env
    return func


def make_multiple_envs_wrapped(render_mode = "rgb_array", num_envs: int = 1) -> SubprocVecEnv:
    """Return a SB3 SubprocVecEnv of fully wrapped games"""

    assert num_envs > 0, "Invalid environment count!"
    if num_envs > 1:
        assert render_mode != "human", "Don't use human render mode for more than one environment! Use the Renderer instead!"

    envs = SubprocVecEnv(
        [make_env_wrapped_func(render_mode) for i in range(num_envs)]
    )
    envs.num_envs = num_envs
    return envs
