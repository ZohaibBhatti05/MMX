import retro #type: ignore

from MMX.wrappers import MegamanXObservationWrapper, MegamanXActionSpaceWrapper

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


valid_stages = {
    "intro_stage" : "IntroStage",
    "sting_chameleon" : "StingChameleon"
}


def make_single_env(render_mode = "human"):
    env = retro.make(
        game='MegamanX-Snes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=render_mode
    )
    return env


def make_recording_env(state):

    print("states/" + state + "1.state")

    env = retro.make(
        game='MegamanX-Snes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode="human",
        record = '.',
        state = "states/" + state + "1.state",
    )
    env = MegamanXActionSpaceWrapper(env)
    env = MegamanXObservationWrapper(env)
    return env


# create an environment to be trained on or tested on
def make_mmx_env(
    type: str = "train",
    stage: str = "sting_chameleon",
    checkpointed: bool = True,
    num_envs: int = 16,
):
    assert stage in valid_stages.keys(), "state not valid, pick one of: " + " ".join(str(s) for s in valid_stages.keys())

    state = valid_stages[stage]

    # training environment
    if type == "train":
        assert num_envs > 0, "number of environments must be greater than 0!"
        envs = SubprocVecEnv(
            [make_env_wrapped_func(state, checkpointed, i) for i in range(num_envs)]
        )
        return envs

    # testing environment
    elif type == "test":
        env = make_recording_env(state=state)
        return env
    
    else:
        raise Exception("environment type must be one of: train, test")


def make_env_wrapped_func(state, checkpointed, i):

    def func():
        # get which save state to load
        state_index = i % 3 if checkpointed else 0
        state_index += 1

        # get the specific state for this environment
        current_state = state + str(state_index)

        # create the environment
        env = retro.make(
            game='MegamanX-Snes',
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            render_mode="rgb_array",

            state = "states/" + current_state + ".state",
        )

        # wrap environment
        env = MegamanXActionSpaceWrapper(env)
        env = MegamanXObservationWrapper(env)

        return env
    return func