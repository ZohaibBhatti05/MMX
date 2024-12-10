import retro #type: ignore

from MMX.wrappers import MegamanXObservationWrapper, MegamanXActionSpaceWrapper

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


valid_stages = {
    "intro_stage" : "IntroStage",
    "sting_chameleon" : "StingChameleon",
    "sc_boss" : "StingChameleon3",
}


def make_single_env(render_mode = "human"):
    env = retro.make(
        game='MegamanX-Snes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=render_mode
    )
    return env


def make_recording_env(state):
    env = retro.make(
        game='MegamanX-Snes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode="human",
        record = '.',
        state = "states/" + state + "1.state",
    )
    intro = (state == "IntroStage")
    env = MegamanXActionSpaceWrapper(env, simple=intro)
    env = MegamanXObservationWrapper(env, frameskip_prob=0) # dont frameskip when testing agent
    env.num_envs = 1    # set this here so stable-retro doesnt warn about wrappers
    return env


# create an environment to be trained on or tested on
def make_mmx_env(
    type: str = "train",
    stage: str = "sting_chameleon",
    checkpointed: bool = True,
    num_envs: int = 16,
):
    assert stage in valid_stages.keys(), "state not valid, pick one of: " + ", ".join(str(s) for s in valid_stages.keys())

    state = valid_stages[stage]

    # training environment
    if type == "train":
        assert num_envs > 0, "number of environments must be greater than 0!"
        envs = SubprocVecEnv(
            [make_env_wrapped_func(state, checkpointed, i, stage) for i in range(num_envs)]
        )
        return envs

    # testing environment
    elif type == "test":
        env = make_recording_env(state=state)
        return env
    
    else:
        raise Exception("environment type must be one of: train, test")


def make_env_wrapped_func(state, checkpointed, i, scen):

    def func():
        # get which save state to load
        state_index = i % 3 if checkpointed else 0
        state_index += 1

        # get the specific state for this environment
        current_state = state + str(state_index)

        # speecial case: training against just the boss
        if state == "StingChameleon3":
            current_state = state
        

        # create the environment
        env = retro.make(
            game='MegamanX-Snes',
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            render_mode="rgb_array",
            state = "states/" + current_state + ".state",
            scenario = "scenarios/" + scen
        )

        # wrap environment
        intro = (scen == "intro_stage")
        env = MegamanXActionSpaceWrapper(env, simple=intro)
        env = MegamanXObservationWrapper(env)

        return env
    return func