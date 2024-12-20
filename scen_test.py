import numpy as np

import retro #type: ignore
import MMX

from MMX.utils import Renderer
from MMX.wrappers import MegamanXObservationWrapper


if __name__ == "__main__":

    env = retro.make(
        game='MegamanX-Snes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode="human",
        state="states/StingChameleon3",
        scenario="scenarios/sting_chameleon",
        obs_type = retro.Observations.RAM
    )

    env = MegamanXObservationWrapper(env, image = False)
    print(env.observation_space.shape)

    s, _ = env.reset()

    for i in range(1000):
        a = env.action_space.sample()
        s, r, _, _, _ = env.step(a)


    env.close()