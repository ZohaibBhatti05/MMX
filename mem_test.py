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

    for i in range(10):
        a = env.action_space.sample()
        s, r, _, _, _ = env.step(a)

        print(s[2989])
        print(s[2990])

        print(2**8 * s[2990] + s[2989]) # 2nd byte THEN first


    env.close()