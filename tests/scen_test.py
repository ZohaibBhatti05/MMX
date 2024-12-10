import numpy as np

import retro #type: ignore
import MMX

from MMX.utils import Renderer


if __name__ == "__main__":

    env = retro.make(
        game='MegamanX-Snes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode="human",
        state="states/StingChameleon3",
        scenario="scenarios/sting_chameleon"
    )
    
    env.reset()

    for i in range(1000):
        a = env.action_space.sample()
        _, r, _, _, _ = env.step(a)

        print(r)

    env.close()