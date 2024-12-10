from torch.utils.tensorboard import SummaryWriter
import numpy as np

import MMX

import MMX.wrappers
import MMX.wrappers.observation_wrappers as wraps


from MMX.utils import Renderer


if __name__ == "__main__":

    envs = MMX.make_multiple_envs_wrapped(num_envs=4)
    renderer = Renderer(envs, 2, 2, scale=2.0)
    envs.reset()

    for i in range(670):
        actions = np.array([1,2,5,0])
        renderer.render(envs.get_images())

        _, _, dones, infos = envs.step(actions)

        if dones.any():
            for info in infos:
                if "episode" in info:
                    print(info["episode"])
    
    envs.close()
    renderer.close_display()
    print("DONE!")