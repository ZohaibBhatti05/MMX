import numpy as np
from MMX import make_mmx_env, MegamanXObservationWrapper, MegamanXActionSpaceWrapper


from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage


import pygame
import numpy as np

num_envs = 16

def display_images(data):

    screen.fill((0,0,0))

    for i in range(0, 4):
        for j in range(0, 4):
            surf = pygame.surfarray.make_surface(data[(i * 4) + j])
            surf = pygame.transform.rotate(surf, 90)
            surf = pygame.transform.flip(surf, False, True)

            x = 256 * i
            y = 224 * j

            screen.blit(surf, (x, y))

    pygame.display.flip()


if __name__ == "__main__":


    def make_env():
        env = make_mmx_env(render_mode="rgb_array")
        env = MegamanXObservationWrapper(env)
        env = MegamanXActionSpaceWrapper(env)
        return env

    venv = SubprocVecEnv(num_envs * [make_env])
    venv.reset()

    pygame.init()
    screen = pygame.display.set_mode((1024, 896))


    for timestep in range(10000):

        actions = np.array([venv.action_space.sample() for i in range(num_envs)])

        venv.step(actions)

        img = venv.get_images()

        display_images(img)

    venv.close()

    pygame.quit()

