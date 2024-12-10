import numpy as np
from MMX import make_mmx_env, MegamanXObservationWrapper, MegamanXActionSpaceWrapper


from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage


import numpy as np

num_envs = 16


if __name__ == "__main__":


    def make_env():
        env = make_mmx_env(render_mode="rgb_array")
        env = MegamanXObservationWrapper(env, framestack=7)
        env = MegamanXActionSpaceWrapper(env)
        return env

    venv = SubprocVecEnv(num_envs * [make_env])
    venv.reset()

    for timestep in range(1):

        actions = np.array([venv.action_space.sample() for i in range(num_envs)])

        obs, _, _, _, = venv.step(actions)

        print(np.shape(obs))

        img = venv.get_images()
        print(np.shape(img))

    venv.close()

