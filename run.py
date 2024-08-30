from MMX import make_mmx_env
from MMX import MegamanXActionSpaceWrapper
from MMX import MegamanXObservationWrapper

env = make_mmx_env(render_mode="human")

print(env.observation_space.shape)
print(env.action_space)

env = MegamanXActionSpaceWrapper(env)

print(env.observation_space.shape)
print(env.action_space)

env.reset()
for i in range(1000):
    action = env.action_space.sample()

    obs, rew, term, trunc, info = env.step(action)

    print(obs.shape)
