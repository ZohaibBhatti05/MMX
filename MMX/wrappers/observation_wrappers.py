"""
Observation wrappers for use in backend training (basically anything that directly changes the observation taken from env.step)
"""

import gymnasium
from gymnasium.wrappers import FrameStack, ResizeObservation, GrayScaleObservation, TimeLimit, RecordEpisodeStatistics
import numpy as np

class MegamanXObservationWrapper(gymnasium.Wrapper):
    """
    A wrapper that does the following (by default):

    - Resize observation to 128x128 pixels
    - Convert observation to grayscale
    - Stack 4 frames into the observation
    - Add a 10% chance to repeat each action for 4 frames
    - Record episode length and cumulative reward
    """

    def __init__(self, env, size = 128, framestack = 4, max_episode_steps = 18_000, stochastic_frame_skips = 2, frameskip_prob = 0.25):

        env = ResizeObservation(env, [size, size])
        env = GrayScaleObservation(env)
        env = FrameStack(env, framestack, True)
        env = TimeLimit(env, max_episode_steps = max_episode_steps)
        env = RecordEpisodeStatistics(env, deque_size = 10)
        env = StochasticFrameSkip(env, stochastic_frame_skips, frameskip_prob)

        super().__init__(env)


class StochasticFrameSkip(gymnasium.Wrapper):
    
    def __init__(self, env, repeat_count, repeat_prob):
        gymnasium.Wrapper.__init__(self, env)
        self.repeat_count = repeat_count
        self.repeat_prob = repeat_prob
        self.rng = np.random.RandomState()

        self.is_skipping = False
        self.current_action = None
        self.skip_count = 0
        

    def step(self, action):
        if self.rng.random() > self.repeat_prob:    # dont repeat actions
            return self.env.step(action)
        else:
            # repeat action repeat_count times
            total_reward = 0
            for i in range(self.repeat_count):
                state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break   # halt inputs if done halfway through
                    
            return state, total_reward, terminated, truncated, info

        
        




__all__ = ["MegamanXObservationWrapper"]