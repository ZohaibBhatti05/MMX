"""
Redefine the action space of the environment to button combos actually used in the game.

Adapted from https://github.com/Farama-Foundation/stable-retro/blob/master/retro/examples/discretizer.py
"""

import gymnasium as gym
import numpy as np

import retro #type: ignore

simple_combos = [
    [], # NOOP

    ["LEFT"],
    ["LEFT", "B"],
    ["LEFT", "Y"],
    ["LEFT", "B", "Y"],
    
    ["RIGHT"],
    ["RIGHT", "B"],
    ["RIGHT", "Y"],
    ["RIGHT", "B", "Y"],

    ["B"],
    ["Y"],
    ["B", "Y"],
]

complex_combos = [
    [], # NOOP

    ["LEFT"],
    ["LEFT", "B"],
    ["LEFT", "Y"],
    ["LEFT", "B", "Y"],
    
    ["RIGHT"],
    ["RIGHT", "B"],
    ["RIGHT", "Y"],
    ["RIGHT", "B", "Y"],

    ["B"],
    ["Y"],
    ["B", "Y"],

    ["LEFT", "A"],
    ["LEFT", "B", "A"],
    ["LEFT", "Y", "A"],
    ["LEFT", "B", "Y", "A"],
    
    ["RIGHT", "A"],
    ["RIGHT", "B", "A"],
    ["RIGHT", "Y", "A"],
    ["RIGHT", "B", "Y", "A"],

    ["B", "UP"],   # used for getting out of ride armour

    # dashing without holding a direction is functionally identical to dashing with direction so is omitted
]

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()
    
class MegamanXActionSpaceWrapper(Discretizer):
    """
    Action space that only includes walking, jumping and shooting
    """

    def __init__(self, env, simple = False):

        if simple:
            combos = simple_combos
        else:
            combos = complex_combos

        super().__init__(
            env = env,
            combos = combos
        )

__all__ = ["MegamanXActionWrapper"]