import gymnasium
import numpy as np


class MegamanXMemoryWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self.observation_space = gymnasium.spaces.Box(0, 255, (46,))
        self.shape = (46,)

    def reset(self, seed : int = None, options = None):
        _, info = self.env.reset()
        return np.empty(self.shape), info
    
    def step(self, action):
        
        _, reward, terminated, truncated, info = self.env.step(action)

        # get all relevant variables from info
        data = self.memory(info)
        return data, reward, terminated, truncated, info

    # get all relevant variables from info manually
    def memory(self, old_info):
        info = old_info.copy()

#region remove irrelevant data from dictionary

        del info["num_shots"]
        del info["X_state"]
        del info["health"]

        info.pop("furthest_position", None)
        info.pop("beat_stage", None)
        info.pop("damage_taken", None)

        # EXPERIMENTAL: IGNORING ENEMY TYPE
        del info["enemy1_id"]
        del info["enemy2_id"]
        del info["enemy3_id"]
        del info["enemy4_id"]

#endregion

#region format all useless data as 0

        # Player shots
        if info["X_shot1_exists"] < 1.0:
            info["X_shot1_posX"] = 0.0
            info["X_shot1_posY"] = 0.0
            info["X_shot1_velX"] = 0.0
        if info["X_shot2_exists"] < 1.0:
            info["X_shot2_posX"] = 0.0
            info["X_shot2_posY"] = 0.0
            info["X_shot2_velX"] = 0.0
        if info["X_shot3_exists"] < 1.0:
            info["X_shot3_posX"] = 0.0
            info["X_shot3_posY"] = 0.0
            info["X_shot3_velX"] = 0.0
        if info["X_shot4_exists"] < 1.0:
            info["X_shot4_posX"] = 0.0
            info["X_shot4_posY"] = 0.0
            info["X_shot4_velX"] = 0.0

        # Enemy shots
        if info["enemy_shot1_exists"] < 1.0:
            info["enemy_shot1_posX"] = 0.0
            info["enemy_shot1_posY"] = 0.0
            info["enemy_shot1_velX"] = 0.0
            info["enemy_shot1_velY"] = 0.0
        if info["enemy_shot2_exists"] < 1.0:
            info["enemy_shot2_posX"] = 0.0
            info["enemy_shot2_posY"] = 0.0
            info["enemy_shot2_velX"] = 0.0
            info["enemy_shot2_velY"] = 0.0
        if info["enemy_shot3_exists"] < 1.0:
            info["enemy_shot3_posX"] = 0.0
            info["enemy_shot3_posY"] = 0.0
            info["enemy_shot3_velX"] = 0.0
            info["enemy_shot3_velY"] = 0.0
        if info["enemy_shot4_exists"] < 1.0:
            info["enemy_shot4_posX"] = 0.0
            info["enemy_shot4_posY"] = 0.0
            info["enemy_shot4_velX"] = 0.0
            info["enemy_shot4_velY"] = 0.0

        # Enemies
        if info["enemy1_exists"] < 1.0:
            info["enemy1_posX"] = 0.0
            info["enemy1_posY"] = 0.0
            info["enemy1_health"] = 0.0
        if info["enemy2_exists"] < 1.0:
            info["enemy2_posX"] = 0.0
            info["enemy2_posY"] = 0.0
            info["enemy2_health"] = 0.0
        if info["enemy3_exists"] < 1.0:
            info["enemy3_posX"] = 0.0
            info["enemy3_posY"] = 0.0
            info["enemy3_health"] = 0.0
        if info["enemy4_exists"] < 1.0:
            info["enemy4_posX"] = 0.0
            info["enemy4_posY"] = 0.0
            info["enemy4_health"] = 0.0

        # remove "exists" vals from dictionary
        del info["X_shot1_exists"]
        del info["X_shot2_exists"]
        del info["X_shot3_exists"]
        del info["X_shot4_exists"]
        del info["enemy_shot1_exists"]
        del info["enemy_shot2_exists"]
        del info["enemy_shot3_exists"]
        del info["enemy_shot4_exists"]
        del info["enemy1_exists"]
        del info["enemy2_exists"]
        del info["enemy3_exists"]
        del info["enemy4_exists"]

#endregion

#region reformat certain variables

        # Player direction
        if info["direction_facing"] > 0.0:
            info["direction_facing"] = 1.0
        else:
            info["direction_facing"] = -1.0

        # Wall climb state
        if info["wall_climb_state"] == 1:   # right wall climb
            info["wall_climb_state"] = 1.0
        elif info["wall_climb_state"] == 2: # left wall climb
            info["wall_climb_state"] = -1.0
        else:
            info["wall_climb_state"] = 0.0

#endregion

        data = np.fromiter(info.values(), dtype=float)

        return data
        