import gymnasium
import numpy as np
from numba import njit


class MegamanXMemoryWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self.observation_space = gymnasium.spaces.Box(0, 255, (68,))
        self.shape = (68,)

    def reset(self, seed : int = None, options = None):
        _, info = self.env.reset()
        return np.empty(self.shape), info
    
    def step(self, action):
        
        ram, reward, terminated, truncated, info = self.env.step(action)

        data = extract_memory(ram)

        # data = np.zeros((68,))

        return data, reward, terminated, truncated, info

@njit
def two_byte(ram, index):
    offset = 256
    byte1_index = offset + 1
    return offset * ram[byte1_index] + ram[index]

@njit
def extract_memory(ram):
        
    num_entities = 8

    # addresses
    # i_health = 0x0bcf

    i_X_posX = 0x0bad
    i_X_posY = 0x0bb0

    i_X_velX = 0x0bc2
    i_X_velY = 0x0bc4

    # i_X_shot_data = 0x1228
    i_enemy_data = 0x0e68
    i_enemy_shot_data = 0x1428

    i_offset_posX = 0x5
    i_offset_posY = 0x8
    i_offset_velX = 0x1a
    i_offset_velY = 0x1c

#region extracting from memory
    data = np.empty((68,))

    # player
    data[0] = two_byte(ram, i_X_posX)
    data[1] = two_byte(ram, i_X_posY)
    data[2] = two_byte(ram, i_X_velX)
    data[3] = two_byte(ram, i_X_velY)

    index = 4

    # enemy data
    for i in range(num_entities):
        entity_offset = 64 * i

        # enemy i
        if ram[entity_offset + i_enemy_data]:
            data[index] = two_byte(ram, entity_offset + i_enemy_data + i_offset_posX)
            data[index + 1] = two_byte(ram, entity_offset + i_enemy_data + i_offset_posY)
            data[index + 2] = two_byte(ram, entity_offset + i_enemy_data + i_offset_velX)
            data[index + 3] = two_byte(ram, entity_offset + i_enemy_data + i_offset_velY)
        else:
            data[index] = 0
            data[index + 1] = 0
            data[index + 2] = 0
            data[index + 3] = 0

        # enemy shot i 
        if ram[entity_offset + i_enemy_shot_data]:
            data[index + 4] = two_byte(ram, entity_offset + i_enemy_shot_data + i_offset_posX)
            data[index + 5] = two_byte(ram, entity_offset + i_enemy_shot_data + i_offset_posY)
            data[index + 6] = two_byte(ram, entity_offset + i_enemy_shot_data + i_offset_velX)
            data[index + 7] = two_byte(ram, entity_offset + i_enemy_shot_data + i_offset_velY)
        else:
            data[index + 4] = 0
            data[index + 5] = 0
            data[index + 6] = 0
            data[index + 7] = 0
        
        index += 8
#endregion
    return data
        
