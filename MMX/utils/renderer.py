import pygame
import numpy as np

"""class used to render environments during training"""
class Renderer:

    def __init__(self,
                envs,
                envs_per_col : int, 
                envs_per_row : int, 
                scale : float = 1.0,
                obs_x : int = 256, 
                obs_y : int = 224):

        assert (envs_per_col * envs_per_row == envs.num_envs), "Renderer error: num_envs != envs_per_col * envs_per_row"

        self.num_envs = envs.num_envs
        self.envs_per_row = envs_per_row
        self.envs_per_col = envs_per_col
        
        self.env_width = obs_x * scale
        self.env_height = obs_y * scale

        self.screen_width = self.env_width * envs_per_row
        self.screen_height = self.env_height * envs_per_col

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def render(self, data, transposed = True):

        assert np.shape(data)[0] == self.num_envs, "renderer error: amount of data does not match with number of expected environments"
        
        if transposed:
            surface = pygame.Surface((self.screen_height, self.screen_width))
        else:
            surface = pygame.Surface((self.screen_width, self.screen_height))

        for i in range(self.envs_per_row):
            for j in range(self.envs_per_col):

                # get frame for one environment
                frame = pygame.surfarray.make_surface(data[(i * self.envs_per_row) + j])
                if transposed:
                    frame = pygame.transform.scale(frame, (self.env_height, self.env_width))
                    x = i * self.env_height
                    y = j * self.env_width
                else:
                    frame = pygame.transform.scale(frame, (self.env_width, self.env_height))
                    x = i * self.env_width
                    y = j * self.env_height
                
                # blit it to temp surface
                surface.blit(frame, (x, y))
        
        # rotate entire surface if transposed
        if transposed:
            surface = pygame.transform.rotate(surface, 90)
            surface = pygame.transform.flip(surface, False, True)
        
        self.screen.blit(surface, (0,0))
        pygame.display.flip()

    def render_direct(self, data):

        data = np.swapaxes(data, 1, 2)

        for i in range(self.envs_per_row):
            for j in range(self.envs_per_col):

                # get frame for one environment
                frame = pygame.surfarray.make_surface(data[(i * self.envs_per_row) + j])
                x = i * self.env_width
                y = j * self.env_height
                
                # blit it to temp surface
                self.screen.blit(frame, (x, y))
        
        pygame.display.flip()

    def close_display(self):
        pygame.quit()




        
