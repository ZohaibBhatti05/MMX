from .agent import Agent

class Random_Agent(Agent):
    """Agent that always returns random actions"""
    
    def __init__(self, envs) -> None:
        super().__init__(envs)
    

    def get_policy_action(self, state):
        return [self.action_space.sample() for n in range(self.num_envs)]
