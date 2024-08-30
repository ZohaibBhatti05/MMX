class Agent:
    """Base class for an agent"""

    def __init__(self, envs):
        self.action_space = envs.action_space
        self.num_envs = envs.num_envs

    def get_policy_action(self, state):
        pass

