from dataclasses import dataclass
import time
import retro #type:ignore
import tyro
import MMX
from MMX.agents import PPO_Agent

from retro.scripts.playback_movie import main as render_video #type: ignore

@dataclass
class Args:
    stage: str = "intro_stage"
    """which stage to train on"""

    agent: str = "agent"

if __name__ == "__main__":

    args = tyro.cli(Args)

    stage = args.stage
    agent_name = args.agent

    env = MMX.make_mmx_env("test", stage = stage)
    agent = PPO_Agent(env, None)


    agent.load_from_name(agent_name + ".pt")

    state, _ = env.reset()

    added = 0
    
    for i in range(100000):
        action = agent.get_policy_action(state)
        state, _, term, trunc, info = env.step(action)

        if term or trunc:
            added += 1

        if added == 200:
            break
    
    env.close()