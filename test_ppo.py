import retro #type:ignore
import MMX
from MMX.agents import PPO_Agent

if __name__ == "__main__":
    env = MMX.make_recording_env()
    agent = PPO_Agent(env)
    agent.load()

    state, _ = env.reset()

    added = 0
    
    for i in range(100_000):
        action = agent.get_policy_action(state)
        state, _, term, trunc, info = env.step(action)

        if term or trunc:
            added += 1

        if added == 150:
            break
    
    env.close()