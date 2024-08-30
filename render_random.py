import MMX
from MMX.utils import Renderer
from MMX.agents import Random_Agent


if __name__ == "__main__":
    envs = MMX.make_multiple_envs_wrapped(render_mode = "rgb_array", num_envs = 16)

    agent = Random_Agent(envs)

    renderer = Renderer(envs, 4, 4, scale = 1.5)

    s = envs.reset()

    try:

        for t in range(1000):
            action = agent.get_policy_action(s)

            s, _, _, _ = envs.step(action)

            img_data = envs.get_images()

            renderer.render(img_data, transposed=True)

    finally:
        
        print("closed envs")
        envs.close()

