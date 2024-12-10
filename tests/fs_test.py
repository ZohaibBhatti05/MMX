import time
import numpy
import MMX
import MMX.wrappers as wraps
from MMX.utils import Renderer


if __name__ == "__main__":

    envs = MMX.make_multiple_envs_wrapped(num_envs=16)

    renderer = Renderer(envs, 4, 4)

    envs.reset()

    try:
        start = time.time()
        for i in range(5_000):
            actions = numpy.array([envs.action_space.sample() for i in range(16)])
            envs.step(actions)

            renderer.render_direct(envs.get_images())

        end = time.time() - start

        print(end)

        print(5 * 16 * 1000/end)

    finally:
        envs.close()
        renderer.close_display()