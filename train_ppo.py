from dataclasses import dataclass
import tyro

import MMX
from MMX.agents import PPO_Agent

@dataclass
class Args:
    # render arguments
    render_training: bool = True
    """whether to display the training process"""
    render_fpr: int = 4
    """Number of frames to step between each render"""

    # algorithm specific arguments
    total_timesteps: int = 500_000
    """total timesteps to train for"""
    num_envs: int = 16
    """the number of parallel game environments"""
    checkpoint_rate: int = 50_000
    """number of timesteps to train on between checkpoints"""

    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    learning_rate: float = 2.5e-4
    """the initial learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""

    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # setup envs
    envs = MMX.make_multiple_envs_wrapped(render_mode="rgb_array", num_envs = 16)

    try:
        # setup agent
        agent = PPO_Agent(envs)

        # train model
        agent.train()
    finally:
        envs.close()
        print("envs closed!")