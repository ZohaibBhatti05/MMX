from dataclasses import dataclass
import random
import numpy as np
import torch
import tyro

import MMX
from MMX.agents import PPO_Agent

@dataclass
class Args:
    # render arguments
    render_training: bool = True
    """whether to display the training process"""
    render_fpr: int = 10
    """Number of frames to step between each render"""

    # algorithm specific arguments
    total_timesteps: int = 20_000_000   # yeah this one takes a while unfortunately
    """total timesteps to train for"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_checkpoints: int = 50
    """number of times to save the agent whilst training"""

    steps_per_rollout: int = 1024
    """number of timesteps between policy rollouts"""
    alpha: float = 2.5e-4
    """the initial learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    anneal_learning_rate: bool = True
    """whether to decrease learning rate over time"""

    num_minibatches: int = 32
    """the number of mini-batches"""
    num_epochs: int = 2
    """the K epochs to update the policy"""
    epsilon: float = 0.2
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
    iterations_per_save: int = 0
    """computed in runtime"""

    seed: int = 1

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.steps_per_rollout)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    
    args.num_iterations = args.total_timesteps // (args.steps_per_rollout * args.num_envs)
    args.iterations_per_save = args.num_iterations // args.num_checkpoints
    args.batch_size = args.steps_per_rollout * args.num_envs
    args.minibatch_size = args.batch_size // args.num_minibatches


    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # setup envs
    envs = MMX.make_mmx_env(
        stage = "intro_stage",
        checkpointed = True,
    )

    try:
        # setup agent
        agent = PPO_Agent(envs, args)

        # train model
        agent.train()
    finally:
        envs.close()
        print("envs closed!")