import time
import datetime

import MMX

from .agent import Agent
from MMX.utils.renderer import Renderer

from .ppo import PPONetwork
import torch
import torch.nn as nn
import torch.optim as optim

import retro #type: ignore
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class PPO_Agent(Agent):
    """Class that trains a PPO agent on the supplied environments"""
    def __init__(self, envs):
        super().__init__(envs)

        # keep track of observation shape, action size, and keep a copy of the environments
        self.envs = envs
        self.state_shape = envs.observation_space.shape
        self.action_size = envs.action_space.n

        # get device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # init network and optimiser
        self.agent = PPONetwork(self.action_size).to(self.device)


    def init_memory(self, T, N):
        device = self.device

        self.state_memory = torch.zeros((T + 1, N) + self.state_shape, device=device)
        self.action_memory = torch.zeros((T, N), device=device)
        self.reward_memory = torch.zeros((T, N), device=device)
        self.done_memory = torch.zeros((T + 1, N), device=device)
        self.log_probability_memory = torch.zeros((T, N), device=device)
        self.value_memory = torch.zeros((T + 1, N), device=device)

    def train(self):

#region Params
        total_timesteps = 15_000_000
        self.steps_per_rollout = steps_per_rollout = 1024
        minibatch_count = 32
        num_envs = self.num_envs
        num_epochs = 2

        save_count = 20

        self.num_iterations = num_iterations = total_timesteps // (steps_per_rollout * num_envs)
        iterations_per_save = num_iterations // save_count
        self.batch_size = steps_per_rollout * num_envs
        self.minibatch_size = self.batch_size // minibatch_count

        print(f"num_iterations {num_iterations} :: batch_size {self.batch_size} :: minibatch_size {self.minibatch_size} :: iterations per save {iterations_per_save}")

        self.display_rate = 6

        self.alpha = 2.5e-4
        self.gamma = 0.99
        self.gae_lambda = 0.98
        self.epsilon = 0.2

        self.entropy_coefficient = 0.01
        self.value_function_coefficient = 0.5
        self.max_grad_norm = 0.5

        self.optimiser = optim.Adam(self.agent.parameters(), lr = self.alpha, eps = 1e-5)

#endregion

        try:

            self.load()

            # init memory
            self.init_memory(steps_per_rollout, num_envs)

            # init renderer
            self.renderer = Renderer(self.envs, 4, 4)

            # init summary writer
            self.writer = SummaryWriter()
            
            device = self.device
            self.global_step = 0
            start_time = time.time()
            print("Training start")

            # -- reset environments and store s_0, d_0
            next_state = self.envs.reset()
            next_state = torch.tensor(next_state, device=device)
            next_done = torch.zeros(num_envs, device=device)

            self.state_memory[0] = next_state
            self.done_memory[0] = next_done
            # --

            for iteration in range(1, num_iterations + 1):

                self.anneal_learning_rate(iteration)

                for timestep in range(steps_per_rollout):
                    self.global_step += num_envs
                    next_state, next_done = self.rollout(timestep, next_state)

                advantages, returns = self.compute_gae(next_state)

                # flatten data for batch
                b_states = self.state_memory.reshape((-1,) + self.state_shape)
                b_actions = self.action_memory.reshape(-1)
                b_logprobs = self.log_probability_memory.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = self.value_memory.reshape(-1)

                b_indices = np.arange(self.batch_size)

                for epoch in range(num_epochs):
                    np.random.shuffle(b_indices)
                    self.train_on_batch(b_indices, b_states, b_actions, b_logprobs, b_advantages, b_returns, b_values)

                # log data
                training_time = time.time() - start_time
                timehms = str(datetime.timedelta(seconds = training_time))
                sps = self.global_step / training_time

                print(f"SPS {sps:.2f}  :::  step {self.global_step}  :::  time {timehms}  :::  iteration {iteration} / {num_iterations}")

                # save every now and then
                if iteration % iterations_per_save == 0:
                    self.save()
                    print("saved agent!")

        finally:
            self.writer.close()
            self.renderer.close_display()
            self.save()
            print("done")



    def rollout(self, timestep, next_state):
        # action logic
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(next_state)
            value = value.flatten()
        
        next_state, reward, next_done, infos = self.envs.step(action.cpu().numpy())

        # move state and done back to gpu
        next_state = torch.tensor(next_state, device=self.device)
        next_done = torch.tensor(next_done, device=self.device)

        # store data to memory
        self.state_memory[timestep + 1] = next_state
        self.action_memory[timestep] = action
        self.reward_memory[timestep] = torch.tensor(reward, device=self.device)
        self.done_memory[timestep + 1] = next_done
        self.log_probability_memory[timestep] = logprob
        self.value_memory[timestep] = value
        #

        # render
        if timestep % self.display_rate == 0:
            self.renderer.render_direct(self.envs.get_images())

        # logging
        if next_done.any():
            for info in infos:
                if "episode" in info:
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
                    self.writer.add_scalar("charts/episodic_damage_taken", info["damage_taken"], self.global_step)
                    self.writer.add_scalar("charts/episodic_progress", info["furthest_position"], self.global_step)
                    

        return next_state, next_done
                
    def compute_gae(self, next_state):
        with torch.no_grad():
            self.value_memory[self.steps_per_rollout] = self.agent.get_value(next_state).reshape(1, -1)
            last_gae_lam = 0
            advantages = torch.zeros_like(self.value_memory, device=self.device)

            for t in reversed(range(self.steps_per_rollout)):
                delta = self.reward_memory[t] + (self.gamma * self.value_memory[t+1]) - self.value_memory[t]
                advantages[t] = last_gae_lam = delta + (self.gamma * self.gae_lambda) * (1.0 - self.done_memory[t + 1]) * last_gae_lam
            
            returns = advantages + self.value_memory
        return advantages, returns

    def train_on_batch(self, batch_indices, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns, batch_values):
        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            minibatch_indices = batch_indices[start:end]

            # get minibatch
            minibatch_states = batch_states[minibatch_indices]
            minibatch_actions = batch_actions.long()[minibatch_indices]
            minibatch_logprobs = batch_logprobs[minibatch_indices]
            minibatch_advantages = batch_advantages[minibatch_indices]
            minibatch_returns = batch_returns[minibatch_indices]
            minibatch_values = batch_values[minibatch_indices]

            _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(minibatch_states, minibatch_actions)

            # get log(p(θ)), p(θ)
            log_ratio = newlogprob - minibatch_logprobs
            ratio = log_ratio.exp()

            # normalise advantages
            minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

            # -L_CLIP
            policy_loss = torch.max(
                -minibatch_advantages * ratio,
                -minibatch_advantages * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            ).mean()

            
            newvalue = newvalue.view(-1)
            # (unclipped) L_V
            value_loss = 0.5 * ((newvalue - minibatch_returns) ** 2).mean()

            # H
            entropy_loss = entropy.mean()
            
            loss_PPO = policy_loss - (self.entropy_coefficient * entropy_loss) + (self.value_function_coefficient * value_loss)

            self.optimiser.zero_grad()
            loss_PPO.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimiser.step()

        self.writer.add_scalar("losses/value_loss", value_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", -policy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)

    # decrease learning rate over time
    def anneal_learning_rate(self, iteration):
        frac = 1.0 - (iteration - 1.0) / self.num_iterations
        lrnow = frac * self.alpha
        self.optimiser.param_groups[0]["lr"] = lrnow


    def save(self):
        torch.save(self.agent.state_dict(), "agent.pt")

    def load(self):
        # LOAD CHECKPOINT
        self.agent.load_state_dict(torch.load("agent.pt", weights_only=True))

    def get_policy_action(self, state):        
        state = torch.tensor(state, device=self.device).unsqueeze(0)

        # action logic
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(state)
     
        return action

        
        

        


