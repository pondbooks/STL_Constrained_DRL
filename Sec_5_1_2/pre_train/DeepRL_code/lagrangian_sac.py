import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

import replay_memory
import init_state_memory
import network
import trainer

class LagrangianSAC(trainer.Algorithm):

    def __init__(self, state_shape, action_shape, device=torch.device('cuda'), seed=0,
                batch_size=64, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, lr_entropy=3e-4, lr_kappa=1e-5, threshold=35.,
                pretrain_steps=250000, replay_size=10**6, start_steps=10**4, tau=0.01, alpha=1.0, kappa=1.0, reward_scale=1.0, auto_coef=False):
        super().__init__()


        # Initialize the Replay buffer
        self.buffer = replay_memory.ReplayBuffer(
            buffer_size=replay_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )

        # Initialize the Replay Buffer for initial states 
        self.init_state_buffer = init_state_memory.InitStateBuffer(
            buffer_size=replay_size,
            state_shape=state_shape,
            device=device,
        )

        # Initialize DNNs
        self.actor = network.SACActor(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)
        self.reward_critic = network.SACCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)
        self.reward_critic_target = network.SACCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device).eval()
        self.STL_critic = network.SACCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)
        self.STL_critic_target = network.SACCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device).eval()

        # Flag for auto_coef
        self.auto_coef = auto_coef

        self.alpha = alpha 
        if self.auto_coef: 
            # Target Entropy = −dim(A) https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
            self.target_entropy = -torch.prod(torch.Tensor(action_shape).to(device)).item()
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp() # 1.0

        # Lagrangian multiplier
        self.threshold = threshold
        self.log_kappa = torch.tensor(0.0, requires_grad=True, device=device) # log(kappa)=1.946
        self.kappa = self.log_kappa.exp() # 1.0

        # Initialize target DNNs (reward critic and stl-reward critic)
        self.reward_critic_target.load_state_dict(self.reward_critic.state_dict())
        for param in self.reward_critic_target.parameters(): # Cut the grad information
            param.requires_grad = False
        self.STL_critic_target.load_state_dict(self.STL_critic.state_dict())
        for param in self.STL_critic_target.parameters(): # Cut the grad information
            param.requires_grad = False

        # Define the optimizer (Adam)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_reward_critic = torch.optim.Adam(self.reward_critic.parameters(), lr=lr_critic)
        self.optim_STL_critic = torch.optim.Adam(self.STL_critic.parameters(), lr=lr_critic)
        if self.auto_coef:
            self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_entropy)
        self.optim_kappa = torch.optim.Adam([self.log_kappa], lr=lr_kappa) 
        
        # other hyper-parameter
        self.learning_steps = 0 # learning counter
        self.batch_size = batch_size
        self.pretrain_steps = pretrain_steps
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.reward_scale = reward_scale

    def explore(self, state):
        """ Return a stochastic action a~\pi and its logarithmic probability density \log(\pi(a|s))． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit(self, state):
        """ Return a deterministic action (mean)． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def is_update(self, steps):
        # The agent does not learn its policy for a certain period (start_steps)．
        return steps >= max(self.start_steps, self.batch_size)
    
    def init_state_stock(self, init_state):
        # Add state data sampled from the initial state density．
        self.init_state_buffer.append(init_state)

    def step(self, env, state, t, steps): # function for exploration
        if t == 0:
            self.init_state_buffer.append(state)

        t += 1

        # For a certain period (start_steps), the agent determine an action randomly to collect diverse data．
        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            action, _ = self.explore(state)
        next_state, reward, stl_reward, done, _ = env.step(action)

        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        # Add experience to the replay buffer．
        self.buffer.append(state, action, reward, stl_reward, done_masked, next_state)

        # Reset env at the end of the episode．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, steps): # main algorithm
        self.learning_steps += 1
        states, actions, rewards, stl_rewards, dones, next_states = self.buffer.sample(self.batch_size)

        self.update_reward_critic(states, actions, rewards, dones, next_states)

        if steps == self.pretrain_steps:
            print("===== END PRETRAIN =====")
        
        if steps < self.pretrain_steps:
            self.update_pretrain_STL_critic(states, actions, stl_rewards, dones, next_states)
            self.update_pretrain_actor(states)
        else:
            self.update_finetune_STL_critic(states, actions, stl_rewards, dones, next_states)
            self.update_finetune_actor(states)

        if self.auto_coef:
            self.update_entropy_coef(states)

        if steps >= self.pretrain_steps:   
            init_states = self.init_state_buffer.sample(self.batch_size) 
            self.update_kappa(init_states)

        self.update_target()    
    
    def update_reward_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.reward_critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.reward_critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_reward_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_reward_critic.step()

    def update_pretrain_STL_critic(self, states, actions, stl_rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.STL_critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.STL_critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = stl_rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_STL_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_STL_critic.step()

    def update_finetune_STL_critic(self, states, actions, stl_rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.STL_critic(states, actions)

        with torch.no_grad():
            next_actions, _ = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.STL_critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2)
        target_qs = stl_rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_STL_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_STL_critic.step()


    def update_pretrain_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        STL_qs1, STL_qs2 = self.STL_critic(states, actions)
        loss_actor = (self.alpha * log_pis - torch.min(STL_qs1, STL_qs2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_finetune_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        reward_qs1, reward_qs2 = self.reward_critic(states, actions)
        STL_qs1, STL_qs2 = self.STL_critic(states, actions)
        loss_actor = (self.alpha * log_pis - torch.min(reward_qs1, reward_qs2) - self.kappa * (torch.min(STL_qs1, STL_qs2))).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_entropy_coef(self, states):
        _, log_pis = self.actor.sample(states)
        #alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
        alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()

        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()

        self.alpha = self.log_alpha.exp()

    def update_kappa(self, states):
        actions, _ = self.actor.sample(states)
        STL_qs1, STL_qs2 = self.STL_critic(states, actions)
        STL_qs = torch.min(STL_qs1, STL_qs2)

        #kappa_loss = self.log_kappa * ((STL_qs - self.threshold).detach()).mean()
        kappa_loss = self.log_kappa.exp() * ((STL_qs - self.threshold).detach()).mean()

        self.optim_kappa.zero_grad()
        kappa_loss.backward()
        self.optim_kappa.step()

        self.kappa = self.log_kappa.exp()

    def update_target(self):
        for t, s in zip(self.reward_critic_target.parameters(), self.reward_critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)
        for t, s in zip(self.STL_critic_target.parameters(), self.STL_critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)
    
    def backup_model(self, steps):
        #model = self.actor.to('cpu')
        torch.save(self.actor.state_dict(), 'SAC_STL_Actor_' + str(steps) + '.pth')
        torch.save(self.reward_critic.state_dict(), 'SAC_Reward_Critic_' + str(steps) + '.pth')
        torch.save(self.STL_critic.state_dict(), 'SAC_STL_Critic_' + str(steps) + '.pth')

        






