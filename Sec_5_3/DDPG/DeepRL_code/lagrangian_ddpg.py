from argparse import Action
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

import replay_memory
import init_state_memory
import network
import trainer
from ounoise import OU_NOISE

class LagrangianDDPG(trainer.Algorithm):

    def __init__(self, state_shape, action_shape, device=torch.device('cuda'), seed=0,
                batch_size=64, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, lr_kappa=1e-5, threshold=-35.,
                pretrain_steps=250000, replay_size=10**6, start_steps=10**4, tau=0.01, reward_scale=1.0):
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
        self.actor = network.DDPGActor(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)

        self.actor_target = network.DDPGActor(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device).eval()

        self.reward_critic = network.DDPGCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)

        self.reward_critic_target = network.DDPGCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device).eval()

        self.STL_critic = network.DDPGCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)

        self.STL_critic_target = network.DDPGCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device).eval()

        # Lagrangian multiplier
        self.threshold = threshold # -40.0
        self.log_kappa = torch.tensor(0.0, requires_grad=True, device=device) 
        self.kappa = self.log_kappa.exp() # 1.0

        # Initialize target DNNs (reward critic and stl-reward critic)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for param in self.actor_target.parameters(): # Cut the grad information
            param.requires_grad = False
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
        self.optim_kappa = torch.optim.Adam([self.log_kappa], lr=lr_kappa) 

        # OU NOISE
        self.ounoise = OU_NOISE(action_shape)
        
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
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor.sample(state)
        action += torch.tensor(self.ounoise.noise(), dtype=torch.float, device=self.device)
        action = action.cpu().numpy()[0]
        action = np.clip(action, -1, 1)
        return action
    
    def exploit(self, state):
        """ Return a deterministic action (mean). """
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
            action = self.explore(state)
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
            self.ounoise.reset()

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

        if steps >= self.pretrain_steps:   
            init_states = self.init_state_buffer.sample(self.batch_size) 
            self.update_kappa(init_states)

        self.update_target()    
    
    def update_reward_critic(self, states, actions, rewards, dones, next_states): # reward Critic
        curr_qs = self.reward_critic(states, actions)

        with torch.no_grad():
            #next_actions = self.actor_target.sample(next_states)
            next_actions = self.actor_target(next_states)
            next_qs = self.reward_critic_target(next_states, next_actions)
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic = (curr_qs - target_qs).pow_(2).mean()

        self.optim_reward_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_reward_critic.step()

    def update_pretrain_STL_critic(self, states, actions, stl_rewards, dones, next_states): #STL critic
        curr_stl_qs = self.STL_critic(states, actions)

        with torch.no_grad():
            #next_actions = self.actor_target.sample(next_states)
            next_actions = self.actor_target(next_states)
            next_stl_qs = self.STL_critic_target(next_states, next_actions)
        target_stl_qs = stl_rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_stl_qs

        loss_stl_critic = (curr_stl_qs - target_stl_qs).pow_(2).mean()

        self.optim_STL_critic.zero_grad()
        loss_stl_critic.backward(retain_graph=False)
        self.optim_STL_critic.step()

    def update_finetune_STL_critic(self, states, actions, stl_rewards, dones, next_states): # Same as update_pretrain_STL_critic
        curr_stl_qs = self.STL_critic(states, actions)

        with torch.no_grad():
            #next_actions = self.actor_target.sample(next_states)
            next_actions = self.actor_target(next_states)
            next_stl_qs = self.STL_critic_target(next_states, next_actions)
        target_stl_qs = stl_rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_stl_qs

        loss_stl_critic = (curr_stl_qs - target_stl_qs).pow_(2).mean()

        self.optim_STL_critic.zero_grad()
        loss_stl_critic.backward(retain_graph=False)
        self.optim_STL_critic.step()

    def update_pretrain_actor(self, states):
        #actions = self.actor.sample(states)
        actions = self.actor(states)
        STL_qs = self.STL_critic(states, actions)
        loss_actor = -STL_qs.mean() # Pretrain

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_finetune_actor(self, states):
        #actions = self.actor.sample(states)
        actions = self.actor(states)
        reward_qs = self.reward_critic(states, actions)
        STL_qs = self.STL_critic(states, actions)
        loss_actor = -(reward_qs + self.kappa * STL_qs).mean() # Finetune

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_kappa(self, states):
        #actions = self.actor.sample(states)
        actions = self.actor(states)
        STL_qs = self.STL_critic(states, actions)

        kappa_loss = self.log_kappa.exp() * ((STL_qs - self.threshold).detach()).mean()
        #kappa_loss = self.log_kappa * ((STL_qs - self.threshold).detach()).mean()

        self.optim_kappa.zero_grad()
        kappa_loss.backward()
        self.optim_kappa.step()

        self.kappa = self.log_kappa.exp()

    def update_target(self):
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)
        for t, s in zip(self.reward_critic_target.parameters(), self.reward_critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)
        for t, s in zip(self.STL_critic_target.parameters(), self.STL_critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)
    
    def backup_model(self, steps):
        #model = self.actor.to('cpu')
        torch.save(self.actor.state_dict(), 'DDPG_STL_Actor_' + str(steps) + '.pth')
        torch.save(self.reward_critic.state_dict(), 'DDPG_Reward_Critic_' + str(steps) + '.pth')
        torch.save(self.STL_critic.state_dict(), 'DDPG_STL_Critic_' + str(steps) + '.pth')

        





