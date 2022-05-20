from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from datetime import timedelta
import numpy as np
import torch
import pandas as pd

class Trainer:

    def __init__(self, env, env_test, algo, seed=0, num_steps=6*10**5, eval_interval=10**4, num_eval_episodes=100):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        # Dictionary for results．
        self.returns = {'step': [], 'return': [], 'stl_return': [], 'success_rate':[], 'alpha':[], 'kappa':[]}

        # Num of learning steps
        self.num_steps = num_steps
        # Interval for evaluation
        self.eval_interval = eval_interval
        # Num of episodes for one evaluation．
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        """ Repeat collecting experience data, learning, and evaluating for num_steps. """

        # Stock states sampled from the intial state destribution for initial state Buffer
        for i in range(2000):
            init_state = self.env.reset()
            self.algo.init_state_stock(init_state)
        #print("Num of Data in Init State Buffer "+str(self.algo.init_state_buffer._n))

        self.start_time = time()

        # counter for steps in the episode．
        t = 0

        # initialize env．
        state = self.env.reset()

        # initialize processing time 
        self.processing_time = 0.0

        for steps in range(1, self.num_steps + 1):

            before_processing = time() ##################
            
            state, t = self.algo.step(self.env, state, t, steps) # exploration

            # Update．
            if self.algo.is_update(steps):
                self.algo.update(steps)
            after_processing = time() ##################
            self.processing_time += after_processing - before_processing

            # evaluation．
            if steps % self.eval_interval == 0: # 10000 steps
                self.evaluate(steps)

        print(f'Processing Time: {str(timedelta(seconds=int(self.processing_time)))}')
        #self.save_gif()
    
    #def save_gif(self):
    #    images = []
    #    state = self.env_test.reset()
    #    done = False

    #    while(not done):
    #        images.append(self.env_test.render(mode='rgb_array'))
    #        action = self.algo.exploit(state)
    #        state, reward, stl_reward, done, _ = self.env_test.step(action)
    #    self.display_video(images) 

    #def display_video(self, frames):
    #    #plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    #    plt.figure(figsize=(8, 8), dpi=50)
    #    patch = plt.imshow(frames[0])
    #    plt.axis('off')

    #    def animate(i):
    #        patch.set_data(frames[i])

    #    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    #    anim.save('env.gif', writer='PillowWriter')  

    def evaluate(self, steps):

        returns = [] # sum of discounted rewards 
        stl_returns = [] # sum of discounted stl-rewards
        evaluates = [] # for compute success rate
        GAMMA = 0.99 # discount factor

        for _ in range(self.num_eval_episodes): # 100 times evaluation
            evaluate_val = 1.0
            state = self.env_test.reset()
            eval_temp = self.env_test.evaluate_stl_formula()
            evaluate_val = min(evaluate_val, eval_temp) # \Phi = G\phi
            done = False

            episode_return = 0.0 # rewards
            episode_stl_return = 0.0 # stl rewards
            gamma_count = 0

            while (not done):
                #self.env_test.render() 
                action = self.algo.exploit(state)
                state, reward, stl_reward, done, _ = self.env_test.step(action)
                eval_temp = self.env_test.evaluate_stl_formula()
                evaluate_val = min(evaluate_val, eval_temp) # \Phi = G\phi
                episode_return += (GAMMA**(gamma_count)) * reward
                episode_stl_return += (GAMMA**(gamma_count))  * stl_reward
                gamma_count += 1

            evaluates.append(evaluate_val)
            returns.append(episode_return)
            stl_returns.append(episode_stl_return)

        numpy_alpha = self.algo.alpha.cpu().detach().numpy()  # Entropy Temp
        #numpy_alpha = self.algo.alpha  
        numpy_kappa = self.algo.kappa.cpu().detach().numpy() # Lagrangian multiplier

        mean_return = np.mean(returns)
        mean_stl_return = np.mean(stl_returns)
        success_rate = np.mean(evaluates)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)
        self.returns['stl_return'].append(mean_stl_return)
        self.returns['success_rate'].append(success_rate)
        self.returns['alpha'].append(numpy_alpha)
        self.returns['kappa'].append(numpy_kappa)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'STL Return: {mean_stl_return:<5.1f}   '
              f'Success Rate: {success_rate:<5.2f}   '
              f'Alpha: {numpy_alpha:<5.4f}   '
              f'Kappa: {numpy_kappa:<5.4f}   '
              f'Time: {str(timedelta(seconds=int(self.processing_time)))}') # self.time -> self.processing_time
        if steps % 10000 == 0:    
            self.algo.backup_model(steps)

    def save_result(self):

        datasets = pd.DataFrame(self.returns['return'])
        datasets.to_csv('rewards.csv', mode='w')
        datasets = pd.DataFrame(self.returns['stl_return'])
        datasets.to_csv('stl_rewards.csv', mode='w')
        datasets = pd.DataFrame(self.returns['success_rate'])
        datasets.to_csv('success.csv', mode='w')
        datasets = pd.DataFrame(self.returns['alpha'])
        datasets.to_csv('alpha.csv', mode='w')
        datasets = pd.DataFrame(self.returns['kappa'])
        datasets.to_csv('kappa.csv', mode='w')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))

class Algorithm(ABC):

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def is_update(self, steps):
        pass

    @abstractmethod
    def step(self, env, state, t, steps):
        pass

    @abstractmethod
    def update(self, steps):
        pass

    @abstractmethod
    def backup_model(self, steps):
        pass