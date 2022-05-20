import numpy as np
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
import os

# from gym.envs.classic_control import rendering 

class STL_Problem_GF(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # The area of render is [(0.0,0.0),(0.0,max_y),(max_x,0.0),(max_x,max_y)]
        self._max_x_of_window = 5.0
        self._max_y_of_window = 5.0

        self.dt = 0.1 # The sampling period of the dynamical system.
        self._max_episode_steps = 1000

        # The time bound of subSTL
        hrz_phi_1 = 99.
        hrz_phi_2 = 99.
        hrz_phi = max(hrz_phi_1,hrz_phi_2)
        self.phi_1_timebound = [0.0, hrz_phi_1]
        self.phi_2_timebound = [0.0, hrz_phi_2]
        self.tau = int(hrz_phi + 1) # hrz(phi) + 1
        self.tau_1 = hrz_phi_1
        self.tau_2 = hrz_phi_2

        # log-sum-exp app parameter
        self.beta = 100.

        # robot param
        self.robot_radius = 0.2 #[m]

        # action param
        self.max_velocity = 1.0   # [m/s]
        self.min_velocity = -1.0  # [m/s]
        self.min_angular_velocity = math.radians(-90)  # [rad/s]
        self.max_angular_velocity = math.radians(90) # [rad/s]

        self.num_steps = 0

        # [car_x, car_y, car_yaw, flag_1, flag_2]
        self.high = np.array([np.inf, np.inf, np.pi], dtype=np.float32)
        self.low = np.array([-np.inf, -np.inf, -np.pi], dtype=np.float32)
        self.car_dim = 3
        self.low_extended_space = np.array([-np.inf, -np.inf, -np.pi, -0.5 , -0.5], dtype=np.float32)# return obs dim (tau_mdp)
        self.high_extended_space =  np.array([np.inf, np.inf, np.pi, 0.5 , 0.5], dtype=np.float32)

        # set action_space (velocity[m/s], omega[rad/s])
        self.action_low  = np.array([self.min_velocity, self.min_angular_velocity]) 
        self.action_high = np.array([self.max_velocity, self.max_angular_velocity]) 

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape =(2,), dtype=np.float32)
        self.observation_space = spaces.Box(\
            low=self.low,\
            high=self.high,\
            dtype=np.float32 
        )
        self.extended_state_space = spaces.Box(\
            low=self.low_extended_space,\
            high=self.high_extended_space,\
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.vis_lidar = True
        self.num_steps = 0 # step counter

        # ##################
        # initial state region
        self.init_low_x = 0.5
        self.init_low_y = 0.5
        self.init_high_x = 2.5
        self.init_high_y = 2.5

        # phi_1
        self.stl_1_low_x = 3.5
        self.stl_1_low_y = 3.5
        self.stl_1_high_x = 4.5
        self.stl_1_high_y = 4.5

        #phi_2
        self.stl_2_low_x = 3.5
        self.stl_2_low_y = 1.5
        self.stl_2_high_x = 4.5
        self.stl_2_high_y = 2.5

        # better region
        self.better_region_low_x = 0.5
        self.better_region_low_y = 0.5
        self.better_region_high_x = 4.5
        self.better_region_high_y = 4.5
        # ##################

    def reset(self): 
        # initial state [x(m), y(m), yaw(rad)]
        init_x = self.np_random.uniform(low=self.init_low_x, high=self.init_high_x)
        init_y = self.np_random.uniform(low=self.init_low_y, high=self.init_high_y)
        init_rad = self.np_random.uniform(low=-np.pi/2, high=np.pi/2)

        self.past_tau_trajectory = [] # past trajectory memory [s_{t-tau+1},s_{t-tau+2},...,s_{t}]
        for i in range(self.tau): # The length is tau
            current_state = np.array([init_x, init_y, init_rad])
            self.past_tau_trajectory.append(current_state)

        self.num_steps = 0 # counter of exploration steps

        self.state = np.array([init_x, init_y, init_rad]) # the current state of the car

        self.observation = self.preprocess(self.past_tau_trajectory) # preprocessing
        
        self.done = False
        self.success_val = 0.0

        return self.observation

    def step(self, action): 

        stl_reward = self.STLreward(self.past_tau_trajectory) # stl reward
        reward = self.reward(self.state, action) # low-level reward

        noise_w0 = 0.1*np.random.normal(0,1) 
        noise_w1 = 0.1*np.random.normal(0,1)
        noise_w2 = 0.1*np.random.normal(0,1)

        # next step ================================================
        self.state[0] += (action[0] * math.cos(self.state[2]) + noise_w0) * self.dt
        self.state[1] += (action[0] * math.sin(self.state[2]) + noise_w1) * self.dt
        self.state[2] += (action[1] + noise_w2) * self.dt 

        if self.state[2] < -np.pi:
            #self.state[2] += math.pi * 2.0 # Original Code
            self.state[2] += np.pi * 2.0
        elif math.pi < self.state[2]:
            #self.state[2] -= math.pi * 2.0 # Original Code
            self.state[2] -= np.pi * 2.0
        #======================================================================

        self.past_tau_trajectory = self.past_tau_trajectory[1:]
        self.past_tau_trajectory.append(self.state.copy())

        self.observation = self.preprocess(self.past_tau_trajectory) # return to the agent
        
        self.num_steps += 1

        if self.num_steps == self._max_episode_steps:
            return_done = True
            self.reset() 
        else:
            return_done = False

        return self.observation, reward, stl_reward, return_done, {}

    def preprocess(self, tau_state): # return pre-processed tau_mdp state
        tau_num = len(tau_state)
        assert tau_num == self.tau, "dim of tau-state is wrong."
        obs = np.zeros(self.car_dim + 2)
        obs[0] = tau_state[tau_num-1][0] - (self._max_x_of_window/2)
        obs[1] = tau_state[tau_num-1][1] - (self._max_y_of_window/2)
        obs[2] = tau_state[tau_num-1][2]

        f1 = 0.0
        f2 = 0.0 
        for i in range(tau_num):
            if self.subSTL_1_robustness(tau_state[i]) >= 0:
                f1 = 1.0
            else:
                f1 = max(f1 - 1/(float(self.tau_1 + 1)), 0.0)
            if self.subSTL_2_robustness(tau_state[i]) >= 0:
                f2 = 1.0
            else:
                f2 = max(f2 - 1/(float(self.tau_2 + 1)), 0.0)
        obs[3] = f1 - 0.5
        obs[4] = f2 - 0.5
        return obs

    def STLreward(self, tau_state): 
        tau_num = len(tau_state)
        phi_1_rob = -500. # Tentative value
        phi_2_rob = -500.

        for i in range(tau_num):
            temp_1_rob = self.subSTL_1_robustness(tau_state[i])
            temp_2_rob = self.subSTL_2_robustness(tau_state[i])
            phi_1_rob = max(phi_1_rob, temp_1_rob)
            phi_2_rob = max(phi_2_rob, temp_2_rob)

        return_val = min(phi_1_rob, phi_2_rob) # GF

        # indicator I(x)
        if return_val >= 0:
            return_val = 1.0
        else:
            return_val = 0.0

        stl_reward = -np.exp(-self.beta*return_val)

        return stl_reward

    def reward(self, state, action):
        #print(action)
        return_val = - action[0]**2
        return_val += - action[1]**2
        return_val += self.better_region(state)
        return return_val

    def evaluate_stl_formula(self): # called by trainer to evaluate satisfying the specification
        if self.num_steps >= self.tau-1: # after 99 steps.
            tau_num = len(self.past_tau_trajectory)
            phi_1_rob = -500. # Tentative value
            phi_2_rob = -500.

            for i in range(tau_num):
                temp_1_rob = self.subSTL_1_robustness(self.past_tau_trajectory[i])
                temp_2_rob = self.subSTL_2_robustness(self.past_tau_trajectory[i])
                phi_1_rob = max(phi_1_rob, temp_1_rob)
                phi_2_rob = max(phi_2_rob, temp_2_rob)

            returns = min(phi_1_rob, phi_2_rob) # GF

            if returns >= 0:
                returns = 1.0
            else:
                returns = 0.0

        else: # t < tau-1 
            returns = 1.0 

        return returns

    def subSTL_1_robustness(self, state):
        psi1 = state[0] - self.stl_1_low_x # 3.5 < x < 4.5
        psi2 = self.stl_1_high_x - state[0]
        psi3 = state[1] - self.stl_1_low_y # 3.5 < y < 4.5
        psi4 = self.stl_1_high_y - state[1]
        robustness = min(psi1, psi2)
        robustness = min(robustness, psi3)
        robustness = min(robustness, psi4)
        return robustness
    
    def subSTL_2_robustness(self, state):
        psi1 = state[0] - self.stl_2_low_x # 3.5 < x < 4.5
        psi2 = self.stl_2_high_x - state[0]
        psi3 = state[1] - self.stl_2_low_y # 1.5 < y < 2.5
        psi4 = self.stl_2_high_y - state[1]
        robustness = min(psi1, psi2)
        robustness = min(robustness, psi3)
        robustness = min(robustness, psi4)
        return robustness

    def better_region(self,state):
        psi1 = state[0] - self.better_region_low_x  # 0.5 < x < 4.5
        psi2 = self.better_region_high_x - state[0]
        psi3 = state[1] - self.better_region_low_y # 0.5 < y < 4.5
        psi4 = self.better_region_high_y - state[1]
        robustness = min(psi1, psi2)
        robustness = min(robustness, psi3)
        robustness = min(robustness, psi4)
        robustness = min(robustness, 0.0)
        return robustness

    ''' def render(self, mode='human', close=False): # 環境を可視化する
        screen_width  = 300
        screen_height = 300

        rate_x = screen_width / self._max_x_of_window
        rate_y = screen_height / self._max_y_of_window 

        rate_init_l = self.init_low_x / self._max_x_of_window
        rate_init_r = self.init_high_x / self._max_x_of_window
        rate_init_t = self.init_high_y / self._max_y_of_window
        rate_init_b = self.init_low_y / self._max_y_of_window
        rate_stl_1_l = self.stl_1_low_x / self._max_x_of_window
        rate_stl_1_r = self.stl_1_high_x / self._max_x_of_window
        rate_stl_1_t = self.stl_1_high_y / self._max_y_of_window
        rate_stl_1_b = self.stl_1_low_y / self._max_y_of_window
        rate_stl_2_l = self.stl_2_low_x / self._max_x_of_window
        rate_stl_2_r = self.stl_2_high_x / self._max_x_of_window
        rate_stl_2_t = self.stl_2_high_y / self._max_y_of_window
        rate_stl_2_b = self.stl_2_low_y / self._max_y_of_window
        rate_better_region_l = self.better_region_low_x / self._max_x_of_window
        rate_better_region_r = self.better_region_high_x / self._max_x_of_window
        rate_better_region_t = self.better_region_high_y / self._max_y_of_window
        rate_better_region_b = self.better_region_low_y / self._max_y_of_window

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)


            # base
            b1_l, b1_r, b1_t, b1_b = rate_better_region_l*screen_width, rate_better_region_r*screen_width, rate_better_region_t*screen_height, rate_better_region_b*screen_height  # screen のサイズで与える．
            self.b1 = [(b1_l,b1_b), (b1_l,b1_t), (b1_r,b1_t), (b1_r,b1_b)]
            b1 = rendering.make_polygon(self.b1)
            self.b1trans = rendering.Transform()
            b1.add_attr(self.b1trans)
            b1.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(b1)

            # start
            start_l, start_r, start_t, start_b = rate_init_l*screen_width, rate_init_r*screen_width, rate_init_t*screen_height, rate_init_b*screen_height  # screen のサイズで与える．
            self.start_area = [(start_l,start_b), (start_l,start_t), (start_r,start_t), (start_r,start_b)]
            start = rendering.make_polygon(self.start_area)
            self.start_area_trans = rendering.Transform()
            start.add_attr(self.start_area_trans)
            start.set_color(0.8, 0.5, 0.5)
            self.viewer.add_geom(start)

            # goal_1
            g1_l, g1_r, g1_t, g1_b = rate_stl_1_l*screen_width, rate_stl_1_r*screen_width, rate_stl_1_t*screen_height, rate_stl_1_b*screen_height  # screen のサイズで与える．
            self.v1 = [(g1_l,g1_b), (g1_l,g1_t), (g1_r,g1_t), (g1_r,g1_b)]
            g1 = rendering.make_polygon(self.v1)
            self.g1trans = rendering.Transform()
            g1.add_attr(self.g1trans)
            g1.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(g1)

            # goal_2
            g2_l, g2_r, g2_t, g2_b = rate_stl_2_l*screen_width, rate_stl_2_r*screen_width, rate_stl_2_t*screen_height, rate_stl_2_b*screen_height  # screen のサイズで与える．
            self.v2 = [(g2_l,g2_b), (g2_l,g2_t), (g2_r,g2_t), (g2_r,g2_b)]
            g2 = rendering.make_polygon(self.v2)
            self.g2trans = rendering.Transform()
            g2.add_attr(self.g2trans)
            g2.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(g2)

            head_x = (self.robot_radius) * rate_x
            head_y = 0.0 * rate_y
            tail_left_x = (self.robot_radius*np.cos((5/6)+np.pi)) * rate_x
            tail_left_y = (self.robot_radius*np.sin((5/6)+np.pi)) * rate_y
            tail_right_x = (self.robot_radius*np.cos(-(5/6)+np.pi)) * rate_x
            tail_right_y = (self.robot_radius*np.sin(-(5/6)+np.pi)) * rate_y
            self.car_v = [(head_x,head_y), (tail_left_x,tail_left_y), (tail_right_x,tail_right_y)]
            car = rendering.FilledPolygon(self.car_v)
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            car.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(car)

        car_x = self.state[0] * rate_x
        car_y = self.state[1] * rate_y
   

        self.cartrans.set_translation(car_x, car_y)
        self.cartrans.set_rotation(self.state[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array') '''

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

