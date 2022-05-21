# coding: UTF-8
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__) + '/../')) # Add the path

import numpy as np
import torch
import random

import fixed_seed # the module that fixes the seed 

import gym 
gym.logger.set_level(40) # Hide warning log

import gym_pathplan # import Self-made module

import trainer
import lagrangian_sac

def main():
    ENV_ID = 'GFPathPlan-v0'

    SEED =  0 # set SEED 0, 1, 2, ..., 9
    
    NUM_STEPS = 6 * 10 ** 5
    EVAL_INTERVAL = 10 ** 4
    NUM_EVAL_EPISODES = 100

    env = gym.make(ENV_ID)
    env_test = gym.make(ENV_ID)

    fixed_seed.fixed_seed_function(SEED)
    
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    env_test.seed(2**31-SEED)
    env_test.action_space.seed(2**31-SEED)
    env_test.observation_space.seed(2**31-SEED)

    print(env.observation_space.shape) # Original system state
    print(env.extended_state_space.shape) # Pre-processed state

    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    LEARNING_RATE_ENT = 3e-4
    LEARNING_RATE_KAPPA = 1e-5
    REPLAY_BUFFER_SIZE = 10**5
    TAU = 0.01
    REWARD_SCALE = 1.0
    AUTO_COEF = True

    # Proposed method parameter
    THRESHOLD = -40.
    #PRETRAIN_STEPS = 300000 # with pretrain
    PRETRAIN_STEPS = 0 # without pretrain

    algo = lagrangian_sac.LagrangianSAC(
        state_shape=env.extended_state_space.shape,
        action_shape=env.action_space.shape,
        seed=SEED,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr_actor=LEARNING_RATE_ACTOR,
        lr_critic=LEARNING_RATE_CRITIC,
        lr_entropy=LEARNING_RATE_ENT,
        lr_kappa = LEARNING_RATE_KAPPA,
        threshold = THRESHOLD,
        pretrain_steps = PRETRAIN_STEPS, 
        replay_size=REPLAY_BUFFER_SIZE,
        tau=TAU,
        reward_scale=REWARD_SCALE,
        auto_coef=AUTO_COEF,
    )

    Lagrangian_SAC_trainer = trainer.Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        seed=SEED,
        num_steps=NUM_STEPS,
        eval_interval=EVAL_INTERVAL,
        num_eval_episodes=NUM_EVAL_EPISODES,
    )

    Lagrangian_SAC_trainer.train() # training
    Lagrangian_SAC_trainer.save_result() # saving result
    
    env.close()
    env_test.close()

if __name__ == "__main__":
    main()
