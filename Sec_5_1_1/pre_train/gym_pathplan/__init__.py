from gym.envs.registration import register

register(
    id='GFPathPlan-v0',
    entry_point='gym_pathplan.envs:STL_Problem_GF',
)
