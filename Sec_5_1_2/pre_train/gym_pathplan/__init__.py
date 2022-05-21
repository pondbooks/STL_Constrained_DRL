from gym.envs.registration import register

register(
    id='FGPathPlan-v0',
    entry_point='gym_pathplan.envs:STL_Problem_FG',
)