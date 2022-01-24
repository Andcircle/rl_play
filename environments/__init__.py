from gym.envs.registration import register

register(
    id='Scaling-v0',
    entry_point='envionments.scaling:ScalingEnv',
    max_episode_steps=250,
    # reward_threshold=25.0,
)

register(
    id='Continuous-Scaling-v0',
    entry_point='environments.scaling:ContinuousScalingEnv',
    max_episode_steps=2000,
    # reward_threshold=25.0,
)