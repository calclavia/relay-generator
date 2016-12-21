from gym.envs.registration import register
from relay_env import RelayEnv

register(
    id='relay-generator-v0',
    entry_point='relay_generator:RelayEnv',
    timestep_limit=1000,
)
