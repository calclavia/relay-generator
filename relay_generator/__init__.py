from gym.envs.registration import register
from .relay_env import RelayEnv
from .util import BlockType

register(
    id='relay-generator-v0',
    entry_point='relay_generator:RelayEnv'
)
