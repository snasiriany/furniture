""" Define all environments and provide helper functions to load environments. """

from furniture.env.base import make_env, make_vec_env

# register all environment to use
from .furniture_baxter import FurnitureBaxterEnv
from .furniture_sawyer import FurnitureSawyerEnv
from .furniture_cursor import FurnitureCursorEnv
from .furniture_baxter_block import FurnitureBaxterBlockEnv

from furniture.env.furniture_cursor_rl import FurnitureCursorRLEnv

# OpenAI gym interface
from gym.envs.registration import register


# # add cursor environment to Gym
# register(
#     id='furniture-cursor-v0',
#     entry_point='env.furniture_gym:FurnitureGym',
#     kwargs={'name': 'FurnitureCursorEnv',
#             'furniture_id': 0,
#             'background': 'Lab',
#             'port': 1050}
# )
#
#
# # add sawyer environment to Gym
# register(
#     id='furniture-sawyer-v0',
#     entry_point='env.furniture_gym:FurnitureGym',
#     kwargs={'name': 'FurnitureSawyerEnv',
#             'furniture_name': 'swivel_chair_0700',
#             'background': 'Industrial',
#             'port': 1050}
# )
#
#
# # add baxter environment to Gym
# register(
#     id='furniture-baxter-v0',
#     entry_point='env.furniture_gym:FurnitureGym',
#     kwargs={'name': 'FurnitureBaxterEnv',
#             'furniture_id': 1,
#             'background': 'Interior',
#             'port': 1050}
# )

