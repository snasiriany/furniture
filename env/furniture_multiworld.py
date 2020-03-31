""" Gym wrapper for the IKEA Furniture Assembly Environment. """

import gym
import numpy as np

from env import make_env
from config.furniture import get_default_config

from gym.spaces import Box, Dict

from multiworld.core.multitask_env import MultitaskEnv

class FurnitureMultiworld(MultitaskEnv):
    """
    Multiworld wrapper class for Furniture assmebly environment.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwarg: configurations for the environment.
        """
        config = get_default_config()

        name = kwargs['name']
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # create an environment
        self._wrapped_env = make_env(name, config)

        orig_obs_space = self._wrapped_env.observation_space
        robot_low = -1 * np.ones(orig_obs_space['robot_ob'])
        robot_high = np.ones(orig_obs_space['robot_ob'])
        object_low = -1 * np.ones(orig_obs_space['object_ob'])
        object_high = np.ones(orig_obs_space['object_ob'])

        # covert observation space
        obs_space = gym.spaces.Box(
            np.concatenate((robot_low, object_low)),
            np.concatenate((robot_high, object_high)),
            dtype=np.float32
        )
        self.observation_space = Dict([
            ('observation', obs_space),
            ('desired_goal',obs_space),
            ('achieved_goal', obs_space),
            ('state_observation', obs_space),
            ('state_desired_goal', obs_space),
            ('state_achieved_goal', obs_space),
            ('proprio_observation', orig_obs_space['robot_ob']),
            ('proprio_desired_goal', orig_obs_space['robot_ob']),
            ('proprio_achieved_goal', orig_obs_space['robot_ob']),
        ])

        # covert action space
        dof = self._wrapped_env.dof
        low = -1 * np.ones(dof)
        high = np.ones(dof)
        self.action_space = Box(low=low, high=high)

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def reset(self):
        obs = self._wrapped_env.reset()
        return self.__covert_to_multiworld_obs(obs)

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)
        return self.__covert_to_multiworld_obs(obs), reward, done, info

    def __get_obs(self):
        obs = self._wrapped_env._get_obs()
        return self.__covert_to_multiworld_obs(obs)

    def __covert_to_multiworld_obs(self, obs):
        flat_obs = np.concatenate((obs['robot_ob'], obs['object_ob']))
        robot_dim = obs['robot_ob'].size
        state_goal = np.zeros(self.observation_space.spaces['state_desired_goal'].low.size)

        return dict(
            observation=flat_obs,
            desired_goal=state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs[:robot_dim],
            proprio_desired_goal=state_goal[:robot_dim],
            proprio_achieved_goal=flat_obs[:robot_dim],
        )

    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        return np.zeros(len(obs['state_observation']))

    def sample_goals(self, batch_size):
        return None

    def get_goal(self):
        return None
