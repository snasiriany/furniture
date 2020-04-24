""" Gym wrapper for the IKEA Furniture Assembly Environment. """

import gym
import numpy as np

from furniture.env import make_env
from furniture.config.furniture import get_default_config

from gym.spaces import Box, Dict

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import concatenate_box_spaces

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
            # if hasattr(config, key):
            #     setattr(config, key, value)
            setattr(config, key, value)

        # create an environment
        self._wrapped_env = make_env(name, config)

        orig_obs_space = self._wrapped_env.observation_space
        robot_low = -1 * np.ones(orig_obs_space['robot_ob'])
        robot_high = np.ones(orig_obs_space['robot_ob'])
        object_low = -1 * np.ones(orig_obs_space['object_ob'])
        object_high = np.ones(orig_obs_space['object_ob'])

        # covert observation space
        robot_space = Box(robot_low, robot_high, dtype=np.float32)
        object_space = Box(object_low, object_high, dtype=np.float32)
        obs_space = concatenate_box_spaces(robot_space, object_space)

        if self._connector_ob_type is not None:
            if self._connector_ob_type == "dist":
                dim = self._wrapped_env.n_connectors * 1 // 2
            elif self._connector_ob_type == "diff":
                dim = self._wrapped_env.n_connectors * 3 // 2
            elif self._connector_ob_type == "pos":
                dim = self._wrapped_env.n_connectors * 3
            else:
                raise NotImplementedError
            connector_space = Box(-1 * np.ones(dim), 1 * np.ones(dim), dtype=np.float32)
            obs_space = concatenate_box_spaces(obs_space, connector_space)
        if self._config.num_connected_ob:
            num_connected_space = Box(np.array([0]), np.array([100]), dtype=np.float32)
            obs_space = concatenate_box_spaces(obs_space, num_connected_space)

        self.observation_space = Dict([
            ('observation', obs_space),
            ('desired_goal',obs_space),
            ('achieved_goal', obs_space),
            ('state_observation', obs_space),
            ('state_desired_goal', obs_space),
            ('state_achieved_goal', obs_space),
            ('proprio_observation', robot_space),
            ('proprio_desired_goal', robot_space),
            ('proprio_achieved_goal', robot_space),
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
        if self._connector_ob_type is not None:
            flat_obs = np.concatenate((flat_obs, obs['connector_ob']))
        if self._config.num_connected_ob:
            flat_obs = np.concatenate((flat_obs, obs['num_connected_ob']))

        robot_dim = obs['robot_ob'].size
        # state_goal = np.zeros(self.observation_space.spaces['state_desired_goal'].low.size)
        state_goal = self._state_goal.copy()

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
        return self._wrapped_env.compute_rewards(actions, obs, prev_obs, reward_type)

    def get_env_state(self):
        return self.get_state()

    def set_env_state(self, state):
        qpos, qvel = state
        self.set_state(qpos, qvel)

    def sample_goals(self, batch_size):
        assert False

    def get_goal(self):
        return {
            'desired_goal': self._state_goal.copy(),
            'state_desired_goal': self._state_goal.copy(),
        }

    def get_image(self, width=84, height=84, camera_name=None):
        return self.sim.render(
            camera_name=self._camera_name,
            width=width,
            height=height,
            depth=False
        )[::-1,:,:]
