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

    def __init__(self, goal_sampling_mode='assembled', **kwargs):
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

        self.goal_sampling_mode = goal_sampling_mode

        # create an environment
        self._wrapped_env = make_env(name, config)

        # orig_obs_space = self._wrapped_env.observation_space
        # robot_low = -1 * np.ones(orig_obs_space['robot_ob'])
        # robot_high = np.ones(orig_obs_space['robot_ob'])
        # object_low = -1 * np.ones(orig_obs_space['object_ob'])
        # object_high = np.ones(orig_obs_space['object_ob'])

        ### hack in the observation space ###
        b = np.array(self._wrapped_env._config.boundary)
        obj_low = list(-b.copy())
        obj_low[2] = -0.05
        obj_high = list(b.copy())

        cursor_low = list(-b.copy())
        cursor_low[2] = self._wrapped_env._move_speed / 2
        cursor_high = list(b.copy())

        robot_low = np.array(cursor_low + cursor_low + [0, 0])
        robot_high = np.array(cursor_high + cursor_high + [1, 1])

        object_low = np.array(obj_low * self._wrapped_env.n_objects)
        object_high = np.array(obj_high * self._wrapped_env.n_objects)

        # covert observation space
        robot_space = Box(robot_low, robot_high, dtype=np.float32)
        object_space = Box(object_low, object_high, dtype=np.float32)
        obs_space = concatenate_box_spaces(robot_space, object_space)


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
        self._state_goal = self.sample_goal()['state_desired_goal']
        return self.__covert_to_multiworld_obs(obs)

    def step(self, action):
        obs, _, done, info = self._wrapped_env.step(action)
        obs = self.__covert_to_multiworld_obs(obs)
        reward = self.compute_reward(action, obs)
        return obs, reward, done, info

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

            oracle_connector_info=self._get_oracle_connector_info(),
            oracle_robot_info=self._get_oracle_robot_info(),
        )

    def compute_rewards(self, actions, obs):
        return self._wrapped_env.compute_rewards(actions, obs)

    def get_env_state(self):
        return self.get_state()

    def set_env_state(self, state):
        qpos, qvel = state
        self.set_state(qpos, qvel)

    def sample_goals(self, batch_size):
        assert self.goal_sampling_mode in ['assembled', 'uniform', 'assembled_random']
        if batch_size == 1 and self.goal_sampling_mode in ['assembled', 'assembled_random']:
            goals = self._wrapped_env.sample_goal_for_rollout(self.goal_sampling_mode)[None]
        else:
            # b = np.array(self._wrapped_env._env_config["boundary"])
            b = np.array(self._wrapped_env._config.boundary)
            low = -b.copy()
            low[2] = -0.05
            high = b.copy()

            goals = np.zeros((batch_size, len(self.observation_space.spaces['state_desired_goal'].low)))
            goals[:, 0:3] = np.random.uniform(low, high, (batch_size, 3))
            goals[:, 3:6] = np.random.uniform(low, high, (batch_size, 3))
            for i in range(self._wrapped_env.n_objects):
                start = 8 + i * self._wrapped_env._obj_dim
                goals[:, start:start+3] = np.random.uniform(low, high, (batch_size, 3))
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

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
