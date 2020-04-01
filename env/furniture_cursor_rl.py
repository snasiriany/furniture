""" Define cursor environment class FurnitureCursorEnv. """

from collections import OrderedDict

import numpy as np

from env.furniture_cursor import FurnitureCursorEnv


class FurnitureCursorRLEnv(FurnitureCursorEnv):
    """
    Cursor environment.
    """

    def __init__(self, config):
        super().__init__(config)

    def _step(self, a):
        ob, reward, done, _ = super()._step(a)

        info = self._get_info()

        return ob, reward, done, info

    def _get_info(self):
        obs = self._get_obs()
        obj1 = obs["object_ob"][:7]
        obj2 = obs["object_ob"][7:]
        robot1 = obs["robot_ob"][:4]
        robot2 = obs["robot_ob"][4:]
        info = dict(
            robot1_obj1_dist=np.linalg.norm(obj1[:3] - robot1[:3]),
            robot2_obj2_dist=np.linalg.norm(obj2[:3] - robot2[:3]),
        )
        return info

    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        object_ob = obs["state_observation"][:,:14]
        robot_ob = obs["state_observation"][:, 14:]

        obj1 = object_ob[:,:7]
        obj2 = object_ob[:,7:]
        robot1 = robot_ob[:,:4]
        robot2 = robot_ob[:,4:]

        robot1_obj1_dist = np.linalg.norm(obj1[:,:3] - robot1[:,:3], axis=1)
        robot2_obj2_dist = np.linalg.norm(obj2[:,:3] - robot2[:,:3], axis=1)

        return robot1_obj1_dist + robot2_obj2_dist
