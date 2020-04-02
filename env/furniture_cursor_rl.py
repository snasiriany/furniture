from collections import OrderedDict

import numpy as np

import env.transform_utils as T

from env.furniture_cursor import FurnitureCursorEnv


class FurnitureCursorRLEnv(FurnitureCursorEnv):
    """
    Cursor environment.
    """

    def __init__(self, config):
        super().__init__(config)

    @property
    def dof(self):
        """
        Returns the DoF of the curosr agent.
        """
        assert self._control_type == 'ik'
        if self._config.small_action_space:
            dof = (2 + 0 + 0) * 2 + 0  # (move, rotate, select) * 2 + connect
            return dof
        else:
            return super().dof

    def _step(self, a):
        if self._config.small_action_space:
            low_level_a = np.zeros(super().dof)
            low_level_a[0:2] = a[0:2]
            low_level_a[7:9] = a[2:4]
        else:
            low_level_a = a.copy()
            if self._config.control_degrees == '2d':
                low_level_a[2:7] = 0
                low_level_a[9:15] = 0
            elif self._config.control_degrees == '3d':
                low_level_a[3:7] = 0
                low_level_a[10:15] = 0
            elif self._config.control_degrees == '3d+rot':
                low_level_a[6] = 0
                low_level_a[13] = 0
                low_level_a[14] = 0

        ob, reward, done, _ = super()._step(low_level_a)

        info = self._get_info()

        return ob, reward, done, info

    def _get_info(self):
        obs = self._get_obs()
        robot1 = obs["robot_ob"][0:3]
        robot2 = obs["robot_ob"][3:6]
        obj1 = obs["object_ob"][:7]
        obj2 = obs["object_ob"][7:]
        info = dict(
            robot1_obj1_dist=np.linalg.norm(obj1[:3] - robot1[:3]),
            robot2_obj2_dist=np.linalg.norm(obj2[:3] - robot2[:3]),
        )
        return info

    def _place_objects(self):
        """
        Returns the fixed initial positions and rotations of furniture parts.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        if self._config.fixed_reset:
            pos_init = [[0.3, 0.3, 0.05], [-0.3, 0.3, 0.05]]
            quat_init = [[1, 0, 0, 0], [1, 0, 0, 0]]
            return pos_init, quat_init
        else:
            return super()._place_objects()

    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        ### For multiworld envs only! ###
        robot_ob = obs["state_observation"][:,0:8]
        object_ob = obs["state_observation"][:,8:22]

        robot1 = robot_ob[:,0:3]
        robot2 = robot_ob[:,3:6]
        obj1 = object_ob[:,:7]
        obj2 = object_ob[:,7:]

        robot1_obj1_dist = np.linalg.norm(obj1[:,:3] - robot1[:,:3], axis=1)
        robot2_obj2_dist = np.linalg.norm(obj2[:,:3] - robot2[:,:3], axis=1)

        if self._config.reward_type == "robot1_obj1":
            return - (robot1_obj1_dist)
        elif self._config.reward_type == "robot2_obj2":
            return - (robot2_obj2_dist)
        elif self._config.reward_type == "robot1_obj1_and_robot2_obj2":
            return - (robot1_obj1_dist + robot2_obj2_dist)
        else:
            raise NotImplementedError
