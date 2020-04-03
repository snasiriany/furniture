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

        self._state_goal = None

    @property
    def dof(self):
        """
        Returns the DoF of the curosr agent.
        """
        assert self._control_type == 'ik'
        if self._config.tight_action_space:
            move = rotate = select = connect = 0
            if self._config.control_degrees == '2d':
                move = 2
            elif self._config.control_degrees == '3d':
                move = 3
            elif self._config.control_degrees == '3d+rot':
                move = 3
                rotate = 3
            elif self._config.control_degrees == '2d+select':
                move = 2
                select = 1
            else:
                raise NotImplementedError

            dof = (move + rotate + select) * 2 + connect
            return dof
        else:
            return super().dof

    def reset(self, furniture_id=None, background=None):
        obs = super().reset(furniture_id=furniture_id, background=background)

        # set the goal
        robot_ob = obs['robot_ob'].copy()
        object_ob = obs['object_ob'].copy()

        robot_goal = np.zeros(robot_ob.size)
        object_goal = np.zeros(object_ob.size)

        if self._config.task_type == "reach_obj":
            robot_goal[0:3] = object_ob[0:3]
            robot_goal[3:6] = object_ob[7:10]
            robot_goal[6] = 0
            robot_goal[7] = 0
            object_goal = object_ob
            self._state_goal = np.concatenate((robot_goal, object_goal))
        elif self._config.task_type == "reach_obj_and_latch":
            robot_goal[0:3] = object_ob[0:3]
            robot_goal[3:6] = object_ob[7:10]
            robot_goal[6] = 1
            robot_goal[7] = 1
            object_goal = object_ob
            self._state_goal = np.concatenate((robot_goal, object_goal))
        else:
            raise NotImplementedError

        return obs

    def _step(self, a):
        if self._config.tight_action_space:
            low_level_a = np.zeros(super().dof)
            if self._config.control_degrees == '2d':
                low_level_a[0:2] = a[0:2]
                low_level_a[7:9] = a[2:4]
            elif self._config.control_degrees == '3d':
                raise NotImplementedError
            elif self._config.control_degrees == '3d+rot':
                raise NotImplementedError
            else:
                raise NotImplementedError
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
            elif self._config.control_degrees == '2d+select':
                low_level_a[2:6] = 0
                low_level_a[9:13] = 0
                low_level_a[14] = 0
            else:
                raise NotImplementedError

        ob, reward, done, _ = super()._step(low_level_a)

        info = self._get_info()

        return ob, reward, done, info

    def _get_info(self):
        obs = self._get_obs()
        cursor1 = obs["robot_ob"][0:3]
        cursor2 = obs["robot_ob"][3:6]
        obj1 = obs["object_ob"][:7]
        obj2 = obs["object_ob"][7:]
        info = dict(
            cursor1_obj1_dist=np.linalg.norm(obj1[:3] - cursor1[:3]),
            cursor2_obj2_dist=np.linalg.norm(obj2[:3] - cursor2[:3]),
            cursor1_latched=obs["robot_ob"][6],
            cursor2_latched=obs["robot_ob"][7],
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
        # robot_ob = obs["state_observation"][:,0:8]
        # object_ob = obs["state_observation"][:,8:22]
        #
        # cursor1 = robot_ob[:,0:3]
        # cursor2 = robot_ob[:,3:6]
        # obj1 = object_ob[:,:7]
        # obj2 = object_ob[:,7:]
        #
        # cursor1_obj1_dist = np.linalg.norm(obj1[:,:3] - cursor1[:,:3], axis=1)
        # cursor2_obj2_dist = np.linalg.norm(obj2[:,:3] - cursor2[:,:3], axis=1)
        #
        # if self._config.reward_type == "cursor1_obj1":
        #     return - (cursor1_obj1_dist)
        # elif self._config.reward_type == "cursor2_obj2":
        #     return - (cursor2_obj2_dist)
        # elif self._config.reward_type == "cursor1_obj1_and_cursor2_obj2":
        #     return - (cursor1_obj1_dist + cursor2_obj2_dist)
        # else:
        #     raise NotImplementedError


        state = obs["state_observation"]
        goal = obs["state_desired_goal"]

        dist = np.linalg.norm(state - goal, axis=1)
        return -dist