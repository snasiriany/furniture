from collections import OrderedDict

import numpy as np

from furniture.env.furniture_cursor import FurnitureCursorEnv
from . import transform_utils as T


class FurnitureCursorRLEnv(FurnitureCursorEnv):
    """
    Cursor environment.
    """

    def __init__(self, config):
        super().__init__(config)

        self._state_goal = None

        for k in self._env_config.keys():
            if k in self._config:
                self._env_config[k] = getattr(self._config, k)

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
            elif self._config.control_degrees == '3dpos+3drot+select':
                move = 3
                rotate = 3
                select = 1
            elif self._config.control_degrees == '3dpos+3drot+select+connect':
                move = 3
                rotate = 3
                select = 1
                connect = 1
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
        ### sample goal by finding valid configuration in sim ###
        if self._config.task_type in ["reach+latch+move_obj", "latch+move_obj", "move_obj"]:
            if self._config.goal_type == 'fixed':
                object_goal = np.zeros(14)
                object_goal[0:3] = [0.3, 0.3, 0.05]
                object_goal[7:10] = [-0.3, 0.3, 0.05]
                object_goal[3:7] = [1, 0, 0, 0]
                object_goal[10:14] = [1, 0, 0, 0]
            elif self._config.goal_type == 'reset':
                object_goal = super().reset(furniture_id=furniture_id, background=background)["object_ob"].copy()
            else:
                raise NotImplementedError
        else:
            object_goal = None # defer setting goal to later

        ### reset the env
        obs = super().reset(furniture_id=furniture_id, background=background)
        if self._config.task_type in ["move_obj"]:
            low_level_a = np.zeros(15)
            low_level_a[6] = 1
            low_level_a[13] = 1
            obs, _, _, _ = super()._step(low_level_a)

        # set the goal
        robot_ob = obs['robot_ob'].copy()
        object_ob = obs['object_ob'].copy()

        robot_goal = np.zeros(robot_ob.size)

        if self._config.task_type == "reach_obj":
            robot_goal[0:3] = object_ob[0:3]
            robot_goal[3:6] = object_ob[7:10]
            robot_goal[6] = 0
            robot_goal[7] = 0
            object_goal = object_ob
        elif self._config.task_type in ["reach_obj+latch", "latch"]:
            robot_goal[0:3] = object_ob[0:3]
            robot_goal[3:6] = object_ob[7:10]
            robot_goal[6] = 1
            robot_goal[7] = 1
            object_goal = object_ob
        elif self._config.task_type in ["reach+latch+move_obj", "latch+move_obj", "move_obj"]:
            robot_goal[0:3] = object_goal[0:3]
            robot_goal[3:6] = object_goal[7:10]
            robot_goal[6] = 1
            robot_goal[7] = 1
        else:
            raise NotImplementedError

        self._state_goal = np.concatenate((robot_goal, object_goal))
        if self._config.num_connected_ob:
            self._state_goal = np.concatenate((self._state_goal, np.zeros(1)))

        return obs

    def _step(self, a):
        if self._config.tight_action_space:
            low_level_a = np.zeros(super().dof)
            if self._config.control_degrees == '2d':
                low_level_a[0:2] = a[0:2]
                low_level_a[7:9] = a[2:4]
            elif self._config.control_degrees == '2d+select':
                low_level_a[0:2] = a[0:2]
                low_level_a[7:9] = a[2:4]
                low_level_a[6] = a[4]
                low_level_a[13] = a[5]
            elif self._config.control_degrees == '3dpos+3drot+select':
                low_level_a[0:7] = a[0:7]
                low_level_a[7:14] = a[7:14]
            elif self._config.control_degrees == '3dpos+3drot+select+connect':
                low_level_a[0:7] = a[0:7]
                low_level_a[7:14] = a[7:14]
                low_level_a[14] = a[14]
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
            elif self._config.control_degrees == '3dpos+3drot+select':
                low_level_a[14] = 0
            elif self._config.control_degrees == '3dpos+3drot+select+connect':
                pass
            else:
                raise NotImplementedError

        if self._config.task_type in ["move_obj"]:
            low_level_a[6] = 1
            low_level_a[13] = 1

        ob, reward, done, _ = super()._step(low_level_a)

        info = self._get_info(action=a)

        return ob, reward, done, info

    def _get_info(self, action=None):
        obs = self._get_obs()
        cursor1_ob = np.concatenate((obs["robot_ob"][0:3], obs["robot_ob"][6:7]))
        cursor2_ob = np.concatenate((obs["robot_ob"][3:6], obs["robot_ob"][7:8]))
        obj1_ob = obs["object_ob"][0:7]
        obj2_ob = obs["object_ob"][7:14]

        cursor1_goal = np.concatenate((self._state_goal[0:3], self._state_goal[6:7]))
        cursor2_goal = np.concatenate((self._state_goal[3:6], self._state_goal[7:8]))
        obj1_goal = self._state_goal[8:15]
        obj2_goal = self._state_goal[15:22]

        state_ob = np.concatenate((obs["robot_ob"], obs["object_ob"]))
        if self._config.num_connected_ob:
            state_ob = np.concatenate((state_ob, obs["num_connected_ob"]))

        cursor1_obj1_dist = np.linalg.norm(cursor1_ob[:3] - obj1_ob[:3])
        cursor2_obj2_dist = np.linalg.norm(cursor2_ob[:3] - obj2_ob[:3])
        cursor1_latched = cursor1_ob[3]
        cursor2_latched = cursor2_ob[3]
        cursor1_xyz_dist = np.linalg.norm(cursor1_ob[:3] - cursor1_goal[:3])
        cursor2_xyz_dist = np.linalg.norm(cursor2_ob[:3] - cursor2_goal[:3])
        cursor1_latch_dist = np.linalg.norm(cursor1_ob[3] - cursor1_goal[3])
        cursor2_latch_dist = np.linalg.norm(cursor2_ob[3] - cursor2_goal[3])
        obj1_xyz_dist = np.linalg.norm(obj1_ob[:3] - obj1_goal[:3])
        obj2_xyz_dist = np.linalg.norm(obj2_ob[:3] - obj2_goal[:3])
        obj1_quat_dist = np.linalg.norm(obj1_ob[3:7] - obj1_goal[3:7])
        obj2_quat_dist = np.linalg.norm(obj2_ob[3:7] - obj2_goal[3:7])
        state_distance = np.linalg.norm(state_ob - self._state_goal)

        info = dict(
            cursor1_obj1_dist=cursor1_obj1_dist,
            cursor2_obj2_dist=cursor2_obj2_dist,
            cursor_obj_dist=cursor1_obj1_dist + cursor2_obj2_dist,

            cursor1_latched=cursor1_latched,
            cursor2_latched=cursor2_latched,
            cursor_latched=cursor1_latched + cursor2_latched,

            cursor1_xyz_dist=cursor1_xyz_dist,
            cursor2_xyz_dist=cursor2_xyz_dist,
            cursor_xyz_dist=cursor1_xyz_dist + cursor2_xyz_dist,

            cursor1_latch_dist=cursor1_latch_dist,
            cursor2_latch_dist=cursor2_latch_dist,
            cursor_latch_dist=cursor1_latch_dist + cursor2_latch_dist,

            obj1_xyz_dist=obj1_xyz_dist,
            obj2_xyz_dist=obj2_xyz_dist,
            obj_xyz_dist=obj1_xyz_dist + obj2_xyz_dist,

            obj1_quat_dist=obj1_quat_dist,
            obj2_quat_dist=obj2_quat_dist,
            obj_quat_dist=obj1_quat_dist + obj2_quat_dist,

            state_distance=state_distance,
        )

        if self._config.num_connected_ob:
            info["num_connected"] = obs["num_connected_ob"]

        if action is not None:
            info["action_mag_mean"] = np.mean(np.abs(action))
            info["action_mag_max"] = np.max(np.abs(action))

        return info

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        qpos = state_goal[8:22].copy()
        qvel = np.zeros(self.sim.model.nv)
        self.set_state(qpos, qvel)

    def _place_objects(self):
        """
        Returns the fixed initial positions and rotations of furniture parts.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        if self._config.reset_type=='fixed':
            pos_init = [[0.3, 0.3, 0.05], [-0.3, 0.3, 0.05]]
            quat_init = [[1, 0, 0, 0], [1, 0, 0, 0]]
        elif self._config.reset_type=='var_2dpos':
            pos_init, _ = self.mujoco_model.place_objects()
            quat_init = []
            for i, body in enumerate(self._object_names):
                rotate = self._rng.randint(0, 10, size=3)
                quat_init.append(list(T.euler_to_quat(rotate)))
        elif self._config.reset_type=='var_2dpos+var_1drot':
            pos_init, _ = self.mujoco_model.place_objects()
            quat_init = []
            for i, body in enumerate(self._object_names):
                rotate = self._rng.randint([0, 0, 0], [1, 1, 360], size=3)
                quat_init.append(list(T.euler_to_quat(rotate)))
        else:
            raise NotImplementedError

        return pos_init, quat_init

    def _initialize_robot_pos(self):
        """
        Initializes robot posision with random noise perturbation
        """
        if self._config.task_type in ["latch", "latch+move_obj", "move_obj"]:
            for i, obj_name in enumerate(self._object_names):
                obj_pos = self._get_pos(obj_name)
                obj_quat = self._get_quat(obj_name)

                if i == 0:
                    cursor_name = 'cursor0'
                elif i == 1:
                    cursor_name = 'cursor1'
                else:
                    raise NotImplementedError

                self._set_pos(cursor_name, [obj_pos[0], obj_pos[1], self._move_speed / 2])
        else:
            self._set_pos('cursor0', [-0.2, 0., self._move_speed / 2])
            self._set_pos('cursor1', [0.2, 0., self._move_speed / 2])

    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        ### For multiworld envs only! ###
        state = obs["state_observation"]
        goal = obs["state_desired_goal"]

        if self._config.reward_type == 'state_distance':
            dist = np.linalg.norm(state - goal, axis=1)
        elif self._config.reward_type == 'cursor_distance':
            dist = np.linalg.norm(state[:,:8] - goal[:,:8], axis=1)
        elif self._config.reward_type == 'object_distance':
            dist = np.linalg.norm(state[:,8:22] - goal[:,8:22], axis=1)
        elif self._config.reward_type == 'object1_distance':
            dist = np.linalg.norm(state[:,8:15] - goal[:,8:15], axis=1)
        elif self._config.reward_type == 'object1_distance+num_connected':
            assert self._config.num_connected_ob
            num_connected_rew = np.linalg.norm(state[:,22:23] - goal[:,22:23], axis=1) * self._config.num_connected_reward_scale
            dist = np.linalg.norm(state[:,8:15] - goal[:,8:15], axis=1) - num_connected_rew
        elif self._config.reward_type == 'object_xyz_distance':
            state_xyz = np.concatenate((state[:,8:11], state[:,15:18]), axis=1)
            goal_xyz = np.concatenate((goal[:, 8:11], goal[:, 15:18]), axis=1)
            dist = np.linalg.norm(state_xyz - goal_xyz, axis=1)
        elif self._config.reward_type == 'object1_xyz_distance':
            dist = np.linalg.norm(state[:,8:11] - goal[:, 8:11], axis=1)
        elif self._config.reward_type == 'object_distance+latch_distance':
            dist1 = np.linalg.norm(state[:,8:22] - goal[:,8:22], axis=1)
            dist2 = np.linalg.norm(state[:, 6:8] - goal[:, 6:8], axis=1)
            dist = dist1 + dist2
        else:
            raise NotImplementedError

        return -dist