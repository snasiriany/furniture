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

        assert self._config.task_type in [
            "reach+select+move",
            "reach+move",
            "select+move",
            "move",

            "reach+select+connect+move",
            "reach+select+connect",
            "select+connect",
            "reach+connect",
            "reach+connect+move",
            "connect",
            "select+connect+move",
            "connect+move",
        ]
        
        assert "connect" not in self._config.reward_type

        if "light_logging" in self._config:
            self._light_logging = self._config.light_logging
        else:
            self._light_logging = False

    @property
    def dof(self):
        """
        Returns the DoF of the curosr agent.
        """
        assert self._control_type == 'ik'
        if self._config.tight_action_space:
            move = rotate = select = connect = 0
            if self._config.control_degrees == '2dpos':
                move = 2
            elif self._config.control_degrees == '3dpos':
                move = 3
            elif self._config.control_degrees == '3dpos+3drot':
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
            elif self._config.control_degrees == '2dpos+select':
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
        if self._config.goal_type == 'fixed':
            assert self.n_objects == 2
            object_goal = np.zeros(14)
            object_goal[0:3] = [0.3, 0.3, 0.05]
            object_goal[7:10] = [-0.3, 0.3, 0.05]
            object_goal[3:7] = [1, 0, 0, 0]
            object_goal[10:14] = [1, 0, 0, 0]
        elif self._config.goal_type == 'reset':
            object_goal = super().reset(furniture_id=furniture_id, background=background)["object_ob"].copy()
        else:
            raise NotImplementedError

        ### reset the env
        obs = super().reset(furniture_id=furniture_id, background=background)
        if "select" not in self._config.task_type:
            low_level_a = np.zeros(15)
            low_level_a[6] = 1
            low_level_a[13] = 1
            obs, _, _, _ = super()._step(low_level_a)

        # set the goal
        robot_ob = obs['robot_ob'].copy()
        object_ob = obs['object_ob'].copy()

        robot_goal = np.zeros(robot_ob.size)
        robot_goal[0:3] = object_goal[0:3]
        robot_goal[3:6] = object_goal[7:10]
        robot_goal[6] = 1
        robot_goal[7] = 1

        self._state_goal = np.concatenate((robot_goal, object_goal))
        if self._config.num_connected_ob:
            self._state_goal = np.concatenate((self._state_goal, np.zeros(1)))

        return obs

    def _step(self, a):
        if self._config.tight_action_space:
            low_level_a = np.zeros(super().dof)
            if self._config.control_degrees == '2dpos':
                low_level_a[0:2] = a[0:2]
                low_level_a[7:9] = a[2:4]
            elif self._config.control_degrees == '2dpos+select':
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
            elif self._config.control_degrees == '3dpos':
                raise NotImplementedError
            elif self._config.control_degrees == '3dpos+3drot':
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            low_level_a = a.copy()
            if self._config.control_degrees == '2dpos':
                low_level_a[2:7] = 0
                low_level_a[9:15] = 0
            elif self._config.control_degrees == '3dpos':
                low_level_a[3:7] = 0
                low_level_a[10:15] = 0
            elif self._config.control_degrees == '3dpos+3drot':
                low_level_a[6] = 0
                low_level_a[13] = 0
                low_level_a[14] = 0
            elif self._config.control_degrees == '2dpos+select':
                low_level_a[2:6] = 0
                low_level_a[9:13] = 0
                low_level_a[14] = 0
            elif self._config.control_degrees == '2dpos+select+connect':
                low_level_a[2:6] = 0
                low_level_a[9:13] = 0
            elif self._config.control_degrees == '3dpos+select+connect':
                low_level_a[3:6] = 0
                low_level_a[10:13] = 0
            elif self._config.control_degrees == '3dpos+3drot+select':
                low_level_a[14] = 0
            elif self._config.control_degrees == '3dpos+3drot+select+connect':
                pass
            else:
                raise NotImplementedError

        if "select" not in self._config.task_type:
            low_level_a[6] = 1
            low_level_a[13] = 1

        ob, reward, done, _ = super()._step(low_level_a)

        info = self._get_info(action=a)

        return ob, reward, done, info

    def _get_info(self, action=None):
        obs = self._get_obs()

        state_ob = np.concatenate((obs["robot_ob"], obs["object_ob"]))
        if self._config.num_connected_ob:
            state_ob = np.concatenate((state_ob, obs["num_connected_ob"]))
        state_distance = np.linalg.norm(state_ob - self._state_goal)

        cursor1_ob = np.concatenate((obs["robot_ob"][0:3], obs["robot_ob"][6:7]))
        cursor2_ob = np.concatenate((obs["robot_ob"][3:6], obs["robot_ob"][7:8]))
        cursor1_goal = np.concatenate((self._state_goal[0:3], self._state_goal[6:7]))
        cursor2_goal = np.concatenate((self._state_goal[3:6], self._state_goal[7:8]))

        cursor1_selected = cursor1_ob[3]
        cursor2_selected = cursor2_ob[3]
        cursor1_xyz_dist = np.linalg.norm(cursor1_ob[:3] - cursor1_goal[:3])
        cursor2_xyz_dist = np.linalg.norm(cursor2_ob[:3] - cursor2_goal[:3])
        cursor1_select_dist = np.linalg.norm(cursor1_ob[3] - cursor1_goal[3])
        cursor2_select_dist = np.linalg.norm(cursor2_ob[3] - cursor2_goal[3])

        info = dict(
            cursor_selected=cursor1_selected + cursor2_selected,
            cursor_xyz_dist=cursor1_xyz_dist + cursor2_xyz_dist,
            cursor_select_dist=cursor1_select_dist + cursor2_select_dist,

            state_distance=state_distance,
        )
        if not self._light_logging:
            info.update(dict(
                cursor1_selected=cursor1_selected,
                cursor2_selected=cursor2_selected,

                cursor1_xyz_dist=cursor1_xyz_dist,
                cursor2_xyz_dist=cursor2_xyz_dist,

                cursor1_select_dist=cursor1_select_dist,
                cursor2_select_dist=cursor2_select_dist,
            ))

        obj_ob = obs["object_ob"]
        obj_goal = self._state_goal[8:8 + self.n_objects * 7]

        obj_xyz_dist, obj_quat_dist, obj_in_bounds = 0, 0, 0
        bv = 0.40
        for i in range(self.n_objects):
            start_idx = i*7
            xyz_dist = np.linalg.norm(obj_ob[start_idx:start_idx + 3] - obj_goal[start_idx:start_idx + 3])
            quat_dist = np.linalg.norm(obj_ob[start_idx + 3:start_idx + 7] - obj_goal[start_idx + 3:start_idx + 7])
            in_bounds = np.all(
                (obj_ob[start_idx:start_idx + 3] >= [-bv, -bv, -bv]) & (obj_ob[start_idx:start_idx + 3] <= [bv, bv, bv])
            ).astype(float)

            if not self._light_logging:
                info['obj{}_xyz_dist'.format(i + 1)] = xyz_dist
                info['obj{}_quat_dist'.format(i + 1)] = quat_dist

            info['obj{}_in_bounds'.format(i + 1)] = in_bounds

            obj_xyz_dist += xyz_dist
            obj_quat_dist += quat_dist
            obj_in_bounds += obj_in_bounds

        info['obj_xyz_dist'] = obj_xyz_dist
        info['obj_quat_dist'] = obj_quat_dist
        info['obj_in_bounds'] = obj_in_bounds

        if self._config.num_connected_ob:
            info["num_connected"] = obs["num_connected_ob"]

        if action is not None:
            info["action_mag_mean"] = np.mean(np.abs(action))
            info["action_mag_max"] = np.max(np.abs(action))

        return info

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        if self._obj_joint_type == 'free':
            qpos = state_goal[8 : 8+7*self.n_objects].copy()
        elif self._obj_joint_type == 'slide':
            positions = []
            for i in range(self.n_objects):
                start_idx = 8 + i*7
                positions.append(state_goal[start_idx:start_idx+3])
            qpos = np.concatenate(positions)
        else:
            raise NotImplementedError
        qvel = np.zeros(self.sim.model.nv)
        self.set_state(qpos, qvel)

    def _place_objects(self):
        """
        Returns the fixed initial positions and rotations of furniture parts.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        assert self._config.reset_type in [
            'fixed',
            'var_2dpos',
            'var_2dpos+objs_near',
            'var_2dpos+no_rot',
            'var_2dpos+var_1drot',
        ]

        if self._config.reset_type=='fixed':
            assert (self.n_objects == 2)
            pos_init = [[0.3, 0.3, 0.05], [-0.3, 0.3, 0.05]]
            quat_init = [[1, 0, 0, 0], [1, 0, 0, 0]]
        else:
            pos_init, _ = self.mujoco_model.place_objects()
            quat_init = []

            if self._config.reset_type=='var_2dpos':
                for i, body in enumerate(self._object_names):
                    rotate = self._rng.randint(0, 10, size=3)
                    quat_init.append(list(T.euler_to_quat(rotate)))
            elif self._config.reset_type=='var_2dpos+objs_near':
                pos_init[1] = pos_init[0].copy()
                offset1 = [0.4, 0, 0]
                offset2 = np.random.uniform([-0.1, -0.1, 0], [0.1, 0.1, 0])
                pos_init[1] = [x + y + z for x, y, z in zip(pos_init[1], offset1, offset2)]
                for i, body in enumerate(self._object_names):
                    rotate = self._rng.randint(0, 10, size=3)
                    quat_init.append(list(T.euler_to_quat(rotate)))
            elif self._config.reset_type=='var_2dpos+no_rot':
                for i, body in enumerate(self._object_names):
                    rotate = [0, 0, 0]
                    quat_init.append(list(T.euler_to_quat(rotate)))
            elif self._config.reset_type=='var_2dpos+var_1drot':
                for i, body in enumerate(self._object_names):
                    rotate = self._rng.randint([0, 0, 0], [1, 1, 360], size=3)
                    quat_init.append(list(T.euler_to_quat(rotate)))

        return pos_init, quat_init

    def _initialize_robot_pos(self):
        """
        Initializes robot posision with random noise perturbation
        """
        if "reach" not in self._config.task_type:
            while True:
                cursor0_idx, cursor1_idx = np.random.choice(self.n_objects, 2, replace=False)
                if self._can_connect(self._object_names[cursor0_idx], self._object_names[cursor1_idx]):
                    break


            for i, obj_name in enumerate(self._object_names):
                body_ids = [self.sim.model.body_name2id(obj_name)]
                sites = []
                for j, site in enumerate(self.sim.model.site_names):
                    if 'conn_site' in site:
                        if self.sim.model.site_bodyid[j] in body_ids:
                            sites.append(site)
                site_pos = self._site_xpos_xquat(sites[0])[0:3]

                if i == cursor0_idx:
                    cursor_name = 'cursor0'
                elif i == cursor1_idx:
                    cursor_name = 'cursor1'
                else:
                    cursor_name = None

                # obj_pos = self._get_pos(obj_name)
                # self._set_pos(cursor_name, [site_pos[0], site_pos[1], self._move_speed / 2])
                if cursor_name is not None:
                    self._set_pos(cursor_name, [site_pos[0], site_pos[1], max(self._move_speed / 2, site_pos[2])])
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
            dist = np.zeros(len(state))
            for i in range(self.n_objects):
                dist += np.linalg.norm(state[:,8+7*i:8+7*i+7] - goal[:,8+7*i:8+7*i+7], axis=1)
        elif self._config.reward_type == 'object1_distance':
            dist = np.linalg.norm(state[:,8:15] - goal[:,8:15], axis=1)
        elif self._config.reward_type == 'object_xyz_distance':
            dist = np.zeros(len(state))
            for i in range(self.n_objects):
                dist += np.linalg.norm(state[:, 8 + 7 * i:8 + 7 * i + 3] - goal[:, 8 + 7 * i:8 + 7 * i + 3], axis=1)
        elif self._config.reward_type == 'object1_xyz_distance':
            dist = np.linalg.norm(state[:,8:11] - goal[:, 8:11], axis=1)
        elif self._config.reward_type == 'object1_xyz_in_bounds':
            bv = 0.40
            inbounds = np.all((state[:,8:11] >= [-bv, -bv, -bv]) & (state[:,8:11] <= [bv, bv, bv]), axis=1)
            dist = -inbounds.astype(float)
        elif self._config.reward_type == 'object_in_bounds':
            bv = 0.40
            dist = np.zeros(len(state))
            for i in range(self.n_objects):
                inbounds = np.all(
                    (state[:,8+7*i:8+7*i+3] >= [-bv, -bv, -bv]) & (state[:,8+7*i:8+7*i+3] <= [bv, bv, bv]), axis=1
                )
                dist -= inbounds.astype(float)
        elif self._config.reward_type == 'object_distance+select_distance':
            dist = np.zeros(len(state))
            for i in range(self.n_objects):
                dist += np.linalg.norm(state[:,8+7*i:8+7*i+7] - goal[:,8+7*i:8+7*i+7], axis=1)
            dist += np.linalg.norm(state[:, 6:8] - goal[:, 6:8], axis=1)
        else:
            raise NotImplementedError

        if "connect" in self._config.task_type:
            num_connected_rew = state[:, -1] * self._config.num_connected_reward_scale
            dist = dist - num_connected_rew

        return -dist