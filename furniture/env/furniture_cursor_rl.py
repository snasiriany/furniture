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
            elif self._config.control_degrees == '3dpos+select+connect':
                move = 3
                rotate = 0
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
        if not hasattr(self, "_task_types"):
            self._task_types = self._config.task_type.split('+')
            for t in self._task_types:
                assert t in [
                    "reach",
                    "reach2",
                    "move2",
                    "select",
                    "select2",
                    "connect",
                ]

        if not hasattr(self, "_obj_dim"):
            if self._obj_joint_type == 'slide':
                self._obj_dim = 3
            else:
                self._obj_dim = 7

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
        elif self._config.goal_type == 'zeros':
            if not hasattr(self, 'n_objects'):
                super().reset(furniture_id=furniture_id, background=background)
            object_goal = np.zeros(self.n_objects*self._obj_dim)
        elif self._config.goal_type == 'assembled':
            pass
        else:
            raise NotImplementedError

        ### reset the env
        obs = super().reset(furniture_id=furniture_id, background=background)
        if "select" in self._task_types:
            pass
        elif "select2" in self._task_types:
            low_level_a = np.zeros(15)
            low_level_a[6] = 1
            obs, _, _, _ = super()._step(low_level_a)
        else:
            low_level_a = np.zeros(15)
            low_level_a[6] = 1
            low_level_a[13] = 1
            obs, _, _, _ = super()._step(low_level_a)

        # if self._config.goal_type == 'assembled':
        #     assert len(self._anchor_objects) == 1
        #     assert self._obj_joint_type == 'slide'
        #     object_goal = obs['object_ob'].copy()
        #     anchor_id = self._object_name2id[self._anchor_objects[0]]
        #     conn_info = self._get_oracle_connector_info()
        #     conn_info_dim = len(conn_info) // self.n_connectors
        #     for obj_id, obj_name in enumerate(self._object_names):
        #         if obj_id == anchor_id:
        #             continue
        #
        #         conn1_idx = self._obj_ids_to_connector_idx[obj_id][anchor_id]
        #         conn2_idx = self._obj_ids_to_connector_idx[anchor_id][obj_id]
        #         conn1_pos = conn_info[(conn1_idx + 1) * conn_info_dim - 3: (conn1_idx + 1) * conn_info_dim]
        #         conn2_pos = conn_info[(conn2_idx + 1) * conn_info_dim - 3: (conn2_idx + 1) * conn_info_dim]
        #
        #         start = obj_id * 3
        #         end = start + 3
        #         object_goal[start:end] = conn2_pos + self._get_qpos(obj_name) - conn1_pos
        #
        # # set the goal
        # robot_ob = obs['robot_ob'].copy()
        # object_ob = obs['object_ob'].copy()
        #
        # robot_goal = np.zeros(robot_ob.size)
        # if self._config.goal_type is not 'zeros':
        #     robot_goal[0:3] = self._sample_cursor_position()
        #     robot_goal[3:6] = self._sample_cursor_position()
        #     robot_goal[6] = 1
        #     robot_goal[7] = 1

        self._state_goal = self.sample_goal_for_rollout()

        # self._state_goal = np.concatenate((robot_goal, object_goal))
        # if self._connector_ob_type is not None:
        #     if self._connector_ob_type == "dist":
        #         dim = self.n_connectors * 1 // 2
        #     elif self._connector_ob_type == "diff":
        #         dim = self.n_connectors * 3 // 2
        #     elif self._connector_ob_type == "pos":
        #         dim = self.n_connectors * 3
        #     else:
        #         raise NotImplementedError
        #     dim += self.n_connectors // 2
        #     self._state_goal = np.concatenate((self._state_goal, np.zeros(dim)))
        # if self._config.num_connected_ob:
        #     self._state_goal = np.concatenate((self._state_goal, np.zeros(1)))

        return obs

    def sample_goal_for_rollout(self):
        obs = self._get_obs()
        if self._config.goal_type == 'assembled':
            assert len(self._anchor_objects) == 1
            assert self._obj_joint_type == 'slide'
            object_goal = obs['object_ob'].copy()
            anchor_id = self._object_name2id[self._anchor_objects[0]]
            conn_info = self._get_oracle_connector_info()
            conn_info_dim = len(conn_info) // self.n_connectors
            for obj_id, obj_name in enumerate(self._object_names):
                if obj_id == anchor_id:
                    continue

                conn1_idx = self._obj_ids_to_connector_idx[obj_id][anchor_id]
                conn2_idx = self._obj_ids_to_connector_idx[anchor_id][obj_id]
                conn1_pos = conn_info[(conn1_idx + 1) * conn_info_dim - 3: (conn1_idx + 1) * conn_info_dim]
                conn2_pos = conn_info[(conn2_idx + 1) * conn_info_dim - 3: (conn2_idx + 1) * conn_info_dim]

                start = obj_id * 3
                end = start + 3
                object_goal[start:end] = conn2_pos + self._get_qpos(obj_name) - conn1_pos
        else:
            raise NotImplementedError

        robot_goal = np.zeros(8)
        if self._config.goal_type is not 'zeros':
            robot_goal[0:3] = self._sample_cursor_position()
            robot_goal[3:6] = self._sample_cursor_position()
            robot_goal[6] = 1
            robot_goal[7] = 1

        return np.concatenate((robot_goal, object_goal))

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
            elif self._config.control_degrees == '3dpos+select+connect':
                low_level_a[0:3] = a[0:3]
                low_level_a[7:10] = a[3:6]
                low_level_a[6] = a[6]
                low_level_a[13] = a[7]
                low_level_a[14] = a[8]
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

        if "select" in self._task_types:
            pass
        elif "select2" in self._task_types:
            low_level_a[6] = 1
        else:
            low_level_a[6] = 1
            low_level_a[13] = 1

        if "connect" not in self._task_types:
            low_level_a[14] = 1

        if "move2" in self._task_types:
            low_level_a[0:6] = 0

        ob, reward, done, _ = super()._step(low_level_a)

        info = self._get_info()

        return ob, reward, done, info

    def _get_info(self):
        obs = self._get_obs()

        state_ob = np.concatenate((obs["robot_ob"], obs["object_ob"]))
        if self._connector_ob_type is not None:
            state_ob = np.concatenate((state_ob, obs["connector_ob"]))
        if self._config.num_connected_ob:
            state_ob = np.concatenate((state_ob, obs["num_connected_ob"]))
        state_distance = np.linalg.norm(state_ob - self._state_goal)

        cursor1_ob = np.concatenate((obs["robot_ob"][0:3], obs["robot_ob"][6:7]))
        cursor2_ob = np.concatenate((obs["robot_ob"][3:6], obs["robot_ob"][7:8]))

        cursor1_selected = cursor1_ob[3]
        cursor2_selected = cursor2_ob[3]

        info = dict(
            cursor1_dist=np.linalg.norm(obs["robot_ob"][0:3] - self._state_goal[0:3]),
            cursor2_dist=np.linalg.norm(obs["robot_ob"][3:6] - self._state_goal[3:6]),

            cursor1_selected=cursor1_selected,
            cursor2_selected=cursor2_selected,
            cursor_selected=cursor1_selected + cursor2_selected,

            state_distance=state_distance,
        )

        obj_ob = obs["object_ob"]
        obj_goal = self._state_goal[8:8 + self.n_objects * self._obj_dim]

        obj_xyz_dist = 0
        for i in range(self.n_objects):
            start_idx = i*self._obj_dim
            xyz_dist = np.linalg.norm(obj_ob[start_idx:start_idx + 3] - obj_goal[start_idx:start_idx + 3])

            if not self._light_logging:
                info['obj{}_xyz_dist'.format(i)] = xyz_dist

            obj_xyz_dist += xyz_dist

        info['obj_xyz_dist'] = obj_xyz_dist

        return info

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        if self._obj_joint_type == 'free':
            qpos = state_goal[8 : 8+self._obj_dim*self.n_objects].copy()
        elif self._obj_joint_type == 'slide':
            positions = []
            for (i, obj_name) in enumerate(self._object_names):
                start_idx = 8 + i*self._obj_dim
                position = state_goal[start_idx:start_idx + 3].copy()

                from furniture.env.mjcf_utils import string_to_array
                position -= string_to_array(
                    self.mujoco_objects[obj_name].worldbody.find("./body[@name='%s']" % obj_name).get("pos"))

                positions.append(position)
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

    def _sample_cursor_position(self):
        return np.random.uniform(low=[-0.5, -0.5, self._move_speed / 2], high=[0.5, 0.5, 0.5])

    def _initialize_robot_pos(self):
        """
        Initializes robot posision with random noise perturbation
        """
        if "reach" in self._task_types:
            self._set_pos('cursor0', self._sample_cursor_position()) #[-0.2, 0., self._move_speed / 2]
            self._set_pos('cursor1', self._sample_cursor_position()) #[0.2, 0., self._move_speed / 2]
        else:
            if self._task_connect_sequence is not None:
                cursor0_idx, cursor1_idx = self._task_connect_sequence[0:2]
            else:
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

            if "reach2" in self._task_types:
                self._set_pos('cursor1', self._sample_cursor_position()) # [0.2, 0., self._move_speed / 2]

    def compute_rewards(self, actions, obs):
        ### For multiworld envs only! ###
        state = obs["state_achieved_goal"]
        goal = obs["state_desired_goal"]

        # assert (len(state) == 1) or ('mask' in obs)
        rewards = self._config.reward_type.split('+')

        dist = np.zeros(len(state))

        for reward in rewards:
            if reward == 'state_distance':
                dist += np.linalg.norm(state - goal, axis=1)
            elif reward == 'object_distance':
                dist += np.linalg.norm(
                    state[:,8:8+self._obj_dim*self.n_objects] -
                    goal[:,8:8+self._obj_dim*self.n_objects],
                    axis=1
                )
            elif reward == 'object1_distance':
                dist += np.linalg.norm(
                    state[:,8:8+self._obj_dim] -
                    goal[:,8:8+self._obj_dim],
                    axis=1
                )
            elif reward == 'select_distance':
                dist += np.linalg.norm(state[:, 6:8] - goal[:, 6:8], axis=1)
            else:
                raise NotImplementedError

        return -dist

    def _select_object(self, cursor_i):
        """
        Selects an object within cursor_i
        """
        for obj_name in self._object_names:
            is_selected = False
            obj_group = self._find_group(obj_name)
            for selected_obj in self._cursor_selected:
                if selected_obj and obj_group == self._find_group(selected_obj):
                    is_selected = True

            min_pos, max_pos = self._get_bounding_box(obj_name, obj_only=True)
            cursor_pos = self._get_pos('cursor%d' % cursor_i)
            is_in_bounded_area = False
            if (cursor_pos >= np.array(min_pos)).all() and \
                    (cursor_pos <= np.array(max_pos)).all():
                is_in_bounded_area = True

            if not is_selected and (self.on_collision('cursor%d' % cursor_i, obj_name) or is_in_bounded_area):
                return obj_name
        return None

    def goal_conditioned_diagnostics(self, paths, goals):
        from collections import OrderedDict, defaultdict
        from multiworld.envs.env_util import create_stats_ordered_dict
        statistics = OrderedDict()
        stat_to_lists = defaultdict(list)
        for path, goal in zip(paths, goals):
            difference = path['observations'] - goal
            for i in range(2):
                distance_key = 'cursor{}_dist'.format(i)
                xyz_dist = np.linalg.norm(difference[:, i*3:i*3 + 3], axis=-1)
                stat_to_lists[distance_key].append(xyz_dist)
            for i in range(self.n_objects):
                start_idx = 8 + i * self._obj_dim
                xyz_dist = np.linalg.norm(difference[:, start_idx:start_idx + 3], axis=-1)
                distance_key = 'obj{}_dist'.format(i)
                stat_to_lists[distance_key].append(xyz_dist)
        for stat_name, stat_list in stat_to_lists.items():
            statistics.update(create_stats_ordered_dict(
                'env_infos/{}'.format(stat_name),
                stat_list,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'env_infos/final/{}'.format(stat_name),
                [s[-1:] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'env_infos/initial/{}'.format(stat_name),
                [s[:1] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        return statistics