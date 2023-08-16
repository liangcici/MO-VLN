import numpy as np
import json
import math
import os

from envs.ue5.env import Env
import envs.utils.pose as pu


class ObjectGoalEnv(Env):

    def __init__(self, args, rank):
        self.args = args
        self.rank = rank

        # Loading dataset info file
        self.split = args.split
        self.episodes_dir = args.episodes_dir

        # Episode Dataset info
        eps_data_file = self.episodes_dir + "{split}.json".format(split=self.split)
        with open(eps_data_file, 'r') as f:
            self.eps_data = json.load(f)
        self.eps_data_idx = None
        self.goal_name = None
        self.goal_best_position = None
        self.goal_shortest_distance = None
        self.starting_position = None
        self.eps_data_idx = 0

        replace_obj_name_file = os.path.join('data_preprocess', '{}_objectname_replaced.json'.format(self.split))
        if os.path.exists(replace_obj_name_file):
            with open(replace_obj_name_file, 'r') as f:
                self.replace_obj_names = json.load(f)
        else:
            self.replace_obj_names = None
        with open('data_preprocess/name_replace_dict.json', 'r') as f:
            self.replace_names = json.load(f)

        # Initializations
        self.episode_no = 0

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.stopped = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None

        self.success_dist = args.success_dist * 100.

        super().__init__(args)

    def load_new_episode(self):
        """The function loads a fixed episode from the episode dataset. This
        function is used for evaluating a trained model on the val split.
        """

        # Load episode info
        episode = self.eps_data[self.eps_data_idx]
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        pos = episode["start_position"]
        self.starting_position = pos

        self.goal_name = episode['goal']['name']
        self.goal_cat_id = episode['goal']['cat_id']
        self.goal_best_position = episode['goal']['best_position']
        self.goal_shortest_distance = episode['goal']['shortest_distance']
        self.prev_distance = self.goal_shortest_distance

        rgb, depth = self.get_obs()
        obs = {'rgb': rgb, 'depth': depth}
        return obs

    def reset(self):
        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.trajectory_states = []

        # Set info
        self.load_new_episode()

        msg = super().reset(self.starting_position)

        self.info['time'] = self.timestep
        self.info['yaw'] = msg.rotation.Yaw
        self.info['goal_name'] = self.goal_name
        self.info['goal_cat_id'] = self.goal_cat_id
        self.info['sensor_pose'] = [0., 0., 0.]

        rgb, depth = self.get_obs()
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = [self.msg.location.X, self.msg.location.Y, self.msg.rotation.Yaw * np.pi / 180.]

        return state, self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action = action["action"]
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            # action = 3

        rgb, depth, msg = super().step(action)
        done = self.get_done()
        rew = self.get_reward()

        spl, success, dist = 0., 0., 0.
        if done:
            spl, success, dist = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success

        rgb = rgb.astype(np.uint8)
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info['time'] = self.timestep
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, -do]

        return state, rew, done, self.info

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        dist = 100000
        g_x, g_y = self.msg.location.X, self.msg.location.Y
        for i in range(len(self.get_objects())):
            obj = self.msg.objects[i]
            name = obj.name

            if name in self.replace_names:
                name = self.replace_names[name]
            if self.replace_obj_names is not None:
                name = self.replace_obj_names[name]

            if name.lower() == self.goal_name.lower():
                o_x, o_y = obj.location.X, obj.location.Y
                dist_i = pu.get_l2_distance(g_x, g_y, o_x, o_y)
                if dist_i < dist:
                    dist = dist_i

        if dist < self.success_dist:
            success = 1
        else:
            success = 0
            dist = pu.get_l2_distance(g_x, g_y, self.goal_best_position[0], self.goal_best_position[1])
        spl = min(success * self.goal_shortest_distance / self.starting_distance, 1)
        return spl, success, dist / 100.

    def get_done(self):
        if self.info['time'] >= self.args.max_episode_length - 1:
            done = True
        elif self.starting_distance > 10000:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_reward(self):
        curr_loc_x, curr_loc_y = self.msg.location.X, self.msg.location.Y
        self.curr_distance = pu.get_l2_distance(curr_loc_x, curr_loc_y, self.goal_best_position[0],
                                                self.goal_best_position[1])

        reward = (self.prev_distance - self.curr_distance) * \
            self.args.reward_coeff

        self.prev_distance = self.curr_distance
        return reward

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = [self.msg.location.X, self.msg.location.Y, self.msg.rotation.Yaw * np.pi / 180.]
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        # convert to meters
        dx, dy = dx / 100., -dy / 100.
        return dx, dy, do


