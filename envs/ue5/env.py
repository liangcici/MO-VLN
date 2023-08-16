import time
import random
import numpy as np
import grpc
import GrabSim_pb2_grpc
import GrabSim_pb2

import envs.utils.pose as pu


class Env():

    def __init__(self, args):
        channel = grpc.insecure_channel('localhost:30001',options=[
                    ('grpc.max_send_message_length', 1024*1024*1024),
                    ('grpc.max_receive_message_length', 1024*1024*1024)
                ])

        self.stub = GrabSim_pb2_grpc.GrabSimStub(channel)
        self.version = args.version
        self.num_processes = args.num_processes
        self.map_id = args.map_id

        self.initworld = self.stub.SetWorld(GrabSim_pb2.BatchMap(count=self.num_processes, mapID=self.map_id))

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.matrix = None
        self.msg = self.set_world()

        self.turn_angle = args.turn_angle
        self.forward_cm = args.forward_cm
        self.starting_distance = 0.000001

    def reset(self, location):
        msg = self.set_world()
        msg = self.stub.Do(GrabSim_pb2.Action(
                    scene=0,
                    action=GrabSim_pb2.Action.ActionType.WalkTo,
                    values=[location[0], location[1], location[2], -1, 500]
                ))
        self.msg = msg
        self.starting_distance = 0.000001
        return self.msg

    def set_world(self):
        msg = self.stub.Reset(GrabSim_pb2.ResetParams())
        # msg = self.stub.Reset(GrabSim_pb2.ResetParams())
        # msg = self.stub.Reset(GrabSim_pb2.ResetParams())
        time.sleep(3)
        return msg

    def reset_world(self, location):
        msg = self.set_world()
        p_x, p_y, yaw = location
        msg = self.stub.Do(GrabSim_pb2.Action(
            action=GrabSim_pb2.Action.WalkTo,
            values=[p_x, p_y, yaw, -1, 350]
        ))
        print('reset world', msg.info, msg.location)
        while msg.info == 'Unreachable':
            p_x += random.uniform(-1., 1.) * 100.
            p_y += random.uniform(-1., 1.) * 100.
            msg = self.set_world()
            msg = self.stub.Do(GrabSim_pb2.Action(
                action=GrabSim_pb2.Action.ActionType.WalkTo,
                values=[p_x, p_y, yaw, -1, 500]
            ))
            print('reset world', msg.info, msg.location)
        self.msg = msg
        return self.msg

    def get_obs(self):
        images = self.stub.Capture(GrabSim_pb2.CameraList(scene=0, cameras=[
            GrabSim_pb2.CameraName.Head_Depth, GrabSim_pb2.CameraName.Head_Color,
            GrabSim_pb2.CameraName.Head_Segment
        ])).images
        depth = np.frombuffer(images[0].data, dtype=images[0].dtype).reshape(
            (images[0].height, images[0].width, images[0].channels))
        rgb = np.frombuffer(images[1].data, dtype=images[1].dtype).reshape(
            (images[1].height, images[1].width, images[1].channels))
        # convert to BGR format
        rgb = rgb[:, :, [2, 1, 0]]

        self.fx = images[0].parameters.fx
        self.fy = images[0].parameters.fy
        self.cx = images[0].parameters.cx
        self.cy = images[0].parameters.cy
        self.matrix = np.array(images[0].parameters.matrix).reshape((4, 4)).transpose()

        return rgb, depth

    def get_seg(self):
        message = self.stub.Capture(GrabSim_pb2.CameraList(scene=0, cameras=[
            GrabSim_pb2.CameraName.Head_Segment]))
        images = message.images
        seg = np.frombuffer(images[0].data, dtype=images[0].dtype).reshape(
            (images[0].height, images[0].width, images[0].channels))

        items = message.info.split(';')
        seg_object_names = {}
        for item in items:
            key, value = item.split(':')
            seg_object_names[int(key)] = value
        return seg, seg_object_names

    def get_objects(self):
        scene = self.stub.Observe(GrabSim_pb2.SceneID(value=0))
        return scene.objects

    def inquire_length(self, o_x, o_y):
        msg = self.stub.Do(GrabSim_pb2.Action(
                                scene=0,
                                action=GrabSim_pb2.Action.ActionType.WalkTo,
                                values=[o_x, o_y, 0, 0, 350]
                            ))
        if msg.info == 'Unreachable':
            return -1
        else:
            message = msg.info.split(';')
            distance = float(message[0][22:])
            return distance

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (int): 0: stop, 1: forward (0.5m), 2: left, 3: right
        """
        scene = self.msg
        p_x, p_y = scene.location.X, scene.location.Y
        if action == 1:
            yaw = scene.rotation.Yaw * np.pi / 180.
            new_p_x = p_x + self.forward_cm * np.cos(yaw)
            new_p_y = p_y + self.forward_cm * np.sin(yaw)

            self.msg = self.stub.Do(GrabSim_pb2.Action(
                scene=0,
                action=GrabSim_pb2.Action.ActionType.WalkTo,
                values=[new_p_x, new_p_y, scene.rotation.Yaw, -1, 100]
            ))
        elif action == 2:
            new_yaw = scene.rotation.Yaw - self.turn_angle
            self.msg = self.stub.Do(GrabSim_pb2.Action(
                scene=0,
                action=GrabSim_pb2.Action.ActionType.WalkTo,
                values=[p_x, p_y, new_yaw, -1, 100]
            ))
        elif action == 3:
            new_yaw = scene.rotation.Yaw + self.turn_angle
            self.msg = self.stub.Do(GrabSim_pb2.Action(
                scene=0,
                action=GrabSim_pb2.Action.ActionType.WalkTo,
                values=[p_x, p_y, new_yaw, -1, 100]
            ))

        print('action: ', action)
        message = self.msg.info
        print(message)
        print('position: ', self.msg.location, '; rotation: ', self.msg.rotation.Yaw)

        if message == 'Unreachable':
            print('Cannot reach!')
        elif message == 'Failed':
            raise ValueError('Failed! Restart the simulator!')
        elif action == 1 and message != 'AlreadyAtGoal':
            self.starting_distance += pu.get_l2_distance(p_x, p_y, self.msg.location.X, self.msg.location.Y)
        rgb, depth = self.get_obs()

        if action in [2, 3] and message == 'Unreachable':
            self.msg = self.reset_world(location=(self.msg.location.X + random.uniform(-10., 10.),
                                                  self.msg.location.Y + random.uniform(-10., 10.), new_yaw + 1))
        elif action == 1 and message == 'AlreadyAtGoal':
            self.msg = self.reset_world(location=(new_p_x + random.uniform(-1., 1.) * 100.,
                                                  new_p_y + random.uniform(-1., 1.) * 100.,
                                                  self.msg.rotation.Yaw + 1))

        return rgb, depth, self.msg


