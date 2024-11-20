"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # Run environment step
        env.step(action)
        env.render()
        # print(env.robots[0].recent_ee_pose.last[3:])

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="robot0_eye_in_hand", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)


# sim.model.camera_name2id(camera_name)
# def quaternion_multiply_xyzw(q, r):
#     x = q[3] * r[0] + q[0] * r[3] + q[1] * r[2] - q[2] * r[1]
#     y = q[3] * r[1] - q[0] * r[2] + q[1] * r[3] + q[2] * r[0]
#     z = q[3] * r[2] + q[0] * r[1] - q[1] * r[0] + q[2] * r[3]
#     w = q[3] * r[3] - q[0] * r[0] - q[1] * r[1] - q[2] * r[2]
#     return np.array([x, y, z, w])

"""
def make_pose(translation, rotation):

    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.
    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation
    Returns:
        pose (np.array): a 4x4 homogeneous matrix

    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose
PyDev console: using IPython 8.18.1
def mat2quat(rmat):

    Converts given rotation matrix to quaternion.
    Args:
        rmat (np.array): 3x3 rotation matrix
    Returns:
        np.array: (x,y,z,w) float quaternion angles

    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]
r = env.env.viewer.sim.data.cam_xmat[5].reshape(3,3)
t = env.env.viewer.sim.data.cam_xpos[5]
camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
R = pose @ camera_axis_correction
Traceback (most recent call last):
  File "/home/bocehu/mambaforge/envs/equidiffpo/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 3550, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-5-8ef7e3bef29a>", line 4, in <module>
    R = pose @ camera_axis_correction
NameError: name 'pose' is not defined
pose = make_pose(t,r)
camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
R = pose @ camera_axis_correction
R 
Out[8]: 
array([[ 4.96537180e-05, -9.91483967e-01,  1.30228805e-01,
        -8.69140046e-02],
       [-9.99999950e-01, -8.99621509e-05, -3.03637239e-04,
        -9.62390857e-03],
       [ 3.12767117e-04, -1.30228783e-01, -9.91483921e-01,
         1.11610486e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
Q = mat2quat(R)
env.robots[0].recent_ee_pose.last[3:]
Out[10]: array([0.70561224, 0.70558077, 0.04606798, 0.0463133 ])
Q 
Out[11]: array([-0.70562446,  0.705575  , -0.04625037,  0.04603197], dtype=float32)
r = env.env.viewer.sim.data.cam_xmat[5].reshape(3,3)
q  = mat2quat(r)
q 
Out[14]: array([ 0.04603197, -0.04625037, -0.705575  ,  0.70562446], dtype=float32)
Demonstration is unsuccessful and has NOT been saved
DataCollectionWrapper: making folder at /tmp/1730063585_7150733/ep_1730064289_4663804
r = env.env.viewer.sim.data.cam_xmat[5].reshape(3,3)
t = env.env.viewer.sim.data.cam_xpos[5]
pose = make_pose(t,r)
camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
R = pose @ camera_axis_correction
Q = mat2quat(R)
Q 
Out[21]: array([-0.5985041 ,  0.7385595 , -0.30813602,  0.03707973], dtype=float32)
env.robots[0].recent_ee_pose.last[3:]
Out[22]: array([0.73844796, 0.59854084, 0.03736645, 0.30829725])
def quaternion_multiply_xyzw(q, r):
    x = q[3] * r[0] + q[0] * r[3] + q[1] * r[2] - q[2] * r[1]
    y = q[3] * r[1] - q[0] * r[2] + q[1] * r[3] + q[2] * r[0]
    z = q[3] * r[2] + q[0] * r[1] - q[1] * r[0] + q[2] * r[3]
    w = q[3] * r[3] - q[0] * r[0] - q[1] * r[1] - q[2] * r[2]
    return np.array([x, y, z, w])
q_unknown = quaternion_multiply_xyzw(np.array([0, 0, 1, 0]), Q)
q_unknow
Traceback (most recent call last):
  File "/home/bocehu/mambaforge/envs/equidiffpo/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 3550, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-25-99f4d2e38c6c>", line 1, in <module>
    q_unknow
NameError: name 'q_unknow' is not defined
q_unknown
Out[26]: array([-0.73855948, -0.59850413,  0.03707973,  0.30813602])
env.robots[0].recent_ee_pose.last[3:]
Out[27]: array([0.73844796, 0.59854084, 0.03736645, 0.30829725])
Q 
Out[28]: array([-0.5985041 ,  0.7385595 , -0.30813602,  0.03707973], dtype=float32)
q_unknown = quaternion_multiply_xyzw(np.array([0, 0, 1, 0]), env.robots[0].recent_ee_pose.last[3:])
q_unknown
Out[30]: array([-0.59854084,  0.73844796,  0.30829725, -0.03736645])
q_unknown = quaternion_multiply_xyzw(np.array([0,0,0.7071,0.7071]), env.robots[0].recent_ee_pose.last[3:])
q_unknown
Out[32]: array([0.09892833, 0.94538479, 0.2444188 , 0.19157517])
q_unknown = quaternion_multiply_xyzw(np.array([0, 0, -1, 0]), env.robots[0].recent_ee_pose.last[3:])
q_unknown
Out[34]: array([ 0.59854084, -0.73844796, -0.30829725,  0.03736645])
r 
Out[35]: 
array([[-0.28083581,  0.86121057, -0.42361255],
       [-0.90691298, -0.09369   ,  0.41076882],
       [ 0.3140702 ,  0.49953832,  0.80735456]])
env.robots[0].recent_ee_pose.last[3:]
Out[36]: array([0.73844796, 0.59854084, 0.03736645, 0.30829725])
r = env.env.viewer.sim.data.cam_xmat[5].reshape(3,3)
t = env.env.viewer.sim.data.cam_xpos[5]
pose = make_pose(t,r)
camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
R = pose @ camera_axis_correction
Q = mat2quat(R)
Q 
Out[42]: array([-0.5985041 ,  0.7385595 , -0.30813602,  0.03707973], dtype=float32)
env.robots[0].recent_ee_pose.last[3:]
Out[43]: array([0.73844796, 0.59854084, 0.03736645, 0.30829725])
q_unknown = quaternion_multiply_xyzw(np.array([0, 0, 1, 0]), Q)
q_unknown
Out[45]: array([-0.73855948, -0.59850413,  0.03707973,  0.30813602])
q_unknown = quaternion_multiply_xyzw(np.array([0, 0, -1, 0]), Q)
q_unknown
Out[47]: array([ 0.73855948,  0.59850413, -0.03707973, -0.30813602])
Q
Out[48]: array([-0.5985041 ,  0.7385595 , -0.30813602,  0.03707973], dtype=float32)
q_unknown
Out[49]: array([ 0.73855948,  0.59850413, -0.03707973, -0.30813602])
env.robots[0].recent_ee_pose.last[3:]
Out[50]: array([0.73844796, 0.59854084, 0.03736645, 0.30829725])
q_unknown2 = quaternion_multiply_xyzw(np.array([0.7071,0,0,0.7071]), Q)
q_unknown2 
Out[52]: array([-0.39698319,  0.74011839,  0.30435243,  0.44942134])
R 
Out[53]: 
array([[-0.28083581, -0.86121057,  0.42361255, -0.11272299],
       [-0.90691298,  0.09369   , -0.41076882, -0.13176905],
       [ 0.3140702 , -0.49953832, -0.80735456,  1.12337252],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
q_unknown2 = quaternion_multiply_xyzw(np.array([0,0, -0.7071,0.7071]), Q)
q_unknown2 
Out[55]: array([ 0.09903314,  0.94543768, -0.24410205, -0.1916639 ])
R_z = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
R 
Out[57]: 
array([[-0.28083581, -0.86121057,  0.42361255, -0.11272299],
       [-0.90691298,  0.09369   , -0.41076882, -0.13176905],
       [ 0.3140702 , -0.49953832, -0.80735456,  1.12337252],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
M_rotated = np.dot(R, R_z)
Q = mat2quat(M_rotated)
Q 
Out[60]: array([ 0.9454467 , -0.09903408,  0.24410442,  0.19166575], dtype=float32)
Q = mat2quat(R)
Q 
Out[62]: array([-0.5985041 ,  0.7385595 , -0.30813602,  0.03707973], dtype=float32)
env.robots[0].recent_ee_pose.last[3:]
Out[63]: array([0.73844796, 0.59854084, 0.03736645, 0.30829725])
q_unknown = quaternion_multiply_xyzw(np.array([0, 0, -1, 0]), Q)
q_unknown
Out[65]: array([ 0.73855948,  0.59850413, -0.03707973, -0.30813602])
q_unknown = quaternion_multiply_xyzw(Q, np.array([0, 0, -1, 0]))
q_unknown
Out[67]: array([-0.73855948, -0.59850413, -0.03707973, -0.30813602])
q_unknown = quaternion_multiply_xyzw(Q, np.array([0, 0, 1, 0]))
q_unknown
Out[69]: array([0.73855948, 0.59850413, 0.03707973, 0.30813602])

def mat2quat(rmat):

    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def make_pose(translation, rotation):

    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose

r = env.env.viewer.sim.data.cam_xmat[5].reshape(3,3)
t = env.env.viewer.sim.data.cam_xpos[5]
pose = make_pose(t,r)
camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
R = pose @ camera_axis_correction
Q = mat2quat(R)
Q_final = quaternion_multiply_xyzw(Q, np.array([0, 0, 1, 0]))


"""
