"""
mission_node.py — Student entry point.
"""

import os
import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient

import yaml
import math
from pathlib import Path
from geometry_msgs.msg import PoseStamped

from geometry_msgs.msg import Twist, TwistStamped
from ament_index_python.packages import get_package_share_directory
from irobot_create_msgs.action import Undock
from std_msgs.msg import Empty

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose

# our imports
import py_trees
from py_trees.common import Status
from nav2_msgs.action import Spin
from std_srvs.srv import Empty as EmptyServer
from geometry_msgs.msg import PoseWithCovarianceStamped
from py_trees.decorators import FailureIsSuccess
from nav2_msgs.action import DriveOnHeading, BackUp
from geometry_msgs.msg import Point
import traceback
# ─────────────────────────────────────────────────────────────────────────────
# WHERE THE MISSION GOES, PER GRADE
#
# Every grade collects the cube from the same source box. They differ only in
# which box it is placed on, and config/shelves.yaml holds all of them.
#
# The grade comes from a ROS parameter rather than being written in here, so it
# cannot silently disagree with the grade the simulation was launched with. Set
# both from one place:
#
#     GRADE=c pixi run mission            # world, odometry, localization
#     GRADE=c pixi run mission-node       # this node
#
# MissionNode logs the grade it is running as, so a mismatch is visible in the
# first line of output rather than showing up as the robot driving to the wrong
# box twenty metres away.
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_BOX = 'shelf_7_ID11'

DROP_BOX_BY_GRADE = {
    'e': 'shelf_7_ID10',   # target-1, near the start
    'c': 'shelf_7_ID20',   # target-2, across the warehouse
    'a': 'shelf_7_ID20',   # target-2, same as C
}

ARM_SAFE = [0.0, -0.5, 0.2, 0.0, 0.0, 0.0] 
ARM_PICK = [0.0, 1.4, 1.85, 0.0, 0.0, 0.0]
ARM_PLACE = [0.0, 1.8, 2.5, 0.0, 0.0, 0.0]

COVARIANCE = 0.07

def load_shelf(name: str) -> PoseStamped:
    for shelf in load_shelves():
        if shelf['name'] == name:
            return shelf['pose']
    known = [s['name'] for s in load_shelves()]
    raise KeyError(f"no box named '{name}' in shelves.yaml; known boxes: {known}")


def load_shelves(priority_first: bool = False) -> list[dict]:
    pkg_share = get_package_share_directory('warehouse_inventory_robot')
    yaml_path = Path(pkg_share) / 'config' / 'shelves.yaml'

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    shelves = []
    for shelf in data.get('shelves', []):
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose.position.x = float(shelf['pose']['x'])
        ps.pose.position.y = float(shelf['pose']['y'])
        ps.pose.position.z = 0.0
        yaw = float(shelf['pose'].get('yaw', 0.0))
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        shelves.append({
            'name':      shelf['name'],
            'marker_id': int(shelf['marker_id']),
            'priority':  shelf.get('priority', 'normal'),
            'pose':      ps,
        })

    if priority_first:
        shelves.sort(key=lambda s: 0 if s['priority'] == 'high' else 1)

    return shelves

def load_home_base() -> PoseStamped:
    pkg_share = get_package_share_directory('warehouse_inventory_robot')
    yaml_path = Path(pkg_share) / 'config' / 'shelves.yaml'

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    hb = data['home_base']
    ps = PoseStamped()
    ps.header.frame_id = 'map'
    ps.pose.position.x = float(hb['pose']['x'])
    ps.pose.position.y = float(hb['pose']['y'])
    yaw = float(hb['pose'].get('yaw', 0.0))
    ps.pose.orientation.z = math.sin(yaw / 2.0)
    ps.pose.orientation.w = math.cos(yaw / 2.0)
    return ps



class ActionLeaf(py_trees.behaviour.Behaviour):
    def __init__(self, name, node, client, timeout=360.0):
        super().__init__(name)
        self.node = node
        self.client = client
        self.timeout = timeout

    def make_goal(self):
        raise NotImplementedError

    def initialise(self):
        self.goal_future = None
        self.result_future = None
        self.handle = None
        self.t = time.time()
        self.node.get_logger().info(f'[BT] start: {self.name}')

    def update(self):
        if time.time() > self.t + self.timeout:
            self.node.get_logger().error(f'{self.name} timed out !!')
            if self.handle is not None:
                self.handle.cancel_goal_async()
            return Status.FAILURE

        if self.goal_future is None:
            if self.client.server_is_ready():
                self.goal_future = self.client.send_goal_async(self.make_goal())
            return Status.RUNNING

        if self.result_future is None:
            if not self.goal_future.done():
                return Status.RUNNING
            
            self.handle = self.goal_future.result()

            if not self.handle.accepted:
                self.node.get_logger().error(f'{self.name} goal rejected!')
                return Status.FAILURE
            self.result_future = self.handle.get_result_async()
            return Status.RUNNING

        if not self.result_future.done():
            return Status.RUNNING

        result = self.result_future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().error(f'{self.name}: status {result.status}')
            return Status.FAILURE

        self.node.get_logger().info(
            f'[BT] done: {self.name} ({time.time() - self.t:.1f}s)')
        return Status.SUCCESS

    def terminate(self, status):
        if status == Status.INVALID and self.handle is not None:
            self.handle.cancel_goal_async()

class UndockLeaf(ActionLeaf):
    def make_goal(self):
        return Undock.Goal()
    

class NavigateTo(ActionLeaf):
    def __init__(self, name, node, pose):
        super().__init__(name, node, node._nav_client, timeout=1000.0)
        self.pose = pose

    def make_goal(self):
        goal = NavigateToPose.Goal()
        self.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose = self.pose
        return goal

class MoveArm(ActionLeaf):
    def __init__(self, name, node, angles, duration=5):
        super().__init__(name, node, node._arm_client, timeout=180.0)
        self.angles = angles
        self.duration = duration

    def make_goal(self):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [f'arm_joint{i}' for i in range(1, 7)]
        point = JointTrajectoryPoint()
        point.positions = self.angles
        point.time_from_start = Duration(sec=self.duration)
        goal.trajectory.points.append(point)
        return goal

class Vacuum(py_trees.behaviour.Behaviour):
    def __init__(self, name, node, enable):
        super().__init__(name)
        self.node = node
        self.enable = enable

    def initialise(self):
        if self.enable:
            self.node._attach_pub.publish(Empty())
        else:
            self.node._detach_pub.publish(Empty())
        self.t = time.time()

    def update(self):
        if time.time() > self.t + 2.0:
            return Status.SUCCESS
        return Status.RUNNING


class SpinLeaf(ActionLeaf):
    def __init__(self, name, node, yaw=3.0):
        super().__init__(name, node, node._spin_client, timeout=120.0)
        self.yaw = yaw

    def make_goal(self):
        goal = Spin.Goal()
        goal.target_yaw = self.yaw
        return goal


class Rescatter(py_trees.behaviour.Behaviour):
    def __init__(self, name, node):
        super().__init__(name)
        self.node = node

    def initialise(self):
        self.future = None

    def update(self):
        if self.future is None:
            if not self.node._rescatter_client.service_is_ready():
                return Status.RUNNING
            self.future = self.node._rescatter_client.call_async(EmptyServer.Request())
            return Status.RUNNING
        return Status.SUCCESS if self.future.done() else Status.RUNNING


class Check(py_trees.behaviour.Behaviour):
    def __init__(self, name, node, key, warn=False):
        super().__init__(name)
        self.node, self.key, self.warn = node, key, warn

    def update(self):
        ok = self.node.state[self.key]
        if not ok and self.warn:
            self.node.get_logger().warn(
                f'{self.name} FAILED, cov = {self.node.last_cov}')  
        return Status.SUCCESS if ok else Status.FAILURE
    
class SetFlag(py_trees.behaviour.Behaviour):
    def __init__(self, name, node, **flags):
        super().__init__(name)
        self.node, self.flags = node, flags

    def update(self):
        self.node.state.update(self.flags)
        self.node.get_logger().info(
            f'[BT] flags: {self.flags}')
        return Status.SUCCESS

# class DriveLeaf(ActionLeaf):
#     def __init__(self, name, node, dist=3.0, speed=0.3):
#         super().__init__(name, node, node._drive_client, timeout=60.0)
#         self.dist, self.speed = dist, speed

#     def make_goal(self):
#         goal = DriveOnHeading.Goal()
#         goal.target = Point(x=self.dist, y=0.0, z=0.0)
#         goal.speed = self.speed
#         goal.time_allowance = Duration(sec=10)
#         return goal
    
# class BackupLeaf(ActionLeaf):
#     def __init__(self, name, node, dist=2.6, speed=0.15):
#         super().__init__(name, node, node._backup_client, timeout=40.0)
#         self.dist, self.speed = dist, speed

#     def make_goal(self):
#         goal = BackUp.Goal()
#         goal.target = Point(x=self.dist, y=0.0, z=0.0)
#         goal.speed = self.speed
#         goal.time_allowance = Duration(sec=10)
#         return goal

class DriveLeaf(ActionLeaf):
    def __init__(self, name, node, dist=1.0, speed=0.2):
        super().__init__(name, node, node._drive_client, timeout=40.0)
        self.dist, self.speed = dist, speed

    def make_goal(self):
        goal = DriveOnHeading.Goal()
        goal.target = Point(x=self.dist, y=0.0, z=0.0)
        goal.speed = self.speed
        goal.time_allowance = Duration(sec=20)
        return goal


class BackupLeaf(ActionLeaf):
    def __init__(self, name, node, dist=1.0, speed=0.2):
        super().__init__(name, node, node._backup_client, timeout=40.0)
        self.dist, self.speed = dist, speed

    def make_goal(self):
        goal = BackUp.Goal()
        goal.target = Point(x=self.dist, y=0.0, z=0.0)
        goal.speed = self.speed
        goal.time_allowance = Duration(sec=20)
        return goal
    
# def move_a_bit(node, name):
#     return FailureIsSuccess(name=f'{name} move', child=sel(f'{name} drive or back', [
#         DriveLeaf(f'{name} drive', node),
#         BackupLeaf(f'{name} backup', node),
#     ], memory=True))

def move_a_bit(node, name):
    return FailureIsSuccess(name=f'{name} move',
                            child=DriveLeaf(f'{name} drive', node, dist=2.0))

def seq(name, children, memory=True):
    return py_trees.composites.Sequence(name=name, memory=memory, children=children)


def sel(name, children, memory=False):
    return py_trees.composites.Selector(name=name, memory=memory, children=children)


def ensure(node, key, action):
    return sel(f'ensure {key}', [Check(f'{key}?', node, key), action])


def undock(node):
    return seq('undock', [
        MoveArm('arm safe', node, ARM_SAFE),
        UndockLeaf('undock', node, node._undock_client),
        SetFlag('undocked!', node, undocked=True, arm_safe=False)])


def arm_to(node, name, angles, safe):
    return seq(name, [MoveArm(name, node, angles),
                      SetFlag(f'{name} done', node, arm_safe=safe)])


# def localize(node):
#     return seq('localize', [
#         Rescatter('global init', node),
#         py_trees.decorators.Retry('retry spins',
#             settle(node, 'localize', spins=6), num_failures=3),
#         SetFlag('localized!', node, localized=True)])

# def localize(node):
#     return py_trees.decorators.Retry('retry localize', seq('localize', [
#         Rescatter('global init', node),
#         settle(node, 'localize', rounds=3),
#         SetFlag('localized!', node, localized=True)]), num_failures=2)

# def localize(node):
#     return seq('localize', [
#         Rescatter('global init', node),
#         py_trees.decorators.Retry('retry settle',
#             settle(node, 'localize', rounds=3), num_failures=3),
#         SetFlag('localized!', node, localized=True)])

def localize(node):
    # fast check if not moved
    verify = seq('verify pose', [
        Check('converged?', node, 'converged'),
        SpinLeaf('verify spin', node, yaw=3.0),
        Check('still converged?', node, 'converged')])

    # rescatter if not
    full = seq('full localize', [
        Rescatter('global init', node),
        py_trees.decorators.Retry('retry settle',
            settle(node, 'localize', rounds=3), num_failures=3)])

    return seq('localize', [
        sel('verify or relocalize', [verify, full], memory=True),
        SetFlag('localized!', node, localized=True)])

def move_to(node, name, pose, here, other, refine=False):
    nav = [NavigateTo(name, node, pose)]
    if refine:
        nav += [NavigateTo(f'{name} refine', node, pose)]
    nav.append(SetFlag(f'{name} arrived', node, **{here: True, other: False}))

    preconditions = seq(f'{name} preconditions', [
        ensure(node, 'undocked', undock(node)),
        ensure(node, 'localized', localize(node)),
        ensure(node, 'arm_safe', arm_to(node, 'arm up', ARM_SAFE, True)),
        seq(f'{name} drive', nav),
    ], memory=False)
    return sel(f'{name}?', [Check(f'at {here}?', node, here), preconditions])

# def settle(node, name, spins=4):
#     steps = [FailureIsSuccess(name=f'{name} spin {i}', child=sel(
#              f'{name} spin? {i}',
#              [Check('converged?', node, 'converged'),
#               seq(f'{name} spin+drive {i}', [
#                 SpinLeaf(f'{name} spin {i}', node, yaw=1.5),
#                 move_a_bit(node, f'{name} {i}')])],
#              memory=True))
#          for i in range(spins)]
#     return seq(f'{name} settle',
#                [SetFlag(f'{name} forget', node, converged=False)]
#                + steps
#                + [Check(f'{name} really converged?', node, 'converged', warn=True)])

def spin_step(node, name, i):
    return FailureIsSuccess(name=f'{name} spin {i}', child=sel(
        f'{name} spin? {i}',
        [Check('converged?', node, 'converged'),
         SpinLeaf(f'{name} spin {i}', node, yaw=1.5)],
        memory=True))



def settle(node, name, rounds=3):
    steps = [
        sel(f'{name} round {i}', [
            Check('converged?', node, 'converged'),
            seq(f'{name} look and move {i}', [
                FailureIsSuccess(name=f'{name} spin {i}',
                    child=SpinLeaf(f'{name} spin {i}', node, yaw=3.0)),
                sel(f'{name} done or move {i}', [
                    Check('converged?', node, 'converged'),
                    move_a_bit(node, f'{name} {i}')]),
            ]),
        ], memory=True)
        for i in range(rounds)
    ]
    return seq(f'{name} settle',
               [SetFlag(f'{name} forget', node, converged=False)]
               + steps
               + [Check(f'{name} really converged?', node, 'converged', warn=True)])


def build_full_tree(node, pick_pose, drop_pose):
    get_cube = sel('have cube?', [
        Check('has_cube?', node, 'has_cube'),
        seq('get cube', [
            move_to(node, 'go to cube', pick_pose, 'at_source', 'at_target'),
            arm_to(node, 'arm to pick', ARM_PICK, False),
            Vacuum('grab', node, True),
            arm_to(node, 'arm picked up', ARM_SAFE, True),
            SetFlag('got cube', node, has_cube=True)])])

    place_cube = sel('placed?', [
        Check('placed?', node, 'placed'),
        seq('place cube', [
            move_to(node, 'go target', drop_pose, 'at_target', 'at_source', refine=True),
            arm_to(node, 'arm place down', ARM_PLACE, False),
            Vacuum('release', node, False),
            arm_to(node, 'arm back up', ARM_SAFE, True),
            SetFlag('cube placed', node, placed=True)])])

    return seq('Mission', [get_cube, place_cube])


class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')

        # Which grade this run is for. Read from the GRADE environment
        # variable, so one spelling works whether you go through
        # `GRADE=c pixi run mission-node` or call ros2 run yourself inside a
        # pixi shell. An explicit -p grade:=c overrides it. Use the same value
        # you launched the simulation with.
        self.declare_parameter('grade', os.environ.get('GRADE', 'e'))
        self.grade = str(self.get_parameter('grade').value).strip().lower()
        if self.grade not in DROP_BOX_BY_GRADE:
            self.get_logger().warn(
                f"Unknown grade '{self.grade}'; falling back to 'e'. "
                f"Valid grades: {sorted(DROP_BOX_BY_GRADE)}")
            self.grade = 'e'

        self.source_box = SOURCE_BOX
        self.drop_box = DROP_BOX_BY_GRADE[self.grade]

        self.get_logger().info(
            f"Mission node started for grade '{self.grade}'. "
            f'Collect from {self.source_box}, place on {self.drop_box}.')

        self._attach_pub = self.create_publisher(Empty, '/vacuum_gripper/attach', 10)
        self._detach_pub = self.create_publisher(Empty, '/vacuum_gripper/detach', 10)
        self._undock_client = ActionClient(self, Undock, '/undock')
        self._arm_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/lite6_traj_controller/follow_joint_trajectory'
        )

        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._spin_client = ActionClient(self, Spin, 'spin')
        self._rescatter_client = self.create_client(EmptyServer, '/reinitialize_global_localization')
        self.state = dict(undocked=False,
                        localized=False,
                        converged=False,
                        arm_safe=False,
                        at_source=False,
                        at_target=False,
                        has_cube=False,
                        placed=False)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10)
        self.last_cov = None
        self._drive_client = ActionClient(self, DriveOnHeading, 'drive_on_heading')
        self._backup_client = ActionClient(self, BackUp, 'backup')

    def undock_robot(self):
        # TODO: Implement undocking logic using the Undock action client (self._undock_client).
        #       Return True once the base is undocked, False if it refused.

        if not self._undock_client.wait_for_server(timeout_sec=500.0):
            self.get_logger().error('undock action server is not available')
            return False

        send = self._undock_client.send_goal_async(Undock.Goal())
        rclpy.spin_until_future_complete(self, send, timeout_sec=100.0)
        if not send.done():
            self.get_logger().error('undock goal not acknowledged')
            return False
        
        acceptance = send.result()
        if not acceptance.accepted:
            self.get_logger().error('undock goal rejected')
            return False
        
        future = acceptance.get_result_async()
        rclpy.spin_until_future_complete(self, future, timeout_sec=300.0)
        if not future.done():
            self.get_logger().error('undock try timed out')
            return False

        result = future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f'undock ended with bad status {result.status}')
            return False

        if result.result.is_docked:
            self.get_logger().error('undocked but still docked (?)')
            return False
        
        return True
        # raise NotImplementedError('undock_robot() is yours to write')

    def go_to_pose(self, pose_stamped):
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f"Navigating to x: {pose_stamped.pose.position.x}, y: {pose_stamped.pose.position.y}")
        # TODO: Implement navigation to the given pose using Nav2's NavigateToPose action.
        #       Return True once the robot has arrived, False if it did not. Callers
        #       read the return value as "did this work", so falling off the end and
        #       returning None counts as failure.

        if not self._nav_client.wait_for_server(timeout_sec=50.0):
            self.get_logger().error('navigation server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped
        send = self._nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=50.0)
        if not send.done():
            self.get_logger().error('navigation goal not acknowledged')
            return False

        acceptance = send.result()
        if not acceptance.accepted:
            self.get_logger().error('navigation goal rejected')
            return False

        future = acceptance.get_result_async()
        rclpy.spin_until_future_complete(self, future, timeout_sec=360.0)
        if not future.done():
            self.get_logger().error('navigation reached max time')
            acceptance.cancel_goal_async()
            return False

        result = future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f'navigation finished with status {result.status}')
            return False
        return True

        

    def toggle_vacuum(self, enable=True):
        state = "ENGAGING" if enable else "RELEASING"
        self.get_logger().info(f'{state} vacuum gripper...')
        (self._attach_pub if enable else self._detach_pub).publish(Empty())
        time.sleep(1.5)

    def move_arm_to_joint_angles(self, angles, duration_sec=4):
        """Generic helper function to send the arm to any 6-DOF joint configuration."""
        if not self._arm_client.wait_for_server(timeout_sec=120.0):
            self.get_logger().error(
                'Arm action server /lite6_traj_controller/follow_joint_trajectory '
                'never appeared. Is lite6_traj_controller active? Check with: '
                'ros2 control list_controllers')
            return False


        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            'arm_joint1', 'arm_joint2', 'arm_joint3', 
            'arm_joint4', 'arm_joint5', 'arm_joint6'
        ]

        point = JointTrajectoryPoint()
        point.positions = angles 
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        goal_msg.trajectory.points.append(point)

        # Every wait below is bounded, so a misbehaving controller will show an
        # error instead of an indefinite hang with no output.
        send_goal_future = self._arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=15.0)
        if not send_goal_future.done():
            self.get_logger().error('Arm trajectory goal was never acknowledged!')
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Arm trajectory goal was rejected!')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=180.0)
        if not result_future.done():
            self.get_logger().error('Arm trajectory did not finish in time!')
            return False
        return True

    def _amcl_cb(self, msg):
        covariance = msg.pose.covariance
        self.state['converged'] = (covariance[0] < COVARIANCE )and (covariance[7] < COVARIANCE) and (covariance[35] < COVARIANCE)
        self.last_cov = (covariance[0], covariance[7])
        self.get_logger().info(
            f'[AMCL] cov x={covariance[0]:.3f} y={covariance[7]:.3f} '
            f'yaw={covariance[35]:.3f} converged={self.state["converged"]}',
            throttle_duration_sec=2.0)
    # =========================================================================
    # THE MAIN MISSION SEQUENCE
    # =========================================================================


    def run_mission(self):
        self.get_logger().info('Starting mission.')

        # Ensure gripper starts in a known-detached state
        time.sleep(10.0)  # allow ROS->bridge->gz discovery to complete across all hops
        self._detach_pub.publish(Empty())
        time.sleep(10.0)

        # Load waypoints. Which box the cube is placed on depends on the grade,
        # so both boxes are looked up BY NAME through the constants at the top
        # of this file rather than by their position in shelves.yaml. Indexing
        # into the list instead would tie the mission to the order of that file
        # and quietly send the robot to the wrong box when it changed.
        home_base = load_home_base()
        pick_pose = load_shelf(self.source_box)
        drop_pose = load_shelf(self.drop_box)

        tree = py_trees.trees.BehaviourTree(build_full_tree(self, pick_pose, drop_pose))
        snapshot = py_trees.visitors.SnapshotVisitor()
        tree.visitors.append(snapshot)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            tree.tick()

            if snapshot.visited != snapshot.previously_visited:
                print(py_trees.display.unicode_tree(
                    tree.root, show_status=True,
                    visited=snapshot.visited,
                    previously_visited=snapshot.previously_visited,
                    show_only_visited=True))

            status = tree.root.status
            if status in (Status.SUCCESS, Status.FAILURE):
                self.get_logger().info(f'Mission finished: {status.name}')
                print(py_trees.display.unicode_tree(tree.root, show_status=True))
                break

        return True



def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()

    try:
        node.run_mission()
    except KeyboardInterrupt:
        node.get_logger().info("Mission interrupted by user.")
    except Exception as e:
        node.get_logger().fatal(f"Mission failed: {str(e)}")
        node.get_logger().fatal(f"Mission failed:\n{traceback.format_exc()}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()