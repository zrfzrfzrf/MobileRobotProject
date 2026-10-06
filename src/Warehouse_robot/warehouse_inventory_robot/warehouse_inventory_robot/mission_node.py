"""
mission_node.py — Student entry point (grade C: behaviour tree).

Layout of this file
  1. constants            where to go, which arm poses              (arm poses: TODO(core))
  2. loaders              shelves.yaml -> PoseStamped               (given)
  3. AsyncActionCall      non-blocking action wrapper               (given, read it!)
  4. MissionNode          ROS clients / publishers / tick loop      (given)
  5. behaviour leaves     UndockLeaf is a worked example; the rest  (TODO(core))
  6. build_tree()         assemble the tree                         (TODO(core))
  7. main()

Search for TODO(core) to find everything that is left for you.

Rule of thumb for a BT: update() is called on every tick (10 Hz) and must
return immediately. Never call time.sleep() or spin_until_future_complete()
inside a leaf; start the work in initialise(), poll it in update(), and
return RUNNING until it finishes.
"""

import os
import math
from pathlib import Path

import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

import py_trees
from py_trees.common import Status

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectoryPoint

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
#
# The grade comes from a ROS parameter / the GRADE environment variable, so it
# cannot silently disagree with the grade the simulation was launched with:
#
#     GRADE=c pixi run mission            # world, odometry, nav stack
#     GRADE=c pixi run mission-node       # this node
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_BOX = 'shelf_7_ID11'

DROP_BOX_BY_GRADE = {
    'e': 'shelf_7_ID10',   # target-1, near the start
    'c': 'shelf_7_ID20',   # target-2, across the warehouse
    'a': 'shelf_7_ID20',   # target-2, same as C
}

TICK_PERIOD_SEC = 0.1          # the BT is ticked at 10 Hz (simulated time)
UNDOCK_TIMEOUT_SEC = 60.0      # all timeouts below are measured in SIM time
NAV_TIMEOUT_SEC = 400.0
ARM_TIMEOUT_SEC = 60.0
VACUUM_SETTLE_SEC = 1.5        # wait after attach/detach so the cube settles

ARM_JOINT_NAMES = [
    'arm_joint1', 'arm_joint2', 'arm_joint3',
    'arm_joint4', 'arm_joint5', 'arm_joint6',
]

# Joint angles in radians, in the order of ARM_JOINT_NAMES.
#
# ARM_SAFE is the pose the arm spawns in (config/initial_positions.yaml). Forward
# kinematics from the URDF puts its tool tip 0.46 m ahead of and 0.73 m above the
# robot, so it is NOT compact. It is only a known-good starting point; see
# ARM_SAFE_COMPACT if the base rocks while driving.
ARM_SAFE = [0.0, 0.87, 1.57, 0.0, -1.57, 0.0]

# ESTIMATES from numerical inverse kinematics on the URDF (tool pointing straight
# down, joint 6 left at 0), assuming the robot stands exactly on the goal pose in
# shelves.yaml: tool tip 0.34 m ahead of base_link and level with the cube top
# (z = 0.38 m). NOT yet tried in the simulator; verify, then nudge by hand.
ARM_PICK = [-0.059, 1.274, 1.568, 0.0, 0.293, 0.0]
# Both boxes are approached with the same stand-off, so the cube is carried back
# to the same spot relative to the robot, which is now above the target box.
ARM_PLACE = ARM_PICK

# Hover 12 cm above ARM_PICK. Going straight from ARM_SAFE to ARM_PICK can sweep
# the tool through the box; move here first, then down (and back up after).
ARM_PRE_PICK = [-0.059, 0.831, 1.378, 0.0, 0.547, 0.0]

# Tool tip 0.15 m ahead of and 0.62 m above base_link, over the robot body.
ARM_SAFE_COMPACT = [0.0, -0.417, 0.466, 0.0, 0.883, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOADERS (given)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 3. NON-BLOCKING ACTION CALL (given)
#
# ROS actions are asynchronous: you send a goal, the server accepts it, works
# for a while, then returns a result. A BT leaf can't block for that, so this
# small state machine hides the plumbing:
#
#     start(goal)  ->  WAITING  (server not up yet; retried on every poll)
#                  ->  SENDING  (goal sent, waiting for accept/reject)
#                  ->  ACTIVE   (accepted, waiting for the result)
#                  ->  SUCCEEDED | FAILED
#
# Call poll() once per tick; it never blocks. A call that takes longer than
# timeout_sec of sim time is cancelled and reported as FAILED.
# ─────────────────────────────────────────────────────────────────────────────

class AsyncActionCall:
    IDLE = 'IDLE'
    WAITING = 'WAITING'
    SENDING = 'SENDING'
    ACTIVE = 'ACTIVE'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'

    def __init__(self, ros_node, client, label, timeout_sec):
        self._ros = ros_node
        self._client = client
        self._label = label
        self._timeout = timeout_sec
        self.state = self.IDLE
        self.result = None          # the action's Result message, once finished
        self._goal = None
        self._goal_future = None
        self._handle = None
        self._result_future = None
        self._t0 = 0.0

    def start(self, goal):
        self.cancel()
        self._goal = goal
        self.result = None
        self._goal_future = None
        self._handle = None
        self._result_future = None
        self._t0 = self._ros.now_sec()
        self.state = self.WAITING

    def poll(self) -> str:
        if self.state in (self.WAITING, self.SENDING, self.ACTIVE):
            if self._ros.now_sec() - self._t0 > self._timeout:
                self._ros.get_logger().error(
                    f'[{self._label}] timed out after {self._timeout:.0f}s (sim time)')
                self.cancel()
                self.state = self.FAILED
                return self.state

        if self.state == self.WAITING and self._client.server_is_ready():
            self._goal_future = self._client.send_goal_async(self._goal)
            self.state = self.SENDING

        if self.state == self.SENDING and self._goal_future.done():
            self._handle = self._goal_future.result()
            if not self._handle.accepted:
                self._ros.get_logger().error(f'[{self._label}] goal rejected')
                self.state = self.FAILED
            else:
                self._result_future = self._handle.get_result_async()
                self.state = self.ACTIVE

        if self.state == self.ACTIVE and self._result_future.done():
            wrapped = self._result_future.result()
            self.result = wrapped.result
            if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
                self.state = self.SUCCEEDED
            else:
                self._ros.get_logger().warn(
                    f'[{self._label}] finished with action status {wrapped.status}')
                self.state = self.FAILED

        return self.state

    def cancel(self):
        """Abort an in-flight call. Safe to call in any state."""
        if self.state == self.SENDING and self._goal_future is not None:
            def _cancel_when_accepted(fut):
                handle = fut.result()
                if handle.accepted:
                    handle.cancel_goal_async()
            self._goal_future.add_done_callback(_cancel_when_accepted)
            self.state = self.IDLE
        elif self.state == self.ACTIVE and self._handle is not None:
            self._handle.cancel_goal_async()
            self.state = self.IDLE
        elif self.state == self.WAITING:
            self.state = self.IDLE

    @property
    def in_flight(self) -> bool:
        return self.state in (self.WAITING, self.SENDING, self.ACTIVE)


# ─────────────────────────────────────────────────────────────────────────────
# 4. THE ROS NODE (given)
#
# Owns every ROS client / publisher / subscriber and the tick loop. Leaves
# reach the robot only through this object (they hold it as self.ros).
# ─────────────────────────────────────────────────────────────────────────────

class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')

        # Read from the GRADE environment variable, so one spelling works
        # whether you go through `GRADE=c pixi run mission-node` or call
        # ros2 run yourself inside a pixi shell. An explicit -p grade:=c wins.
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

        # Actuators / interfaces -------------------------------------------
        self._attach_pub = self.create_publisher(Empty, '/vacuum_gripper/attach', 10)
        self._detach_pub = self.create_publisher(Empty, '/vacuum_gripper/detach', 10)
        self.undock_client = ActionClient(self, Undock, '/undock')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/lite6_traj_controller/follow_joint_trajectory')

        # True / False once the first /dock_status message arrives, None before.
        # Handy as a BT condition ("is the robot undocked?").
        self.is_docked = None
        self.create_subscription(
            DockStatus, 'dock_status', self._on_dock_status, qos_profile_sensor_data)

        # BT bookkeeping -----------------------------------------------------
        self.tree = None
        self.done = False
        self.success = False
        self._last_snapshot = ''
        self._timer = None

    # --- small helpers leaves use ------------------------------------------

    def now_sec(self) -> float:
        """Current time in seconds on the node clock (sim time with use_sim_time)."""
        return self.get_clock().now().nanoseconds / 1e9

    def _on_dock_status(self, msg: DockStatus):
        self.is_docked = bool(msg.is_docked)

    def set_vacuum(self, enable: bool):
        """Publish attach/detach once. Does not wait; the leaf must wait itself."""
        state = 'ENGAGING' if enable else 'RELEASING'
        self.get_logger().info(f'{state} vacuum gripper...')
        (self._attach_pub if enable else self._detach_pub).publish(Empty())

    def make_nav_goal(self, pose: PoseStamped) -> NavigateToPose.Goal:
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        return goal

    def make_arm_goal(self, angles, duration_sec=4) -> FollowJointTrajectory.Goal:
        if angles is None:
            raise ValueError('arm pose is None; fill in ARM_PICK / ARM_PLACE at the top')
        if len(angles) != len(ARM_JOINT_NAMES):
            raise ValueError(f'expected {len(ARM_JOINT_NAMES)} joint angles, got {len(angles)}')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(a) for a in angles]
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=0)
        goal.trajectory.points.append(point)
        return goal

    # --- the tick loop ------------------------------------------------------

    def run_mission(self):
        """Build the tree and start ticking it. Returns immediately; main() spins."""
        self.get_logger().info('Starting mission.')

        # Boxes are looked up BY NAME (constants above), never by position in
        # the yaml, so reordering shelves.yaml cannot send the robot to the
        # wrong box.
        home_base = load_home_base()
        pick_pose = load_shelf(self.source_box)
        drop_pose = load_shelf(self.drop_box)

        root = build_tree(self, home_base, pick_pose, drop_pose)
        self.tree = py_trees.trees.BehaviourTree(root)
        self._timer = self.create_timer(TICK_PERIOD_SEC, self._tick)

    def _tick(self):
        self.tree.tick()
        root = self.tree.root

        # Print the tree only when something changed, not 10 times a second.
        snapshot = py_trees.display.unicode_tree(root, show_status=True)
        if snapshot != self._last_snapshot:
            self.get_logger().info('\n' + snapshot)
            self._last_snapshot = snapshot

        if root.status in (Status.SUCCESS, Status.FAILURE):
            self.success = root.status == Status.SUCCESS
            self.get_logger().info(
                'MISSION SUCCEEDED.' if self.success else 'MISSION FAILED.')
            self._timer.cancel()
            self.done = True


# ─────────────────────────────────────────────────────────────────────────────
# 5. BEHAVIOUR LEAVES
#
# Lifecycle of a py_trees leaf (see py_trees docs, "Behaviours"):
#   initialise()  called when the leaf is entered, i.e. the first tick after it
#                 was not RUNNING. Start your work here.
#   update()      called every tick while the leaf is the active one. Return
#                 Status.RUNNING / SUCCESS / FAILURE. Must not block.
#   terminate(s)  called when the leaf stops: after SUCCESS/FAILURE, or when it
#                 is interrupted (s == Status.INVALID). Cancel work here.
# ─────────────────────────────────────────────────────────────────────────────

def call_to_status(state: str) -> Status:
    """Map an AsyncActionCall state onto a BT status."""
    if state == AsyncActionCall.SUCCEEDED:
        return Status.SUCCESS
    if state in (AsyncActionCall.FAILED, AsyncActionCall.IDLE):
        return Status.FAILURE
    return Status.RUNNING


class UndockLeaf(py_trees.behaviour.Behaviour):
    """WORKED EXAMPLE. Read this, then write the other leaves the same way."""

    def __init__(self, ros_node: MissionNode, name='Undock'):
        super().__init__(name)
        self.ros = ros_node
        self.call = AsyncActionCall(
            ros_node, ros_node.undock_client, 'undock', UNDOCK_TIMEOUT_SEC)

    def initialise(self):
        if self.ros.is_docked is False:
            # Already off the dock (e.g. the mission was restarted): nothing to do.
            return
        self.call.start(Undock.Goal())

    def update(self):
        if self.ros.is_docked is False and not self.call.in_flight:
            return Status.SUCCESS
        state = self.call.poll()
        if state == AsyncActionCall.SUCCEEDED:
            return Status.SUCCESS
        if state in (AsyncActionCall.FAILED, AsyncActionCall.IDLE):
            return Status.FAILURE
        return Status.RUNNING

    def terminate(self, new_status):
        # INVALID means we were interrupted while still running.
        if new_status == Status.INVALID:
            self.call.cancel()


class NavigateToLeaf(py_trees.behaviour.Behaviour):
    """Drive to `pose` (map frame) with Nav2's NavigateToPose action.

    Obstacle avoidance is NOT done here: Nav2 replans and avoids the walking
    person by itself. This leaf only sends the goal and reports the outcome.
    """

    def __init__(self, ros_node: MissionNode, pose: PoseStamped, name='NavigateTo'):
        super().__init__(name)
        self.ros = ros_node
        self.pose = pose
        self.call = AsyncActionCall(
            ros_node, ros_node.nav_client, name, NAV_TIMEOUT_SEC)

    def initialise(self):
        self.call.start(self.ros.make_nav_goal(self.pose))

    def update(self):
        return call_to_status(self.call.poll())

    def terminate(self, new_status):
        if new_status == Status.INVALID:
            self.call.cancel()


class MoveArmLeaf(py_trees.behaviour.Behaviour):
    """Move the arm to a joint configuration (list of 6 angles, radians)."""

    def __init__(self, ros_node: MissionNode, angles, name='MoveArm', duration_sec=4):
        super().__init__(name)
        self.ros = ros_node
        self.angles = angles
        self.duration_sec = duration_sec
        self.call = AsyncActionCall(
            ros_node, ros_node.arm_client, name, ARM_TIMEOUT_SEC)

    def initialise(self):
        self.call.start(self.ros.make_arm_goal(self.angles, self.duration_sec))

    def update(self):
        state = self.call.poll()
        if state == AsyncActionCall.FAILED and self.call.result is not None:
            # The controller has a 0.01 rad goal tolerance (arm_controllers.yaml);
            # a non-zero error_code here usually means it was not met in time.
            self.ros.get_logger().error(
                f'[{self.name}] error_code={self.call.result.error_code} '
                f'{self.call.result.error_string}')
        return call_to_status(state)

    def terminate(self, new_status):
        if new_status == Status.INVALID:
            self.call.cancel()


class SetVacuumLeaf(py_trees.behaviour.Behaviour):
    """Attach (True) or detach (False) the cube, then wait for it to settle."""

    def __init__(self, ros_node: MissionNode, enable: bool, name='SetVacuum',
                 settle_sec=VACUUM_SETTLE_SEC):
        super().__init__(name)
        self.ros = ros_node
        self.enable = enable
        self.settle_sec = settle_sec
        self._t0 = 0.0

    def initialise(self):
        self.ros.set_vacuum(self.enable)
        self._t0 = self.ros.now_sec()

    def update(self):
        # Sim-time wait, no time.sleep(): sleeping would freeze every other
        # callback and use wall time, which is wrong in a simulation.
        if self.ros.now_sec() - self._t0 >= self.settle_sec:
            return Status.SUCCESS
        return Status.RUNNING


# ─────────────────────────────────────────────────────────────────────────────
# 6. THE TREE
# ─────────────────────────────────────────────────────────────────────────────

def build_tree(ros_node: MissionNode, home_base, pick_pose, drop_pose):
    """Return the root behaviour of the mission tree.

    Building blocks (py_trees 2.x):
        py_trees.composites.Sequence(name='...', memory=True, children=[...])
        py_trees.composites.Selector(name='...', memory=False, children=[...])
        py_trees.decorators.Retry(name='...', child=leaf, num_failures=3)

    memory=True means a Sequence resumes at the child that was RUNNING instead
    of re-running the children that already succeeded (you do NOT want to
    undock again after arriving at the box).

    Suggested order for grade C:
        detach (known gripper state) -> undock -> navigate(pick_pose)
        -> arm to ARM_PICK -> attach -> arm to ARM_SAFE
        -> navigate(drop_pose) -> arm to ARM_PLACE -> detach -> arm to ARM_SAFE

    Questions to answer before you code it (the TAs will ask):
      * which leaves deserve a Retry, and how many times?
      * why is the arm folded to ARM_SAFE before driving?
      * what happens to the cube if the second navigation fails?
    """
    # Fail now, not after the robot has already driven to the box.
    if ARM_PICK is None or ARM_PLACE is None:
        raise ValueError('ARM_PICK / ARM_PLACE are still None; set them at the top')

    # Every leaf object may appear in the tree only once, and names show up in
    # the log, so the repeated actions (detach, arm to safe) get distinct names.
    root = py_trees.composites.Sequence(name='Mission', memory=True, children=[
        SetVacuumLeaf(ros_node, enable=False, name='DetachCubeInit'),
        UndockLeaf(ros_node),
        py_trees.decorators.Retry(
            'RetryGoSource',
            NavigateToLeaf(ros_node, pick_pose, name='NavigateToPick'),
            num_failures=3),
        MoveArmLeaf(ros_node, ARM_PICK, name='MoveArmPick'),
        SetVacuumLeaf(ros_node, enable=True, name='AttachCube'),
        MoveArmLeaf(ros_node, ARM_SAFE, name='MoveArmSafeAfterPick'),
        py_trees.decorators.Retry(
            'RetryGoTarget',
            NavigateToLeaf(ros_node, drop_pose, name='NavigateToDrop'),
            num_failures=3),
        MoveArmLeaf(ros_node, ARM_PLACE, name='MoveArmPlace'),
        SetVacuumLeaf(ros_node, enable=False, name='DetachCube'),
        MoveArmLeaf(ros_node, ARM_SAFE, name='MoveArmSafeAfterPlace'),
    ])
    return root


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()

    try:
        node.run_mission()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Mission interrupted by user.')
    except Exception as e:
        node.get_logger().fatal(f'Mission failed: {e!r}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
