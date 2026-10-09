"""
mission_node.py — Student entry point (grades E, C and A: behaviour tree).

Layout of this file
  1. constants            where to go, arm poses, thresholds
  2. loaders              shelves.yaml -> PoseStamped, approach points
  3. MapMatcher           does the laser scan agree with the map at a pose?
  4. AsyncActionCall      non-blocking action wrapper
  5. MissionNode          ROS interfaces, world-state queries, tick loop
  6. behaviour leaves     conditions, undock, navigate, arm, vacuum, localize
  7. build_tree()         the mission as a backward-chained behaviour tree
  8. main()

Rule of thumb for a BT: update() is called on every tick (10 Hz) and must
return immediately. Never call time.sleep() or spin_until_future_complete()
inside a leaf; start the work in initialise(), poll it in update(), and
return RUNNING until it finishes.
"""

import os
import copy
import math
from pathlib import Path

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

import py_trees
from py_trees.common import Status
import tf2_ros

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, TwistStamped
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus
from nav2_msgs.action import DriveOnHeading, NavigateToPose, Spin
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv
from trajectory_msgs.msg import JointTrajectoryPoint

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
#
# The grade comes from a ROS parameter / the GRADE environment variable, so it
# cannot silently disagree with the grade the simulation was launched with:
#
#     GRADE=a pixi run mission            # world, odometry, nav stack, AMCL
#     GRADE=a pixi run mission-node       # this node
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
VACUUM_READY_TIMEOUT_SEC = 30.0  # give up if the gripper bridge never subscribes
APPROACH_BACKOFF_M = 0.5       # approach point this far behind each box pose
# Final alignment at a box, by laser: robot centre to the box's front face. Both
# box poses in shelves.yaml stand exactly this far off (the arm poses rely on
# it). Measured, so AMCL's along-track error (up to ~0.3 m after a long drive
# in grade A) does not end up as the cube landing off the box's edge.
BOX_FACE_DIST_M = 0.22
ALIGN_TOL_M = 0.02
ALIGN_MAX_CORRECTION_M = 0.5   # bigger than this: what we see is not the box
ALIGN_SPEED = 0.05
ALIGN_TIMEOUT_SEC = 20.0

# "Is the robot at this box?" Looser than Nav2's goal tolerance (0.08 m), so
# that localization noise while the arm works cannot flip the answer.
AT_BOX_TOL_M = 0.25
AT_BOX_TOL_RAD = 0.35
ARM_SAFE_TOL_RAD = 0.05        # "is the arm folded?" per-joint tolerance

# --- localization (grade A) ---------------------------------------------------
# AMCL's own confidence: x/y variance [m^2] and yaw variance [rad^2]. Only
# meant to rule out a cloud split between places (two clusters 1 m apart give
# ~0.25, the look-alike aisles 11 m apart ~30); whether the place is right is
# the scan check's job. Stricter values re-localized needlessly, because along
# a wall or an aisle the spread stays large while the scan matches well:
# 0.05-0.09 next to a long wall (0.91-0.96 match, four re-localizations,
# 6.4 min), 0.13-0.15 in the 18 m aisle between two big shelves (0.83-0.93).
LOC_COV_XY = 0.2
LOC_COV_YAW = 0.05
# Scan/map consistency at the pose AMCL reports (see MapMatcher). AMCL can be
# confident and still wrong: this warehouse repeats itself, and a symmetric
# wrong mode can have a tiny covariance. Measured in the simulator: correct
# estimates (AMCL error up to ~0.2 m and a few degrees) 0.73-0.99, wrong places
# (the symmetric aisles 11.2 m away, random poses, turned around) 0.03-0.4.
# 0.75 was too close to the correct side: it rejected right answers and cost a
# full re-localization each time.
LOC_SCORE_OK = 0.6
# While driving, declare the robot lost only if the score stays below this for
# LOC_LOST_SEC (a pedestrian can hide part of the scan for a while).
LOC_SCORE_LOST = 0.4
LOC_LOST_SEC = 8.0
# Scores taken while the robot turns faster than this do not count towards
# "lost": turning on the spot makes the wheel odometry slip, and AMCL corrects
# the heading only at its next update (seen: 22 deg off for a few seconds,
# score 0.18 at an otherwise correct pose, a needless re-localization).
LOC_SEARCH_TIMEOUT_SEC = 240.0  # one global-localization attempt
EXPLORE_STEP_M = 1.5           # drive this far between look-around spins
EXPLORE_SPEED = 0.25
SPIN_TIMEOUT_SEC = 60.0
LOST_CHECK_MAX_TURN_RATE = 0.3  # rad/s
# The charging dock is only 10 cm tall, below the lidar, so no scan shows it. Its
# collision box is DOCK_AHEAD_M straight ahead of the docked robot. In grade A the
# TA moves robot AND dock to a random spot, and the first exploration drives of
# the localization search must not run into it (seen: the Create 3 bump reflex
# took over, the wheels slipped, and the pose estimate was thrown off).
DOCK_AHEAD_M = 0.2
DRIVE_TIMEOUT_SEC = 40.0

ARM_JOINT_NAMES = [
    'arm_joint1', 'arm_joint2', 'arm_joint3',
    'arm_joint4', 'arm_joint5', 'arm_joint6',
]

# All poses come from inverse kinematics on the URDF (tool pointing straight down,
# joint 6 left at 0) and were checked in the simulator: the tool tip (TF frame
# arm_link_tcp) lands within 1 mm of the intended point.
#
# ARM_SAFE: tool tip 0.15 m ahead of and 0.62 m above base_link, over the robot
# body, which keeps the centre of mass low and central while driving. The spawn
# pose (config/initial_positions.yaml) holds the arm 0.46 m forward, 0.73 m high.
ARM_SAFE = [0.0, -0.417, 0.466, 0.0, 0.883, 0.0]

# ARM_PICK: tool tip 0.34 m ahead of base_link and level with the cube top
# (z = 0.38 m), i.e. on the cube when the robot stands on the source box pose.
ARM_PICK = [-0.059, 1.274, 1.568, 0.0, 0.293, 0.0]
# The cube hangs under the tool tip, so it lands where the tip is. The target box
# (1.0 x 0.5 m) has its near edge only ~0.22 m ahead of the robot and its centre at
# 0.47 m, which the arm cannot reach with the tool pointing down (limit ~0.43 m).
# Tip at 0.40 m ahead, z = 0.39 m: ~15 cm inside the near edge.
ARM_PLACE = [0.0, 1.498, 2.098, 0.0, 0.6, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOADERS
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


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def approach_pose(pose: PoseStamped, back: float) -> PoseStamped:
    """The same pose moved `back` metres backwards along its own heading.

    A box's working pose leaves only ~3 cm between the robot and the box, so
    the robot must arrive head-on. Driving to this point first, already facing
    the box, makes the last leg a short straight line; arriving from the side
    instead brushes the box and Nav2's collision monitor halts the robot.
    """
    p = copy.deepcopy(pose)
    q = p.pose.orientation
    yaw = 2.0 * math.atan2(q.z, q.w)
    p.pose.position.x -= back * math.cos(yaw)
    p.pose.position.y -= back * math.sin(yaw)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAP MATCHER
#
# AMCL reports a pose and a covariance. A small covariance only says the
# particles agree with each other, not that they are right: in a warehouse
# whose aisles repeat, the whole cloud can collapse onto the wrong aisle. This
# checks the answer independently, using only the map and the laser.
#
# For every beam, ray-cast the map from the laser's pose to get the range the
# map predicts, and compare with the range measured:
#   about equal          -> agree
#   measured SHORTER     -> something not in the map is in the way (the dock,
#                           a box, a person): no evidence either way, ignored
#   measured LONGER, or  -> the beam went THROUGH a mapped wall: impossible at
#   no return although      the right pose, strong evidence of a wrong one
#   a wall is in range
# score = agree / (agree + through). Ignoring the occluded beams is what makes
# this robust to the unmapped objects around the robot; a first version that
# counted "endpoints near a wall" read only 0.88 at the true pose next to the
# relocated dock and rejected correct estimates.
# ─────────────────────────────────────────────────────────────────────────────

class MapMatcher:

    def __init__(self, map_yaml: str):
        with open(map_yaml) as f:
            meta = yaml.safe_load(f)
        img = self._read_pgm(os.path.join(os.path.dirname(map_yaml), meta['image']))
        if meta.get('negate', 0):
            img = 255 - img
        self.occupied = (255.0 - img.astype(float)) / 255.0 >= float(meta['occupied_thresh'])
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        self.h, self.w = self.occupied.shape

    @staticmethod
    def _read_pgm(path: str) -> np.ndarray:
        raw = open(path, 'rb').read()
        fields, i = [], 0
        while len(fields) < 4:              # magic, width, height, maxval
            while raw[i:i + 1].isspace():
                i += 1
            if raw[i:i + 1] == b'#':        # comment line
                while raw[i:i + 1] != b'\n':
                    i += 1
                continue
            j = i
            while not raw[j:j + 1].isspace():
                j += 1
            fields.append(raw[i:j])
            i = j
        if fields[0] != b'P5':
            raise ValueError(f'{path}: only binary PGM (P5) is supported')
        w, h = int(fields[1]), int(fields[2])
        return np.frombuffer(raw[i + 1:i + 1 + w * h], np.uint8).reshape(h, w)

    def expected_ranges(self, x, y, angles, range_max):
        """Ray-cast the map: range to the first occupied cell along each angle."""
        steps = np.arange(0.05, range_max, 0.8 * self.res)
        px = x + np.outer(np.cos(angles), steps)
        py = y + np.outer(np.sin(angles), steps)
        col = np.floor((px - self.ox) / self.res).astype(int)
        row = self.h - 1 - np.floor((py - self.oy) / self.res).astype(int)
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        occ = np.ones(px.shape, bool)               # off the map counts as a wall
        occ[inside] = self.occupied[row[inside], col[inside]]
        hit = occ.any(axis=1)
        return np.where(hit, steps[occ.argmax(axis=1)], range_max)

    def consistency(self, x, y, yaw, ranges, beam_angles, range_min, range_max):
        """Score in [0, 1] for a laser at (x, y, yaw) in the map, or None.

        beam_angles are in the laser frame. See the section comment.
        """
        r = np.asarray(ranges, float)
        expected = self.expected_ranges(x, y, yaw + np.asarray(beam_angles), range_max)
        valid = np.isfinite(r) & (r > range_min) & (r < 0.98 * range_max)
        tol = 0.2 + 0.03 * np.where(valid, np.minimum(r, expected), expected)
        agree = valid & (np.abs(r - expected) <= tol)
        through = (valid & (r > expected + tol)) | (~valid & (expected < range_max - 0.5))
        n = int(agree.sum() + through.sum())
        if n < 30:
            return None
        return float(agree.sum()) / n


# ─────────────────────────────────────────────────────────────────────────────
# 4. NON-BLOCKING ACTION CALL
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


def call_to_status(state: str) -> Status:
    """Map an AsyncActionCall state onto a BT status."""
    if state == AsyncActionCall.SUCCEEDED:
        return Status.SUCCESS
    if state in (AsyncActionCall.FAILED, AsyncActionCall.IDLE):
        return Status.FAILURE
    return Status.RUNNING


# ─────────────────────────────────────────────────────────────────────────────
# 5. THE ROS NODE
#
# Owns every ROS client / publisher / subscriber and the tick loop, and answers
# the questions the tree's conditions ask about the world ("is the arm
# folded?", "is the robot localized?"). Leaves reach the robot only through
# this object (they hold it as self.ros).
# ─────────────────────────────────────────────────────────────────────────────

class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')

        # Read from the GRADE environment variable, so one spelling works
        # whether you go through `GRADE=a pixi run mission-node` or call
        # ros2 run yourself inside a pixi shell. An explicit -p grade:=a wins.
        self.declare_parameter('grade', os.environ.get('GRADE', 'e'))
        self.grade = str(self.get_parameter('grade').value).strip().lower()
        if self.grade not in DROP_BOX_BY_GRADE:
            self.get_logger().warn(
                f"Unknown grade '{self.grade}'; falling back to 'e'. "
                f"Valid grades: {sorted(DROP_BOX_BY_GRADE)}")
            self.grade = 'e'

        self.source_box = SOURCE_BOX
        self.drop_box = DROP_BOX_BY_GRADE[self.grade]
        # Only grade A starts somewhere unknown; E and C get an exact map->odom.
        self.needs_localization = self.grade == 'a'

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
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.drive_client = ActionClient(self, DriveOnHeading, 'drive_on_heading')
        self.reinit_client = self.create_client(EmptySrv, '/reinitialize_global_localization')
        # Direct velocity commands, only for the last few centimetres in front of
        # a box (Nav2 is idle then; its own collision checks refuse to drive that
        # close to anything).
        self._cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        # World state ----------------------------------------------------------
        # True / False once the first /dock_status message arrives, None before.
        self.is_docked = None
        self.create_subscription(
            DockStatus, 'dock_status', self._on_dock_status, qos_profile_sensor_data)
        self.arm_angles = None      # latest arm joint positions, ARM_JOINT_NAMES order
        self.create_subscription(
            JointState, 'joint_states', self._on_joint_states, qos_profile_sensor_data)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Facts the robot cannot sense, so the tree remembers them: the gripper
        # reports nothing back, and the cube's position is not observable
        # without ground truth (which the assignment forbids).
        self.holding_cube = False
        self.cube_placed = False

        # Localization (grade A) ---------------------------------------------
        self.amcl_cov = None        # (var x, var y, var yaw) of the latest /amcl_pose
        self.match_score = 0.0      # see MapMatcher; refreshed from /scan
        self.match_score_seq = 0    # bumped on every fresh score
        self.localized = not self.needs_localization
        self._low_score_since = None
        self._scan = None
        self._last_score_t = -1e9
        self.dock_xy_odom = None    # where the dock is, so exploring avoids it
        self.turn_rate = 0.0
        self.create_subscription(Odometry, 'odom', self._on_odom, 10)
        self.create_subscription(LaserScan, 'scan', self._on_scan_any, qos_profile_sensor_data)
        if self.needs_localization:
            map_yaml = os.path.join(
                get_package_share_directory('warehouse_inventory_robot'),
                'maps', 'warehouse.yaml')
            self.matcher = MapMatcher(map_yaml)
            self.create_subscription(
                PoseWithCovarianceStamped, 'amcl_pose', self._on_amcl_pose, 10)

        # BT bookkeeping -----------------------------------------------------
        self.tree = None
        self.done = False
        self.success = False
        self._last_snapshot = ''
        self._timer = None

    # --- callbacks ----------------------------------------------------------

    def now_sec(self) -> float:
        """Current time in seconds on the node clock (sim time with use_sim_time)."""
        return self.get_clock().now().nanoseconds / 1e9

    def _on_dock_status(self, msg: DockStatus):
        self.is_docked = bool(msg.is_docked)

    def _on_joint_states(self, msg: JointState):
        d = dict(zip(msg.name, msg.position))
        if all(n in d for n in ARM_JOINT_NAMES):
            self.arm_angles = [d[n] for n in ARM_JOINT_NAMES]

    def _on_odom(self, msg: Odometry):
        self.turn_rate = msg.twist.twist.angular.z

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        c = msg.pose.covariance
        self.amcl_cov = (c[0], c[7], c[35])

    def _on_scan_any(self, msg: LaserScan):
        self._scan = msg
        if self.needs_localization:
            self._on_scan(msg)

    def _on_scan(self, msg: LaserScan):
        if self.now_sec() - self._last_score_t >= 0.25:   # 4 Hz is plenty
            self._last_score_t = self.now_sec()
            s = self._compute_match_score(msg)
            if s is not None:
                self.match_score = s
                self.match_score_seq += 1

    # --- world-state queries (used by the tree's conditions) ---------------

    def lookup(self, target: str, source: str):
        """(x, y, yaw) of `source` in `target`, latest available, or None."""
        try:
            t = self.tf_buffer.lookup_transform(target, source, Time())
        except Exception:
            return None
        tr = t.transform.translation
        return tr.x, tr.y, yaw_of(t.transform.rotation)

    def robot_pose(self):
        """Robot pose in the map frame (AMCL's estimate in grade A)."""
        return self.lookup('map', 'base_link')

    def is_undocked(self) -> bool:
        return self.is_docked is False

    def arm_is_safe(self) -> bool:
        return self.arm_angles is not None and all(
            abs(a - b) < ARM_SAFE_TOL_RAD for a, b in zip(self.arm_angles, ARM_SAFE))

    def at_pose(self, pose: PoseStamped) -> bool:
        p = self.robot_pose()
        if p is None:
            return False
        goal_yaw = 2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w)
        return (math.hypot(p[0] - pose.pose.position.x, p[1] - pose.pose.position.y)
                < AT_BOX_TOL_M and abs(angle_diff(p[2], goal_yaw)) < AT_BOX_TOL_RAD)

    def amcl_confident(self) -> bool:
        c = self.amcl_cov
        return c is not None and c[0] < LOC_COV_XY and c[1] < LOC_COV_XY and c[2] < LOC_COV_YAW

    def pose_agrees_with_scan(self) -> bool:
        return self.amcl_confident() and self.match_score >= LOC_SCORE_OK

    def is_localized(self) -> bool:
        """Condition 'Localized?' with hysteresis.

        Set True only by the localization subtree (after a verification spin).
        Dropped when the scan stops matching the map for LOC_LOST_SEC, or when
        AMCL's own spread blows up; a single bad scan (pedestrian, unmapped box)
        is not enough.
        """
        if not self.needs_localization:
            return True
        if not self.localized:
            return False
        now = self.now_sec()
        if abs(self.turn_rate) > LOST_CHECK_MAX_TURN_RATE:
            pass                    # turning: this score says little, see LOC_LOST_SEC
        elif self.match_score < LOC_SCORE_LOST:
            if self._low_score_since is None:
                self._low_score_since = now
            elif now - self._low_score_since > LOC_LOST_SEC:
                self.get_logger().warn(
                    f'Localization lost: scan/map match {self.match_score:.2f} for '
                    f'{LOC_LOST_SEC:.0f}s')
                self.localized = False
        else:
            self._low_score_since = None
        c = self.amcl_cov
        if c is not None and (c[0] > 1.0 or c[1] > 1.0):
            self.get_logger().warn(f'Localization lost: AMCL covariance {c[0]:.2f}, {c[1]:.2f}')
            self.localized = False
        return self.localized

    def _compute_match_score(self, scan: LaserScan):
        # TF at the scan's own timestamp: while spinning at 1 rad/s, using the
        # latest transform instead would smear a 5 m return by ~0.25 m.
        try:
            t = self.tf_buffer.lookup_transform('map', scan.header.frame_id,
                                                Time.from_msg(scan.header.stamp))
        except Exception:
            return None
        a = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        tr = t.transform.translation
        return self.matcher.consistency(tr.x, tr.y, yaw_of(t.transform.rotation),
                                        scan.ranges, a, scan.range_min, scan.range_max)

    def explore_heading(self):
        """Relative heading [rad] towards the most open space, or None.

        Uses only the latest scan (in base_link). Skips directions that pass
        near the dock, which is right next to the robot after undocking and too
        low for the lidar to see (see DOCK_AHEAD_M).
        """
        scan = self._scan
        laser = self.lookup('base_link', scan.header.frame_id) if scan else None
        odom = self.lookup('odom', 'base_link')
        if scan is None or laser is None:
            return None
        r = np.asarray(scan.ranges, float)
        r = np.where(np.isfinite(r), r, scan.range_max)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment + laser[2]
        best, best_clear = None, 0.0
        for cand in np.radians(np.arange(-180, 180, 15)):
            # clearance = shortest return within +-15 deg of the candidate
            sector = np.abs(np.arctan2(np.sin(a - cand), np.cos(a - cand))) < math.radians(15)
            clear = float(r[sector].min()) if sector.any() else 0.0
            if self.dock_xy_odom is not None and odom is not None:
                hx = odom[0] + EXPLORE_STEP_M * math.cos(odom[2] + cand)
                hy = odom[1] + EXPLORE_STEP_M * math.sin(odom[2] + cand)
                mx, my = (odom[0] + hx) / 2, (odom[1] + hy) / 2
                if min(math.hypot(hx - self.dock_xy_odom[0], hy - self.dock_xy_odom[1]),
                       math.hypot(mx - self.dock_xy_odom[0], my - self.dock_xy_odom[1])) < 0.6:
                    continue
            if clear > best_clear:
                best, best_clear = float(cand), clear
        if best is None or best_clear < EXPLORE_STEP_M + 0.5:
            return None
        return best

    def front_distance(self):
        """Distance [m] from base_link to whatever is straight ahead, or None.

        Median of the returns within +-6 deg of the robot's heading.
        """
        scan = self._scan
        laser = self.lookup('base_link', scan.header.frame_id) if scan else None
        if scan is None or laser is None:
            return None
        r = np.asarray(scan.ranges, float)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment + laser[2]
        ahead = (np.abs(np.arctan2(np.sin(a), np.cos(a))) < math.radians(6)) & \
            np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
        if ahead.sum() < 3:
            return None
        return float(np.median(r[ahead])) + laser[0]     # laser sits behind base_link

    def drive(self, v: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(v)
        self._cmd_pub.publish(msg)

    def remember_dock(self):
        """Called while still docked: the dock is DOCK_AHEAD_M straight ahead."""
        odom = self.lookup('odom', 'base_link')
        if odom is not None and self.dock_xy_odom is None:
            self.dock_xy_odom = (odom[0] + DOCK_AHEAD_M * math.cos(odom[2]),
                                 odom[1] + DOCK_AHEAD_M * math.sin(odom[2]))

    # --- goal builders ----------------------------------------------------------

    def vacuum_ready(self, enable: bool) -> bool:
        """True once the Gazebo bridge subscribes to the attach/detach topic.

        A message published before that is silently dropped. That matters: the
        gripper's DetachableJoint starts ATTACHED to the cube, so a lost initial
        detach leaves the robot welded to a cube 7.5 m away, which tilts the base
        and stops it from turning on the spot.
        """
        pub = self._attach_pub if enable else self._detach_pub
        return pub.get_subscription_count() > 0

    def set_vacuum(self, enable: bool, log: bool = True):
        """Publish attach/detach once. Does not wait; the leaf must wait itself."""
        if log:
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

    def make_spin_goal(self, yaw: float) -> Spin.Goal:
        goal = Spin.Goal()
        goal.target_yaw = float(yaw)
        goal.time_allowance = Duration(sec=int(SPIN_TIMEOUT_SEC))
        return goal

    def make_drive_goal(self, dist: float) -> DriveOnHeading.Goal:
        goal = DriveOnHeading.Goal()
        goal.target = Point(x=float(dist), y=0.0, z=0.0)
        goal.speed = EXPLORE_SPEED
        goal.time_allowance = Duration(sec=int(DRIVE_TIMEOUT_SEC))
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
# 6. BEHAVIOUR LEAVES
#
# Lifecycle of a py_trees leaf (see py_trees docs, "Behaviours"):
#   initialise()  called when the leaf is entered, i.e. the first tick after it
#                 was not RUNNING. Start your work here.
#   update()      called every tick while the leaf is the active one. Return
#                 Status.RUNNING / SUCCESS / FAILURE. Must not block.
#   terminate(s)  called when the leaf stops: after SUCCESS/FAILURE, or when it
#                 is interrupted (s == Status.INVALID). Cancel work here.
# ─────────────────────────────────────────────────────────────────────────────

class Condition(py_trees.behaviour.Behaviour):
    """SUCCESS if check() is true, FAILURE otherwise. Never RUNNING."""

    def __init__(self, name, check):
        super().__init__(name)
        self.check = check

    def update(self):
        return Status.SUCCESS if self.check() else Status.FAILURE


class SetFlag(py_trees.behaviour.Behaviour):
    """Record a fact the robot cannot sense (e.g. 'holding the cube')."""

    def __init__(self, ros_node, attr, value, name=None):
        super().__init__(name or f'{attr}={value}')
        self.ros, self.attr, self.value = ros_node, attr, value

    def update(self):
        setattr(self.ros, self.attr, self.value)
        return Status.SUCCESS


class UndockLeaf(py_trees.behaviour.Behaviour):
    """Leave the charging dock. Succeeds at once if the robot is already off it."""

    def __init__(self, ros_node: MissionNode, name='Undock'):
        super().__init__(name)
        self.ros = ros_node
        self.call = AsyncActionCall(
            ros_node, ros_node.undock_client, 'undock', UNDOCK_TIMEOUT_SEC)

    def initialise(self):
        self.call.cancel()
        self._t0 = self.ros.now_sec()

    def update(self):
        if self.ros.is_docked is None:            # no /dock_status yet
            if self.ros.now_sec() - self._t0 > UNDOCK_TIMEOUT_SEC:
                return Status.FAILURE
            return Status.RUNNING
        if self.ros.is_docked is False and not self.call.in_flight:
            return Status.SUCCESS
        if self.call.state == AsyncActionCall.IDLE:
            self.ros.remember_dock()            # still on the dock: see DOCK_AHEAD_M
            self.call.start(Undock.Goal())
        return call_to_status(self.call.poll())

    def terminate(self, new_status):
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
        self._t0 = None

    def initialise(self):
        self._t0 = None          # not sent yet: wait for the bridge to subscribe
        self._t_enter = self.ros.now_sec()

    def update(self):
        now = self.ros.now_sec()
        if self._t0 is None:
            if not self.ros.vacuum_ready(self.enable):
                if now - self._t_enter > VACUUM_READY_TIMEOUT_SEC:
                    self.ros.get_logger().error(
                        f'[{self.name}] nobody subscribes to the gripper topic')
                    return Status.FAILURE
                return Status.RUNNING
            self.ros.set_vacuum(self.enable)
            self._t0 = now
            return Status.RUNNING
        # Re-send while settling: cheap insurance against a dropped message, and
        # attach/detach are idempotent on the Gazebo side.
        self.ros.set_vacuum(self.enable, log=False)
        # Sim-time wait, no time.sleep(): sleeping would freeze every other
        # callback and use wall time, which is wrong in a simulation.
        if now - self._t0 >= self.settle_sec:
            return Status.SUCCESS
        return Status.RUNNING


class AlignToBoxLeaf(py_trees.behaviour.Behaviour):
    """Creep forward/back until the box's front face is BOX_FACE_DIST_M ahead.

    Uses the laser only. If nothing plausible is ahead (the box not where it
    should be), leaves the robot where Nav2 put it rather than guess.
    """

    def __init__(self, ros_node: MissionNode, name='AlignToBox'):
        super().__init__(name)
        self.ros = ros_node

    def initialise(self):
        self._t0 = self.ros.now_sec()

    def update(self):
        d = self.ros.front_distance()
        if d is None:
            return Status.RUNNING if self.ros.now_sec() - self._t0 < 2.0 else Status.SUCCESS
        err = d - BOX_FACE_DIST_M
        if abs(err) > ALIGN_MAX_CORRECTION_M:
            self.ros.get_logger().warn(
                f'[{self.name}] nearest thing ahead is {d:.2f} m away; not aligning')
            self.ros.drive(0.0)
            return Status.SUCCESS
        if abs(err) <= ALIGN_TOL_M or self.ros.now_sec() - self._t0 > ALIGN_TIMEOUT_SEC:
            self.ros.drive(0.0)
            self.ros.get_logger().info(f'[{self.name}] box face {d:.3f} m ahead')
            return Status.SUCCESS
        self.ros.drive(math.copysign(min(ALIGN_SPEED, 2.0 * abs(err) + 0.01), err))
        return Status.RUNNING

    def terminate(self, new_status):
        self.ros.drive(0.0)


class ReinitParticlesLeaf(py_trees.behaviour.Behaviour):
    """Global localization: ask AMCL to spread its particles over the whole map."""

    def __init__(self, ros_node: MissionNode, name='ScatterParticles'):
        super().__init__(name)
        self.ros = ros_node

    def initialise(self):
        self.future = None
        self._t0 = self.ros.now_sec()
        self.ros.localized = False

    def update(self):
        if self.future is None:
            if not self.ros.reinit_client.service_is_ready():
                if self.ros.now_sec() - self._t0 > 60.0:
                    self.ros.get_logger().error('AMCL global localization service never appeared')
                    return Status.FAILURE
                return Status.RUNNING
            self.ros.get_logger().info('Scattering AMCL particles over the whole map')
            self.future = self.ros.reinit_client.call_async(EmptySrv.Request())
            return Status.RUNNING
        if not self.future.done():
            return Status.RUNNING
        # AMCL's covariance messages arrive a little later; forget the old one.
        self.ros.amcl_cov = None
        return Status.SUCCESS


class SearchLeaf(py_trees.behaviour.Behaviour):
    """Look around and move until AMCL is confident AND agrees with the scan.

    A spin shows the lidar every direction and makes AMCL update (it only
    updates after the robot has moved or turned enough); driving to a new spot
    adds views that a symmetric wrong guess cannot explain. Repeats
        full spin -> turn to the most open direction -> drive EXPLORE_STEP_M
    and succeeds as soon as the condition holds, even mid-manoeuvre.

    Fails early when AMCL is confident but the scan keeps contradicting the
    map: the particles have all collapsed onto a wrong place, AMCL cannot get
    out of that by itself, and the retry around this leaf scatters them again
    (seen in the simulator: confident at 28 m from the truth).
    """

    WRONG_MODE_SCANS = 8          # ~2 s of fresh scores at 4 Hz

    def __init__(self, ros_node: MissionNode, name='LookAroundUntilConfident'):
        super().__init__(name)
        self.ros = ros_node
        self.spin = AsyncActionCall(ros_node, ros_node.spin_client, 'spin', SPIN_TIMEOUT_SEC)
        self.drive = AsyncActionCall(ros_node, ros_node.drive_client, 'drive', DRIVE_TIMEOUT_SEC)

    def initialise(self):
        self._t0 = self.ros.now_sec()
        self._phase = 'look'
        self._seq = self.ros.match_score_seq
        self._wrong = 0
        self.spin.start(self.ros.make_spin_goal(2.0 * math.pi))

    def update(self):
        if self.ros.match_score_seq != self._seq:            # a fresh score
            self._seq = self.ros.match_score_seq
            confident_but_wrong = (self.ros.amcl_confident()
                                   and self.ros.match_score < LOC_SCORE_LOST)
            self._wrong = self._wrong + 1 if confident_but_wrong else 0
            if self._wrong >= self.WRONG_MODE_SCANS:
                self.ros.get_logger().warn(
                    'AMCL is confident but the scan contradicts the map '
                    f'(match {self.ros.match_score:.2f}): wrong place, scattering again')
                return Status.FAILURE
        if self.ros.pose_agrees_with_scan():
            self.ros.get_logger().info(
                f'AMCL confident (cov {self.ros.amcl_cov}) '
                f'and scan matches map ({self.ros.match_score:.2f})')
            return Status.SUCCESS
        if self.ros.now_sec() - self._t0 > LOC_SEARCH_TIMEOUT_SEC:
            self.ros.get_logger().warn('Global localization attempt timed out')
            return Status.FAILURE

        if self._phase == 'look':
            if self.spin.poll() in (AsyncActionCall.SUCCEEDED, AsyncActionCall.FAILED,
                                    AsyncActionCall.IDLE):
                heading = self.ros.explore_heading()
                if heading is None:              # boxed in: just look again
                    self.spin.start(self.ros.make_spin_goal(math.pi))
                else:
                    self._phase = 'turn'
                    self.spin.start(self.ros.make_spin_goal(heading))
        elif self._phase == 'turn':
            if self.spin.poll() in (AsyncActionCall.SUCCEEDED, AsyncActionCall.FAILED,
                                    AsyncActionCall.IDLE):
                self._phase = 'drive'
                self.drive.start(self.ros.make_drive_goal(EXPLORE_STEP_M))
        elif self._phase == 'drive':
            # A drive aborted by an obstacle is fine: look around from here.
            if self.drive.poll() in (AsyncActionCall.SUCCEEDED, AsyncActionCall.FAILED,
                                     AsyncActionCall.IDLE):
                self._phase = 'look'
                self.spin.start(self.ros.make_spin_goal(2.0 * math.pi))
        return Status.RUNNING

    def terminate(self, new_status):
        self.spin.cancel()
        self.drive.cancel()


class VerifyLeaf(py_trees.behaviour.Behaviour):
    """Look in four directions; the scan must match the map in every one.

    A wrong-but-similar place can match from one viewpoint, almost never from
    all four. Each look is taken standing still: scored while turning, even a
    correct estimate reads only ~0.6, because a few tens of milliseconds
    between the scan and the pose make degrees of heading error at 1 rad/s
    (measured: true pose 0.97 at rest, median 0.6 while spinning). So:
        score -> quarter turn -> settle -> score -> ... (4 views)
    Success marks the robot localized.
    """

    VIEWS = 4
    SETTLE_SEC = 0.6

    def __init__(self, ros_node: MissionNode, name='VerifyByLooking'):
        super().__init__(name)
        self.ros = ros_node
        self.spin = AsyncActionCall(ros_node, ros_node.spin_client, 'verify turn',
                                    SPIN_TIMEOUT_SEC)

    def initialise(self):
        self._scores = []
        self._phase = 'settle'
        self._t = self.ros.now_sec()
        self._seq = None

    def _fail(self, why):
        self.ros.get_logger().warn(f'Verification failed: {why}')
        self.ros.localized = False
        return Status.FAILURE

    def update(self):
        now = self.ros.now_sec()
        if self._phase == 'settle':                 # standing still, let things settle
            if now - self._t >= self.SETTLE_SEC:
                self._phase, self._seq = 'score', self.ros.match_score_seq
            return Status.RUNNING
        if self._phase == 'score':                  # take the next fresh score
            if self.ros.match_score_seq == self._seq:
                return Status.RUNNING
            sc = self.ros.match_score
            self._scores.append(sc)
            if sc < LOC_SCORE_OK:
                return self._fail(f'view {len(self._scores)} matches the map only {sc:.2f}')
            if len(self._scores) >= self.VIEWS:
                if not self.ros.amcl_confident():
                    return self._fail(f'AMCL not confident (cov {self.ros.amcl_cov})')
                self.ros.localized = True
                self.ros._low_score_since = None
                p = self.ros.robot_pose()
                if p is not None:
                    views = ', '.join(f'{v:.2f}' for v in self._scores)
                    self.ros.get_logger().info(
                        f'LOCALIZED at ({p[0]:.2f}, {p[1]:.2f}, {math.degrees(p[2]):.0f} deg); '
                        f'map match in {self.VIEWS} directions: {views}')
                return Status.SUCCESS
            self._phase = 'turn'
            self.spin.start(self.ros.make_spin_goal(math.pi / 2))
            return Status.RUNNING
        # turning
        if self.spin.poll() in (AsyncActionCall.SUCCEEDED, AsyncActionCall.FAILED,
                                AsyncActionCall.IDLE):
            self._phase, self._t = 'settle', now
        return Status.RUNNING

    def terminate(self, new_status):
        self.spin.cancel()


# ─────────────────────────────────────────────────────────────────────────────
# 7. THE TREE — backward chained
#
# Built from one pattern (postcondition-precondition-action, "PPA"):
#
#     Fallback[ postcondition?,  Sequence[ precondition PPAs..., action ] ]
#
# Ticking from the root, a condition that already holds short-circuits its
# whole subtree, and an unmet precondition is fixed by its own PPA before the
# action runs. The conditions are re-checked on EVERY tick (memory=False), so
# the tree reacts: if localization is lost while driving, the drive is halted
# and the robot re-localizes before carrying on.
#
#   Mission     = Fallback[ CubeOnTarget?, Sequence[ HoldingCube PPA, AtTarget PPA, Place ] ]
#   HoldingCube = Fallback[ HoldingCube?,  Sequence[ AtSource PPA, Pick ] ]
#   AtSource / AtTarget
#               = Fallback[ At(box)?,      Sequence[ ArmSafe PPA, Undocked PPA,
#                                                    Localized PPA, Move(box) ] ]
#   ArmSafe     = Fallback[ ArmSafe?,      FoldArm ]
#   Undocked    = Fallback[ Undocked?,     Undock ]
#   Localized   = Fallback[ Localized?,    Localize ]    (only grade A has work to do)
#
# "Arm in safe pos" and "Undocked" are preconditions of every Move, as the
# assignment asks. Pick and Place do not end by folding the arm: the next Move's
# ArmSafe precondition does that, which is the point of backward chaining.
# ─────────────────────────────────────────────────────────────────────────────

def ppa(name, postcondition, preconditions, action):
    """Fallback[postcondition, Sequence[preconditions..., action]] (reactive).

    One rule on top of the textbook pattern: while its own action is running,
    the postcondition does not end it. Many postconditions turn true *during*
    the action -- /dock_status reports "undocked" as soon as the robot starts
    to back off, the robot is within the at-box tolerance before Nav2 has
    finished, the arm passes the folded pose just before it stops -- and
    re-checking them every tick would cancel the action half-way (both
    happened: spinning on the dock, stopping 0.3 m short of a box). The
    preconditions inside the action branch are still re-checked every tick, so
    e.g. losing localization still interrupts a drive.
    """
    do = py_trees.composites.Sequence(name=f'{name}: do', memory=False,
                                      children=list(preconditions) + [action])
    check = postcondition.check
    postcondition.check = lambda: do.status != Status.RUNNING and check()
    return py_trees.composites.Selector(name=name, memory=False, children=[postcondition, do])


def go_to_box(ros_node: MissionNode, box_pose, label):
    """Retry( Sequence( approach point -> box pose -> laser alignment ) ).

    A failed attempt restarts from the approach point, so a retry always ends
    with a straight head-on approach rather than a sideways one.
    """
    return py_trees.decorators.Retry(
        f'RetryGo{label}',
        py_trees.composites.Sequence(name=f'Go{label}', memory=True, children=[
            NavigateToLeaf(ros_node, approach_pose(box_pose, APPROACH_BACKOFF_M),
                           name=f'NavigateTo{label}Approach'),
            NavigateToLeaf(ros_node, box_pose, name=f'NavigateTo{label}'),
            AlignToBoxLeaf(ros_node, name=f'AlignTo{label}Box'),
        ]),
        num_failures=3)


def localize(ros_node: MissionNode):
    """Become localized, cheaply if possible.

    1. If AMCL already agrees with the scan (e.g. the robot was not moved, or a
       brief loss), just verify it by looking in four directions.
    2. Otherwise global localization: scatter particles, look around and move
       until confident, then verify. Retried with a fresh scatter, because a
       verification failure means the particles settled on the wrong place.
    """
    quick = py_trees.composites.Sequence(name='KeepCurrentEstimate', memory=True, children=[
        Condition('ScanMatchesMap?', lambda: ros_node.match_score >= LOC_SCORE_OK),
        VerifyLeaf(ros_node, name='VerifyCurrent'),
    ])
    full = py_trees.decorators.Retry('RetryGlobalLocalization', py_trees.composites.Sequence(
        name='GlobalLocalization', memory=True, children=[
            ReinitParticlesLeaf(ros_node),
            SearchLeaf(ros_node),
            VerifyLeaf(ros_node),
        ]), num_failures=5)
    return py_trees.composites.Selector(name='Localize', memory=True, children=[quick, full])


def move_preconditions(ros_node: MissionNode, label):
    """The PPAs every Move needs: arm folded, off the dock, localized."""
    return [
        ppa(f'ArmSafe ({label})', Condition('ArmSafe?', ros_node.arm_is_safe), [],
            MoveArmLeaf(ros_node, ARM_SAFE, name=f'FoldArm ({label})')),
        ppa(f'Undocked ({label})', Condition('Undocked?', ros_node.is_undocked),
            [], UndockLeaf(ros_node, name=f'Undock ({label})')),
        ppa(f'Localized ({label})', Condition('Localized?', ros_node.is_localized),
            [], localize(ros_node)),
    ]


def build_tree(ros_node: MissionNode, home_base, pick_pose, drop_pose):
    """Return the root behaviour of the mission tree (see the section comment)."""
    # Fail now, not after the robot has already driven to the box.
    if ARM_PICK is None or ARM_PLACE is None:
        raise ValueError('ARM_PICK / ARM_PLACE are still None; set them at the top')

    at_source = ppa(
        'AtSource', Condition('AtSource?', lambda: ros_node.at_pose(pick_pose)),
        move_preconditions(ros_node, 'source'), go_to_box(ros_node, pick_pose, 'Source'))
    at_target = ppa(
        'AtTarget', Condition('AtTarget?', lambda: ros_node.at_pose(drop_pose)),
        move_preconditions(ros_node, 'target'), go_to_box(ros_node, drop_pose, 'Target'))

    pick = py_trees.composites.Sequence(name='Pick', memory=True, children=[
        MoveArmLeaf(ros_node, ARM_PICK, name='ArmToCube'),
        SetVacuumLeaf(ros_node, enable=True, name='AttachCube'),
        SetFlag(ros_node, 'holding_cube', True, name='holding=True'),
    ])
    place = py_trees.composites.Sequence(name='Place', memory=True, children=[
        MoveArmLeaf(ros_node, ARM_PLACE, name='ArmToTargetBox'),
        SetVacuumLeaf(ros_node, enable=False, name='DetachCube'),
        SetFlag(ros_node, 'holding_cube', False, name='holding=False'),
        SetFlag(ros_node, 'cube_placed', True, name='placed=True'),
    ])
    holding = ppa('HoldingCube', Condition('HoldingCube?', lambda: ros_node.holding_cube),
                  [at_source], pick)
    mission = ppa('CubeOnTarget', Condition('CubeOnTarget?', lambda: ros_node.cube_placed),
                  [holding, at_target], place)

    return py_trees.composites.Sequence(name='Root', memory=True, children=[
        # The gripper's DetachableJoint starts the simulation ATTACHED to the
        # cube (the robot is welded to a cube 7.5 m away until released).
        SetVacuumLeaf(ros_node, enable=False, name='ReleaseGripperWeld'),
        mission,
        ppa('ArmSafeAtEnd', Condition('ArmSafe?', ros_node.arm_is_safe), [],
            MoveArmLeaf(ros_node, ARM_SAFE, name='FoldArmAtEnd')),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
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
