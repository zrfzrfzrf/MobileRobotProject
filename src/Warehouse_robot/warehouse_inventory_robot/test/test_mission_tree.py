"""Logic tests for the mission behaviour tree. No simulator needed.

A small fake world stands in for ROS: action servers succeed (or fail, as
scripted) after one simulated second and change the fake world accordingly
(the arm moves, the robot drives, undocks, gets localized). Time is a counter
the test advances, so ordering, retries and the tree's reactions can be checked
in a fraction of a second. Needs py_trees; ROS itself is stubbed out if absent.

    python -m pytest test/test_mission_tree.py
"""

import math
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip('py_trees')
pytest.importorskip('numpy')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rclpy  # noqa: F401
except ImportError:
    # Stand-ins for the ROS packages mission_node imports. Only names used at
    # import time matter; the fakes below replace everything else.
    for mod in ('rclpy', 'rclpy.node', 'rclpy.action', 'rclpy.qos', 'rclpy.time', 'tf2_ros',
                'action_msgs', 'action_msgs.msg', 'ament_index_python',
                'ament_index_python.packages', 'builtin_interfaces', 'builtin_interfaces.msg',
                'control_msgs', 'control_msgs.action', 'geometry_msgs', 'geometry_msgs.msg',
                'irobot_create_msgs', 'irobot_create_msgs.action', 'irobot_create_msgs.msg',
                'nav2_msgs', 'nav2_msgs.action', 'nav_msgs', 'nav_msgs.msg',
                'sensor_msgs', 'sensor_msgs.msg', 'sensor_msgs_py', 'sensor_msgs_py.point_cloud2',
                'std_msgs', 'std_msgs.msg', 'std_srvs', 'std_srvs.srv',
                'trajectory_msgs', 'trajectory_msgs.msg'):
        sys.modules[mod] = MagicMock()
    sys.modules['rclpy.node'].Node = type('Node', (), {})
    sys.modules['action_msgs.msg'].GoalStatus = SimpleNamespace(
        STATUS_SUCCEEDED=4, STATUS_ABORTED=6)

from py_trees.common import Status  # noqa: E402

from warehouse_inventory_robot import mission_node as mn  # noqa: E402

SUCCEEDED, ABORTED = 4, 6


def mkpose(x, y, yaw):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='map', stamp=None),
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=0.0),
                             orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2),
                                                         w=math.cos(yaw / 2))))


def xy(pose):
    return (round(pose.pose.position.x, 2), round(pose.pose.position.y, 2))


# The real C/A-grade boxes: both are approached facing +y.
PICK = mkpose(1.98, 7.15, math.pi / 2)
DROP = mkpose(13.24, -22.07, math.pi / 2)
HOME = mkpose(0.0, 0.0, 0.0)
PICK_APPROACH, DROP_APPROACH = (1.98, 6.65), (13.24, -22.57)   # 0.5 m behind, facing the box
SPAWN_ARM = [0.0, 0.87, 1.57, 0.0, -1.57, 0.0]


class FakeFuture:
    def __init__(self, ros, delay, value, on_done=None):
        self._ros, self._ready_at, self._value = ros, ros.t + delay, value
        self._on_done, self._fired = on_done, False

    def done(self):
        ready = self._ros.t >= self._ready_at
        if ready and self._on_done and not self._fired:
            self._fired = True
            self._on_done()
        return ready

    def result(self):
        return self._value

    def add_done_callback(self, fn):
        fn(self)


class FakeHandle:
    accepted = True

    def __init__(self, ros, kind, status, effect):
        self._ros, self._kind, self._status, self._effect = ros, kind, status, effect

    def get_result_async(self):
        result = SimpleNamespace(error_code=0, error_string='')
        on_done = self._effect if self._status == SUCCEEDED else None
        return FakeFuture(self._ros, 1.0, SimpleNamespace(status=self._status, result=result),
                          on_done)

    def cancel_goal_async(self):
        self._ros.events.append((self._kind, 'cancel'))


class FakeClient:
    """Stands in for an ActionClient; `outcomes` is the status of each goal in turn."""

    def __init__(self, ros, kind, outcomes=()):
        self._ros, self._kind, self._outcomes = ros, kind, list(outcomes)

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):
        self._ros.events.append((self._kind, goal))
        if self._kind == 'undock' and self._ros.dock_status_flips_early:
            self._ros.is_docked = False
        if self._kind == 'nav' and self._ros.nav_pose_flips_early:
            self._ros.apply('nav', goal)
        status = self._outcomes.pop(0) if self._outcomes else SUCCEEDED
        effect = lambda: self._ros.apply(self._kind, goal)  # noqa: E731
        return FakeFuture(self._ros, 0.0, FakeHandle(self._ros, self._kind, status, effect))


class FakeService:
    def __init__(self, ros):
        self._ros = ros

    def service_is_ready(self):
        return True

    def call_async(self, req):
        self._ros.events.append(('reinit', None))
        self._ros.confident, self._ros.match_score, self._ros.spins_seen = False, 0.0, 0
        return FakeFuture(self._ros, 0.0, None)


class FakeRos:
    """The parts of MissionNode the tree touches, backed by a toy world."""

    def __init__(self, nav_outcomes=(), docked=True, grade='c', spins_to_converge=2,
                 dock_status_flips_early=False, nav_pose_flips_early=False):
        self.t = 0.0
        self.events = []
        self.is_docked = docked
        self.arm = list(SPAWN_ARM)
        self.pose = (0.0, 0.0, 0.0)
        self.needs_localization = grade == 'a'
        self.localized = not self.needs_localization
        self.confident = False
        self.match_score = 0.0
        self.match_score_seq = 0
        self.amcl_cov = (1.0, 1.0, 1.0)
        self.spins_seen = 0
        self.spins_to_converge = spins_to_converge
        # The real /dock_status says "undocked" as soon as the robot starts to
        # back off, long before the Undock action finishes.
        self.dock_status_flips_early = dock_status_flips_early
        # Likewise the robot is within the at-box tolerance before Nav2 is done.
        self.nav_pose_flips_early = nav_pose_flips_early
        self._low_score_since = None
        self.holding_cube = False
        self.cube_placed = False
        self.aligned = 0
        self.undock_client = FakeClient(self, 'undock')
        self.nav_client = FakeClient(self, 'nav', nav_outcomes)
        self.arm_client = FakeClient(self, 'arm')
        self.spin_client = FakeClient(self, 'spin')
        self.drive_client = FakeClient(self, 'drive')
        self.reinit_client = FakeService(self)

    # the toy world -------------------------------------------------------
    def apply(self, kind, goal):
        if kind == 'arm':
            self.arm = list(goal)
        elif kind == 'undock':
            self.is_docked = False
        elif kind == 'nav':
            q = goal.pose.orientation
            self.pose = (goal.pose.position.x, goal.pose.position.y,
                         2 * math.atan2(q.z, q.w))
        elif kind == 'spin' and self.needs_localization:
            self.spins_seen += 1
            if self.spins_seen >= self.spins_to_converge:
                self.confident, self.match_score, self.amcl_cov = True, 1.0, (0.01, 0.01, 0.01)

    # MissionNode API -------------------------------------------------------
    def now_sec(self):
        return self.t

    def get_logger(self):
        return MagicMock()

    def is_undocked(self):
        return self.is_docked is False

    def arm_is_safe(self):
        return all(
            abs(a - b) < mn.ARM_SAFE_TOL_RAD for a, b in zip(self.arm, mn.ARM_SAFE))

    def at_pose(self, pose):
        q = pose.pose.orientation
        yaw = 2 * math.atan2(q.z, q.w)
        return (math.hypot(self.pose[0] - pose.pose.position.x,
                           self.pose[1] - pose.pose.position.y) < mn.AT_BOX_TOL_M
                and abs(mn.angle_diff(self.pose[2], yaw)) < mn.AT_BOX_TOL_RAD)

    def robot_pose(self):
        return self.pose

    def is_localized(self):
        return True if not self.needs_localization else self.localized

    def pose_agrees_with_scan(self):
        return self.confident and self.match_score >= mn.LOC_SCORE_OK

    def amcl_confident(self):
        return self.confident

    def explore_heading(self):
        return 0.5

    def front_distance(self):
        self.aligned += 1
        return mn.BOX_FACE_DIST_M

    def drive(self, v):
        pass

    def remember_dock(self):
        pass

    def vacuum_ready(self, enable):
        return True

    def set_vacuum(self, enable, log=True):
        # The leaf re-sends while settling; record one event per change.
        if not self.events or self.events[-1] != ('vacuum', enable):
            self.events.append(('vacuum', enable))

    def make_nav_goal(self, pose):
        return pose

    def make_arm_goal(self, angles, duration_sec=4):
        return angles

    def make_spin_goal(self, yaw):
        return yaw

    def make_drive_goal(self, dist):
        return dist


def run(ros, max_ticks=20000, hook=None):
    root = mn.build_tree(ros, HOME, PICK, DROP)
    tree = mn.py_trees.trees.BehaviourTree(root)
    for _ in range(max_ticks):
        ros.match_score_seq += 1               # a fresh scan score every tick
        tree.tick()
        if hook:
            hook(ros)
        if root.status in (Status.SUCCESS, Status.FAILURE):
            break
        ros.t += 0.1
    return root.status, ros.events


def kinds(events):
    return [e[0] for e in events if e[1] != 'cancel']


def test_grade_c_runs_every_step_in_order():
    status, events = run(FakeRos(grade='c'))
    assert status == Status.SUCCESS
    assert kinds(events) == [
        'vacuum',                      # release the start-up weld
        'arm', 'undock',               # Move preconditions: fold arm, leave the dock
        'nav', 'nav', 'arm', 'vacuum',  # approach + source, pick, attach
        'arm',                         # the next Move's ArmSafe precondition folds it
        'nav', 'nav', 'arm', 'vacuum',  # approach + target, place, detach
        'arm']                         # fold at the end
    assert [xy(e[1]) for e in events if e[0] == 'nav'] == [
        PICK_APPROACH, xy(PICK), DROP_APPROACH, xy(DROP)]
    assert [e[1] for e in events if e[0] == 'arm'] == [
        mn.ARM_SAFE, mn.ARM_PICK, mn.ARM_SAFE, mn.ARM_PLACE, mn.ARM_SAFE]
    assert [e[1] for e in events if e[0] == 'vacuum'] == [False, True, False]


def test_grade_a_localizes_after_undocking_and_before_driving():
    status, events = run(FakeRos(grade='a'))
    assert status == Status.SUCCESS
    k = kinds(events)
    assert k.index('undock') < k.index('reinit') < k.index('nav')
    # look-around spins, then the four-view verification (3 quarter turns),
    # all before the first drive
    assert k[k.index('reinit'):k.index('nav')].count('spin') >= 5
    assert k.count('reinit') == 1          # localized once, kept through the mission


def test_preconditions_are_checked_before_every_move():
    status, events = run(FakeRos(grade='c', docked=False))
    assert status == Status.SUCCESS
    assert 'undock' not in kinds(events)   # already off the dock: Undocked? holds


def test_undock_is_not_cut_short_when_dock_status_flips_early():
    # Regression: the reactive tree used to see 'Undocked?' turn true while the
    # Undock action was still running, cancel it, and start spinning on the dock.
    status, events = run(FakeRos(grade='a', dock_status_flips_early=True))
    assert status == Status.SUCCESS
    assert ('undock', 'cancel') not in events
    k = kinds(events)
    assert k.count('undock') == 1 and k.index('undock') < k.index('reinit')


def test_drive_is_not_cut_short_when_robot_is_already_near_the_box():
    # Regression: 'AtSource?' turned true within 0.25 m of the box, the tree
    # cancelled the running drive, and the laser alignment never ran.
    status, events = run(FakeRos(grade='c', nav_pose_flips_early=True))
    assert status == Status.SUCCESS
    assert ('nav', 'cancel') not in events
    assert [xy(e[1]) for e in events if e[0] == 'nav'] == [
        PICK_APPROACH, xy(PICK), DROP_APPROACH, xy(DROP)]


def test_every_box_ends_with_laser_alignment():
    ros = FakeRos(grade='c')
    status, _ = run(ros)
    assert status == Status.SUCCESS and ros.aligned >= 2


def test_confident_but_wrong_amcl_is_rescattered():
    # AMCL collapses onto a wrong place: confident, but the scan contradicts the
    # map. The search must give up on it quickly and scatter the particles again.
    ros = FakeRos(grade='a', spins_to_converge=10 ** 6)
    ros.confident, ros.amcl_cov, ros.match_score = True, (0.01, 0.01, 0.01), 0.1

    def converge_after_second_scatter(r):
        if [e[0] for e in r.events].count('reinit') >= 2 and not r.match_score:
            r.confident, r.match_score, r.amcl_cov = True, 1.0, (0.01, 0.01, 0.01)
        elif [e[0] for e in r.events].count('reinit') == 1:
            r.confident, r.match_score, r.amcl_cov = True, 0.1, (0.01, 0.01, 0.01)

    status, events = run(ros, hook=converge_after_second_scatter)
    assert status == Status.SUCCESS
    assert kinds(events).count('reinit') == 2
    # the wrong place was abandoned long before the search timeout
    assert ros.t < mn.LOC_SEARCH_TIMEOUT_SEC


def test_lost_localization_halts_driving_and_relocalizes():
    lost = {'done': False}

    def lose_it_mid_drive(ros):
        # Once the robot is driving to the target, pretend AMCL went wrong.
        navs = [e for e in ros.events if e[0] == 'nav' and e[1] != 'cancel']
        if not lost['done'] and len(navs) == 3:
            lost['done'] = True
            ros.localized, ros.confident, ros.match_score = False, False, 0.0

    status, events = run(FakeRos(grade='a'), hook=lose_it_mid_drive)
    assert status == Status.SUCCESS
    assert ('nav', 'cancel') in events     # the running drive was halted
    assert kinds(events).count('reinit') == 2
    assert ('vacuum', True) in events and events[-1][0] == 'arm'


def test_failed_navigation_is_retried():
    status, events = run(FakeRos(nav_outcomes=[ABORTED, ABORTED, SUCCEEDED]))
    assert status == Status.SUCCESS
    assert [xy(e[1]) for e in events if e[0] == 'nav'] == [
        PICK_APPROACH, PICK_APPROACH, PICK_APPROACH, xy(PICK), DROP_APPROACH, xy(DROP)]


def test_failed_final_leg_restarts_from_the_approach_point():
    status, events = run(FakeRos(nav_outcomes=[SUCCEEDED, ABORTED]))
    assert status == Status.SUCCESS
    assert [xy(e[1]) for e in events if e[0] == 'nav'][:4] == [
        PICK_APPROACH, xy(PICK), PICK_APPROACH, xy(PICK)]


def test_mission_fails_without_picking_when_navigation_keeps_failing():
    status, events = run(FakeRos(nav_outcomes=[ABORTED] * 3))
    assert status == Status.FAILURE
    assert ('vacuum', True) not in events
    assert mn.ARM_PICK not in [e[1] for e in events if e[0] == 'arm']


def test_unset_arm_pose_is_rejected_before_anything_moves(monkeypatch):
    monkeypatch.setattr(mn, 'ARM_PICK', None)
    ros = FakeRos()
    with pytest.raises(ValueError):
        mn.build_tree(ros, HOME, PICK, DROP)
    assert ros.events == []


def test_map_matcher_separates_agree_occluded_and_see_through(tmp_path):
    # 10 x 10 cell map, 1 m cells, one wall column at x = 5..6
    img = bytearray([254] * 100)
    for row in range(10):
        img[row * 10 + 5] = 0
    (tmp_path / 'm.pgm').write_bytes(b'P5\n10 10\n255\n' + bytes(img))
    (tmp_path / 'm.yaml').write_text(
        'image: m.pgm\nresolution: 1.0\norigin: [0, 0, 0]\nnegate: 0\n'
        'occupied_thresh: 0.65\nfree_thresh: 0.1\n')
    m = mn.MapMatcher(str(tmp_path / 'm.yaml'))
    np = mn.np
    # laser at (2, 5) facing +x: the map predicts the wall 3 m ahead
    assert abs(m.expected_ranges(2.0, 5.0, np.array([0.0]), 12.0)[0] - 3.0) < 0.9
    beams = np.zeros(40)                     # 40 beams, all straight ahead
    agree = m.consistency(2.0, 5.0, 0.0, np.full(40, 3.0), beams, 0.1, 12.0)
    occluded = m.consistency(2.0, 5.0, 0.0, np.r_[np.full(35, 3.0), np.full(5, 1.0)],
                             beams, 0.1, 12.0)
    through = m.consistency(2.0, 5.0, 0.0, np.r_[np.full(20, 3.0), np.full(20, 8.0)],
                            beams, 0.1, 12.0)
    assert agree == 1.0
    assert occluded == 1.0                   # something in front of the wall: ignored
    assert through == 0.5                    # beams through the wall count against
