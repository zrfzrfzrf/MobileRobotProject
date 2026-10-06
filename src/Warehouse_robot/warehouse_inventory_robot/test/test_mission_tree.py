"""Logic tests for the mission behaviour tree. No simulator needed.

The action servers are replaced by scripted fakes, and time is a counter the
test advances, so the tree's ordering, retries and failure behaviour can be
checked in a second. Needs py_trees; ROS itself is stubbed out if absent.

    python -m pytest test/test_mission_tree.py
"""

import math
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip('py_trees')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rclpy  # noqa: F401
except ImportError:
    # Stand-ins for the ROS packages mission_node imports. Only names used at
    # import time matter; the fakes below replace everything else.
    for mod in ('rclpy', 'rclpy.node', 'rclpy.action', 'rclpy.qos', 'action_msgs',
                'action_msgs.msg', 'ament_index_python', 'ament_index_python.packages',
                'builtin_interfaces', 'builtin_interfaces.msg', 'control_msgs',
                'control_msgs.action', 'geometry_msgs', 'geometry_msgs.msg',
                'irobot_create_msgs', 'irobot_create_msgs.action',
                'irobot_create_msgs.msg', 'nav2_msgs', 'nav2_msgs.action',
                'std_msgs', 'std_msgs.msg', 'trajectory_msgs', 'trajectory_msgs.msg'):
        sys.modules[mod] = MagicMock()
    sys.modules['rclpy.node'].Node = type('Node', (), {})
    sys.modules['action_msgs.msg'].GoalStatus = SimpleNamespace(
        STATUS_SUCCEEDED=4, STATUS_ABORTED=6)

from py_trees.common import Status  # noqa: E402

from warehouse_inventory_robot import mission_node as mn  # noqa: E402

SUCCEEDED, ABORTED = 4, 6


class FakeFuture:
    def __init__(self, ros, delay, value):
        self._ros, self._ready_at, self._value = ros, ros.t + delay, value

    def done(self):
        return self._ros.t >= self._ready_at

    def result(self):
        return self._value

    def add_done_callback(self, fn):
        fn(self)


class FakeHandle:
    accepted = True

    def __init__(self, ros, kind, status):
        self._ros, self._kind, self._status = ros, kind, status

    def get_result_async(self):
        result = SimpleNamespace(error_code=0, error_string='', is_docked=False)
        return FakeFuture(self._ros, 1.0, SimpleNamespace(status=self._status, result=result))

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
        status = self._outcomes.pop(0) if self._outcomes else SUCCEEDED
        return FakeFuture(self._ros, 0.0, FakeHandle(self._ros, self._kind, status))


class FakeRos:
    def __init__(self, nav_outcomes=(), docked=True):
        self.t = 0.0
        self.events = []
        self.is_docked = docked
        self.undock_client = FakeClient(self, 'undock')
        self.nav_client = FakeClient(self, 'nav', nav_outcomes)
        self.arm_client = FakeClient(self, 'arm')

    def now_sec(self):
        return self.t

    def get_logger(self):
        return MagicMock()

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


def mkpose(x, y, yaw):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='map', stamp=None),
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=0.0),
                             orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2),
                                                         w=math.cos(yaw / 2))))


def xy(pose):
    return (round(pose.pose.position.x, 2), round(pose.pose.position.y, 2))


# The real C-grade boxes: both are approached facing +y.
PICK = mkpose(1.98, 7.15, math.pi / 2)
DROP = mkpose(13.24, -22.07, math.pi / 2)
HOME = mkpose(0.0, 0.0, 0.0)
PICK_APPROACH, DROP_APPROACH = (1.98, 6.65), (13.24, -22.57)   # 0.5 m behind, facing the box


@pytest.fixture(autouse=True)
def arm_poses(monkeypatch):
    monkeypatch.setattr(mn, 'ARM_PICK', [1.0] * 6)
    monkeypatch.setattr(mn, 'ARM_PLACE', [2.0] * 6)


def run(ros, max_ticks=2000):
    root = mn.build_tree(ros, HOME, PICK, DROP)
    tree = mn.py_trees.trees.BehaviourTree(root)
    for _ in range(max_ticks):
        tree.tick()
        if root.status in (Status.SUCCESS, Status.FAILURE):
            break
        ros.t += 0.1
    return root.status, ros.events


def test_happy_path_runs_every_step_in_order():
    status, events = run(FakeRos())
    assert status == Status.SUCCESS
    assert [e[0] for e in events] == [
        'vacuum', 'arm', 'undock', 'nav', 'nav', 'arm', 'vacuum', 'arm',
        'nav', 'nav', 'arm', 'vacuum', 'arm']
    # each box: approach point first, then the head-on final leg
    assert [xy(e[1]) for e in events if e[0] == 'nav'] == [
        PICK_APPROACH, xy(PICK), DROP_APPROACH, xy(DROP)]
    assert [e[1] for e in events if e[0] == 'arm'] == [
        mn.ARM_SAFE, [1.0] * 6, mn.ARM_SAFE, [2.0] * 6, mn.ARM_SAFE]
    assert [e[1] for e in events if e[0] == 'vacuum'] == [False, True, False]


def test_failed_navigation_is_retried():
    status, events = run(FakeRos(nav_outcomes=[ABORTED, ABORTED, SUCCEEDED]))
    assert status == Status.SUCCESS
    assert [xy(e[1]) for e in events if e[0] == 'nav'] == [
        PICK_APPROACH, PICK_APPROACH, PICK_APPROACH, xy(PICK), DROP_APPROACH, xy(DROP)]


def test_failed_final_leg_restarts_from_the_approach_point():
    # approach ok, final leg aborted -> the retry drives to the approach point again
    status, events = run(FakeRos(nav_outcomes=[SUCCEEDED, ABORTED]))
    assert status == Status.SUCCESS
    assert [xy(e[1]) for e in events if e[0] == 'nav'][:4] == [
        PICK_APPROACH, xy(PICK), PICK_APPROACH, xy(PICK)]


def test_mission_fails_without_moving_arm_when_navigation_keeps_failing():
    status, events = run(FakeRos(nav_outcomes=[ABORTED] * 3))
    assert status == Status.FAILURE
    kinds = [e[0] for e in events]
    assert kinds.count('arm') == 1          # only the initial fold to ARM_SAFE
    assert ('vacuum', True) not in events


def test_undock_is_skipped_when_already_undocked():
    status, events = run(FakeRos(docked=False))
    assert status == Status.SUCCESS
    assert 'undock' not in [e[0] for e in events]


def test_unset_arm_pose_is_rejected_before_anything_moves(monkeypatch):
    monkeypatch.setattr(mn, 'ARM_PICK', None)
    ros = FakeRos()
    with pytest.raises(ValueError):
        mn.build_tree(ros, HOME, PICK, DROP)
    assert ros.events == []
