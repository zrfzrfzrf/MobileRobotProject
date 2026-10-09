import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare



# Run by lifecycle_activator() in its own process, with the node names as
# arguments. One process for all the nodes: starting a `ros2 lifecycle` CLI per
# step (an earlier version) cost several minutes of bring-up.
_ACTIVATOR = r"""
import sys, time
import rclpy
from rclpy.node import Node
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState

rclpy.init()
node = Node('lifecycle_activator_' + sys.argv[1])

def call(client, request, timeout):
    # None when the service is missing or the reply never comes: retry then.
    if not client.wait_for_service(timeout_sec=timeout):
        return None
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    return future.result() if future.done() else None

for name in sys.argv[2:]:
    get = node.create_client(GetState, '/' + name + '/get_state')
    change = node.create_client(ChangeState, '/' + name + '/change_state')
    deadline = time.time() + 300.0
    while True:
        if time.time() > deadline:
            print(name + ' never became active', file=sys.stderr, flush=True)
            sys.exit(1)
        state = call(get, GetState.Request(), 10.0)
        if state is None:
            continue
        label = state.current_state.label
        if label == 'active':
            print(name + ': active', flush=True)
            break
        request = ChangeState.Request()
        if label == 'unconfigured':
            request.transition.id = Transition.TRANSITION_CONFIGURE
        elif label == 'inactive':
            request.transition.id = Transition.TRANSITION_ACTIVATE
        else:                       # mid-transition: look again shortly
            time.sleep(0.5)
            continue
        call(change, request, 30.0)
"""


def lifecycle_activator(name, nodes, condition=None):
    """Bring lifecycle nodes to `active`, one by one, retrying until they get there.

    Stands in for nav2_lifecycle_manager, which gives each change_state call a
    hardcoded deadline: on a loaded machine (Gazebo loading meshes, software
    rendering) a node's reply misses it ("failed to send response to
    /<node>/change_state"), the manager gives up, and the node stays
    `unconfigured`/`inactive` for good -- no /map, or no navigation at all.
    Here every call has a timeout and the real state is read again before each
    step, so a reply that was lost or merely slow costs a retry, not the run.
    """
    return ExecuteProcess(cmd=['python3', '-c', _ACTIVATOR, name] + list(nodes),
                          name=name, output='screen', condition=condition)


def build_nav2_params():
    """Nav2's parameters: the TurtleBot 4 file plus the changes this mission needs.

    Made here, in code, because this launch file and mission_node.py are all that
    is handed in: a yaml of our own (config/nav2_params.yaml, an earlier version)
    would not be there. Written to a temporary file for navigation_launch.py.
    """
    with open(os.path.join(get_package_share_directory('turtlebot4_navigation'),
                           'config', 'nav2.yaml')) as f:
        params = yaml.safe_load(f)

    # Odometry: grade A has no /odom_gt, only the wheel odometry.
    params['bt_navigator']['ros__parameters']['odom_topic'] = 'odom'

    controller = params['controller_server']['ros__parameters']
    controller['general_goal_checker']['xy_goal_tolerance'] = 0.08
    controller['general_goal_checker']['yaw_goal_tolerance'] = 0.1
    # Pure Pursuit instead of the default MPPI (the assignment page recommends it).
    controller['FollowPath'] = {
        'plugin': 'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
        'desired_linear_vel': 0.4,
        'lookahead_dist': 0.6,
        'min_lookahead_dist': 0.3,
        'max_lookahead_dist': 0.9,
        'lookahead_time': 1.5,
        'transform_tolerance': 0.1,
        'use_velocity_scaled_lookahead_dist': False,
        'min_approach_linear_velocity': 0.05,
        'approach_velocity_scaling_dist': 0.6,
        'use_collision_detection': True,
        'max_allowed_time_to_collision_up_to_carrot': 1.0,
        'use_regulated_linear_velocity_scaling': True,
        'use_cost_regulated_linear_velocity_scaling': False,
        'regulated_linear_scaling_min_radius': 0.9,
        'regulated_linear_scaling_min_speed': 0.25,
        'use_rotate_to_heading': True,
        'rotate_to_heading_angular_vel': 1.0,
        'rotate_to_heading_min_angle': 0.785,
        'max_angular_accel': 3.2,
        'allow_reversing': False,
        'max_robot_pose_search_dist': 1.0,
    }

    # Speed. The limit is the LINEAR ACCELERATION, not the top speed: pulling away
    # hard pitches the arm-laden robot back, the front cliff sensors lose the
    # floor, and the Create 3 cliff reflex takes over (it backs up in arcs at
    # 0.1 m/s, ignoring Nav2, again and again). Measured on 7 m trips, 4 each:
    #   0.30 m/s, 2.5 m/s^2:  28.8 s (x3), 32.5 s       (some cliff events)
    #   0.40 m/s, 2.5 m/s^2:  22.6 / 57 / 66 / 76 s     (47-67 cliff events)
    #   0.40 m/s, 0.5 m/s^2:  22.6 s (x4)                (none)
    #   0.40 m/s, 0.25 m/s^2: 22.8-23.0 s (x4)           (none)
    #   0.45 m/s, 0.25 m/s^2: 20.8 s (x4)                (none)
    # 0.45 would sit at the wheel limit (0.46 m/s, safety_override "full" in
    # create3_nodes.launch.py), so 0.40 at 0.25 m/s^2.
    smoother = params['velocity_smoother']['ros__parameters']
    smoother['odom_topic'] = 'odom'
    smoother['max_velocity'] = [0.40, 0.0, 1.0]
    smoother['min_velocity'] = [-0.40, 0.0, -1.0]
    smoother['max_accel'] = [0.25, 0.0, 3.2]
    smoother['max_decel'] = [-0.25, 0.0, -3.2]

    # No static map in the local costmap: in grade A map -> odom is wrong until
    # AMCL has converged, and a misplaced map there would block the spins of the
    # localization search.
    local = params['local_costmap']['local_costmap']['ros__parameters']
    local['plugins'] = ['voxel_layer', 'inflation_layer']

    tmp = tempfile.NamedTemporaryFile('w', suffix='_nav2_params.yaml', delete=False)
    yaml.safe_dump(params, tmp)
    tmp.close()
    return tmp.name


def generate_launch_description():
    pkg_share = get_package_share_directory('warehouse_inventory_robot')

    # Selects the grade this run is configured for; see simulation.launch.py
    # for what it changes in the world and in odometry. Here it decides who
    # publishes map -> odom, and which box the mission should treat as the drop
    # target.
    # Defaults to the GRADE environment variable so that one spelling works
    # everywhere: `GRADE=c pixi run mission` and, inside a pixi shell,
    # `GRADE=c ros2 launch ...`. An explicit grade:=c on the command line still
    # wins over it.
    grade_arg = DeclareLaunchArgument(
        'grade', default_value=EnvironmentVariable('GRADE', default_value='e'),
        choices=['e', 'c', 'a'],
        description='Which grade to configure the mission for (default: $GRADE, or e)')
    grade = LaunchConfiguration('grade')
    grade_is_a = PythonExpression(["'", grade, "' == 'a'"])

    headless_arg = DeclareLaunchArgument(
        'headless', default_value=EnvironmentVariable('HEADLESS', default_value='false'),
        choices=['true', 'false'],
        description='Run Gazebo headless (server only); RViz still shows the scene')

    # Must match the world simulation.launch.py selects, AND the name declared
    # inside that .sdf. relocate_robot teleports through
    # /world/<name>/set_pose, so a mismatch here means every click is silently
    # ignored.
    world = PythonExpression(
        ["'warehouse_dynamic' if '", grade, "' in ('c', 'a') else 'warehouse'"])

    x_pose_arg = DeclareLaunchArgument('x_pose', default_value='0.0')
    y_pose_arg = DeclareLaunchArgument('y_pose', default_value='0.0')

    # Simulation Layer
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('warehouse_inventory_robot'),
                'launch', 'simulation.launch.py'
            ])
        ]),
        launch_arguments={
            'grade': grade,
            'x': LaunchConfiguration('x_pose'),
            'y': LaunchConfiguration('y_pose'),
            'headless': LaunchConfiguration('headless'),
        }.items()
    )

    # E and C only. Their odometry is Gazebo's exact pose.
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_static_tf',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': True}],
        condition=UnlessCondition(grade_is_a),
    )

    # Examiner tool, A grade only. Lets the robot be placed anywhere on the map
    # by clicking in RViz with the "Publish Point" tool.
    relocate_robot = Node(
        package='warehouse_inventory_robot',
        executable='relocate_robot',
        name='relocate_robot',
        output='screen',
        parameters=[{'world': world}],
        condition=IfCondition(grade_is_a),
    )

    # The arm controller's type comes from config/arm_controller_types.yaml,
    # loaded into controller_manager by gz_ros2_control at construction, so it is
    # already defined by the time this spawner runs.
    arm_traj_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "lite6_traj_controller", 
            "-c", "/controller_manager",
            "--param-file", os.path.join(pkg_share, 'config', 'arm_controllers.yaml')
        ],
        output="screen",
    )

    # Bringup is gated on observed readiness, not on fixed delays. Wall-clock
    # delays are machine-dependent.
    wait_for_sim = Node(
        package='warehouse_inventory_robot',
        executable='wait_for_ready',
        name='wait_for_sim',
        output='screen',
        arguments=['--label', 'sim',
                   '--node', 'controller_manager',
                   '--call-service', '/controller_manager/list_controllers',
                   '--service', '/controller_manager/set_parameters',
                   '--topic', '/scan',
                   '--timeout', '300'],
    )

    # Map server. Serves /map to Nav2's static costmap layer. It is a lifecycle
    # node, so a lifecycle manager has to configure and activate it.
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': os.path.join(pkg_share, 'maps', 'warehouse.yaml'),
            'use_sim_time': True,
        }],
    )
    map_lifecycle_manager = lifecycle_activator('activate_map_server', ['map_server'])

    # Navigation layer: nav2_bringup's navigation_launch.py (planner, controller,
    # behaviors, bt_navigator, ...). Its parameters come from build_nav2_params().
    #
    # autostart is off: its lifecycle manager is the component that fails on a
    # loaded machine (see lifecycle_activator), and its 4 s heartbeat deadline is
    # too tight for a simulation running below real time -- the first planning
    # request starved collision_monitor's heartbeat and the manager shut the whole
    # stack down ("CRITICAL FAILURE: SERVER collision_monitor IS DOWN").
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': build_nav2_params(),
            'use_composition': 'False',
            'autostart': 'false',
        }.items(),
    )
    # Same node list as navigation_launch.py (Jazzy), in startup order.
    nav2_lifecycle_manager = lifecycle_activator('activate_nav2', [
        'controller_server', 'smoother_server', 'planner_server', 'route_server',
        'behavior_server', 'velocity_smoother', 'collision_monitor', 'bt_navigator',
        'waypoint_follower', 'docking_server'])

    # AMCL, grade A only (E and C get an exact map -> odom from the static
    # publisher above). It publishes map -> odom, correcting the wheel odometry
    # that drifts in grade A. Parameters inline so the solution needs no extra
    # file. Where they differ from turtlebot4_navigation/config/localization.yaml:
    #   max_particles 20000  global localization spreads particles over the
    #                        whole ~30 x 50 m map; 2000 is far too sparse.
    #   max_beams 120        more evidence per update, sharper likelihood.
    #   laser_max_range 12   the simulated RPLIDAR's real range (was 100).
    #   update_min_d/_a      update after 0.1 m / 0.15 rad, so a spin on the
    #                        spot feeds AMCL many scans.
    #   recovery_alpha_*  0  no random particles once converged: the mission
    #                        checks the estimate against the map itself and
    #                        re-scatters the particles when it is wrong.
    # set_initial_pose at the spawn point: right if the robot was not moved,
    # and harmless if it was (the mission detects the mismatch and runs
    # global localization).
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'global_frame_id': 'map',
            'scan_topic': 'scan',
            'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
            'laser_model_type': 'likelihood_field',
            'laser_max_range': 12.0,
            'laser_min_range': 0.2,
            'max_beams': 120,
            'laser_likelihood_max_dist': 2.0,
            'sigma_hit': 0.2,
            'z_hit': 0.5,
            'z_rand': 0.5,
            'z_max': 0.05,
            'z_short': 0.05,
            'min_particles': 2000,
            'max_particles': 20000,
            'pf_err': 0.05,
            'pf_z': 0.99,
            'alpha1': 0.2, 'alpha2': 0.2, 'alpha3': 0.2, 'alpha4': 0.2, 'alpha5': 0.2,
            'update_min_d': 0.1,
            'update_min_a': 0.15,
            'resample_interval': 1,
            'recovery_alpha_slow': 0.0,
            'recovery_alpha_fast': 0.0,
            'transform_tolerance': 1.0,
            'tf_broadcast': True,
            'save_pose_rate': 0.5,
            'set_initial_pose': True,
            'initial_pose': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0},
        }],
        condition=IfCondition(grade_is_a),
    )
    amcl_lifecycle_manager = lifecycle_activator(
        'activate_amcl', ['amcl'], condition=IfCondition(grade_is_a))

    return LaunchDescription([
        grade_arg, x_pose_arg, y_pose_arg, headless_arg,

        simulation_launch,

        relocate_robot,

        # Start watching for the simulation to come up.
        wait_for_sim,

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=wait_for_sim,
                on_exit=[
                    arm_traj_spawner,
                    static_map_to_odom,
                    map_server,
                    map_lifecycle_manager,
                    amcl,
                    amcl_lifecycle_manager,
                    nav2,
                    nav2_lifecycle_manager,
                ],
            )
        ),
    ])
