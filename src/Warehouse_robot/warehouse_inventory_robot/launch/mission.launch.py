import os
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



def lifecycle_activator(name, nodes):
    """Bring lifecycle nodes to `active`, one by one, retrying until they get there.

    Stands in for nav2_lifecycle_manager, which gives each change_state call a
    hardcoded deadline: on a loaded machine (Gazebo loading meshes, software
    rendering) a node's reply misses it ("failed to send response to
    /<node>/change_state"), the manager gives up, and the node stays
    `unconfigured`/`inactive` for good -- no /map, or no navigation at all.
    `ros2 lifecycle set` waits as long as it takes, and the loop re-checks the
    real state, so a transition that was only slow is not mistaken for a failure.
    """
    script = (
        'for n in ' + ' '.join(nodes) + '; do ok=0; '
        'for i in $(seq 1 90); do '
        'st=$(ros2 lifecycle get /$n 2>/dev/null); '
        'case "$st" in '
        'active*) ok=1; break;; '
        'unconfigured*) ros2 lifecycle set /$n configure >/dev/null 2>&1;; '
        'inactive*) ros2 lifecycle set /$n activate >/dev/null 2>&1;; '
        'esac; sleep 2; done; '
        'if [ $ok = 1 ]; then echo "$n: active"; '
        'else echo "$n never became active" >&2; exit 1; fi; done')
    return ExecuteProcess(cmd=['bash', '-c', script], name=name, output='screen')


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
    # behaviors, bt_navigator, ...). The parameter file is our copy in
    # config/nav2_params.yaml (Pure Pursuit instead of MPPI), so tune it there.
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
            'params_file': os.path.join(pkg_share, 'config', 'nav2_params.yaml'),
            'use_composition': 'False',
            'autostart': 'false',
        }.items(),
    )
    # Same node list as navigation_launch.py (Jazzy), in startup order.
    nav2_lifecycle_manager = lifecycle_activator('activate_nav2', [
        'controller_server', 'smoother_server', 'planner_server', 'route_server',
        'behavior_server', 'velocity_smoother', 'collision_monitor', 'bt_navigator',
        'waypoint_follower', 'docking_server'])

    # TODO: AMCL. For A grade only. The other grades get map -> odom from the static publisher
    # above, which is exact. Remember to launch amcl only for A grade.

    # TODO: You might also want to wait for map server and/or amcl to be ready.
    #
    # A fixed delay is fine for ordering things. It cannot fix one failure you
    # may hit if you use nav2_lifecycle_manager, though, so it is worth knowing
    # about in advance: it gives its lifecycle service calls a hardcoded
    # deadline, and on a cold start (Gazebo still loading meshes) map_server's
    # reply can miss it. The manager then never sends ACTIVATE and map_server
    # stays `inactive` indefinitely -- no /map, an empty RViz, and the costmaps
    # complaining "Can't update static costmap layer". Waiting longer does not
    # help once the manager has given up, so if you see that, check the state
    # and finish the transition yourself:
    #
    #     ros2 lifecycle get /map_server
    #     ros2 lifecycle set /map_server activate

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
                    nav2,
                    nav2_lifecycle_manager,
                ],
            )
        ),
    ])