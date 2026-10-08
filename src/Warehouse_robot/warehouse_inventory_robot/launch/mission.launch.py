import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from nav2_common.launch import RewrittenYaml
import yaml
import tempfile
from launch.actions import TimerAction

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

    # TODO: Navigation Layer

    nav2_params_file = build_nav2_params()

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'),
                        'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': nav2_params_file,
        }.items(),
    )
    # TODO: AMCL. For A grade only. The other grades get map -> odom from the static publisher
    # above, which is exact. Remember to launch amcl only for A grade.

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[nav2_params_file, {
            'use_sim_time': True,
            'set_initial_pose': True,
            'initial_pose': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0},
        }],
        condition=IfCondition(grade_is_a),
    )

    amcl_life = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='amcl_lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': True, 'autostart': True, 'node_names': ['amcl']}],
        condition=IfCondition(grade_is_a),
    )


    # TODO: Map server.
    # NOTE: We provide a map at src/Warehouse_robot/warehouse_inventory_robot/maps
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': os.path.join(pkg_share, 'maps', 'warehouse.yaml'),
            'use_sim_time': True
            }]
    )
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
    map_life = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='map_lifecycle_manager',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    delayed_arm_spawner = TimerAction(
        period=10.0,
        actions=[arm_traj_spawner]
    )
    return LaunchDescription([
        grade_arg, x_pose_arg, y_pose_arg, headless_arg,

        simulation_launch,

        relocate_robot,

        # Start watching for the simulation to come up.
        wait_for_sim,

        map_server,
        map_life,

        amcl,
        amcl_life,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=wait_for_sim,
                on_exit=[
                    delayed_arm_spawner,
                    static_map_to_odom,
                    navigation,
                ],
            )
        ),
    ])

def build_nav2_params():
    src = os.path.join(get_package_share_directory('nav2_bringup'),
                       'params', 'nav2_params.yaml')
    with open(src) as f:
        params = yaml.safe_load(f)

    odom = '/odom' if os.environ.get('GRADE', 'e').lower() == 'a' else '/odom_gt'
    # pure pursuit
    params['controller_server']['ros__parameters']['FollowPath'] = {
        'plugin': 'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
        'desired_linear_vel': 0.3,
        'lookahead_dist': 0.6,
        'min_lookahead_dist': 0.3,
        'max_lookahead_dist': 0.9,
        'use_velocity_scaled_lookahead_dist': True,
        'use_rotate_to_heading': True,
        'rotate_to_heading_angular_vel': 1.0,
        'allow_reversing': False,
        'use_collision_detection': True,
    }

    params['controller_server']['ros__parameters']['progress_checker'] = {
        'plugin': 'nav2_controller::SimpleProgressChecker',
        'required_movement_radius': 0.1,
        'movement_time_allowance': 40.0,
    }

    params['controller_server']['ros__parameters']['general_goal_checker'] = {
        'plugin': 'nav2_controller::SimpleGoalChecker',
        'xy_goal_tolerance': 0.05,
        'yaw_goal_tolerance': 0.05,
        'stateful': True,
    }

    params['controller_server']['ros__parameters']['controller_frequency'] = 10.0

    params['local_costmap']['local_costmap']['ros__parameters']['update_frequency'] = 5.0
    params['local_costmap']['local_costmap']['ros__parameters']['publish_frequency'] = 2.0
    params['global_costmap']['global_costmap']['ros__parameters']['update_frequency'] = 1.0
    params['global_costmap']['global_costmap']['ros__parameters']['publish_frequency'] = 1.0

    # params['bt_navigator']['ros__parameters']['odom_topic'] = '/odom_gt'
    # params['velocity_smoother']['ros__parameters']['odom_topic'] = '/odom_gt'
    # params['controller_server']['ros__parameters']['odom_topic'] = '/odom_gt'
    
    # use unstamped
    params['collision_monitor']['ros__parameters']['cmd_vel_out_topic'] = 'cmd_vel_unstamped'

    amcl = params['amcl']['ros__parameters']
    amcl['min_particles'] = 5000
    amcl['max_particles'] = 20000
    amcl['update_min_a'] = 0.1
    amcl['update_min_d'] = 0.1
    amcl['recovery_alpha_slow'] = 0.001
    amcl['recovery_alpha_fast'] = 0.1


    params['bt_navigator']['ros__parameters']['odom_topic'] = odom
    params['velocity_smoother']['ros__parameters']['odom_topic'] = odom
    params['controller_server']['ros__parameters']['odom_topic'] = odom

    vs = params['velocity_smoother']['ros__parameters']
    vs['max_velocity'] = [0.6, 0.0, 0.6]
    vs['min_velocity'] = [-0.6, 0.0, -0.6]
    vs['max_accel']    = [0.1, 0.0, 0.4]
    vs['max_decel']    = [-0.1, 0.0, -0.4]
    tmp = tempfile.NamedTemporaryFile('w', suffix='_nav2_params.yaml', delete=False)
    yaml.safe_dump(params, tmp)
    tmp.close()
    return tmp.name
