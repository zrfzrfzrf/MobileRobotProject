# 实验室机器运行命令（C 级）

以下命令假设仓库在 `~/Desktop/MobileRobotProject`。如果放在别处，把 `cd` 那一行改成你的路径。

## 一、第一次准备（只做一次）

拉取最新代码：

```bash
cd ~/Desktop/MobileRobotProject && git pull
```

如果这个目录不是从 GitHub clone 的（`git pull` 报错），重新 clone：

```bash
cd ~/Desktop && git clone https://github.com/zrfzrfzrf/MobileRobotProject.git && cd MobileRobotProject
```

进入课程环境（只在实验室机器上有效，`student-shell` 上没有 `dd2410` 模块）：

```bash
source /etc/profile.d/modules.sh && module add dd2410 && pixi shell
```

安装 `py_trees`（实验室模块里没有）：

```bash
pip install --user --break-system-packages py_trees
```

检查磁盘配额（编译会写入几百 MB）：

```bash
fs quota
```

完整编译（第一次约 6 到 10 分钟）：

```bash
MAKEFLAGS=-j2 CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --base-paths src/Warehouse_robot --parallel-workers 4
```

## 二、每次运行

### 终端 1：启动仿真

```bash
source install/setup.bash && source scripts/env_vars.sh
```

```bash
GRADE=c ros2 launch warehouse_inventory_robot mission.launch.py
```

**等日志里出现 `docking_server: active` 再进行下一步**（导航栈全部就绪，约 1 到 2 分钟）。

### 终端 2：运行任务（新开一个终端）

```bash
cd ~/Desktop/MobileRobotProject && source /etc/profile.d/modules.sh && module add dd2410 && pixi shell
```

```bash
source install/setup.bash && source scripts/env_vars.sh
```

```bash
GRADE=c ros2 run warehouse_inventory_robot mission_node --ros-args -p use_sim_time:=true
```

### 应该看到的

- 终端 2 不断打印行为树，各步骤依次变成 `[✓]`：
  放开吸盘 → 收臂 → undock → 源箱子预备点 → 源箱子 → 伸臂 → 吸住 → 收臂 →
  目标箱子预备点 → 目标箱子 → 伸臂 → 放开 → 收臂
- 最后一行是 `MISSION SUCCEEDED.`
- Gazebo 里方块落在远处的目标箱子上。

## 三、重跑之前

两个终端都按 `Ctrl-C`，然后在终端 1 执行（Gazebo 不一定随 `Ctrl-C` 退出，不清理会让下一次启动出怪问题）：

```bash
bash scripts/cleanup.sh
```

## 四、改了代码之后

拉代码后只重新编译任务包（几秒钟）：

```bash
git pull && colcon build --base-paths src/Warehouse_robot --packages-select warehouse_inventory_robot
```

然后重新 `source install/setup.bash`，按第二节运行。

## 五、逻辑测试（不需要仿真）

```bash
cd src/Warehouse_robot/warehouse_inventory_robot && python -m pytest test/test_mission_tree.py -q; cd -
```

预期：`6 passed`。

## 六、出问题时检查

| 现象 | 处理 |
|---|---|
| 终端 1 一直没有 `docking_server: active`，或出现 `never became active` | `Ctrl-C`，执行 `bash scripts/cleanup.sh` 后重新启动 |
| 任务节点停在 `Starting mission.` 不动 | 仿真没起来或已经挂掉（没有 `/clock`）。检查终端 1，必要时清理后重启 |
| 某一步变成失败、任务结束 | 记下终端 2 最后打印的那棵树，看是哪一步失败 |
| 登录卡住或图形界面卡死 | 多半是磁盘配额满了，用 `fs quota` 检查，清理 `~/.cache` |

## 七、使用队友的代码（A 级，分支 `teammate-grade-a`）

队友的 `mission_node.py` 和 `mission.launch.py` 放在单独的分支 `teammate-grade-a` 上。
两套代码的包名和可执行文件名相同，不能同时编译，所以通过切换分支来选用，
`main` 分支保持我们自己的 C 级版本。

切换到队友的代码并重新编译任务包：

```bash
git fetch && git checkout teammate-grade-a && colcon build --base-paths src/Warehouse_robot --packages-select warehouse_inventory_robot
```

切回我们自己的版本：

```bash
git checkout main && colcon build --base-paths src/Warehouse_robot --packages-select warehouse_inventory_robot
```

切换后都要重新 `source install/setup.bash`。

### 用队友的代码跑 A 级

**必须用 `GRADE=a` 这种环境变量写法**：他的启动文件靠环境变量 `GRADE`
来决定里程计话题，只写 `grade:=a` 不够。

终端 1：

```bash
GRADE=a ros2 launch warehouse_inventory_robot mission.launch.py
```

等日志里出现 `Managed nodes are active`（他用的是 Nav2 自带的生命周期管理器）。
然后在 RViz 里选 **Publish Point** 工具，在地图上点一个位置，把机器人传送过去
（这一步模拟 TA 的操作，要在启动任务节点之前做；不要用 2D Pose Estimate）。

终端 2：

```bash
GRADE=a ros2 run warehouse_inventory_robot mission_node --ros-args -p use_sim_time:=true
```

任务节点启动后会先等约 20 秒再开始（他代码里的固定等待），这是正常的。
之后会做全局定位：撒满粒子、原地转、必要时往前开一段，直到 AMCL 收敛。

在这个分支上，第五节的逻辑测试不适用（测试是针对我们版本的代码写的）。
