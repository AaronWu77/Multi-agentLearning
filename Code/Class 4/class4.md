# 基于 IsaacGym 的 Unitree G1 机器人平地行走实验 #

### 安装流程 ###
1. 前期准备：Isaac Gym Preview 4 Release 目前仅支持 Linux 操作系统（本地运行安装双系统，请勿使用 WSL，系统版本推荐 Ubuntu 20.04/22.04）
2. 使用 miniconda/anaconda 创建新的 python 虚拟环境，- `conda create -n locomotion python=3.8`
3. 安装 pytorch，- `pip3 install torch torchvision torchaudio`
4. 安装 Isaac Gym
    - 运行以下命令: -`cd isaacgym/python && pip install -e .`
    - 运行 Isaac Gym 提供的样例: -`cd examples && python 1080_balls_of_solitude.py`
    - 遇到问题可以查看本地网页文档 `isaacgym/docs/index.html`
5. 安装 rsl_rl (用于足式机器人强化学习训练的 PPO 实现)
    - 运行以下命令: -`cd rsl_rl && pip install -e .`
6. 安装 legged_gym
    - 运行以下命令: -`cd legged_gym && pip install -e .`


### 代码结构 ###
训练环境由环境文件 `legged_robot.py` 和配置文件 `legged_robot_config.py` 组成，配置文件包含环境参数 `LeggedRobotCfg` 和强化学习算法的训练参数 `LeggeRobotCfgPPo`

### 代码运行 ###
1. 如果本地设备配备了 NVIDIA 显卡，可以运行以下命令在图形界面渲染训练环境：-`python legged_gym/legged_gym/scripts/train.py --task=g1 --rl_device=cuda:0 --sim_device=cuda:0 --num_envs=4`，否则需要将 `--rl_device` 和 `--sim_device` 都设置成 `cpu`（目前 Unitree G1 官方提供的 urdf 文件使用 cpu  进行物理仿真会失真影响训练）
2. 实际训练时显存最低 6GB，推荐 8GB 以上，（显存最低 6GB，推荐 8GB 以上），运行以下命令进行训练: -`python legged_gym/legged_gym/scripts/train.py --task=g1 --rl_device=cuda:0 --sim_device=cuda:0 --headless`，此时环境数量为 2048
3. 修改文件 `legged_gym/envs/g1/g1_config.py` 中的 `G1RoughCfg` 类成员，达到设备运行性能/奖励的平衡点
    - 环境数量：`G1RoughCfg.env.num_envs`
    - 交互步数：`G1RoughCfgPPO.runner.num_steps_per_env`
4. 调整 `g1_config.py` 中的奖励设置 `G1RoughCfg.rewards.scales` ，调整强化学习策略的行为，奖励函数的实现方式可以查阅 `legged_gym/envs/base/legged_robot.py` 中带有 `_reward_` 前缀的类函数，以下是部分奖励函数的作用
    - 任务奖励：`tracking_lin_vel` 和 `tracking_ang_vel` 分别表示 xy 方向的线速度（平面移动）和 yaw 方向的角速度（左右转动）
    - 行为约束：
        - 摔倒惩罚：`lin_vel_z` 和 `ang_vel_xy` 分别惩罚 z 轴方向的线速度和roll & pitch 轴旋转的角速度
        - 平滑限制：`torques` 和 `dof_acc` 分别惩罚电机力矩和关节加速度
        - 步态控制：`feet_air_time` 鼓励增加抬腿时间，减少小碎步
        - 碰撞惩罚和限位惩罚：`collision` 惩罚机器人的特定部位的肢体碰撞， `dof_pos_limits` 惩罚关节限位


### 常见问题 ###
1. 如果遇到: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`，请运行以下命令: `sudo apt install libpython3.8`
此外，还需要设置环境变量 `export LD_LIBRARY_PATH=/path/to/libpython/directory` / `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`

