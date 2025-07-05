# 基于 IsaacGym 的 Unitree G1 机器人复杂地形行走实验 #

### 地形调整 ###
1. 前期准备：Isaac Gym Preview 4 Release 目前仅支持 Linux 操作系统（本地运行安装双系统，请勿使用 WSL，也可以使用**云服务器 + VNC**）
2. 地形调整位置位于 `legged_gym/legged_gym/envs/g1/g1_config.py` 的 `terrain` 类中，将 `mesh_type = "plane"` 改为 `mesh_type = "trimesh"` 即可调整至复杂地形
3. 复杂地形分为 `smooth slope`、`rough slope`、`stairs up`、`stairs down`、`discrete`、`stepping stones` 和 `gaps`，`terrain_dict` 以字典形式指定了每种地形的比例（**请注意，`terrain_proportions` 的顺序是固定的，且全部地形比例的加起来小于或等于1**），可以通过将某种地形的比例直接设置为1，然后在可视化界面中观察该类地形的特征