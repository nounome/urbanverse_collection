# 场景验证状态与相机配置



## 1. 哪些场景跑到了哪一步

需要区分“机器狗走完全程”和“三路 RGB、深度、位姿同步数据全部验收”。**目前没有场景完成正式 1920×1536、10 FPS 全路线数据验收。** 正式配置采用原始标定尺寸，不再要求 720p。

以下为原服务器 RTX 3090＋Isaac Sim 4.5 的已有记录，供参考。场景参数入口是 [scenes.json](scenes.json)。

| 场景 | 汽车/行人/二轮车 | 已有验证范围 |
|---|---:|---|
| Scene01 | 1/12/6 | 无相机短测（3 秒） |
| Scene02 | 1/12/6 | 无相机短测（3 秒） |
| Scene03 | 7/40/1 | 全路线运行（约 100 m） |
| Scene04 | 2/12/3 | 无相机短测（3 秒） |
| Scene05 | 1/12/6 | 无相机短测（3 秒） |
| Scene06 | 0/40/16 | 三相机短测（10 秒） |
| Scene07 | 0/40/9 | 全路线运行（约 100 m） |
| Scene08 | 4/40/20 | 三相机短测（10 秒） |
| Scene09 | 21/40/20 | 三相机短测（10 秒） |
| Scene10 | 3/40/2 | 三相机短测（10 秒） |
| Scene11 | 0/12/4 | 无相机短测（3 秒） |
| Scene12 | 0/40/13 | 三相机短测（10 秒） |




## 2. 当前三相机装在哪里

三台相机在 Fabric 初始化前以 Isaac Lab `CameraCfg` 挂到 `{ENV_REGEX_NS}/Robot/base/<camera_id>`，跟随机身完整姿态。坐标相对 Go2 `base`：X 向前、Y 向左、Z 向上，单位米。

| 相机 ID | 类型 | 相对位置 XYZ（m） | 实际朝向 |
|---|---|---|---|
| `camera_tm_pinhole` | 中央针孔＋OpenCV radtan 映射 | `[0.235, 0, 0.22]` | 大致前向，使用标定 ROS optical 外参 |
| `camera_tm1_fisheye` | 左鱼眼，原生等距鱼眼＋OpenCV fisheye 映射 | `[-0.29, 0.12, 0.18]` | 水平向左外侧 |
| `camera_tm2_fisheye` | 右鱼眼，原生等距鱼眼＋OpenCV fisheye 映射 | `[-0.29, -0.12, 0.18]` | 水平向右外侧 |

因此左右相机是机身偏后、向两侧看，不是简单的“前向针孔左右各放一台”。

### 修改位置与朝向

标定源在 [sensor_suite.py 的 CAMERA_CALIBRATIONS](tools/simulation_qualification/isaac45_urbanverse_sensor_suite.py)：

- `position_vehicle_xyz_m`：安装位置。例如前相机的 X 从 `0.235` 改成 `0.30`，就是向机身前方再移动 6.5 cm。
- `rpy_vehicle_camera_rad`：角度单位是弧度。中央相机按 ROS 光学坐标外参解释，旋转次序 `Rz(yaw) * Ry(pitch) * Rx(roll)`；光学坐标为 X 向右、Y 向下、Z 向前。
- 左右相机有专门兼容逻辑：`add_navigation_camera_sensors()` 实际使用 `side_yaw = -rpy[2]`，并采用 `convention="world"` 的水平朝向。**当前左右 roll/pitch 不会直接控制安装俯仰**；要调整侧相机俯仰，须同时修改该函数的四元数计算，不能只改 RPY 数组。

安装实现见 [go2_route_capture.py 的 add_navigation_camera_sensors](tools/urbanverse/dynamic_agents/integration/go2_route_capture.py)。改挂载父节点时，要同步修改 [three_camera_writer.py](tools/urbanverse/dynamic_agents/rendering/three_camera_writer.py) 的外参计算：当前默认按 `base` 位姿计算相机世界位姿及重叠标签。

### 修改分辨率与帧率

当前统一入口在 [collect.py 的 command](collect.py)，不是独立相机配置 JSON：

| 入口字段 | 控制内容 |
|---|---|
| `THREE_CAMERA_WIDTH` / `THREE_CAMERA_HEIGHT` | 三台相机统一输出宽高 |
| `OVERVIEW_FPS` | 当前联合入口的采样节拍，也控制三相机采样 |
| `GO2_THREE_CAMERA` | 开启/关闭整组三相机 |

`collect.py` 会设置这些环境变量，因此仅在 Shell 外层设置同名变量不会覆盖其 profile 赋值；应修改 `command()` 的 profile 参数。当前 `preview` 为 480×384、5 FPS；`formal` 为 **1920×1536、10 FPS**。

正式配置与原始标定 `image_size` 完全一致，保留 K/D、视场及现有安装外参，不做裁剪、拉伸或鱼眼焦距放大。预览按 1/4 比例缩放 K，D 不变。保留严格宽高比检查；若后续降低成本，可采用同为 5:4 的尺寸，例如 960×768，而不是直接改为 16:9。

原始尺寸每台相机的像素数是 480×384 预览的 16 倍，帧率又从 5 提升至 10 FPS，因此每秒输出像素量是预览的 32 倍；这不是显存或耗时的线性预测。原始尺寸的 RTX 负载、同步与完整路线仍需实测。

### 修改相机类型和数量

当前还不是任意相机组合的一键配置接口。修改时必须同时考虑以下位置：

1. `CAMERA_CALIBRATIONS`：列表条数、唯一 ID、`native_projection_type`、K/D、位置和姿态。
2. `add_navigation_camera_sensors()`：创建 `PinholeCameraCfg` 或 `FisheyeCameraCfg`，并正确设置安装姿态。
3. [camera_calibration.py 的 build_inverse_map](tools/urbanverse/dynamic_agents/rendering/camera_calibration.py)：现在按 `role == center_forward_pinhole` 选择针孔映射，其余默认鱼眼。**只把左相机的 projection 改成 pinhole 不够**；需将映射分支改为按明确的相机模型选择，并使用对应 K/D。
4. `ThreeCameraWriter` 已按传入 definitions 循环写数据，但 [capture_contract.py 的 CAMERAS](tools/urbanverse/dynamic_agents/admission/capture_contract.py) 固定要求当前三个 ID。删减、新增或改名都要同步调整后续视频/训练读取程序。

例如只保留中央相机进行 RGB＋深度采集，应筛选该相机定义，并同步调整契约要求；不要只打开已有的单前向视频开关，它不是同一套 RGB＋深度数据交付接口。相机数量变化后仍需在 reset 前注册传感器，不改成 reset 后临时创建 RTX 相机。
