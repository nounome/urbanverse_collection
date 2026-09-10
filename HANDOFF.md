# 场景验证状态、部署与相机配置

适用于 `urbanverse_collection`，核对日期：2026-09-10，代码基线：`39417d8`。安装、下载和采集命令见 [README](README.md)；本文用于接手维护和调整采集参数。

## 1. 哪些场景跑到了哪一步

需要区分“机器狗走完全程”和“三路 RGB、深度、位姿同步数据全部验收”。**目前没有场景完成正式 1280×720、10 FPS 全路线数据验收。**

以下为原服务器 RTX 3090＋Isaac Sim 4.5 的已有记录，不是新服务器或 Docker 验收。场景参数入口是 [scenes.json](scenes.json)。

| 场景 | 汽车/行人/二轮车 | 已有验证范围 | 下一项待验证 |
|---|---:|---|---|
| Scene01 | 1/12/6 | 路线地面预检、3 秒无相机联合运动 | 三相机短测、全路线采集 |
| Scene02 | 1/12/6 | 新 50 m 路线地面预检、3 秒无相机联合运动 | 实走完整 50 m、三相机采集 |
| Scene03 | 7/40/1 | 约 100 m 三相机联合长测，机器狗完成路线 | 新 writer 全程同步验证、正式参数验证 |
| Scene04 | 2/12/3 | 调整起点后的路线地面预检、3 秒无相机联合运动 | 三相机短测、全路线采集 |
| Scene05 | 1/12/6 | 路线地面预检、3 秒无相机联合运动 | 三相机短测、全路线采集 |
| Scene06 | 0/40/16 | 10 秒低成本三相机联合短测 | 全路线及正式参数采集 |
| Scene07 | 0/40/9 | 约 100 m 三相机联合长测，机器狗完成路线 | 新 writer 全程同步验证、正式参数验证 |
| Scene08 | 4/40/20 | 10 秒低成本三相机联合短测 | 全路线及正式参数采集 |
| Scene09 | 21/40/20 | 当前自动化配置完成 10 秒三相机短测 | 当前配置全路线及正式参数采集 |
| Scene10 | 3/40/2 | 10 秒低成本三相机联合短测 | 全路线及正式参数采集 |
| Scene11 | 0/12/4 | 25 m 路线地面预检、3 秒无相机联合运动 | 实走完整路线、三相机采集 |
| Scene12 | 0/40/13 | 10 秒低成本三相机联合短测 | 全路线及正式参数采集 |

补充说明：

- Scene03/07 的旧长测出现部分相机缓存同步问题，所以不能把“走完约 100 m”写成“完整同步训练数据通过”。现有 writer 已调整采样刷新逻辑，但完整新链路仍需长测。
- Scene09 还有旧配置下的正式第三人称/单针孔视频，不能替代当前 21/40/20 主体配置的三相机全程验收。
- 新仓库曾逐一运行 12 场景的 3 秒无相机测试：全部有正常行走记录；11 个场景自动验收退出 0。Scene03 退出 2：3 秒内仅生成 1/7 辆车，未满足“所有汽车至少生成一次”的生命周期检查，并非仿真崩溃。短测验收条件仍需按此区别处理，不必为了凑检查改变汽车配置。
- 无相机入口的渲染开关冲突已修复并保留。`39417d8` 清理后完成配置、入口和异地目录检查，未重跑 GPU 联合仿真。
- `three_camera_passed=true` 表示已有相机阶段记录；12 场景的 `synchronized_long_capture_passed` 均为 `false`。不将该标志改成通过，直到对应数据实际验收。

接手顺序：环境短测 → Scene01/02/04/05/11 三相机短测 → 修正下文的 720p 标定适配 → 单场景全程同步采集 → 批量新路线。动态主体配置固定，随机种子只改变 Go2 路线。

## 2. Docker 是否合适

可以容器化，适合多台服务器重复部署。镜像固定 Isaac Sim、Isaac Lab、Python、PyTorch 和系统依赖，接手者不必重复手工配置 Python 环境。宿主机仍需安装兼容的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit；容器不能修复显卡与 Isaac Sim 版本本身的不兼容。

NVIDIA 提供 Isaac Sim 4.5 官方容器 `nvcr.io/nvidia/isaac-sim:4.5.0`，并推荐用于远程无界面部署，见 [4.5 官方容器安装说明](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_container.html)。不要使用 `latest` 代替本项目的 4.5 基线。

本仓库目前没有 Dockerfile 或已验收镜像。建议实现时：

1. 优先在镜像内复现现有 Python 3.10＋pip 安装布局，沿用 `setup_environment.sh` 的锁定版本，保持 `repos/isaac45_probe/.venv` 和 `repos/IsaacLab` 相对布局。容器内仓库位置可以固定，宿主机挂载路径可以不同。
2. 若直接基于 NVIDIA 官方镜像，需适配其 Python/扩展目录与本项目 runner、Kit 模板，不要在镜像里混装两套 Isaac Sim 后假定兼容。
3. 镜像包含程序和依赖；场景/人物/权重、转换缓存、输出与 Kit 缓存使用卷挂载。只读源资产与可写缓存分开，避免把大数据打进镜像或被挂载覆盖掉镜像内虚拟环境。
4. 保留 `collect.py` / `batch_collect.py` 命令接口。核对容器内 GPU 编号、`nvidia-smi`、PCI 与设备节点检查的可见范围；不能直接假定容器编号等于宿主机编号。
5. 验证一次无相机短测和三相机短测，再验证正式配置及全路线。镜像发布与第三方资产再分发按各自许可处理，不打包个人 GitHub 凭据。

Docker 减少的是安装差异，不会取消相机同步、路线可行性或画面质量检查。本节是容器化方案，不是已经提供可运行镜像。

## 3. 当前三相机装在哪里

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

`collect.py` 会设置这些环境变量，因此仅在 Shell 外层设置同名变量不会覆盖其 profile 赋值；应修改 `command()` 的 profile 参数。当前 `preview` 为 480×384、5 FPS；`formal` 的目标赋值为 1280×720、10 FPS。

**720p 的待修复问题：**标定 `image_size` 为 1920×1536（5:4），当前 `strict_document_calibration=True` 只接受原尺寸或同比例缩放。480×384 可以通过；1280×720（16:9）会被 `add_navigation_camera_sensors()` 拒绝，尚不能直接用该 formal 配置采集。不要只关闭检查后沿用不匹配的标定。

要生成 16:9，应明确采用裁剪还是重新设计视场，并同步修改 `image_size`、K、原生投影和像素重映射。裁剪示例：原标定中心裁成 1920×1080（左上角偏移 `x0=0,y0=228`），再按 `s=2/3` 缩放至 1280×720：

```text
fx' = s*fx       fy' = s*fy
cx' = s*(cx-x0)  cy' = s*(cy-y0)
```

纯裁剪/等比例缩放时 D 不变；RGB、深度和有效区域必须采用同一几何变换。此方案需实现与预览验证，不代表当前代码已支持该输出链路。

### 修改相机类型和数量

当前还不是任意相机组合的一键配置接口。修改时必须同时考虑以下位置：

1. `CAMERA_CALIBRATIONS`：列表条数、唯一 ID、`native_projection_type`、K/D、位置和姿态。
2. `add_navigation_camera_sensors()`：创建 `PinholeCameraCfg` 或 `FisheyeCameraCfg`，并正确设置安装姿态。
3. [camera_calibration.py 的 build_inverse_map](tools/urbanverse/dynamic_agents/rendering/camera_calibration.py)：现在按 `role == center_forward_pinhole` 选择针孔映射，其余默认鱼眼。**只把左相机的 projection 改成 pinhole 不够**；需将映射分支改为按明确的相机模型选择，并使用对应 K/D。
4. `ThreeCameraWriter` 已按传入 definitions 循环写数据，但 [capture_contract.py 的 CAMERAS](tools/urbanverse/dynamic_agents/admission/capture_contract.py) 固定要求当前三个 ID。删减、新增或改名都要同步调整契约检查和后续视频/训练读取程序。

例如只保留中央相机进行 RGB＋深度采集，应筛选该相机定义，并同步调整契约要求；不要只打开已有的单前向视频开关，它不是同一套 RGB＋深度数据交付接口。相机数量变化后仍需在 reset 前注册传感器，不改成 reset 后临时创建 RTX 相机。

## 4. 鱼眼黑边如何减少

当前管线是“原生鱼眼渲染 → OpenCV 鱼眼畸变映射”。超过有效视角或映射到源图外的像素会被明确置为黑色，深度为正无穷，不是有效观测。原生鱼眼最大 FOV 为 190°，逆映射半角上限为 95°；两处定义需要一致。

优先方案是增大左右相机 K 中的 `fx`、`fy`，保持主点 `cx,cy` 和 D 不变，先试原值的 **1.15 倍**，再比较 **1.30 倍**。图像内容会放大，黑边变少，但画面所覆盖的角度变小；这属于更换相机内参，不是无代价提升画质。修改标定源后，当前代码会据此生成原生焦距和像素映射，不能只改最后写出的标定 JSON。

当前左鱼眼在 480×384 下的 CPU 逆映射有效像素比例为：

| fx/fy 倍率 | 映射有效比例 |
|---:|---:|
| 1.00 | 76.64% |
| 1.15 | 91.64% |
| 1.30 | 98.55% |

这些数值只验证映射数学范围，**不是 RTX 图片实测的无黑边比例**，也不包括机身遮挡或场景本身的暗区。调整后应分别观察左右画面，检查身体遮挡、侧向覆盖和深度。

另一方案是裁剪无效边缘并同步更新 K、图像尺寸和深度；不能只裁展示视频而声称训练数据也已改变。单纯同比例提高分辨率不会明显改变黑边占比；也不应通过把无效深度填零或拉伸 RGB 掩盖无效区域。

可从 `metadata/three_camera_calibration.json` 的各相机 `projection_mapping.valid_output_ratio` 查看映射覆盖率。原始深度定义为 `distance_to_camera`（沿相机射线的距离，米），不是光轴方向 Z 深度；无效值为 `+inf`。

## 5. 相机修改后的检查

先用低分辨率短测核对三台相机的朝向、机身遮挡、鱼眼有效区域和深度；再抽取连续帧，核对相机随机器狗运动且 RGB、深度、位姿时间对应。保留 `three_camera_calibration.json`、`frame_index.jsonl` 和逐相机重叠标记，与该批图像一同交付。

自动文件契约通过只说明文件和帧记录结构符合要求，不替代像素同步和深度准确性检查。更换镜头、安装位置、数量或裁剪方式后，旧相机验收不能直接沿用。
