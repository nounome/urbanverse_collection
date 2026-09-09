# UrbanVerse Go2 三相机采集

在 UrbanVerse 的 12 个场景中采集 Go2 的三相机 RGB、深度、位姿与轨迹数据。各场景的地图、障碍清理覆盖层以及汽车、行人、二轮车配置固定，通过更换随机种子生成不同的 Go2 路线。

Go2 使用 robot_lab 冻结运动策略，沿 A*＋平滑路线真实迈步。其他动态主体忽略 Go2，彼此按场景配置避让；与 Go2 重叠的区间自动记录。原始场景资产不修改。

使用顺序：安装环境 → 准备资产 → 短测 → 生成 Go2 路线 → 采集 → 检查数据。

## 1. 安装

需要 Linux x86_64、可用的 NVIDIA GPU/驱动、Python 3.10、git、lspci、taskset、ffmpeg。脚本不安装驱动或系统 CUDA，不使用 sudo。

```bash
git clone <本仓库地址> urbanverse_collection
cd urbanverse_collection
bash setup_environment.sh
source repos/isaac45_probe/.venv/bin/activate
```

安装会下载较大的 Isaac Sim/扩展缓存和 PyTorch 包。版本固定为 Isaac Sim 4.5.0.0、Isaac Lab v2.1.1、PyTorch 2.7.0/cu128；其他依赖见 requirements.txt。安装完成后须通过 `pip check`。

目录移动后执行 `python configure.py`，重定位自有代码/配置中继承的绝对路径；不改任何下载的源 USD。不要把旧虚拟环境一起搬过去。

## 2. 准备资产

```bash
python download_assets.py all
python prepare_vehicle_cache.py --gpu 0
python collect.py doctor --assets
```

也可分别指定 `craftbench`、`people`、`go2`、`policy`。默认不做文件哈希校验，检查文件存在、下载大小、解压结构及加载格式。

下载目标：

- `data/urbanverse_craftbench/{raw,extracted}/`：12 个场景，包括动态车辆/二轮车所需 GLB。
- `data/isaacsim_assets_4_5/Isaac/People/`：人物与动画。
- `data/isaacsim_assets_4_5/Isaac/IsaacLab/Robots/Unitree/Go2/`：机器人。
- `data/locomotion_policies/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/`：robot_lab 配置和权重。

可从合法已有下载复制以上目录，无需重新下载。

汽车外观需进行一次 GLB→USD 转换，`prepare_vehicle_cache.py` 生成 `data/vehicle_cache/converted/`。转换使用 GPU，脚本会检查健康状态、显存余量并监测超时。二轮车由联合运行入口自动转换。

## 3. 检查配置与短测

```bash
python collect.py list
python collect.py doctor
python collect.py run --scene scene02 --profile smoke --gpu 0 --dry-run
python collect.py run --scene scene02 --profile smoke --gpu 0
python collect.py run --scene scene02 --profile preview --gpu 0
```

不指定 seed 时使用保存的参考路线。运行模式如下：

| 模式 | 时长 | 相机配置 |
|---|---|---|
| `smoke` | 3 秒 | 无相机 |
| `preview` | 10 秒 | 三相机，480×384、5 FPS |
| `formal` | 到达终点结束，最多 600 秒 | 三相机，1280×720、10 FPS |

时长指仿真时间，不包含场景加载、转换与预热。GPU 健康不一致、余量不足或超时会停止当前任务，不会关闭其他用户的任务。

`--wall-timeout` 控制墙钟秒数；默认 3600。共享 GPU 运行估计增量预算：smoke 5600 MiB、相机 12000 MiB，另留 1536 MiB，运行中监测余量。预算是工程默认值，不是所有场景的显存峰值证明。

## 4. 随机路线和正式采集

```bash
# 只生成路线，不启动仿真
python collect.py plan --scene scene02 --seed 1001
# 生成新路线后运行，路线会接受完整实时源地面检查
python collect.py run --scene scene02 --seed 1001 --profile preview --gpu 0
# 正式参数：1280×720 / 10 FPS，到达终点停止，最多仿真 600 秒
python collect.py run --scene scene02 --seed 1001 --profile formal --gpu 0
```

相机为机身挂载中央针孔、左右鱼眼，采集 RGB、float32 深度、Go2 位姿/轨迹、标定和仿真时间戳，同时记录逐相机动态重叠标签及其他主体轨迹。进入车辆内部的区间保留并标记，使用数据时应单独筛选或处理。

种子只改变 Go2 路线，动态主体使用 scenes.json 引用的固定配置和固定种子。路线长度按各场景参数选择，有些紧凑场景仅配置 25 m 档位。新路线需通过运行时地面与碰撞检查；不满足条件时拒绝运行，不自动放宽安全半径或清理新对象。

## 5. 批量与续跑

```bash
# 先检查任务列表；不会启动仿真
python batch_collect.py --scenes scene03 scene07 --count 5 --profile preview \
  --gpu 0 --journal outputs/batch_preview.json --dry-run
# 正式队列；务必先完成新机与正式分辨率验收
python batch_collect.py --scenes scene03 scene07 --count 5 --profile formal \
  --gpu 0 --journal outputs/batch_formal.json
```

不指定 scenes 时包含 12 个场景。每个任务最多尝试 3 个 Go2 种子，顺序运行；失败保留日志，继续下一个任务。重复同一命令读取 journal，跳过已成功项；修改任务配置需新 journal。资源不足也会失败，不会无限等待或无穷重试。断电后应核查对应 run，避免重复采集已完成但尚未写入 journal 的任务。

## 6. 输出与验收

运行输出在 `outputs/`，实际主机/GPU/版本写入每次 environment.json。继承的子目录名 `rtx3090_isaac45` 仅是兼容路径，换服务器后不能当硬件证据。

- `run_log.txt`、`metadata/summary.json`、`metadata/process_exit.json`：运行与退出结果。
- `metadata/frame_index.jsonl`：三路 RGB/深度实际文件位置、对应帧与姿态时间。
- `metadata/trajectory.jsonl`：真实运动轨迹；相机标定及重叠标签也在 metadata 下。
- `outputs/collection_batches/*/result.json`：本入口的监控和文件契约结果。

检查 `result.json`、`summary.json` 和进程退出码，确认运行结果。自动检查涵盖文件完整性、帧记录及运动状态；还应抽看三路画面、深度和重叠区间，确认像素同步与数据质量。正式采集应确认路线到达终点，不能仅依据短测通过。

## 7. 新服务器注意事项与已知问题

- 环境基线为 RTX 3090、驱动 580.173.02、Isaac Sim 4.5。其他硬件或版本需要单独验证，不直接套用兼容结论。
- 在新服务器重新创建虚拟环境，不搬运旧环境。若安装、下载或 `pip check` 报错，应先处理依赖或网络问题，再运行采集。完整的从零部署仍需在目标机器验证。
- `doctor --assets` 会检查整个车型库所需的 15 款汽车缓存；复制部分缓存可能导致检查失败，应运行转换脚本补齐。源 USD 的外部材质与纹理引用还需在相机预览中检查。
- 每个场景先运行 `smoke`，再运行 `preview`。正式 720p/10 FPS 的全路线采集仍需验证负载和三路数据同步；3 秒无相机通过不能代替此项。批量采集前先完成单条正式参数测试，再小批量验证队列与续跑。
- Scene03 的汽车逐辆生成，3 秒短测可能未生成全部 7 辆。当前验收要求每辆车至少生成一次，因此即使 Go2 跑满 3 秒，也可能返回失败。查看 `spawn_counts` 与 `stop_reason` 区分验收条件不足和运行异常；该短测条件尚未调整，不应直接忽略失败码。

## 许可

场景资产、人物、机器人及策略权重通过上述准备步骤获取，不包含在代码仓库中。

CraftBench 来源：Oatmealliu/UrbanVerse-CraftBench，按其 CC BY-NC 4.0 条款；NVIDIA 资产、Isaac Sim、Isaac Lab 和 rl_sar/策略分别按原许可使用，下载意味着使用者需自行接受相应条款。未把第三方资产或项目代码重新授予 MIT 等许可；对外公开前请由项目所有者确认发布范围。
