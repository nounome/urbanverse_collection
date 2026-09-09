# UrbanVerse Go2 三相机采集

独立交付版：固定 12 个场景的地图、清理覆盖层、汽车/行人/二轮车配置，只更换 Go2 的随机路线。Go2 使用 robot_lab 冻结策略，以 A*＋平滑路线真实迈步；其他动态主体忽略 Go2，彼此避让保持原配置。原始场景不修改。

**当前是整理后的交付候选，不是新服务器上已验收的一键采集版本。** 原服务器为 RTX 3090 + Isaac Sim 4.5：Scene03/07 完成约 100 m（旧长测存在部分相机同步问题）；Scene06/08/09/10/12 完成 10 秒低成本三相机；Scene01/02/04/05/11 完成 3 秒无相机。Scene02 已换成检查通过的 50 m 规划路线，尚未实际走完全程。正式 720p/10 FPS 仍待实测。

## 1. 安装

Linux x86_64、可用的 NVIDIA GPU/驱动、Python 3.10、git、lspci、taskset、ffmpeg。脚本不安装驱动或系统 CUDA，不使用 sudo。历史驱动为 580.173.02；不同硬件仍需先做短测，不保证 RTX 5090 或其他 Isaac Sim 版本兼容。

```bash
git clone <本仓库地址> urbanverse_collection
cd urbanverse_collection
bash setup_environment.sh
source repos/isaac45_probe/.venv/bin/activate
```

安装会下载较大的 Isaac Sim/扩展缓存和 PyTorch 包。Isaac Sim 固定 4.5.0.0、Isaac Lab v2.1.1、PyTorch 2.7.0/cu128；应用包见 requirements.txt。安装脚本本轮仅静态检查，未在空白服务器实装。若 pip check 失败，保留输出，不直接跳过或宣称部署通过。

目录移动后执行 `python configure.py`，重定位自有代码/配置中继承的绝对路径；不改任何下载的源 USD。不要把旧虚拟环境一起搬过去。

## 2. 准备资产

```bash
python download_assets.py all
python prepare_vehicle_cache.py --gpu 0
python collect.py doctor --assets
```

也可分别指定 `craftbench`、`people`、`go2`、`policy`。默认不做文件哈希校验，检查文件存在、下载大小、解压结构及加载格式。代码中部分哈希字段仍是历史来源记录或运行元数据，不是用户必须提供的校验步骤。

下载目标：

- `data/urbanverse_craftbench/{raw,extracted}/`：12 个场景，包括动态车辆/二轮车所需 GLB。
- `data/isaacsim_assets_4_5/Isaac/People/`：人物与动画。
- `data/isaacsim_assets_4_5/Isaac/IsaacLab/Robots/Unitree/Go2/`：机器人。
- `data/locomotion_policies/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/`：robot_lab 配置和权重。

可从合法已有下载复制以上目录，无需重新下载。源场景可能有外部引用，doctor 不能证明全部纹理完整；首次相机测试必须检查画面。下载入口本轮未执行远程下载或验证远端可用性。

汽车外观还需一次 GLB→USD 转换，`prepare_vehicle_cache.py` 复用已标定的转换方式，生成 `data/vehicle_cache/converted/`；它会使用 GPU，先检查健康/显存并监测超时。不依赖原服务器的 `outputs/.../current` 软链接。首次缓存生成也尚未在新服务器验证；二轮车仍由联合入口自动转换。

## 3. 首次新机器验证

```bash
python collect.py list
python collect.py doctor
python collect.py run --scene scene02 --profile smoke --gpu 0 --dry-run
python collect.py run --scene scene02 --profile smoke --gpu 0
python collect.py run --scene scene02 --profile preview --gpu 0
```

不指定 seed 时使用保存的参考路线。`smoke` 是 3 秒无相机，`preview` 是 10 秒三相机 480×384/5 FPS。短测不等于长路线或正式采集通过。五个仅无相机通过的场景仍须先完成 preview。GPU 健康不一致、余量不足或超时会停止，不会关闭别人任务。

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

相机为机身挂载中央针孔、左右鱼眼，读取 RGB、float32 深度、Go2 位姿/轨迹、标定和仿真时间戳，保留逐相机动态重叠标签及其他主体轨迹。没有用后处理移动机器人画面或重置跳段。进入车辆内部的区间保留并标记，不能直接当普通导航训练帧。

种子只改变 Go2 路线，动态主体使用 scenes.json 引用的固定配置和固定种子。当前生成参数保留各场景成功来源；有些紧凑场景仅配置 25 m 档位，并非全场景保证 100 m。新的路线仍可能因实际地面/碰撞检查被拒绝，不自动放宽安全半径或清理新对象。地图检查通过也不证明实际能走完整条路线。

## 5. 批量与续跑

```bash
# 先检查任务列表；不会启动仿真
python batch_collect.py --scenes scene03 scene07 --count 5 --profile preview \
  --gpu 0 --journal outputs/batch_preview.json --dry-run
# 正式队列；务必先完成新机与正式分辨率验收
python batch_collect.py --scenes scene03 scene07 --count 5 --profile formal \
  --gpu 0 --journal outputs/batch_formal.json
```

不指定 scenes 时包含 12 个场景。每个任务最多尝试 3 个 Go2 种子，顺序运行；失败保留日志，继续下一个任务。重复同一命令读取 journal，跳过已成功项；修改任务配置需新 journal。资源不足也会失败，不会无限等待或无穷重试。断电恢复不能保证找回刚完成但尚未写入 journal 的结果，需核查对应 run。批处理入口仅经模拟测试，尚未执行真实批量 RTX 采集。

## 6. 输出与验收

运行输出在 `outputs/`，实际主机/GPU/版本写入每次 environment.json。继承的子目录名 `rtx3090_isaac45` 仅是兼容路径，换服务器后不能当硬件证据。

- `run_log.txt`、`metadata/summary.json`、`metadata/process_exit.json`：运行与退出结果。
- `metadata/frame_index.jsonl`：三路 RGB/深度实际文件位置、对应帧与姿态时间。
- `metadata/trajectory.jsonl`：真实运动轨迹；相机标定及重叠标签也在 metadata 下。
- `outputs/collection_batches/*/result.json`：本入口的监控和文件契约结果。

检查器验证文件完整、帧记录等，不证明像素级同步或深度精度。旧 Scene03/07 的部分缓存滞后问题由新 writer 专项修复过，但新 writer 全路线和正式高分辨率还没有完整验收。交付前应抽看三路画面、深度和特殊区间，不把 JSON 中 passed 当作全部质量要求通过。

开发自检：`PYTHONPATH=tools python -m pytest -q tests tools/urbanverse/dynamic_agents/admission/tests`。

## 许可与交付范围

本仓库不包含原项目研究文档、Git 历史、历史输出、虚拟环境或大资产。`export_manifest.json` 记录导出来源版本，非哈希校验清单。保留依赖代码是为避免更改已测执行链，部分文件仍含未启用的旧功能；支持入口以本 README 为准。

CraftBench 来源：Oatmealliu/UrbanVerse-CraftBench，按其 CC BY-NC 4.0 条款；NVIDIA 资产、Isaac Sim、Isaac Lab 和 rl_sar/策略分别按原许可使用，下载意味着使用者需自行接受相应条款。未把第三方资产或项目代码重新授予 MIT 等许可；对外公开前请由项目所有者确认发布范围。
