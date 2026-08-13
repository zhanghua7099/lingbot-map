# 环境

```
conda activate lingbot-map
```

模型权重在 `checkpoints/lingbot-map.pt`（从 HuggingFace `robbyant/lingbot-map` 下的，用的是均衡版 `lingbot-map.pt`，论文/benchmark/offline demo 都用这个）。

---

# 离线批量跑数据集（demo.py，官方自带）

跑仓库自带的示例场景（`example/courthouse`、`example/university`、`example/loop`），一次性把整段处理完再打开网页查看，功能最全（时间轴拖动、GLB 导出、存视频、截图、天空遮罩）：

```
python demo.py --model_path checkpoints/lingbot-map.pt --image_folder example/courthouse --mask_sky
```

然后打开 `http://localhost:8080`。

---

# 摄像头实时看（demo_live.py，自己写的）

跟 demo.py 的区别：不是先处理完整段再显示，而是每算完一帧立刻推到网页，边走边看点云长出来。

```
python demo_live.py --model_path checkpoints/lingbot-map.pt
```

然后打开 `http://localhost:8080`。

网页 GUI 里能看到：摄像头实时画面预览、Model FPS、已处理帧数、置信度阈值/点大小滑块（实时生效）、Stop 按钮。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--camera_id` | 0 | 用哪个摄像头（`ls /dev/video*` 查） |
| `--camera_width` / `--camera_height` | 1280 / 720 | 请求分辨率，摄像头不支持会自动协商成支持的 |
| `--flip` | 关 | 水平镜像 |
| `--port` | 8080 | viser 网页端口 |
| `--conf_threshold` | 1.5 | 点云置信度过滤阈值 |
| `--downsample_factor` | 4 | 点云降采样，越大越稀疏、网页越流畅 |
| `--max_display_frames` | 150 | 网页里最多同时挂多少帧的点云，超过自动丢最老的 |
| `--camera_frustum_every` | 5 | 每 N 帧画一个相机 frustum，别太密 |
| `--keyframe_interval` | 1 | 只有每 N 帧的 KV 才进缓存，长时间挂机跑建议调大 |
| `--kv_cache_sliding_window` | 64 | KV cache 淘汰窗口，控制显存 |
| `--max_frame_num` | 4096 | 3D RoPE 位置编码上限，到了会自动停止（比 demo.py 默认 1024 调高了，给长时间直播留余量） |
| `--use_sdpa` | 关 | 不用 FlashInfer，改用 PyTorch 原生 attention |

---

# 录制与回看

## 1. 录制 `--record_dir`

跑的时候把每一帧原始摄像头画面存到指定文件夹（编号 `000000.jpg, 000001.jpg...`），跟 GUI 里的帧计数对齐。

```
python demo_live.py --model_path checkpoints/lingbot-map.pt --record_dir ./my_session
```

## 2. 回看有两种方式，跑完录制后会自动打印这两条命令

**重新"live"一遍**（点云重新一帧帧长出来，同一套实时 viewer）：

```
python demo_live.py --model_path checkpoints/lingbot-map.pt --replay_dir ./my_session
```

用录制好的帧顶替摄像头，走一模一样的流式推理 + viser 增量渲染代码路径（不是另外糊的播放器，就是同一份代码，只是数据源换成文件夹）。加 `--replay_loop` 可以循环播放，`--replay_fps 10` 能控制回放节奏（默认是模型多快就放多快）。

**完整版离线查看器**（更推荐，功能更全）：

```
python demo.py --model_path checkpoints/lingbot-map.pt --image_folder ./my_session
```

这个复用的是仓库里本来就有、已经验证过的 PointCloudViewer——自带时间轴拖动条、GLB 导出、存视频（ffmpeg）、截图、天空遮罩等一堆功能，比 live 版能做的丰富得多，所以录像的"精细回看"直接导到这条路上最划算，没有重复造轮子。

`--record_dir` 可以跟 `--remote_source` 一起用（见下面），录的是服务器收到的帧，跟本地摄像头录制是同一套逻辑。

---

# 远程发图片（服务器没摄像头 / 摄像头在笔记本上）

服务器（有 GPU 的机器）：

```
python demo_live.py --model_path checkpoints/lingbot-map.pt --remote_source
```

启动后会打印服务器的局域网 IP 和示例命令。笔记本（有摄像头的机器，不需要装 conda/torch，只要 `pip install opencv-python`）：

```
python remote_camera_client.py --server_host 172.24.36.34 --server_port 9999
```

同时开录制的话：

```
python demo_live.py --model_path checkpoints/lingbot-map.pt --remote_source --record_dir ./my_session
```

## 要点

- 协议：4 字节长度前缀 + JPEG，原始 TCP，没有额外依赖。
- 服务器**只保留最新一帧**，处理跟不上时主动丢旧帧而不是排队，避免延迟越攒越大。
- 断线不会导致服务器崩溃，会打印"等待重连"，笔记本重新连上就接着跑。
- 服务器不在同一局域网、只有 SSH 能连的话，用端口转发代替暴露端口（摄像头流的 9999 端口和 viser 网页的 8080 端口都可以这么转发）：
  ```
  ssh -L 9999:localhost:9999 -L 8080:localhost:8080 user@server
  ```
  然后笔记本上 `--server_host localhost`，浏览器开 `http://localhost:8080`。
- `remote_camera_client.py` 默认会弹一个本地预览窗口（摄像头画面 + 连接状态，按 `q`/`Esc` 退出），方便拿着笔记本走的时候不用一直盯服务器那边的网页。无头环境加 `--no-preview`。

## RealSense 版本

有 RealSense 相机（D4xx/L5xx 系列）的话用这个替代 `remote_camera_client.py`，协议和服务器端完全一样：

```
pip install pyrealsense2 opencv-python
python remote_camera_client_realsense.py --server_host 172.24.36.34
```

只发彩色图给服务器（模型是单目深度预测，没有外部深度输入接口）；本地预览窗口默认会把彩色画面和 RealSense 自己算的深度（伪彩色）拼一起显示，纯粹是本地确认相机状态用，`--no-show_depth` 关掉。多个设备接着用 `--serial_number` 指定（`rs-enumerate-devices` 查序列号）。

---

# 踩过的坑

## 部署环境时

conda 环境创建 + PyTorch/FlashInfer/依赖安装本身很顺利，没有真正卡住的问题，就一个小地方值得记一下：

- **`pip install torch...` 和 `pip install -e .` 都会冒出一段看着像报错的 resolver 警告**：
  ```
  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...
  generate-parameter-library-py 0.7.3 requires pyyaml, which is not installed.
  generate-parameter-library-py 0.7.3 requires typeguard, which is not installed.
  ```
  `generate-parameter-library-py` 跟 lingbot-map 毫无关系（像是 ROS2 那边的包），在全新 conda 环境里冒出来有点莫名其妙，但不影响安装结果，忽略就行——两次都是装完之后 import/跑起来完全正常。

## 跑起来之后（demo.py / demo_live.py 通用）

- **摄像头打开卡死不报错**（demo_live.py 特有）：OpenCV 默认后端在某些 Linux 驱动组合下 `cap.read()` 会无限挂起。`demo_live.py` 已经改成强制 V4L2 后端 + 5 秒读取超时，正常情况不会再卡死；真遇到超时会提示检查 `fuser /dev/video0`（是否被占用）、`v4l2-ctl --list-formats-ext -d /dev/video0`（支持的分辨率）。
- **"Failed to load pretrained weights: [Errno 2] No such file or directory: ''"**：无害，忽略。这是构建 DINOv2 backbone 时尝试加载一份独立的预训练权重，路径默认空字符串所以必然找不到，跟 `checkpoints/lingbot-map.pt` 的加载完全无关，后者是紧接着单独加载的（看后面有没有 "Checkpoint loaded." 和 Missing/Unexpected keys 计数）。
- **GPU 显存不够 / OOM**：`nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv` 看是不是有之前忘记关的 demo/demo_live 进程占着显存，`kill` 掉再跑。
