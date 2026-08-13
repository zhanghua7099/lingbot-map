"""LingBot-MAP live demo: real-time 3D reconstruction from a webcam feed.

Unlike demo.py (which reads a whole image folder / video file, runs
inference_streaming() over the full sequence, then builds a static viser
scene), this script drives the model's causal streaming path one live camera
frame at a time and pushes each frame's point cloud into an already-running
viser server as soon as it's computed — so the browser view grows while you
walk around.

Usage:
    python demo_live.py --model_path /path/to/lingbot-map.pt

Then open http://localhost:8080 in a browser. Ctrl+C or the "Stop" button in
the GUI ends the session.
"""

import argparse
import contextlib
import os
import platform
import socket
import struct
import sys
import threading
import time
from collections import deque

if "--compile" not in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import viser
import viser.transforms as tf

from demo import load_model, postprocess, prepare_for_visualization
from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map


def preprocess_frame(frame_bgr, image_size=518, patch_size=14):
    """In-memory equivalent of load_and_preprocess_images' 'crop' mode.

    Mirrors lingbot_map/utils/load_fn.py:load_and_preprocess_images (mode="crop")
    but operates directly on a captured BGR frame instead of a file path, since
    webcam frames never touch disk.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = frame_rgb.shape[:2]
    new_width = image_size
    new_height = max(patch_size, round(h0 * (new_width / w0) / patch_size) * patch_size)
    resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        resized = resized[start_y:start_y + image_size]
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float() / 255.0
    return tensor  # (3, H, W)


def open_camera(camera_id, width, height):
    """Open a webcam, preferring V4L2 on Linux.

    cv2's default backend auto-probes (often GStreamer first); on some
    Linux/driver combos that probe hangs forever on the first cap.read()
    instead of failing, which is indistinguishable from a genuinely slow
    camera. Forcing V4L2 avoids that class of hang.
    """
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_id}. Check that a webcam is connected "
            f"and accessible (`ls /dev/video*`), and that no other process is using it "
            f"(`fuser /dev/video{camera_id}` on Linux)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera {camera_id} opened via {'V4L2' if backend == cv2.CAP_V4L2 else 'default backend'} "
          f"(requested {width}x{height}, negotiated {actual_w:.0f}x{actual_h:.0f})")
    return cap


def read_with_timeout(cap, timeout=5.0):
    """cap.read() with a wall-clock timeout instead of an indefinite hang.

    The read runs in a daemon thread; if it doesn't return within `timeout`
    seconds we give up and report (False, None) rather than blocking the
    whole script forever. The daemon thread itself may stay stuck in the
    underlying blocking C call, but daemon threads are killed automatically
    when the process exits.
    """
    result = {}

    def _read():
        result["ret"], result["frame"] = cap.read()

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, None
    return result.get("ret", False), result.get("frame")


def get_local_ip():
    """Best-effort LAN IP for the "point remote_camera_client.py at me" hint.

    Doesn't actually send anything — a UDP "connect" just asks the OS to pick
    the outbound route/interface, which is enough to read back its address.
    Falls back to loopback if there's no route (e.g. offline).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class RemoteFrameSource:
    """Receives JPEG frames pushed over TCP from remote_camera_client.py,
    instead of reading a local camera.

    Protocol: each frame is [4-byte big-endian length][JPEG bytes]. Only the
    most recently received frame is kept — if the model can't keep up with
    the incoming rate, older undelivered frames are dropped rather than
    queued, so the view stays live instead of drifting further and further
    behind real time.
    """

    def __init__(self, host="0.0.0.0", port=9999):
        self.port = port
        self._cond = threading.Condition()
        self._latest_frame = None
        self._latest_seq = 0
        self._consumed_seq = 0
        self._stop = False
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((host, port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(1.0)
        print(
            f"Waiting for remote camera client to connect on port {port}. On the laptop with the "
            f"webcam, run:\n"
            f"  python remote_camera_client.py --server_host {get_local_ip()} --server_port {port}"
        )
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self):
        while not self._stop:
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            print(f"Remote camera client connected from {addr[0]}:{addr[1]}")
            self._handle_client(conn)
            if not self._stop:
                print("Remote camera client disconnected; waiting for reconnect...")

    def _handle_client(self, conn):
        conn.settimeout(2.0)
        try:
            while not self._stop:
                header = self._recv_exact(conn, 4)
                if header is None:
                    return
                length = struct.unpack(">I", header)[0]
                if not (0 < length <= 50_000_000):
                    return
                payload = self._recv_exact(conn, length)
                if payload is None:
                    return
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                with self._cond:
                    self._latest_frame = frame
                    self._latest_seq += 1
                    self._cond.notify_all()
        except (ConnectionResetError, socket.timeout, OSError):
            return
        finally:
            conn.close()

    def _recv_exact(self, conn, n):
        buf = bytearray()
        while len(buf) < n:
            if self._stop:
                return None
            try:
                chunk = conn.recv(n - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True

    def get(self, *_args, **_kwargs):
        return 0.0

    def read(self, timeout=None):
        """Blocks (up to `timeout`) for a frame newer than the last one handed out.

        Unlike the local-camera / replay sources, this doesn't need the
        generic read_with_timeout() thread-spawning wrapper — Condition.wait_for
        already supports a native timeout, so waiting a long time for a client
        to connect doesn't leak one blocked thread per attempt.
        """
        with self._cond:
            got = self._cond.wait_for(lambda: self._latest_seq != self._consumed_seq, timeout=timeout)
            if not got:
                return False, None
            self._consumed_seq = self._latest_seq
            return True, self._latest_frame.copy()

    def release(self):
        self._stop = True
        try:
            self._server_sock.close()
        except OSError:
            pass


class FolderReplayCapture:
    """Replays a directory of recorded frames as if it were a live camera.

    Pairs with --record_dir: point --replay_dir at a folder a previous
    --record_dir session wrote, and this drives the exact same streaming +
    viser pipeline against that saved footage instead of live hardware —
    useful for re-watching a session grow again, or for testing without a
    camera at all.
    """

    def __init__(self, folder, fps=0.0, loop=False):
        import glob
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(os.path.join(folder, ext)))
        self.paths = sorted(paths)
        if not self.paths:
            raise RuntimeError(f"No images found in --replay_dir {folder!r}")
        self.idx = 0
        self.loop = loop
        self.frame_interval = (1.0 / fps) if fps and fps > 0 else 0.0
        self._last_t = None
        print(
            f"Replaying {len(self.paths)} recorded frames from {folder} "
            + (f"at {fps:.1f} FPS" if fps else "as fast as the model can process")
            + (", looping" if loop else "")
        )

    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True

    def get(self, *_args, **_kwargs):
        return 0.0

    def read(self):
        if self.idx >= len(self.paths):
            if not self.loop:
                return False, None
            self.idx = 0
        if self.frame_interval > 0 and self._last_t is not None:
            remaining = self.frame_interval - (time.time() - self._last_t)
            if remaining > 0:
                time.sleep(remaining)
        frame = cv2.imread(self.paths[self.idx])
        self.idx += 1
        self._last_t = time.time()
        return (frame is not None), frame

    def release(self):
        pass


def slice_batch(pred_dict, si):
    """Slice a batched (B, S, ...) prediction dict down to a single S=si frame."""
    out = {}
    for k, v in pred_dict.items():
        if torch.is_tensor(v) and v.dim() >= 2:
            out[k] = v[:, si:si + 1]
        else:
            out[k] = v
    return out


class LiveSceneBuilder:
    """Incrementally pushes per-frame point clouds / camera frustums into viser.

    Reuses demo.py's postprocess()/prepare_for_visualization() and the same
    depth-based unprojection PointCloudViewer uses by default (use_point_map=False),
    so a single frame here produces numerically the same point cloud that the
    full offline demo.py would produce for that frame — just displayed the
    moment it's ready instead of after the whole sequence finishes.
    """

    def __init__(self, server, conf_slider, psize_slider, max_display_frames,
                 downsample_factor, frustum_every):
        self.server = server
        self.conf_slider = conf_slider
        self.psize_slider = psize_slider
        self.max_display_frames = max_display_frames
        self.downsample_factor = downsample_factor
        self.frustum_every = frustum_every
        self.pc_handles = deque()
        self.cam_handles = deque()
        self.num_points_total = 0

    def _evict_old(self):
        while len(self.pc_handles) > self.max_display_frames:
            self.pc_handles.popleft().remove()
        max_cams = max(1, self.max_display_frames // max(1, self.frustum_every))
        while len(self.cam_handles) > max_cams:
            self.cam_handles.popleft().remove()

    def add_frame(self, raw_pred, images_batch, frame_idx):
        """raw_pred: model.forward() output dict, batch shape (1, 1, ...).
        images_batch: the same (1, 1, 3, H, W) tensor fed into forward().
        """
        predictions, images_cpu = postprocess(dict(raw_pred), images_batch)
        vis = prepare_for_visualization(predictions, images_cpu)

        depth_map = vis["depth"]          # (1, H, W, 1)
        depth_conf = vis["depth_conf"]    # (1, H, W)
        extrinsic = vis["extrinsic"]      # (1, 3, 4)
        intrinsic = vis["intrinsic"]      # (1, 3, 3)
        images = vis["images"]            # (1, 3, H, W)

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (1,H,W,3)
        colors = images.transpose(0, 2, 3, 1)  # (1, H, W, 3)

        pc = world_points[0].reshape(-1, 3)
        color = colors[0].reshape(-1, 3)
        conf = depth_conf[0].reshape(-1)

        valid = np.isfinite(pc).all(axis=1)
        pc, color, conf = pc[valid], color[valid], conf[valid]
        mask = conf > self.conf_slider.value
        pc, color = pc[mask], color[mask]

        ds = max(1, int(self.downsample_factor))
        if ds > 1 and len(pc) > 0:
            idx = np.arange(0, len(pc), ds)
            pc, color = pc[idx], color[idx]

        self.num_points_total += len(pc)

        handle = self.server.scene.add_point_cloud(
            name=f"/frames/{frame_idx}/pred_pts",
            points=pc,
            colors=color,
            point_size=self.psize_slider.value,
        )
        self.pc_handles.append(handle)

        if frame_idx % self.frustum_every == 0:
            cam_to_world = closed_form_inverse_se3(extrinsic)[0]  # (4,4)
            R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
            focal, pp = float(intrinsic[0, 0, 0]), (float(intrinsic[0, 0, 2]), float(intrinsic[0, 1, 2]))
            q = tf.SO3.from_matrix(R).wxyz
            fov = 2 * np.arctan(pp[0] / focal) if focal > 0 else 1.0
            aspect = pp[0] / pp[1] if pp[1] > 0 else 1.0
            frustum = self.server.scene.add_camera_frustum(
                name=f"/frames/{frame_idx}/camera",
                fov=fov, aspect=aspect, wxyz=q, position=t,
                scale=0.03, color=(80, 200, 255),
            )
            self.cam_handles.append(frustum)

        self._evict_old()
        return len(pc)


def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP live webcam demo")

    # Model (mirrors demo.py's load_model() requirements)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=4096,
                         help="3D-RoPE position range. Live sessions run indefinitely, so this "
                              "is set higher than demo.py's default (1024) to give more runway "
                              "before positions run out.")
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64,
                         help="KV cache eviction window — bounds GPU memory during indefinite streaming.")
    parser.add_argument("--keyframe_interval", type=int, default=1,
                         help="Only every N-th frame's KV is kept in cache. Raise this for long "
                              "sessions to slow down effective cache growth.")
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                         help="Use SDPA backend instead of FlashInfer.")

    # Camera
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--flip", action="store_true", help="Mirror the camera horizontally.")

    # Recording / replay
    parser.add_argument("--record_dir", type=str, default=None,
                         help="Save each raw captured frame here as the session streams. Afterwards "
                              "you can either replay this exact session with --replay_dir, or get the "
                              "full offline viewer (scrub slider, GLB/video export, screenshots) via: "
                              "python demo.py --model_path <ckpt> --image_folder <record_dir>")
    parser.add_argument("--replay_dir", type=str, default=None,
                         help="Replay a folder of previously-recorded frames (e.g. from --record_dir) "
                              "instead of opening a live camera.")
    parser.add_argument("--replay_fps", type=float, default=0.0,
                         help="Playback pacing for --replay_dir. 0 = as fast as the model can process.")
    parser.add_argument("--replay_loop", action="store_true",
                         help="Loop --replay_dir indefinitely instead of stopping at the last frame.")

    # Remote camera (run remote_camera_client.py on the machine with the webcam)
    parser.add_argument("--remote_source", action="store_true",
                         help="Receive frames pushed over the network from remote_camera_client.py "
                              "instead of reading a local camera. Use this when the GPU server has "
                              "no webcam of its own — run remote_camera_client.py on the laptop that does.")
    parser.add_argument("--remote_host", type=str, default="0.0.0.0",
                         help="Bind address for the remote frame receiver.")
    parser.add_argument("--remote_port", type=int, default=9999,
                         help="TCP port the remote frame receiver listens on.")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--max_display_frames", type=int, default=150,
                         help="Older frames' point clouds are dropped once this many are on screen, "
                              "to keep the browser responsive during a long-running session.")
    parser.add_argument("--camera_frustum_every", type=int, default=5,
                         help="Only draw a camera frustum every N frames.")

    args = parser.parse_args()
    args.mode = "streaming"  # load_model() branches on this

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = load_model(args, device)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
        print("WARNING: no CUDA device found; running on CPU will be far too slow for live use.")
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)
    amp_ctx = (lambda: torch.amp.autocast("cuda", dtype=dtype)) if torch.cuda.is_available() \
        else (lambda: contextlib.nullcontext())

    if sum([bool(args.remote_source), bool(args.replay_dir)]) > 1:
        raise ValueError("Pass at most one of --remote_source or --replay_dir.")

    remote_mode = bool(args.remote_source)
    if remote_mode:
        cap = RemoteFrameSource(host=args.remote_host, port=args.remote_port)
    elif args.replay_dir:
        cap = FolderReplayCapture(args.replay_dir, fps=args.replay_fps, loop=args.replay_loop)
    else:
        cap = open_camera(args.camera_id, args.camera_width, args.camera_height)

    def read_frame():
        # RemoteFrameSource.read() has a native timeout (Condition.wait_for); reusing
        # read_with_timeout()'s thread-spawning wrapper for it would leak one blocked
        # daemon thread per call while waiting (possibly minutes) for a client to connect.
        if remote_mode:
            return cap.read(timeout=2.0)
        return read_with_timeout(cap, timeout=5.0)

    last_wait_print = [0.0]

    def wait_notice(msg):
        now = time.time()
        if now - last_wait_print[0] > 3.0:
            print(msg)
            last_wait_print[0] = now

    if args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)
        print(f"Recording raw frames to {args.record_dir}/")

    def maybe_record(frame, idx):
        if args.record_dir:
            cv2.imwrite(os.path.join(args.record_dir, f"{idx:06d}.jpg"), frame)

    print(f"Starting viser server on port {args.port}")
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    status_text = server.gui.add_text("Status", initial_value="Warming up...")
    fps_text = server.gui.add_text("Model FPS", initial_value="0.0")
    frames_text = server.gui.add_text("Frames processed", initial_value="0")
    points_text = server.gui.add_text("Points (last frame)", initial_value="0")
    current_frame_gui = server.gui.add_image(
        np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8),
        label="Live Camera",
    )
    with server.gui.add_folder("Point Cloud"):
        conf_slider = server.gui.add_slider(
            "Confidence Threshold", min=1.0, max=5.0, step=0.01, initial_value=args.conf_threshold
        )
        psize_slider = server.gui.add_slider(
            "Point Size", min=0.00001, max=0.05, step=0.00001, initial_value=args.point_size
        )
    stop_button = server.gui.add_button("Stop / Quit")

    stopping = {"flag": False}

    @stop_button.on_click
    def _(_):
        stopping["flag"] = True

    scene = LiveSceneBuilder(
        server, conf_slider, psize_slider,
        args.max_display_frames, args.downsample_factor, args.camera_frustum_every,
    )

    model.clean_kv_cache()

    print(f"Collecting {args.num_scale_frames} scale frames to bootstrap scale estimation...")
    scale_frames_buf = []
    while len(scale_frames_buf) < args.num_scale_frames:
        ret, frame = read_frame()
        if not ret:
            if remote_mode:
                wait_notice("Waiting for remote camera client to connect and start sending frames...")
                continue
            raise RuntimeError(
                "Failed to read from camera during warmup (no frame within 5s). This usually "
                "means either: (1) another process already has the camera open — check with "
                f"`fuser /dev/video{args.camera_id}`; (2) the requested resolution "
                f"({args.camera_width}x{args.camera_height}) isn't supported — try "
                "`v4l2-ctl --list-formats-ext -d /dev/video0` and pass a supported "
                "--camera_width/--camera_height; or (3) --camera_id is wrong — check `ls /dev/video*`."
            )
        if args.flip:
            frame = cv2.flip(frame, 1)
        current_frame_gui.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        maybe_record(frame, len(scale_frames_buf))
        scale_frames_buf.append(preprocess_frame(frame, args.image_size, args.patch_size))

    # Keep images fp32; autocast (below) handles casting for ops that need it,
    # matching demo.py's inference_streaming() call — casting the input tensor
    # itself would leave postprocess()'s images.numpy() holding an unsupported
    # bf16/fp16 array.
    scale_images = torch.stack(scale_frames_buf).unsqueeze(0).to(device)  # (1, N, 3, H, W)

    with torch.no_grad(), amp_ctx():
        scale_output = model.forward(
            scale_images,
            num_frame_for_scale=args.num_scale_frames,
            num_frame_per_block=args.num_scale_frames,
            causal_inference=True,
        )

    frame_idx = 0
    for si in range(args.num_scale_frames):
        scene.add_frame(slice_batch(scale_output, si), scale_images[:, si:si + 1], frame_idx)
        frame_idx += 1
    del scale_output
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    status_text.value = "Streaming"
    print(f"Scale phase done. Streaming live at http://localhost:{args.port}")
    print("Move the camera around; the point cloud grows in the browser as you go.")
    print("Press the 'Stop / Quit' button in the GUI, or Ctrl+C here, to end.")

    fps_ema = None
    warned_position_limit = False

    try:
        while not stopping["flag"]:
            ret, frame = read_frame()
            if not ret:
                if remote_mode:
                    wait_notice("No frame from remote client (disconnected or lagging behind)...")
                    continue
                print("Camera read failed (or timed out after 5s), stopping.")
                break
            if args.flip:
                frame = cv2.flip(frame, 1)
            current_frame_gui.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            maybe_record(frame, frame_idx)

            if frame_idx >= args.max_frame_num:
                status_text.value = f"Stopped: reached max_frame_num={args.max_frame_num}"
                print(
                    f"Reached --max_frame_num={args.max_frame_num} (3D-RoPE position range); "
                    "stopping to avoid undefined position embeddings. Restart the script (or "
                    "raise --max_frame_num) for a longer session."
                )
                break
            if not warned_position_limit and frame_idx >= int(0.9 * args.max_frame_num):
                warned_position_limit = True
                print(f"WARNING: approaching --max_frame_num={args.max_frame_num}; "
                      "reconstruction will stop soon.")

            img_tensor = preprocess_frame(frame, args.image_size, args.patch_size)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, H, W)

            is_keyframe = (args.keyframe_interval <= 1) or (frame_idx % args.keyframe_interval == 0)
            if not is_keyframe:
                model._set_skip_append(True)

            t0 = time.time()
            with torch.no_grad(), amp_ctx():
                frame_output = model.forward(
                    img_tensor,
                    num_frame_for_scale=args.num_scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = max(time.time() - t0, 1e-6)

            if not is_keyframe:
                model._set_skip_append(False)

            fps_ema = (1.0 / dt) if fps_ema is None else (0.9 * fps_ema + 0.1 * (1.0 / dt))
            fps_text.value = f"{fps_ema:.1f}"

            n_pts = scene.add_frame(frame_output, img_tensor, frame_idx)
            points_text.value = f"{n_pts:,}"
            frame_idx += 1
            frames_text.value = str(frame_idx)
    except KeyboardInterrupt:
        print("\nInterrupted, stopping.")
    finally:
        cap.release()
        model.clean_kv_cache()
        status_text.value = "Stopped"
        print(f"Stopped after {frame_idx} frames.")
        if args.record_dir:
            print(f"Recorded {frame_idx} frames to {args.record_dir}/. To review:")
            print(f"  Replay the live-growing view again:")
            print(f"    python demo_live.py --model_path {args.model_path} --replay_dir {args.record_dir}")
            print(f"  Or get the full offline viewer (scrub slider, GLB/video export, screenshots):")
            print(f"    python demo.py --model_path {args.model_path} --image_folder {args.record_dir}")


if __name__ == "__main__":
    main()
