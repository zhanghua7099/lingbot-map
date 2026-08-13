"""Intel RealSense webcam sender for demo_live.py --remote_source mode.

Same protocol and role as remote_camera_client.py, but captures from an
Intel RealSense camera (D4xx/L5xx series) via pyrealsense2 instead of a
generic UVC webcam via cv2.VideoCapture — useful if that's the camera you
have on the capture machine, e.g. a D435i on a robot or a rig.

Only the color stream is sent to the server: the model does its own
monocular depth estimation and has no input for external depth, so
RealSense's own depth stream is used here purely for a local sanity-check
preview (colorized side-by-side with the color feed), not sent over the wire.

Setup on the capture machine:
    pip install pyrealsense2 opencv-python
    python remote_camera_client_realsense.py --server_host <server IP>

If pipeline.start() raises about an unsupported stream profile, run
`rs-enumerate-devices` (ships with librealsense) to see what your specific
camera supports and pass matching --width/--height/--fps.
"""

import argparse
import socket
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs

RECONNECT_INTERVAL = 2.0


def main():
    parser = argparse.ArgumentParser(description="Send RealSense color frames to a remote demo_live.py server")
    parser.add_argument("--server_host", type=str, required=True,
                         help="IP/hostname of the machine running demo_live.py --remote_source "
                              "(printed by that script on startup).")
    parser.add_argument("--server_port", type=int, default=9999)
    parser.add_argument("--serial_number", type=str, default=None,
                         help="RealSense device serial to use if more than one is plugged in "
                              "(see `rs-enumerate-devices`). Default: first device found.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera_fps", type=int, default=30, help="Requested RealSense stream FPS.")
    parser.add_argument("--fps", type=float, default=15.0,
                         help="Cap the send rate to the server (0 = every captured frame). "
                              "The server only keeps the latest frame anyway, so this mainly "
                              "just saves bandwidth/CPU.")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--flip", action="store_true", help="Mirror the color image horizontally.")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                         help="Show a local window with the color feed + connection status "
                              "(on by default; --no-preview for a headless client).")
    parser.add_argument("--show_depth", action=argparse.BooleanOptionalAction, default=True,
                         help="Also show RealSense's own colorized depth next to the color preview "
                              "(sanity check only — depth is never sent to the server). Ignored if "
                              "--no-preview.")
    args = parser.parse_args()

    config = rs.config()
    if args.serial_number:
        config.enable_device(args.serial_number)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.camera_fps)
    if args.show_depth and args.preview:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.camera_fps)

    pipeline = rs.pipeline()
    try:
        pipeline.start(config)
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not start RealSense stream at {args.width}x{args.height}@{args.camera_fps}: {e}\n"
            f"Run `rs-enumerate-devices` to see supported profiles for your camera and pass "
            f"matching --width/--height/--camera_fps."
        )
    align = rs.align(rs.stream.color) if (args.show_depth and args.preview) else None
    colorizer = rs.colorizer() if (args.show_depth and args.preview) else None

    min_interval = (1.0 / args.fps) if args.fps > 0 else 0.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    window_name = "LingBot-Map RealSense preview  (q or Esc to quit)"

    print(f"RealSense stream started ({args.width}x{args.height}@{args.camera_fps}). "
          f"Target server: {args.server_host}:{args.server_port}")

    sock = None
    connected = False
    last_connect_attempt = 0.0
    last_send = 0.0
    sent_count = 0

    def close_sock():
        nonlocal sock, connected
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        sock = None
        connected = False

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"RealSense frame wait timed out/failed ({e}); stopping.")
                break

            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())  # BGR8, HxWx3 — same layout cv2 expects
            if args.flip:
                frame = cv2.flip(frame, 1)

            now = time.time()

            if not connected and now - last_connect_attempt >= RECONNECT_INTERVAL:
                last_connect_attempt = now
                print(f"Connecting to {args.server_host}:{args.server_port}...")
                try:
                    new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    new_sock.settimeout(3.0)
                    new_sock.connect((args.server_host, args.server_port))
                    new_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    new_sock.settimeout(5.0)
                    sock, connected = new_sock, True
                    print("Connected. Streaming frames...")
                except OSError as e:
                    print(f"Connect failed ({e}); retrying every {RECONNECT_INTERVAL:.0f}s...")

            if connected and (now - last_send >= min_interval):
                ok, buf = cv2.imencode(".jpg", frame, encode_params)
                if ok:
                    data = buf.tobytes()
                    try:
                        sock.sendall(struct.pack(">I", len(data)) + data)
                        last_send = now
                        sent_count += 1
                    except OSError as e:
                        print(f"Send failed ({e}); will reconnect.")
                        close_sock()

            if args.preview:
                status = f"connected, sent {sent_count}" if connected else "reconnecting..."
                color_status = (80, 220, 80) if connected else (60, 60, 230)
                display = frame.copy()
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)

                if args.show_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                        if args.flip:
                            depth_vis = cv2.flip(depth_vis, 1)
                        cv2.putText(depth_vis, "depth (not sent — preview only)", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        if depth_vis.shape[:2] != display.shape[:2]:
                            depth_vis = cv2.resize(depth_vis, (display.shape[1], display.shape[0]))
                        display = np.hstack([display, depth_vis])

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or Esc
                    print("Quit requested from preview window.")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted, stopping.")
    finally:
        close_sock()
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
