"""Webcam sender for demo_live.py --remote_source mode.

Run this on the machine that actually has the webcam (e.g. your laptop) and
point it at the GPU server running:

    python demo_live.py --model_path /path/to/lingbot-map.pt --remote_source

Only dependency is opencv-python (`pip install opencv-python`) — no torch,
no CUDA, no conda env needed on this machine.

Protocol: connects to the server over a plain TCP socket and streams
[4-byte big-endian length][JPEG bytes] frames, one per captured image. The
server only ever keeps the most recent frame, so this client doesn't need to
throttle precisely — --fps just caps bandwidth/CPU use.

By default this also opens a local preview window (what demo_live.py shows
you is the reconstruction, not the raw feed, and it may be a browser tab on
a different machine than the one you're walking around with) — press 'q' or
Esc in that window, or Ctrl+C, to stop. Use --no-preview on a headless client.

If the server isn't directly reachable (e.g. it's behind a firewall and you
only have SSH access), forward the port instead of exposing it:
    ssh -L 9999:localhost:9999 user@server
and then run this client with --server_host localhost.
"""

import argparse
import socket
import struct
import time

import cv2

RECONNECT_INTERVAL = 2.0


def main():
    parser = argparse.ArgumentParser(description="Send local webcam frames to a remote demo_live.py server")
    parser.add_argument("--server_host", type=str, required=True,
                         help="IP/hostname of the machine running demo_live.py --remote_source "
                              "(printed by that script on startup).")
    parser.add_argument("--server_port", type=int, default=9999)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=15.0,
                         help="Cap the send rate (0 = as fast as the camera delivers frames). "
                              "The server only keeps the latest frame anyway, so this mainly "
                              "just saves bandwidth/CPU.")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--flip", action="store_true", help="Mirror the camera horizontally.")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                         help="Show a local window with the raw feed + connection status "
                              "(on by default; --no-preview for a headless client).")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open local camera {args.camera_id}.")

    min_interval = (1.0 / args.fps) if args.fps > 0 else 0.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    window_name = "LingBot-Map camera preview  (q or Esc to quit)"

    print(f"Local camera opened. Target server: {args.server_host}:{args.server_port}")

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
            ret, frame = cap.read()
            if not ret:
                print("Local camera read failed; stopping.")
                break
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
                color = (80, 220, 80) if connected else (60, 60, 230)
                display = frame.copy()
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or Esc
                    print("Quit requested from preview window.")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted, stopping.")
    finally:
        close_sock()
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
