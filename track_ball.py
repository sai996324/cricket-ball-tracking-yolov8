"""Real-time ball tracking with YOLOv8 + OpenCV.

Detects a ball in each frame, follows it across frames and draws its trajectory.

Examples:
  python track_ball.py --source match.mp4
  python track_ball.py --source 0                      # webcam
  python track_ball.py --source match.mp4 --weights runs/detect/train/weights/best.pt --ball-class 0
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

COCO_SPORTS_BALL = 32  # class id of "sports ball" in the COCO dataset


def parse_args():
    p = argparse.ArgumentParser(description="Real-time ball tracking (YOLOv8 + OpenCV)")
    p.add_argument("--source", default="0", help="video path or webcam index (default 0)")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights (.pt)")
    p.add_argument("--ball-class", type=int, default=COCO_SPORTS_BALL,
                   help="class id of the ball (32 for COCO; use your own id for custom weights)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=960, help="inference image size (larger helps small balls)")
    p.add_argument("--trail", type=int, default=40, help="number of past points in the trajectory")
    p.add_argument("--max-gap", type=int, default=8,
                   help="frames the ball may be missing before the trajectory is reset")
    p.add_argument("--max-jump", type=float, default=0.25,
                   help="max allowed jump between detections, as a fraction of frame width")
    p.add_argument("--output", default="output.mp4", help="output video path ('' to disable saving)")
    p.add_argument("--no-show", action="store_true", help="do not open a preview window")
    return p.parse_args()


def best_ball(result, ball_class, last_point, max_jump_px):
    """Pick the most plausible ball detection in a frame.

    Prefers detections close to the previous position; ignores implausible jumps.
    Returns (cx, cy, conf, (x1, y1, x2, y2)) or None.
    """
    candidates = []
    for box in result.boxes:
        if int(box.cls[0]) != ball_class:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        conf = float(box.conf[0])
        candidates.append((cx, cy, conf, (x1, y1, x2, y2)))
    if not candidates:
        return None
    if last_point is None:
        return max(candidates, key=lambda c: c[2])
    lx, ly = last_point
    near = [c for c in candidates if np.hypot(c[0] - lx, c[1] - ly) <= max_jump_px]
    return max(near, key=lambda c: c[2]) if near else None


def draw_trajectory(frame, trail):
    pts = [p for p in trail if p is not None]
    for i in range(1, len(pts)):
        thickness = max(1, int(1 + 4 * i / len(pts)))  # newer segments are thicker
        cv2.line(frame, pts[i - 1], pts[i], (0, 200, 255), thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    model = YOLO(args.weights)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    trail = deque(maxlen=args.trail)
    last_point, missing = None, 0
    frames = detected = 0
    start = time.time()
    max_jump_px = args.max_jump * width

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        ball = best_ball(result, args.ball_class, last_point, max_jump_px)

        if ball:
            cx, cy, conf, (x1, y1, x2, y2) = ball
            last_point, missing = (cx, cy), 0
            detected += 1
            trail.append((int(cx), int(cy)))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ball {conf:.2f}", (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            missing += 1
            if missing > args.max_gap:  # ball lost: start a fresh trajectory
                trail.clear()
                last_point = None

        draw_trajectory(frame, trail)
        elapsed = time.time() - start
        cv2.putText(frame, f"FPS: {frames / elapsed:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("Ball tracking (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if frames:
        print(f"Frames processed : {frames}")
        print(f"Frames with ball : {detected} ({detected / frames:.1%})")
        print(f"Average speed    : {frames / (time.time() - start):.1f} FPS")
        if args.output:
            print(f"Saved video      : {args.output}")
        print("Note: 'Frames with ball' is a detection rate, not accuracy. "
              "For accuracy, label a test set and use evaluate.py.")


if __name__ == "__main__":
    main()
