# Real-Time Ball Tracking System Using Computer Vision

Detects and tracks a cricket ball in video using **Python, OpenCV and YOLOv8**, and draws the ball's trajectory in real time.

## Features
- Ball detection with YOLOv8 on every frame
- Tracking across frames (ignores implausible jumps, resets trajectory when the ball is lost)
- Real-time trajectory visualization and FPS counter
- Works on video files or a webcam; saves an annotated output video
- `train.py` to fine-tune on your own cricket-ball dataset
- `evaluate.py` to measure real precision, recall and mAP on a labelled test set

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Pretrained COCO model (class "sports ball")
python track_ball.py --source match.mp4

# Webcam
python track_ball.py --source 0

# Your own fine-tuned weights (ball class id 0 in your dataset)
python track_ball.py --source match.mp4 --weights runs/detect/train/weights/best.pt --ball-class 0
```
Press `q` to quit the preview window. Useful options: `--conf`, `--imgsz`, `--trail`, `--max-gap`, `--max-jump`, `--no-show`.

## Improving accuracy on cricket balls
A cricket ball is small and fast, so the pretrained COCO model may miss it. To improve:
1. Collect frames from cricket videos and label the ball (Roboflow or LabelImg, YOLO format).
2. Train: `python train.py --data cricket_ball.yaml --epochs 50`
3. Measure: `python evaluate.py --weights runs/detect/train/weights/best.pt --data cricket_ball.yaml`
4. Report the precision/recall/mAP that `evaluate.py` prints for your own test set.

## How it works
1. Read a frame with OpenCV.
2. Run YOLOv8 and keep only the ball class.
3. If several candidates exist, pick the one closest to the previous position.
4. Store the centre points in a fixed-length queue and draw them as a fading trail.
5. Write the annotated frame to the output video.
