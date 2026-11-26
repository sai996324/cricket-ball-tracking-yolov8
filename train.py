"""Fine-tune YOLOv8 on your own cricket-ball dataset (YOLO format).

Run:  python train.py --data cricket_ball.yaml --epochs 50
Best weights are saved to runs/detect/train/weights/best.pt
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dataset yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=960)
    args = p.parse_args()
    YOLO(args.model).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
