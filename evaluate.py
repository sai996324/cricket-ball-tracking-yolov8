"""Measure real detection accuracy of your weights on a LABELLED test set.

Needs a YOLO-format dataset yaml (images + labels). Prints precision, recall and mAP.
Run:  python evaluate.py --weights best.pt --data cricket_ball.yaml
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--data", required=True, help="dataset yaml with a val/test split")
    p.add_argument("--imgsz", type=int, default=960)
    args = p.parse_args()

    metrics = YOLO(args.weights).val(data=args.data, imgsz=args.imgsz)
    print(f"Precision : {metrics.box.mp:.3f}")
    print(f"Recall    : {metrics.box.mr:.3f}")
    print(f"mAP@0.50  : {metrics.box.map50:.3f}")
    print(f"mAP@0.50-0.95 : {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
