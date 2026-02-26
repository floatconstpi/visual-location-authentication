from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SplitImage:
    split: str
    path: Path


@dataclass(frozen=True)
class DatasetLayout:
    data_yaml: Path
    dataset_base: Path
    split_dirs: dict[str, Path]


def resolve_dataset_base(data_yaml: Path, yaml_path_value: str | None) -> Path:
    if yaml_path_value:
        base = Path(yaml_path_value)
        if not base.is_absolute():
            base = (data_yaml.parent / base).resolve()
        return base.resolve()
    return data_yaml.parent.resolve()


def load_layout(data_yaml: Path) -> DatasetLayout:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset yaml: {data_yaml}")

    dataset_base = resolve_dataset_base(data_yaml, payload.get("path"))
    split_dirs: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        raw = payload.get(split_name)
        if raw is None:
            continue
        split_path = Path(raw)
        if not split_path.is_absolute():
            split_path = (dataset_base / split_path).resolve()
        split_dirs[split_name] = split_path

    if not split_dirs:
        raise ValueError(f"No train/val/test entries in {data_yaml}")

    return DatasetLayout(data_yaml=data_yaml, dataset_base=dataset_base, split_dirs=split_dirs)


def collect_images(split_dirs: dict[str, Path]) -> list[SplitImage]:
    images: list[SplitImage] = []
    for split, directory in split_dirs.items():
        if not directory.exists():
            raise FileNotFoundError(f"Split directory does not exist: {directory}")
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                images.append(SplitImage(split=split, path=p))
    return sorted(images, key=lambda x: (x.split, str(x.path)))


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)) + (max(0.0, bx2 - bx1) * max(0.0, by2 - by1)) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def load_gt_boxes(label_path: Path, image_w: int, image_h: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    out: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx = float(parts[1]) * image_w
            cy = float(parts[2]) * image_h
            w = float(parts[3]) * image_w
            h = float(parts[4]) * image_h
        except ValueError:
            continue
        x = cx - (w / 2.0)
        y = cy - (h / 2.0)
        out.append((x, y, w, h))
    return out


def max_iou(pred_box: tuple[float, float, float, float], gt_boxes: Iterable[tuple[float, float, float, float]]) -> float:
    best = 0.0
    for gt in gt_boxes:
        best = max(best, iou_xywh(pred_box, gt))
    return best


def infer_dataset(
    model_path: Path,
    data_yaml: Path,
    output_csv: Path,
    imgsz: int,
    conf: float,
    nms_iou: float,
    device: str,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required. Install with: pip install ultralytics") from exc

    model_path = model_path.resolve()
    data_yaml = data_yaml.resolve()
    output_csv = output_csv.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data yaml: {data_yaml}")

    layout = load_layout(data_yaml)
    images = collect_images(layout.split_dirs)
    if not images:
        raise RuntimeError("No images found in dataset splits.")

    labels_root = layout.dataset_base / "labels"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Data:  {data_yaml}")
    print(f"Out:   {output_csv}")
    print(f"Total images: {len(images)}")

    model = YOLO(str(model_path))
    rows: list[dict[str, str]] = []
    failures = 0

    for idx, item in enumerate(images, start=1):
        split_root = layout.split_dirs[item.split]
        rel_path = item.path.relative_to(split_root)
        label_path = labels_root / item.split / f"{item.path.stem}.txt"
        try:
            result = model.predict(
                source=str(item.path),
                imgsz=imgsz,
                conf=conf,
                iou=nms_iou,
                device=device,
                verbose=False,
            )[0]
            image_h, image_w = result.orig_shape
            gt_boxes = load_gt_boxes(label_path, image_w=image_w, image_h=image_h)
            if result.boxes is None or len(result.boxes) == 0:
                row = {
                    "filename": item.path.stem,
                    "bbox_x": "",
                    "bbox_y": "",
                    "bbox_w": "",
                    "bbox_h": "",
                    "iou": "0.0",
                }
            else:
                confs = result.boxes.conf.tolist()
                best_idx = max(range(len(confs)), key=lambda i: confs[i])
                cx, cy, w, h = result.boxes.xywh[best_idx].tolist()
                x = cx - (w / 2.0)
                y = cy - (h / 2.0)
                pred = (float(x), float(y), float(w), float(h))
                row = {
                    "filename": item.path.stem,
                    "bbox_x": str(int(round(pred[0]))),
                    "bbox_y": str(int(round(pred[1]))),
                    "bbox_w": str(int(round(pred[2]))),
                    "bbox_h": str(int(round(pred[3]))),
                    "iou": f"{max_iou(pred, gt_boxes):.6f}",
                }
            rows.append(row)
        except Exception as exc:
            failures += 1
            print(f"[{idx}/{len(images)}] FAILED: {rel_path} -> {exc}")
            rows.append(
                {
                    "filename": item.path.stem,
                    "bbox_x": "",
                    "bbox_y": "",
                    "bbox_w": "",
                    "bbox_h": "",
                    "iou": "0.0",
                }
            )
        if idx % 20 == 0 or idx == len(images):
            print(f"[{idx}/{len(images)}] processed")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "iou"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Rows={len(rows)} Failures={failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference from a model checkpoint and dataset yaml.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("best.pt"),
        help='Model checkpoint path (default: "best.pt" in CWD).',
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.cwd() / "data.yaml",
        help='YOLO data.yaml path (default: "./data.yaml" in CWD).',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_recomputed.csv"),
        help='Output CSV path (default: "inference_recomputed.csv" in CWD).',
    )
    parser.add_argument("--imgsz", type=int, default=1152, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0" or "cpu".')
    args = parser.parse_args()

    infer_dataset(
        model_path=args.model,
        data_yaml=args.data,
        output_csv=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        nms_iou=args.nms_iou,
        device=args.device,
    )


if __name__ == "__main__":
    main()
