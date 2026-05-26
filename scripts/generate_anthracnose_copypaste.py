from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "strawberry-1-stratified"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Patch:
    image: np.ndarray
    mask: np.ndarray
    source_stem: str


@dataclass
class BackgroundSample:
    image_path: Path
    label_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic anthracnose samples with copy-paste augmentation."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="YOLO dataset root. Default: datasets/strawberry-1-stratified",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Target split to augment. Default: train",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=None,
        help="Splits used to extract anthracnose patches. Default: same as --split",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=1,
        help="Target class id. Anthracnose is 1 in this project.",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=100,
        help="How many synthetic images to create. Default: 100",
    )
    parser.add_argument(
        "--max-patches-per-image",
        type=int,
        default=3,
        help="Maximum lesions pasted into one synthetic image. Default: 3",
    )
    parser.add_argument(
        "--min-patch-size",
        type=int,
        default=14,
        help="Minimum lesion crop width/height in pixels. Default: 14",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=80,
        help="Minimum non-zero mask area in pixels. Default: 80",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="cp_anth",
        help="Filename prefix for synthetic outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect counts only, do not write files.",
    )
    parser.add_argument(
        "--reset-prefix",
        action="store_true",
        help="Remove previously generated outputs that use the same prefix before writing new ones.",
    )
    parser.add_argument(
        "--custom-bg-dir",
        type=Path,
        default=None,
        help="Optional directory of pure background images. If provided, images in this directory are used directly as healthy canvases.",
    )
    return parser.parse_args()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x1 = int(round((cx - w / 2.0) * img_w))
    y1 = int(round((cy - h / 2.0) * img_h))
    x2 = int(round((cx + w / 2.0) * img_w))
    y2 = int(round((cy + h / 2.0) * img_h))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    return x1, y1, x2, y2


def bbox_to_yolo(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def normalized_polygon_to_pixels(
    coords: list[float], img_w: int, img_h: int
) -> np.ndarray:
    points = np.array(coords, dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= img_w
    points[:, 1] *= img_h
    points[:, 0] = np.clip(points[:, 0], 0, img_w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, img_h - 1)
    return np.round(points).astype(np.int32)


def compute_lesion_mask(crop: np.ndarray, min_mask_area: int) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_inv = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    dark_thresh = max(25, int(np.percentile(val, 55)))
    sat_thresh = int(np.percentile(sat, 70))

    dark_mask = (val <= dark_thresh).astype(np.uint8) * 255
    sat_mask = (sat <= sat_thresh).astype(np.uint8) * 255
    mask = cv2.bitwise_and(otsu_inv, cv2.bitwise_or(dark_mask, sat_mask))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    center_mask = np.zeros_like(mask)
    h, w = mask.shape
    cx1, cy1 = int(w * 0.2), int(h * 0.2)
    cx2, cy2 = int(w * 0.8), int(h * 0.8)
    center_mask[cy1:cy2, cx1:cx2] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_mask_area:
            continue
        component = (labels == label_id).astype(np.uint8) * 255
        if cv2.countNonZero(cv2.bitwise_and(component, center_mask)) > 0:
            filtered = cv2.bitwise_or(filtered, component)

    if cv2.countNonZero(filtered) >= min_mask_area:
        return filtered

    fallback = np.zeros_like(mask)
    cv2.ellipse(
        fallback,
        (w // 2, h // 2),
        (max(2, int(w * 0.28)), max(2, int(h * 0.28))),
        0,
        0,
        360,
        255,
        -1,
    )
    return fallback


def extract_anthracnose_patches(
    dataset_dir: Path,
    source_splits: list[str],
    class_id: int,
    min_patch_size: int,
    min_mask_area: int,
) -> list[Patch]:
    patches: list[Patch] = []

    for split in source_splits:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        for label_path in sorted(labels_dir.glob("*.txt")):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            img_h, img_w = image.shape[:2]

            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 5 or int(parts[0]) != class_id:
                    continue

                # YOLO detect: class cx cy w h
                if len(parts) == 5:
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)

                    pad_x = int((x2 - x1) * 0.15)
                    pad_y = int((y2 - y1) * 0.15)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(img_w, x2 + pad_x)
                    y2 = min(img_h, y2 + pad_y)

                    if (x2 - x1) < min_patch_size or (y2 - y1) < min_patch_size:
                        continue

                    crop = image[y1:y2, x1:x2].copy()
                    mask = compute_lesion_mask(crop, min_mask_area)
                    ys, xs = np.where(mask > 0)
                    if xs.size == 0 or ys.size == 0:
                        continue

                    tx1, tx2 = xs.min(), xs.max() + 1
                    ty1, ty2 = ys.min(), ys.max() + 1
                    tight_crop = crop[ty1:ty2, tx1:tx2].copy()
                    tight_mask = mask[ty1:ty2, tx1:tx2].copy()
                else:
                    # YOLO segmentation: class x1 y1 x2 y2 ... xn yn
                    coords = list(map(float, parts[1:]))
                    if len(coords) < 6 or len(coords) % 2 != 0:
                        continue

                    polygon = normalized_polygon_to_pixels(coords, img_w, img_h)
                    x, y, w, h = cv2.boundingRect(polygon)
                    pad_x = int(w * 0.12)
                    pad_y = int(h * 0.12)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(img_w, x + w + pad_x)
                    y2 = min(img_h, y + h + pad_y)

                    if (x2 - x1) < min_patch_size or (y2 - y1) < min_patch_size:
                        continue

                    crop = image[y1:y2, x1:x2].copy()
                    local_polygon = polygon.copy()
                    local_polygon[:, 0] -= x1
                    local_polygon[:, 1] -= y1
                    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [local_polygon], 255)

                    ys, xs = np.where(mask > 0)
                    if xs.size == 0 or ys.size == 0:
                        continue

                    tx1, tx2 = xs.min(), xs.max() + 1
                    ty1, ty2 = ys.min(), ys.max() + 1
                    tight_crop = crop[ty1:ty2, tx1:tx2].copy()
                    tight_mask = mask[ty1:ty2, tx1:tx2].copy()

                if (
                    tight_crop.shape[0] < min_patch_size
                    or tight_crop.shape[1] < min_patch_size
                    or cv2.countNonZero(tight_mask) < min_mask_area
                ):
                    continue

                patches.append(
                    Patch(
                        image=tight_crop,
                        mask=tight_mask,
                        source_stem=image_path.stem,
                    )
                )

    return patches


def collect_healthy_backgrounds(
    dataset_dir: Path,
    split: str,
    custom_bg_dir: str | Path | None = None,
) -> list[BackgroundSample]:
    healthy: list[BackgroundSample] = []

    if custom_bg_dir is not None:
        bg_dir = Path(custom_bg_dir)
        if bg_dir.exists():
            for image_path in sorted(bg_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                    healthy.append(BackgroundSample(image_path=image_path, label_path=None))
            return healthy

    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            healthy.append(BackgroundSample(image_path=image_path, label_path=label_path))
            continue
        if not label_path.read_text(encoding="utf-8").strip():
            healthy.append(BackgroundSample(image_path=image_path, label_path=label_path))

    return healthy


def transform_patch(
    patch: Patch,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    scale = rng.uniform(0.7, 1.35)
    angle = rng.uniform(-35.0, 35.0)
    flip_code = rng.choice([-1, 0, 1, None])

    image = patch.image
    mask = patch.mask
    if flip_code is not None:
        image = cv2.flip(image, flipCode=flip_code)
        mask = cv2.flip(mask, flipCode=flip_code)

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    cos_v = abs(matrix[0, 0])
    sin_v = abs(matrix[0, 1])
    out_w = int((h * sin_v) + (w * cos_v))
    out_h = int((h * cos_v) + (w * sin_v))

    matrix[0, 2] += (out_w / 2.0) - center[0]
    matrix[1, 2] += (out_h / 2.0) - center[1]

    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed_mask = cv2.warpAffine(
        mask,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    transformed_mask = np.where(transformed_mask > 0, 255, 0).astype(np.uint8)
    return transformed_image, transformed_mask


def box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def place_patch(
    background: np.ndarray,
    patch: Patch,
    existing_boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    transformed_image, transformed_mask = transform_patch(patch, rng)
    mask_h, mask_w = transformed_mask.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if mask_h >= bg_h or mask_w >= bg_w:
        return background, None

    ys, xs = np.where(transformed_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return background, None

    patch_bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    margin_x = max(4, int(bg_w * 0.02))
    margin_y = max(4, int(bg_h * 0.02))
    max_x = bg_w - mask_w - margin_x
    max_y = bg_h - mask_h - margin_y
    if max_x <= margin_x or max_y <= margin_y:
        return background, None

    for _ in range(40):
        x = rng.randint(margin_x, max_x)
        y = rng.randint(margin_y, max_y)
        bbox = (
            x + patch_bbox[0],
            y + patch_bbox[1],
            x + patch_bbox[2],
            y + patch_bbox[3],
        )
        if any(box_iou(bbox, prev) > 0.2 for prev in existing_boxes):
            continue

        center = (x + mask_w // 2, y + mask_h // 2)
        mixed = cv2.seamlessClone(
            transformed_image,
            background,
            transformed_mask,
            center,
            cv2.NORMAL_CLONE,
        )
        return mixed, bbox

    return background, None


def next_index(images_dir: Path, prefix: str) -> int:
    max_index = 0
    for image_path in images_dir.glob(f"{prefix}_*.jpg"):
        suffix = image_path.stem.replace(f"{prefix}_", "")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def clear_existing_generated(images_dir: Path, labels_dir: Path, prefix: str) -> int:
    removed = 0
    for image_path in images_dir.glob(f"{prefix}_*.jpg"):
        image_path.unlink()
        removed += 1
    for label_path in labels_dir.glob(f"{prefix}_*.txt"):
        label_path.unlink()
    return removed


def remove_cache_files(dataset_dir: Path, split: str) -> None:
    for cache_path in [
        dataset_dir / split / "labels.cache",
        dataset_dir / split / "images.cache",
    ]:
        if cache_path.exists():
            cache_path.unlink()


def generate(
    dataset_dir: Path,
    split: str,
    source_splits: list[str],
    class_id: int,
    num_outputs: int,
    max_patches_per_image: int,
    min_patch_size: int,
    min_mask_area: int,
    prefix: str,
    seed: int,
    dry_run: bool,
    custom_bg_dir: str | Path | None = None,
    reset_prefix: bool = False,
) -> None:
    patches = extract_anthracnose_patches(
        dataset_dir=dataset_dir,
        source_splits=source_splits,
        class_id=class_id,
        min_patch_size=min_patch_size,
        min_mask_area=min_mask_area,
    )
    backgrounds = collect_healthy_backgrounds(dataset_dir, split, custom_bg_dir=custom_bg_dir)

    print("=" * 72)
    print("Anthracnose Copy-Paste Augmentation")
    print(f"dataset      : {dataset_dir}")
    print(f"target split : {split}")
    print(f"source splits: {source_splits}")
    print(f"class id     : {class_id}")
    print(f"custom bg dir: {custom_bg_dir if custom_bg_dir else 'None'}")
    print(f"patches      : {len(patches)}")
    print(f"healthy bg   : {len(backgrounds)}")
    print(f"outputs      : {num_outputs}")
    print("=" * 72)

    if not patches:
        raise RuntimeError("No anthracnose patches were extracted. Check labels and class id.")
    if not backgrounds:
        raise RuntimeError("No healthy backgrounds found. Provide custom_bg_dir or make sure the target split contains unlabeled / empty-label background images.")

    if dry_run:
        print("[dry-run] No files written.")
        return

    rng = random.Random(seed)
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    removed = 0
    if reset_prefix:
        removed = clear_existing_generated(images_dir, labels_dir, prefix)
    start_idx = next_index(images_dir, prefix)

    written = 0
    attempts = 0
    while written < num_outputs and attempts < num_outputs * 5:
        attempts += 1
        bg = rng.choice(backgrounds)
        background = cv2.imread(str(bg.image_path))
        if background is None:
            continue

        boxes: list[tuple[int, int, int, int]] = []
        num_patches = rng.randint(1, max(1, max_patches_per_image))
        composed = background.copy()

        for _ in range(num_patches):
            patch = rng.choice(patches)
            composed, bbox = place_patch(composed, patch, boxes, rng)
            if bbox is not None:
                boxes.append(bbox)

        if not boxes:
            continue

        out_stem = f"{prefix}_{start_idx + written:04d}"
        out_image = images_dir / f"{out_stem}.jpg"
        out_label = labels_dir / f"{out_stem}.txt"

        cv2.imwrite(str(out_image), composed, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        img_h, img_w = composed.shape[:2]
        lines = []
        for x1, y1, x2, y2 in boxes:
            cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(
                f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )
        out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

    remove_cache_files(dataset_dir, split)

    print(f"written      : {written}")
    print(f"failed tries : {attempts - written}")
    print(f"removed old  : {removed}")
    print(f"output dir   : {images_dir}")
    print(f"label dir    : {labels_dir}")


def main() -> None:
    args = parse_args()
    source_splits = args.source_splits or [args.split]
    generate(
        dataset_dir=args.dataset,
        split=args.split,
        source_splits=source_splits,
        class_id=args.class_id,
        num_outputs=args.num_outputs,
        max_patches_per_image=args.max_patches_per_image,
        min_patch_size=args.min_patch_size,
        min_mask_area=args.min_mask_area,
        prefix=args.prefix,
        seed=args.seed,
        dry_run=args.dry_run,
        custom_bg_dir=args.custom_bg_dir,
        reset_prefix=args.reset_prefix,
    )


if __name__ == "__main__":
    main()
