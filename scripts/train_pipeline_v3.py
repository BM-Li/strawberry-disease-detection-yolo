"""
YOLOv8 草莓病害检测 — 训练管线 v3
=====================================
单数据集版本，基于 the_big 数据集训练。

阶段流水线:
  Stage 1 : 全量训练 — yolov8n.pt 从零训练，含类别权重 + Copy-Paste 增强
  Stage 2 : 冻结 Backbone 微调 — 加载 Stage 1 最佳模型，erasing + 低 LR
  Stage 3 : TTA 评估 — 对 Stage 1 和 Stage 2 做标准 + TTA 评估

用法:
    python scripts/train_pipeline_v3.py           # 全部阶段
    python scripts/train_pipeline_v3.py --stage 1 # 只 Stage 1
    python scripts/train_pipeline_v3.py --stage 2 # 只 Stage 2（需要 Stage 1 模型）
    python scripts/train_pipeline_v3.py --stage 3 # 只 TTA 评估
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from generate_anthracnose_copypaste import generate as generate_anthracnose_copypaste
from generate_leaf_copypaste import generate as generate_leaf_copypaste

# ════════════════════════════════════════════════════════════════
#  项目路径
# ════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
CONFIGS_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"

RUN_TAG = "run_v3_thebig"
RUN_ROOT = RESULTS_DIR / "train_v3_runs" / RUN_TAG
TRAIN_PROJECT = RUN_ROOT / "runs" / "train"
EVAL_PROJECT = RUN_ROOT / "runs" / "eval"
METRICS_DIR = RUN_ROOT / "metrics"
LOG_DIR = RUN_ROOT / "logs"
MODEL_DIR = RUN_ROOT / "models"

STAGE1_RUN_NAME = f"stage1_fulltrain_v3_{RUN_TAG}"
STAGE2_RUN_NAME = f"stage2_finetune_v3_{RUN_TAG}"

MODEL_STAGE1 = MODEL_DIR / f"{STAGE1_RUN_NAME}.pt"
MODEL_STAGE2 = MODEL_DIR / f"{STAGE2_RUN_NAME}.pt"

# 数据集
THE_BIG = DATASETS_DIR / "the_big"
BG_FOR_COPYPASTE = DATASETS_DIR / "bg_for_copypaste"
BG_FOR_LEAFSPOT = DATASETS_DIR / "bg_for_leafspot"

# YAML 配置
CFG = CONFIGS_DIR / "the_big.yaml"

# ════════════════════════════════════════════════════════════════
#  超参数
# ════════════════════════════════════════════════════════════════
BASE_MODEL = "yolov8n.pt"
IMGSZ = 640
BATCH = 10
WORKERS = 5
DEVICE = "auto"
AMP = True
OPTIMIZER = "AdamW"
COS_LR = True

STAGE1_EPOCHS = 200
STAGE1_PATIENCE = 40

STAGE2_EPOCHS = 50
STAGE2_LR0 = 5e-5
STAGE2_LRF = 0.01
STAGE2_PATIENCE = 20
STAGE2_WEIGHT_DECAY = 0.001

# 类别名称
CLASS_NAMES = [
    "Angular Leafspot",
    "Anthracnose Fruit Rot",
    "Gray Mold",
    "Leaf Spot",
    "Powdery Mildew Fruit",
    "Powdery Mildew Leaf",
]
NC = 6

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ════════════════════════════════════════════════════════════════
#  GPU 检测
# ════════════════════════════════════════════════════════════════
def detect_gpu(logger):
    global DEVICE
    import torch

    logger.info(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        DEVICE = 0
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
        logger.info(f"GPU: {name}  |  显存: {mem:.1f} GB  |  CUDA: {torch.version.cuda}")
    else:
        DEVICE = "cpu"
        logger.warning("未检测到 CUDA GPU, 将使用 CPU 训练!")
    logger.info(f"设备: {DEVICE}")


# ════════════════════════════════════════════════════════════════
#  日志
# ════════════════════════════════════════════════════════════════
def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"training_v3_{RUN_TAG}_{ts}.log"

    logger = logging.getLogger("strawberry_pipeline_v3")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"日志文件: {log_file}")
    return logger


# ════════════════════════════════════════════════════════════════
#  类别权重计算
# ════════════════════════════════════════════════════════════════
def compute_class_weights(dataset_path: Path, logger) -> list[float]:
    """基于训练集类别频率计算逆频率权重。
    权重 = median_freq / class_freq，让少数类获得更高权重。
    """
    lbl_dir = dataset_path / "train" / "labels"
    class_counts = Counter()

    for lbl_file in lbl_dir.glob("*.txt"):
        content = lbl_file.read_text().strip()
        if not content:
            continue
        for line in content.split("\n"):
            parts = line.strip().split()
            if parts:
                class_counts[int(parts[0])] += 1

    counts = [class_counts.get(i, 1) for i in range(NC)]
    median = sorted(counts)[len(counts) // 2]

    weights = [round(median / max(c, 1), 2) for c in counts]
    weights = [max(0.5, min(3.0, w)) for w in weights]

    logger.info("类别权重 (基于逆频率):")
    for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, counts, weights)):
        logger.info(f"  class {i} ({name}): {cnt} boxes → weight={w:.2f}")

    return weights


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════
def save_best_model(model, dst: Path, logger):
    dst.parent.mkdir(parents=True, exist_ok=True)
    best = Path(model.trainer.best)
    if best.exists():
        shutil.copy2(best, dst)
        logger.info(f"  最佳模型已保存: {dst}")
        return
    last = Path(model.trainer.last)
    if last.exists():
        shutil.copy2(last, dst)
        logger.info(f"  末轮模型已保存 (best 不存在): {dst}")
    else:
        logger.error("  未找到可保存的模型权重!")


def log_metrics(results, stage: str, logger):
    rd = results.results_dict
    logger.info(f"── {stage} 指标 ──")
    for k in ["metrics/precision(B)", "metrics/recall(B)",
              "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if k in rd:
            logger.info(f"  {k}: {rd[k]:.4f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = METRICS_DIR / f"{stage}.json"
    serializable = {}
    for k, v in rd.items():
        try:
            serializable[k] = float(v)
        except (TypeError, ValueError):
            serializable[k] = str(v)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"  指标已保存: {out_file}\n")


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


# ════════════════════════════════════════════════════════════════
#  数据集验证
# ════════════════════════════════════════════════════════════════
def validate_dataset(logger) -> bool:
    if not THE_BIG.exists():
        logger.error(f"数据集不存在: {THE_BIG}")
        return False

    if not CFG.exists():
        logger.error(f"配置文件不存在: {CFG}")
        return False

    for split in ["train", "valid", "test"]:
        img_dir = THE_BIG / split / "images"
        lbl_dir = THE_BIG / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            logger.error(f"数据集结构不完整: {split}")
            return False
        n_img = len(list(img_dir.iterdir()))
        n_lbl = len(list(lbl_dir.glob("*.txt")))
        logger.info(f"  {split}: {n_img} images, {n_lbl} labels")

    return True


# ════════════════════════════════════════════════════════════════
#  Copy-Paste 增强预处理
# ════════════════════════════════════════════════════════════════
def prepare_copypaste(logger):
    logger.info("=" * 60)
    logger.info("STEP CP: Copy-Paste augmentation (三类病害)")
    logger.info("=" * 60)

    # 炭疽病（已充足，少量补充）
    logger.info("  [1/3] Anthracnose (class 1) — 30 张, 背景: bg_for_copypaste")
    generate_anthracnose_copypaste(
        dataset_dir=THE_BIG,
        split="train",
        source_splits=["train"],
        class_id=1,
        num_outputs=30,
        max_patches_per_image=3,
        min_patch_size=14,
        min_mask_area=80,
        prefix="cp_anth",
        seed=42,
        dry_run=False,
        reset_prefix=True,
        custom_bg_dir=str(BG_FOR_COPYPASTE),
    )

    # 叶斑病（重点补充）
    logger.info("  [2/3] Leaf Spot (class 3) — 120 张, 背景: bg_for_leafspot")
    generate_leaf_copypaste(
        dataset_dir=THE_BIG,
        split="train",
        source_splits=["train"],
        class_id=3,
        num_outputs=120,
        max_patches_per_image=3,
        min_patch_size=14,
        min_mask_area=24,
        prefix="cp_leaf",
        seed=42,
        dry_run=False,
        reset_prefix=True,
        custom_bg_dir=str(BG_FOR_LEAFSPOT),
    )

    # 角斑病（适量补充）
    logger.info("  [3/3] Angular Leafspot (class 0) — 100 张, 背景: bg_for_leafspot")
    generate_leaf_copypaste(
        dataset_dir=THE_BIG,
        split="train",
        source_splits=["train"],
        class_id=0,
        num_outputs=100,
        max_patches_per_image=3,
        min_patch_size=14,
        min_mask_area=24,
        prefix="cp_angular",
        seed=42,
        dry_run=False,
        reset_prefix=True,
        custom_bg_dir=str(BG_FOR_LEAFSPOT),
    )

    logger.info("Copy-Paste augmentation finished.\n")


# ════════════════════════════════════════════════════════════════
#  Stage 1: 全量训练
# ════════════════════════════════════════════════════════════════
def stage1_fulltrain(logger):
    logger.info("=" * 60)
    logger.info("STAGE 1: 全量训练 — yolov8n.pt + the_big + 类别权重")
    logger.info("=" * 60)

    compute_class_weights(THE_BIG, logger)  # logs per-class freq for reference

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(CFG),
        epochs=STAGE1_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        amp=AMP,
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        project=str(TRAIN_PROJECT),
        name=STAGE1_RUN_NAME,
        exist_ok=True,
        patience=STAGE1_PATIENCE,
        save=True,
        save_period=10,
        plots=True,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
    )
    log_metrics(results, STAGE1_RUN_NAME, logger)
    save_best_model(model, MODEL_STAGE1, logger)
    return model


# ════════════════════════════════════════════════════════════════
#  Stage 2: 冻结 Backbone 微调
# ════════════════════════════════════════════════════════════════
def stage2_finetune(logger):
    logger.info("=" * 60)
    logger.info("STAGE 2: 冻结 Backbone 微调 — erasing + 低 LR + mosaic=0")
    logger.info("=" * 60)

    if not MODEL_STAGE1.exists():
        logger.error(f"  Stage 1 模型不存在: {MODEL_STAGE1}")
        logger.error("  请先运行 Stage 1")
        return None

    model = YOLO(str(MODEL_STAGE1))
    logger.info("  冻结前 10 层 (Backbone)")

    results = model.train(
        data=str(CFG),
        epochs=STAGE2_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        amp=AMP,
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        lr0=STAGE2_LR0,
        lrf=STAGE2_LRF,
        weight_decay=STAGE2_WEIGHT_DECAY,
        project=str(TRAIN_PROJECT),
        name=STAGE2_RUN_NAME,
        exist_ok=True,
        patience=STAGE2_PATIENCE,
        save=True,
        save_period=5,
        plots=True,
        # 冻结 backbone
        freeze=10,
        # 轻量增强 + 新增 erasing
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.4,
        dropout=0.1,
    )
    log_metrics(results, STAGE2_RUN_NAME, logger)
    save_best_model(model, MODEL_STAGE2, logger)
    return model


# ════════════════════════════════════════════════════════════════
#  Stage 3: TTA 评估
# ════════════════════════════════════════════════════════════════
def stage3_tta_eval(logger):
    logger.info("=" * 60)
    logger.info("STAGE 3: TTA 评估 — Stage 1 vs. Stage 2")
    logger.info("=" * 60)

    candidates = {
        STAGE1_RUN_NAME: MODEL_STAGE1,
        STAGE2_RUN_NAME: MODEL_STAGE2,
    }

    all_results = {}

    for tag, model_path in candidates.items():
        if not model_path.exists():
            logger.warning(f"  模型不存在, 跳过: {tag} ({model_path})")
            continue

        model = YOLO(str(model_path))

        # 标准评估
        logger.info(f"  [{tag}] 标准评估...")
        res_std = model.val(
            data=str(CFG),
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            split="test",
            project=str(EVAL_PROJECT),
            name=f"eval_{tag}_std",
            exist_ok=True,
            plots=True,
        )

        # TTA 评估
        logger.info(f"  [{tag}] TTA 评估...")
        res_tta = model.val(
            data=str(CFG),
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            split="test",
            augment=True,
            project=str(EVAL_PROJECT),
            name=f"eval_{tag}_tta",
            exist_ok=True,
            plots=True,
        )

        std_d = res_std.results_dict
        tta_d = res_tta.results_dict

        all_results[tag] = {
            "standard": {k: _safe_float(v) for k, v in std_d.items()},
            "tta": {k: _safe_float(v) for k, v in tta_d.items()},
        }

        logger.info(f"  ── {tag} 对比 ──")
        for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            s = std_d.get(k, 0)
            t = tta_d.get(k, 0)
            logger.info(f"    {k}:  std={s:.4f}  tta={t:.4f}  Δ={t - s:+.4f}")

        logger.info(f"  ── {tag} 各类别 AP50 ──")
        if hasattr(res_std, "box"):
            try:
                per_class_ap = res_std.box.ap50
                for i, ap in enumerate(per_class_ap):
                    logger.info(f"    class {i} ({CLASS_NAMES[i]}): AP50={ap:.4f}")
            except Exception:
                pass

    out = METRICS_DIR / f"tta_evaluation_v3_{RUN_TAG}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"  TTA 评估汇总: {out}\n")


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 草莓病害检测 — 训练管线 v3")
    p.add_argument(
        "--stage",
        type=str,
        default="all",
        help="运行指定阶段: 1, 2, 3, 或 all (默认 all)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info("YOLOv8 草莓病害检测 — 训练管线 v3")
    logger.info(f"时间: {datetime.now()}")
    logger.info(f"项目根目录: {ROOT}")
    logger.info(f"运行阶段: {args.stage}")
    logger.info(f"run tag: {RUN_TAG}")
    logger.info(f"run root: {RUN_ROOT}")
    logger.info("─" * 60)
    logger.info("管线概览:")
    logger.info("  Stage 1 — yolov8n.pt 全量训练 (epochs=150, patience=30)")
    logger.info("  Stage 2 — 冻结 backbone 微调 (freeze=10, erasing=0.4, mosaic=0)")
    logger.info("  Stage 3 — TTA 评估 (Stage 1 vs. Stage 2)")
    logger.info("─" * 60)
    detect_gpu(logger)

    for d in [RUN_ROOT, MODEL_DIR, METRICS_DIR, LOG_DIR, TRAIN_PROJECT, EVAL_PROJECT]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("\n验证数据集...")
    if not validate_dataset(logger):
        return

    stage = args.stage

    if stage in ("all", "1"):
        prepare_copypaste(logger)
        stage1_fulltrain(logger)

    if stage in ("all", "2"):
        if stage == "2":
            # 单独运行 Stage 2 时不重新生成 Copy-Paste
            pass
        stage2_finetune(logger)

    if stage in ("all", "3"):
        stage3_tta_eval(logger)

    logger.info("=" * 60)
    logger.info("管线 v3 执行完毕!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
