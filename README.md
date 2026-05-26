# Strawberry Disease Detection with YOLO

This project trains a YOLO-based object detection model for strawberry disease detection. The final training pipeline is `scripts/train_pipeline_v3.py`, with targeted copy-paste augmentation for underrepresented disease classes.

## Main Features

- YOLO training on strawberry disease detection data
- Class-frequency analysis before training
- Targeted copy-paste augmentation for anthracnose, leaf spot, and angular leafspot
- Two-stage training: full training followed by frozen-backbone fine-tuning
- Test-time augmentation evaluation
- Metrics logging and best-model saving

## Project Structure

```text
configs/
  the_big.yaml

scripts/
  train_pipeline_v3.py
  generate_anthracnose_copypaste.py
  generate_leaf_copypaste.py

datasets/
  the_big/
  pd_test/
  bg_for_copypaste/
  bg_for_leafspot/
```

## Dataset

Datasets are not included in this repository because redistribution rights may vary across public dataset sources. Prepare the data locally with the following structure:

```text
datasets/the_big/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels

datasets/bg_for_copypaste/
datasets/bg_for_leafspot/
datasets/pd_test/
```

Public strawberry disease datasets may contain duplicate or near-duplicate images, class imbalance, and train/validation/test distribution bias. Before training, review duplicate samples, class distribution, split strategy, and whether generated augmentation samples look realistic.

## Training

Run the full V3 pipeline:

```bash
python scripts/train_pipeline_v3.py
```

Run a single stage:

```bash
python scripts/train_pipeline_v3.py --stage 1
python scripts/train_pipeline_v3.py --stage 2
python scripts/train_pipeline_v3.py --stage 3
```

Stage 1 performs full training with copy-paste augmentation. Stage 2 fine-tunes from the Stage 1 model. Stage 3 evaluates the saved models with standard validation and TTA.

## Copy-Paste Augmentation

The final pipeline uses:

- `generate_anthracnose_copypaste.py` for anthracnose fruit rot samples
- `generate_leaf_copypaste.py` for leaf spot and angular leafspot samples

Background images are expected under:

```text
datasets/bg_for_copypaste/
datasets/bg_for_leafspot/
```

## Model Weights

Model weights and training outputs are not tracked in Git. Train locally or publish selected weights separately through GitHub Releases.
