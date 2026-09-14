# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.894983378664249

# 6. Current score

0.54223

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13453) has done: 'I fix the Albumentations API breakage by updating `RandomResizedCrop` calls to the v2 signature so augmentations construct correctly. I also fix the missing-weights crash by making the script robust to absent `../input/ensemble-1023/*.pth` files: if weights are not found, it fall back to using `timm` pretrained weights for the same architectures (preserving the same models and ensembling logic) so it can run end-to-end. Finally, I ensure the submission is aligned to `sample_submission.csv` ordering (and keep the filename `submission.csv`) to guarantee a valid Kaggle submission format.'
- What this solution (achieved 0.13453) has done: 'I fix the Albumentations v2 API break causing the crash by replacing the removed `A.Cutout` augmentation with its supported equivalent (`A.CoarseDropout`) while keeping the same augmentation intent. I also correct a logic bug in the focal loss (even if it’s not used for inference here) to prevent future silent misuse: `p_t` must be computed from probabilities, not logits. Finally, I keep the existing ensembling/inference logic intact and ensure the script always writes `submission.csv` with the exact `image_id,label` format aligned to `sample_submission.csv` ordering.'
- What this solution (achieved 0.11771) has done: 'Your very low score is consistent with an inference-time bug: you L2-normalize raw logits across classes, which distorts class probabilities and tends to ruin argmax decisions. To move toward the target accuracy with minimal disruption, I remove that normalization and instead ensemble using softmax probabilities (standard for accuracy metrics) while keeping the same two models, weights-loading fallback, TTA count, and blend weights. I also keep submission ordering aligned to `sample_submission.csv` and ensure the output `submission.csv` remains `image_id,label`. These changes preserve the core modeling logic but fix the post-processing to match the evaluation semantics.'
- What this solution (achieved 0.48954) has done: 'Your score is far below the target, and the biggest likely cause (without changing your core models) is that your fallback weight loading is unintentionally using *ImageNet-1000 heads* (because `pretrained=True` with `num_classes=5` does not reliably give a true 5-class pretrained head across timm models), leaving your 5-class head essentially random. I keep the same two architectures, same ensembling, same TTA, and same augmentations, but change the fallback to: load a proper ImageNet pretrained backbone, then **recreate and keep your existing 5-class randomly initialized head**. I also make the inference faster/cleaner (without changing semantics) by moving `DataParallel(...).to(device)` outside the per-model function loop and using `torch.inference_mode()`; this should not hurt accuracy and helps stability under the 600s limit. These are minimal changes that should substantially increase accuracy toward your target.'
- What this solution (achieved 0.54185) has done: 'Your current score (0.48954) is far below the target (0.89498), and the most likely remaining issue (without changing your model architectures or training approach) is that inference is being done at a different resolution/augmentation regime than what the original checkpoint(s) expected, plus using heavy random augmentations for test-time that can hurt accuracy if not tuned. To move the score upward with minimal semantic change, I (1) switch test-time transforms to a deterministic “resize+center-crop+normalize” baseline (no random flips/transpose/crops), and (2) do standard multi-crop TTA by averaging predictions over a small, deterministic set (center + 4 corners), which is stable and typically improves accuracy for ImageNet-pretrained backbones. I also keep the same two models, the same ensemble weights (0.43/0.57), the same weight-loading/fallback logic, and the same submission alignment to `sample_submission.csv`. These changes are directly targeted at improving classification accuracy while preserving your core ensembling logic.'
- What this solution (achieved 0.54223) has done: 'Your current gap to the target is large (0.54185 → 0.89498), and the most likely remaining blocker (without changing your models/loops/loss) is a train–test preprocessing mismatch: these cassava checkpoints are typically trained at 512 with a **center-crop from a slightly larger resize** (and your current TTA crops are taken from an already 512-resized image, making crops effectively “zoomed in” and distribution-shifted). I keep the same two models, same weight loading, same ensemble weights, and same 5-crop TTA count, but change the test preprocessing to `Resize(int(IMAGE_SIZE/0.92)) + Normalize` so the subsequent 0.92 crops match the intended effective crop geometry. I also ensure the test-time augmentation uses the same `max_pixel_value=255.0` normalization settings consistently. These minimal changes are directly aimed at improving accuracy toward your target while preserving your core inference logic.'
- What this solution (achieved 0.54223) has done: 'Your current score is far below the target, so we should make a small, high-impact fix that doesn’t change your model architectures or training/inference loops: the most likely issue is that your custom checkpoints are not being found due to a wrong `INPUT_PATH`, so you are effectively running with random 5-class heads (backbone-only fallback), which caps accuracy near chance/weak performance. I minimally extend the checkpoint path resolution to search common Kaggle input/working locations (including the dataset folder you actually have) and load the weights when present. I also print the resolved checkpoint paths/status so you can verify you’re no longer falling back unintentionally. Everything else (models, 5-crop TTA, blend weights, submission alignment) stays the same.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so we should make a small, high-impact fix without changing your two-model ensemble or TTA logic: right now you are very likely not loading the intended cassava-trained checkpoints and are instead running ImageNet backbones with random 5-class heads, which caps accuracy around your current range. I minimally extend checkpoint discovery to also search the competition dataset folder (where Kaggle typically mounts additional files), and I fail fast (with a clear message) if the custom weights still cannot be found—because producing a “valid but low-accuracy” submission moves you away from the target. Everything else (architectures, 5-crop TTA, blend weights, transforms, and submission formatting) remains unchanged.'
- What this solution (achieved 0.12407) has done: 'I remove the hard fail that stops execution when the private ensemble checkpoints are missing, because your current environment doesn’t contain `ensemble-1023/` and you need a runnable end-to-end pipeline that produces `submission.csv`. To move accuracy upward toward the target using only minimal, semantics-preserving changes, I keep the same two-model ensemble + 5-crop TTA, but switch the fallback to use strong timm pretrained weights via `pretrained=True` with `num_classes=5` (timm adapt and load the classifier where supported) instead of leaving a random 5-class head. I also make checkpoint discovery more robust by additionally searching inside the provided dataset tree (`/kaggle/input/.../cassava-leaf-disease-classification/...`) so custom weights would be picked up automatically if present. Everything else (architectures, transforms, TTA, blending weights, submission format/order) stays intact.'
- What this solution (achieved 0.54223) has done: 'Your current score is far below the target, and the most likely remaining cause (given your code already fixes TTA geometry and submission alignment) is that you are still effectively using an ImageNet-pretrained backbone with a randomly initialized 5-class head in the fallback path, which caps accuracy. I keep the same two architectures, same ensemble weights, same 5-crop TTA, and same transforms, but change the fallback weight loading to correctly transfer only the pretrained backbone weights (excluding the classifier head) so the model remains valid and much stronger than “random head with partially mismatched weights”. I also use each model’s own `default_cfg` normalization (mean/std) for inference to match the pretrained weights’ expected preprocessing, which is a minimal, metric-aligned change. These are small, directly relevant fixes that should move accuracy substantially upward toward your target without changing the core inference/training approach.'
- What this solution (achieved 0.12407) has done: 'Your score is far below the target, so the smallest meaningful improvement is to ensure you’re not silently falling back to “ImageNet backbone + random 5-class head” (which cap accuracy around where you are). I keep your two-model ensemble, 5-crop TTA, blend weights, and overall inference loop intact, but change the fallback path to use each model’s **timm pretrained weights adapted to 5 classes** when available (so the classifier is not random). If timm cannot provide a 5-class pretrained head for a given model, we still load the backbone-only weights as you currently do. This is a minimal, directly score-relevant change and should move accuracy substantially toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.54223) has done: 'Your current score is far below the target, and the biggest likely cause (without changing your two-model ensemble, TTA count, or training approach) is that you are still usually *not loading cassava-trained checkpoints* and the fallback path is effectively producing near-random predictions. The smallest, directly score-relevant improvement is to use a real cassava-trained fallback that is available inside the competition dataset itself: `tf_efficientnet_b4_ns` has an official timm weight for this task (`tf_efficientnet_b4_ns.tf_in1k_ft_in22k_in1k` is common, but for cassava specifically timm provides `tf_efficientnet_b4_ns` “cassava” weights in many builds via `pretrained_cfg`). We keep your exact architectures and inference loop, but adjust fallback loading to (1) prefer a *cassava* pretrained_cfg if timm exposes it, otherwise (2) use ImageNet pretrained backbone-only as before. This should move accuracy substantially upward toward the target while preserving your core logic and still writing a valid `submission.csv`.'
- What this solution (achieved 0.54223) has done: 'Your score is far below the target, and given your logs indicate you usually don’t have the original cassava checkpoints available, the only realistic way to move toward the target without changing the model architectures/loops is to ensure the fallback uses *actual cassava-trained weights* rather than ImageNet/backbone-only. I minimally extend the fallback loader to prefer timm’s known cassava finetuned weights (`*_cassava` tags) by trying a short, safe list of likely pretrained tags for each model, and only if those aren’t available it keep your existing backbone-only fallback. I also keep your existing per-model normalization and 5-crop TTA exactly as-is, and still write `submission.csv` aligned to `sample_submission.csv`. These changes are directly score-relevant and should move accuracy substantially upward toward the target when cassava pretrained weights are available in this timm build.'
- What this solution (achieved 0.54223) has done: 'Your current score (0.542) is far below the target (0.895), and with your custom ensemble checkpoints still missing, the main limiter is the fallback weights: ImageNet-backbone-only + random 5-class head cap accuracy. I keep the exact same two architectures, same 5-crop TTA, same blending, and same inference loop, but make the fallback try timm’s *cassava-finetuned* checkpoints via `timm.create_model(..., pretrained=True)` and `timm.create_model(..., pretrained_cfg=...)` (when available) **while keeping your 5-class head** so shapes match safely. I also ensure the inference normalization uses the preprocessing that corresponds to the actually-loaded pretrained config (not just the base model default), which is a minimal but high-impact correctness fix. These changes should move accuracy substantially upward toward the target without changing your core logic.'

# 9. Code solution

## === cell 0
import os
import math
import random
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm import tqdm
import timm

warnings.filterwarnings("ignore")



## === cell 1
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"
DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5

TTA = 5  # center + 4 corners


def _resolve_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


TRAIN_CSV_PATH = (
    _resolve_first_existing(
        [
            TRAIN_CSV_PATH,
            "/kaggle/input/cassava-leaf-disease-classification/train.csv",
            "/kaggle/data/cassava-leaf-disease-classification/train.csv",
            "/kaggle/input/train.csv",
            "/kaggle/data/train.csv",
        ]
    )
    or TRAIN_CSV_PATH
)

TEST_IMAGE_PATH = (
    _resolve_first_existing(
        [
            TEST_IMAGE_PATH,
            "/kaggle/input/cassava-leaf-disease-classification/test_images/",
            "/kaggle/data/cassava-leaf-disease-classification/test_images/",
            "/kaggle/input/test_images/",
            "/kaggle/data/test_images/",
        ]
    )
    or TEST_IMAGE_PATH
)

SAMPLE_SUB_PATH = _resolve_first_existing(
    [
        "../input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)


def _resolve_dir_first_existing(paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


INPUT_PATH = (
    _resolve_dir_first_existing(
        [
            INPUT_PATH,
            "../input/ensemble-1023/",
            "/kaggle/input/ensemble-1023/",
            "/kaggle/data/ensemble-1023/",
            "/kaggle/working/ensemble-1023/",
            "/kaggle/working/cassava-leaf-disease-classification/ensemble-1023/",
            "/kaggle/data/cassava-leaf-disease-classification/ensemble-1023/",
            "/kaggle/input/cassava-leaf-disease-classification/ensemble-1023/",
            "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
            "/kaggle/working/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
            "/kaggle/working/cassava-leaf-disease-classification/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
        ]
    )
    or INPUT_PATH
)

DEVICE0 = DEVICES[0] if len(DEVICES) > 0 else torch.device("cpu")

print("Resolved INPUT_PATH:", INPUT_PATH)
print("Resolved TEST_IMAGE_PATH:", TEST_IMAGE_PATH)
print("Resolved TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("Resolved SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    """
    Bug fix: focal term must be computed on probabilities (sigmoid(logits)),
    not directly from logits. This keeps semantics correct if used later.
    """

    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor)
        y = y + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor)

    prob = torch.sigmoid(y_hat)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p_t = y_true * prob + (1 - y_true) * (1 - prob)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




## === cell 3
def lr_tune(epoch, num_epochs=NUM_EPOCHS):
    lr_start = LR_START
    lr_max = LR_MAX
    lr_final = LR_FINAL
    lr_warmup_epoch = 4
    lr_sustain_epoch = 0
    lr_decay_epoch = num_epochs - lr_warmup_epoch - lr_sustain_epoch - 1

    if epoch <= lr_warmup_epoch:
        lr = lr_start + (lr_max - lr_start) * (epoch / lr_warmup_epoch) ** 2.5
    elif epoch < lr_warmup_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        epoch_diff = epoch - lr_warmup_epoch - lr_sustain_epoch
        decay_factor = (epoch_diff / lr_decay_epoch) * math.pi
        decay_factor = (torch.cos(torch.tensor(decay_factor)).numpy() + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr




## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(p=0.5),
        A.CoarseDropout(p=0.5),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)



## === cell 5
_TEST_RESIZE = int(round(IMAGE_SIZE / 0.92))




## === cell 6
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)



## === cell 7
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=False)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)




## === cell 8
def _strip_module_prefix(state_dict):
    if isinstance(state_dict, dict) and any(
        k.startswith("module.") for k in state_dict.keys()
    ):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _get_model_preprocess_cfg(model):
    """
    Change (score improvement, minimal): store the effective preprocessing cfg from the
    weights we actually loaded (cassava tag / pretrained cfg), not just model.default_cfg.
    """
    cfg = (
        getattr(model, "pretrained_cfg", None)
        or getattr(model, "default_cfg", {})
        or {}
    )
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    return {"mean": tuple(mean), "std": tuple(std)}


def _load_pretrained_backbone_only(model, model_name_for_fallback):
    """
    Fallback when no cassava-specific pretrained cfg is available:
    load ImageNet backbone weights but exclude classifier head.
    """
    backbone = timm.create_model(
        model_name_for_fallback, pretrained=True, num_classes=1000
    )
    sd = backbone.state_dict()

    drop_prefixes = ("fc.", "classifier.", "head.", "last_linear.")
    for k in list(sd.keys()):
        if k.startswith(drop_prefixes):
            sd.pop(k)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model._inference_preprocess_cfg = _get_model_preprocess_cfg(backbone)
    return f"fallback_imagenet_backbone_only(missing={len(missing)},unexpected={len(unexpected)})"


def _try_load_timm_pretrained_tag_backbone_aware(model, model_name_for_fallback, tag):
    """
    Change (score improvement, minimal): load weights from a tagged pretrained_cfg model,
    but if the tag is not 5-class, still transfer backbone weights only (exclude head).
    This keeps your existing 5-class head and avoids shape-mismatch while leveraging
    stronger finetuned backbones.
    """
    try:
        m = timm.create_model(
            model_name_for_fallback,
            pretrained=True,
            pretrained_cfg=tag,
        )
        sd = m.state_dict()

        drop_prefixes = ("fc.", "classifier.", "head.", "last_linear.")
        for k in list(sd.keys()):
            if k.startswith(drop_prefixes):
                sd.pop(k)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        model._inference_preprocess_cfg = _get_model_preprocess_cfg(m)
        return f"fallback_timm_tag_backbone_only={tag}(missing={len(missing)},unexpected={len(unexpected)})"
    except Exception:
        return None


def _try_load_timm_pretrained_default_backbone_aware(model, model_name_for_fallback):
    """
    Change (score improvement, minimal): try timm's default pretrained weights (often stronger than
    plain ImageNet1k for some families) and transfer backbone-only into our 5-class head model.
    """
    try:
        m = timm.create_model(model_name_for_fallback, pretrained=True)
        sd = m.state_dict()
        drop_prefixes = ("fc.", "classifier.", "head.", "last_linear.")
        for k in list(sd.keys()):
            if k.startswith(drop_prefixes):
                sd.pop(k)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        model._inference_preprocess_cfg = _get_model_preprocess_cfg(m)
        return f"fallback_timm_pretrained_default_backbone_only(missing={len(missing)},unexpected={len(unexpected)})"
    except Exception:
        return None


def _load_best_available_pretrained_cfg(model, model_name_for_fallback):
    """
    Prefer cassava-trained timm weights when available; otherwise use timm default pretrained;
    if that also fails, keep ImageNet backbone-only fallback.

    This is the smallest change likely to materially improve accuracy when your custom
    checkpoints are missing.
    """
    common_cassava_tags = [
        f"{model_name_for_fallback}.cassava",
        f"{model_name_for_fallback}_cassava",
        f"{model_name_for_fallback}.cassava-5",
        f"{model_name_for_fallback}.cassava_ft",
        f"{model_name_for_fallback}.tf_in1k_ft_in22k_in1k_cassava",
    ]
    for tag in common_cassava_tags:
        res = _try_load_timm_pretrained_tag_backbone_aware(
            model, model_name_for_fallback, tag
        )
        if res is not None:
            return res

    try:
        cfg_names = []
        if hasattr(timm, "list_pretrained"):
            try:
                cfg_names = [x for x in timm.list_pretrained(model_name_for_fallback)]
            except Exception:
                cfg_names = []
        if not cfg_names and hasattr(timm.models, "get_pretrained_cfg_names"):
            try:
                cfg_names = list(
                    timm.models.get_pretrained_cfg_names(model_name_for_fallback)
                )
            except Exception:
                cfg_names = []

        preferred = []
        for n in cfg_names:
            ln = str(n).lower()
            if "cassava" in ln:
                preferred.append(n)

        for tag in preferred[:8]:
            res = _try_load_timm_pretrained_tag_backbone_aware(
                model, model_name_for_fallback, tag
            )
            if res is not None:
                return res

        res = _try_load_timm_pretrained_default_backbone_aware(
            model, model_name_for_fallback
        )
        if res is not None:
            return res

        return _load_pretrained_backbone_only(model, model_name_for_fallback)
    except Exception as e:
        return (
            _load_pretrained_backbone_only(model, model_name_for_fallback)
            + f"; reason={type(e).__name__}"
        )


def _load_weights_or_fallback(model, weight_path, model_name_for_fallback, strict=True):
    """
    If custom weight exists: load it.
    If missing: prefer cassava-trained timm weights when available; else timm default pretrained;
    else ImageNet backbone-only.
    """
    if weight_path is not None and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        state = _strip_module_prefix(state)
        model.load_state_dict(state, strict=strict)

        model._inference_preprocess_cfg = _get_model_preprocess_cfg(model)
        return "loaded_custom"

    return _load_best_available_pretrained_cfg(model, model_name_for_fallback)


def _resolve_weight_file(filename):
    candidates = [
        os.path.join(INPUT_PATH, filename),
        os.path.join("../input/ensemble-1023/", filename),
        os.path.join("/kaggle/input/ensemble-1023/", filename),
        os.path.join("/kaggle/data/ensemble-1023/", filename),
        os.path.join("/kaggle/working/ensemble-1023/", filename),
        os.path.join(
            "/kaggle/working/cassava-leaf-disease-classification/ensemble-1023/",
            filename,
        ),
        os.path.join(
            "/kaggle/input/cassava-leaf-disease-classification/ensemble-1023/", filename
        ),
        os.path.join(
            "/kaggle/data/cassava-leaf-disease-classification/ensemble-1023/", filename
        ),
        os.path.join(
            "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
            filename,
        ),
        os.path.join(
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
            filename,
        ),
        os.path.join(
            "/kaggle/working/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
            filename,
        ),
        os.path.join(
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
            filename,
        ),
        os.path.join(
            "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/cassava-leaf-disease-classification/ensemble-1023/",
            filename,
        ),
    ]
    return _resolve_first_existing(candidates)


resnext_weight_path = _resolve_weight_file(RESNEXT_PATH)
b4_weight_path = _resolve_weight_file(B4_PATH)

print("Resolved RESNEXT weight:", resnext_weight_path)
print("Resolved B4 weight:", b4_weight_path)
if resnext_weight_path is None or b4_weight_path is None:
    print(
        "WARNING: Custom ensemble checkpoints not found; using timm pretrained fallback "
        "(prefer cassava tags; else timm default pretrained; else ImageNet backbone-only)."
    )

status1 = _load_weights_or_fallback(
    my_model_1, resnext_weight_path, model_name1, strict=False
)
status2 = _load_weights_or_fallback(
    my_model_2, b4_weight_path, model_name2, strict=False
)

print("Model1 weights:", status1)
print("Model2 weights:", status2)



## === cell 9
torch.cuda.empty_cache()

if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "sample_submission.csv not found in expected Kaggle input locations."
    )
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_list = sample_sub["image_id"].values


def _five_crop_tensors(img_tensor, crop_size):
    _, H, W = img_tensor.shape
    cs = crop_size
    if cs > H or cs > W:
        return [img_tensor]

    top = 0
    left = 0
    bottom = H - cs
    right = W - cs
    center_top = (H - cs) // 2
    center_left = (W - cs) // 2

    crops = [
        img_tensor[:, top : top + cs, left : left + cs],  # top-left
        img_tensor[:, top : top + cs, right : right + cs],  # top-right
        img_tensor[:, bottom : bottom + cs, left : left + cs],  # bottom-left
        img_tensor[:, bottom : bottom + cs, right : right + cs],  # bottom-right
        img_tensor[
            :, center_top : center_top + cs, center_left : center_left + cs
        ],  # center
    ]
    return crops


def _make_test_base_augs_for_model(model):
    cfg = getattr(
        model, "_inference_preprocess_cfg", None
    ) or _get_model_preprocess_cfg(model)
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    return A.Compose(
        [
            A.Resize(_TEST_RESIZE, _TEST_RESIZE),
            A.Normalize(mean=list(mean), std=list(std), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def _infer_model_proba(model, tta_n, base_augs):
    model.eval()
    preds = []
    crop_size = int(round(IMAGE_SIZE * 0.92))

    with torch.inference_mode():
        for single_image_name in tqdm(
            test_image_list, total=len(test_image_list), desc=f"Infer (TTA={tta_n})"
        ):
            image = Image.open(
                os.path.join(TEST_IMAGE_PATH, single_image_name)
            ).convert("RGB")
            base = base_augs(image=np.array(image))["image"]

            if tta_n <= 1:
                batch = base.unsqueeze(0).to(DEVICE0)
                logits = model(batch)
                proba = F.softmax(logits[0], dim=0).detach().cpu()
                preds.append(proba)
                continue

            crops = _five_crop_tensors(base, crop_size=crop_size)
            crops = crops[: int(tta_n)]
            batch = torch.stack(crops, dim=0).to(DEVICE0)
            logits = model(batch)
            proba = F.softmax(logits, dim=1).mean(dim=0).detach().cpu()
            preds.append(proba)

    return torch.stack(preds, dim=0)


my_model_1 = nn.DataParallel(my_model_1).to(DEVICE0)
my_model_2 = nn.DataParallel(my_model_2).to(DEVICE0)

test_base_augs_1 = _make_test_base_augs_for_model(my_model_1.module)
test_base_augs_2 = _make_test_base_augs_for_model(my_model_2.module)

pred_proba_1 = _infer_model_proba(my_model_1, tta_n=TTA, base_augs=test_base_augs_1)
torch.cuda.empty_cache()
pred_proba_2 = _infer_model_proba(my_model_2, tta_n=TTA, base_augs=test_base_augs_2)
torch.cuda.empty_cache()

final_proba = (pred_proba_1 * 0.43) + (pred_proba_2 * 0.57)
label = final_proba.argmax(dim=-1).numpy().astype(int)

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print("Wrote:", SUBMISSION_PATH, "rows:", len(df_submission))
print(df_submission.head())
