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

0.8945300695074041

# 6. Current score

0.05568

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.57474) has done: 'I fix the Albumentations v2 API break by updating `RandomResizedCrop` to use the new `size=(h, w)` signature (this removes the runtime ValidationError). Then I make the weight-loading robust by pointing `INPUT_PATH` to the provided dataset directory and by adding a safe fallback: if the external `.pth` files are not present, the code still run end-to-end by using ImageNet-pretrained weights for the same architectures (keeping the ensemble logic intact) so a valid `submission.csv` is always produced. I also ensure test images are processed in the exact order of `sample_submission.csv` to avoid any potential misalignment between filenames and predictions. Finally, I make the script work on CPU-only environments by selecting an available device instead of assuming CUDA.'
- What this solution (achieved 0.58371) has done: 'I fix the Albumentations v2 API break causing `A.Cutout` to crash by replacing it with a v2-compatible cutout-style transform (implemented via `CoarseDropout`) while keeping the augmentation intent intact. I also correct the focal-loss helper (even though it isn’t used in inference) to avoid incorrect math if you later enable training, without changing any current inference behavior. Finally, I keep paths and ensemble logic unchanged but make sure the script runs end-to-end and writes a properly formatted `submission.csv` aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.57848) has done: 'Your current score (0.58371) is far below the target (0.89453), and the main reason is that the intended fine-tuned `.pth` weights are not being loaded, so you’re effectively ensembling mostly ImageNet-pretrained (randomly re-initialized head) models. To move the score sharply toward the target with minimal logic change, I (1) fix the dataset paths to the actual provided `/kaggle/input/...` locations, and (2) make weight loading robust to common checkpoint formats by extracting `state_dict` from wrapper dicts and verifying that head weights are actually loaded (otherwise warn). I also (3) fix test-time augmentation to be deterministic-per-pass by using `A.ReplayCompose` so each model gets consistent TTA across passes (same core TTA idea, just correct/reproducible), and (4) use a DataLoader for batched inference to reduce overhead while keeping the same per-image augmentation semantics and predictions. These changes are directly aimed at getting you into the target band by ensuring the correct trained weights are used and inference is consistent.'
- What this solution (achieved 0.57399) has done: 'Your score gap to the target is large, and the main likely cause is still that the fine-tuned `.pth` weights are not actually being found/loaded (so you’re ensembling ImageNet-backbone features with a randomly initialized 5-class head). I make the smallest change that directly increases the chance of loading the intended checkpoints by (1) auto-discovering the checkpoint files anywhere under `/kaggle/input/` when they are not in `INPUT_PATH`, and (2) adding lightweight diagnostics that confirm whether head weights were loaded. I also keep your exact model/ensemble/TTA logic, but make test-time preprocessing a bit more faithful to common cassava inference by removing the extra random geometric flips/transposes in test TTA (these can hurt accuracy when the model wasn’t trained with identical test-time randomness), while still keeping `TTA` via multi-crop/resize randomness. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.56876) has done: 'Your score is far below the target, and the most plausible cause is that the intended fine-tuned checkpoints are still not being applied correctly because key names don’t match the timm model (common issue: checkpoints saved from `DataParallel`, from a wrapper model, or with different module prefixes). I keep your exact ensemble and TTA logic, but make weight loading “key-compatible” by automatically remapping common prefixes (e.g., `model.`, `net.`, `module.`) and by translating head key patterns (`fc.*` / `classifier.*`) to whatever the current timm model uses, so the trained 5-class head actually loads when present. I also add a strict sanity check that aborts the “head not loaded” silent failure by printing a clear diagnostic (but still running end-to-end), because that exact failure mode typically produces ~0.57–0.58 accuracy. No changes to architecture, inference pipeline, or submission format—only checkpoint compatibility improvements.'
- What this solution (achieved 0.56278) has done: 'Your score is far below the target, so we should cautiously increase accuracy without changing the model/ensemble design. The biggest issue is your test-time augmentation uses `ReplayCompose` but you never “replay” the same sampled transforms; each pass is just another random draw, which can add harmful noise for a model not trained for that exact randomness. I keep the exact TTA concept and count, but implement proper replayed TTA per image (sample once, then replay the same transform), and I also apply the same TTA count to model_1 for consistency (still the same architecture and ensemble weights). Finally, I ensure the model outputs are treated consistently as logits by averaging logits then taking `argmax` (no metric/semantics change, just reduces variance).'
- What this solution (achieved 0.51682) has done: 'Your gap to the target is large (0.56278 → 0.89453), and the most likely blocker is still that the intended fine-tuned checkpoints are not being correctly applied—especially the classification head—so the ensemble behaves like near-random heads. I make one score-relevant, minimal change: align the timm classifier/head module name to the checkpoint before loading (for ResNeXt, timm models usually use `fc`, but some training code saves to `classifier` or `head.fc`), and explicitly remap/rename the model’s head to match detected checkpoint keys so the 5-class head loads cleanly. I also remove the L2-normalization of logits before ensembling (keep logits averaging), because normalizing logits destroys class-margin information and commonly hurts accuracy for argmax-based metrics. Everything else (architectures, TTA count, averaging, submission format/paths) remains the same and still produces `submission.csv`.'
- What this solution (achieved 0.52205) has done: 'Your current score is far below the target, so the most likely reason is still that the fine-tuned checkpoints are not being loaded and you’re effectively predicting with randomly initialized 5-class heads. I make a minimal, score-relevant change to aggressively locate and use the intended `.pth` weights by (1) auto-discovering them under `/kaggle/input/` when missing, and (2) handling more real-world checkpoint formats (including nested `ema_state_dict` and `DataParallel`-wrapped keys) plus a safer head-key remap. I also make inference numerically/semantically consistent across `DataParallel` vs non-parallel by always calling the underlying module correctly and ensuring logits shapes are handled robustly, without changing the ensemble, TTA count, or submission format. These changes should move accuracy upward toward the target band while preserving your core pipeline.'
- What this solution (achieved 0.05493) has done: 'Your current score is far below the target, so the most likely issue is still “wrong weights loaded / head not loaded”, which makes predictions near-random. I make a minimal, score-relevant change: ensure the timm models are instantiated with the correct `num_classes=5` head from the start (so checkpoint keys match cleanly), and then load checkpoints via `timm.models.load_checkpoint` when possible (it handles many timm training formats). I also fix one accuracy-hurting bug in your test pipeline: `ReplayCompose` currently applies the *same* transform repeatedly per image (so it’s not real TTA); we switch to independent draws per pass while keeping TTA count and augmentation family intact. Everything else (two-model ensemble, logits averaging, submission format, paths) remains the same.'
- What this solution (achieved 0.05194) has done: 'Your current score (0.05493) is extremely far below the target (0.89453), which strongly indicates a fundamental inference bug rather than “slightly suboptimal” modeling. The smallest, core-logic-preserving fix is to correct the Albumentations input type: in your DataLoader, the numpy image is being silently mangled because default collation converts it to a torch tensor, then you call `.numpy()` on it in a way that breaks the expected `HWC uint8` format for Albumentations. I keep the exact same models, weights-loading, TTA count, and logits-averaging, but change the test loader to return the raw numpy image (no torch collation) and ensure every augmentation sees a proper `np.uint8 (H,W,C)` image. This should move the score sharply upward toward the target band by making predictions actually reflect the trained preprocessing rather than near-random artifacts, while still producing the same `submission.csv` schema.'
- What this solution (achieved 0.05568) has done: 'Your current score (0.05194) is far below the target (0.89453), which indicates predictions are effectively random; the most likely cause left in this script is that test-time preprocessing is mismatched to what the fine-tuned models expect. I make the smallest score-relevant change by using a deterministic, standard validation-style transform for inference (Resize+CenterCrop+Normalize) instead of random `RandomResizedCrop` inside TTA, while keeping TTA count, ensemble weights, and logits-averaging exactly the same. This preserves your core logic (same models, same checkpoints, same TTA loop) but removes harmful randomness that can destroy accuracy when the model wasn’t trained with identical random test-time crops. I also keep the numpy-collate fix and ensure the submission remains aligned to `sample_submission.csv` order and writes `submission.csv`.'
- What this solution (achieved 0.05568) has done: 'Your current score (0.05568) is so far below the target (0.89453) that this is almost certainly a fundamental submission/prediction alignment bug rather than “model quality”. The smallest score-relevant fix is to guarantee that the `image_id` order used for inference matches exactly the `sample_submission.csv` order, and that the submission uses the *same* ordered list that the DataLoader actually iterated over (preventing any silent mismatch). I also add a hard assertion that the predicted row count matches the sample submission row count and that filenames line up, so we fail fast instead of producing a valid-looking but misaligned CSV. These changes preserve your exact models, weights-loading, transforms, TTA loop, and ensembling, but should move accuracy sharply upward toward the target if misalignment was the cause.'
- What this solution (achieved 0.05568) has done: 'Your 0.05568 accuracy is consistent with a fundamental inference bug: you are feeding the models un-rescaled `uint8` images into `A.Normalize`, but in Albumentations v2 `Normalize` defaults `max_pixel_value=1.0` (unlike v1 patterns), so your inputs get scaled incorrectly by ~255× and predictions become near-random. To move sharply toward the target with minimal change and identical core logic, I only fix normalization to explicitly use `max_pixel_value=255.0` in `valid_augs` (already correct in `test_augs`) and in the training pipeline’s `A.Normalize` call for consistency. Everything else (models, checkpoints, TTA loop, ensembling, ordering, and CSV writing) is kept the same.'

# 9. Code solution

## === cell 0
import os
import math
import random

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from tqdm import tqdm
import timm



## === cell 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()



## === cell 2
INPUT_PATH = "/kaggle/input/ensemble-1023/"
if not os.path.isdir(INPUT_PATH):
    INPUT_PATH = "/kaggle/input/cassava-leaf-disease-classification/"

TRAIN_CSV_PATH = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
SAMPLE_SUB_PATH = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

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
TTA = 3



## === cell 3
if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    test_image_list = sample_sub["image_id"].astype(str).values
else:
    test_image_list = np.asarray(
        sorted([image_name for image_name in os.listdir(TEST_IMAGE_PATH)])
    )




## === cell 4
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    """
    Score-neutral for this inference notebook: correct focal-loss computation (only used if training is enabled later).
    """

    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor) + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = y_true.to(dtype=y_hat.dtype, device=y_hat.device)
    y_true = smooth(y_true, smooth_factor)

    bce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1.0 - y_true) * (1.0 - p)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * bce, dim=-1)




## === cell 5
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




## === cell 6
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


x = [i for i in range(NUM_EPOCHS)]
y = [lr_tune(i) for i in x]
plt.plot(x, y)



## === cell 7
_cutout_like = A.CoarseDropout(
    num_holes_range=(1, 8),
    hole_height_range=(IMAGE_SIZE // 16, IMAGE_SIZE // 5),
    hole_width_range=(IMAGE_SIZE // 16, IMAGE_SIZE // 5),
    fill=0,
    p=0.5,
)

train_augs = A.Compose(
    [
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
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
        _cutout_like,
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



## === cell 8
test_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)



## === cell 9
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True, num_classes=OUT_FEATURES)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True, num_classes=OUT_FEATURES)




## === cell 10
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "checkpoint",
            "ema_state_dict",
            "ema",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _find_checkpoint_anywhere(filename: str, root: str = "/kaggle/input"):
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def _strip_known_prefixes(state):
    prefixes = ("module.", "model.", "net.", "encoder.", "backbone.")
    out = state
    changed = True
    while changed:
        changed = False
        if not isinstance(out, dict) or not out:
            break
        keys = list(out.keys())
        for p in prefixes:
            if all(k.startswith(p) for k in keys):
                out = {k[len(p) :]: v for k, v in out.items()}
                changed = True
                break
    if any(k.startswith("module.") for k in out.keys()):
        out = {k.replace("module.", "", 1): v for k, v in out.items()}
    return out


def _try_load_weights(model: nn.Module, weights_path: str):
    """
    Score-relevant: prefer timm's loader first (handles many timm checkpoint formats),
    then fall back to robust manual state_dict loading.
    """
    resolved_path = weights_path

    if not os.path.exists(resolved_path):
        alt = _find_checkpoint_anywhere(
            os.path.basename(weights_path), root="/kaggle/input"
        )
        if alt is not None:
            print(f"[INFO] Found weights via search: {alt}")
            resolved_path = alt

    if not os.path.exists(resolved_path):
        print(
            f"[WARN] Weights not found: {weights_path} (using current model weights)."
        )
        return model

    try:
        from timm.models import load_checkpoint as _timm_load_checkpoint

        _timm_load_checkpoint(model, resolved_path, strict=False)
        print(f"[INFO] Loaded checkpoint via timm.load_checkpoint: {resolved_path}")
        return model
    except Exception as e:
        print(
            f"[WARN] timm.load_checkpoint failed ({type(e).__name__}): {e}. Falling back."
        )

    ckpt = torch.load(resolved_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state.keys()):
        print(
            f"[WARN] Unrecognized checkpoint format at {resolved_path} (using current model weights)."
        )
        return model

    state = _strip_known_prefixes(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    model_sd = model.state_dict()
    likely_head_loaded = any(
        (
            k in state
            and k in model_sd
            and tuple(state[k].shape) == tuple(model_sd[k].shape)
            and len(state[k].shape) == 2
            and state[k].shape[0] == OUT_FEATURES
        )
        for k in state.keys()
    )

    if likely_head_loaded:
        print(
            f"[INFO] Loaded checkpoint (manual; head likely compatible): {resolved_path}"
        )
    else:
        print(
            f"[WARN] Loaded checkpoint (manual) but head compatibility unclear: {resolved_path}"
        )

    if missing:
        print(
            f"[WARN] Missing keys when loading {resolved_path}: {missing[:5]}{'...' if len(missing)>5 else ''}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading {resolved_path}: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
        )

    return model




## === cell 11
torch.cuda.empty_cache()

my_model_1 = _try_load_weights(my_model_1, os.path.join(INPUT_PATH, RESNEXT_PATH))
my_model_2 = _try_load_weights(my_model_2, os.path.join(INPUT_PATH, B4_PATH))

my_model_1.eval()
my_model_2.eval()

if torch.cuda.is_available() and N_GPU > 1:
    my_model_1 = nn.DataParallel(my_model_1).to(DEVICE)
    my_model_2 = nn.DataParallel(my_model_2).to(DEVICE)
else:
    my_model_1 = my_model_1.to(DEVICE)
    my_model_2 = my_model_2.to(DEVICE)




## === cell 12
def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)




## === cell 13
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self, image_names, image_dir):
        self.image_names = list(image_names)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img = _load_image(os.path.join(self.image_dir, name))
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        return name, img


def _collate_keep_numpy(batch):
    names = [b[0] for b in batch]
    imgs = [b[1] for b in batch]
    return names, imgs


test_ds = TestDataset(test_image_list, TEST_IMAGE_PATH)
test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=_collate_keep_numpy,
)


def _predict_with_tta(model, tta_count: int, desc: str):
    preds = []
    seen_names = []
    with torch.no_grad():
        for names, imgs in tqdm(test_loader, desc=desc):
            seen_names.append(names[0])

            img_np = imgs[0]
            if isinstance(img_np, torch.Tensor):
                img_np = img_np.detach().cpu().numpy()
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8, copy=False)

            ans = torch.zeros(OUT_FEATURES, device=DEVICE)

            for _t in range(tta_count):
                out_aug = test_augs(image=img_np)
                test_image = out_aug["image"].unsqueeze(0).to(DEVICE)
                out = model(test_image)
                ans += out.view(-1)

            if tta_count > 1:
                ans /= float(tta_count)

            preds.append(ans.detach().cpu())

    predictions = torch.stack(preds, dim=0)
    return predictions, seen_names


predictions_1, seen_1 = _predict_with_tta(my_model_1, TTA, desc="Predict model_1")
torch.cuda.empty_cache()

predictions_2, seen_2 = _predict_with_tta(my_model_2, TTA, desc="Predict model_2")

assert seen_1 == seen_2, "Mismatch in test iteration order between models."

final_pred = (predictions_1 * 0.5) + (predictions_2 * 0.5)

label = final_pred.argmax(dim=-1).numpy().astype(int)
label_list = list(label)



## === cell 14
df_submission = pd.DataFrame({"image_id": seen_1, "label": label_list})

if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    assert len(df_submission) == len(
        sample_sub
    ), "Row count mismatch vs sample_submission.csv"
    assert (
        df_submission["image_id"].values == sample_sub["image_id"].astype(str).values
    ).all(), "image_id order mismatch vs sample_submission.csv; submission would be misaligned."

df_submission.to_csv(SUBMISSION_PATH, index=False)

print(df_submission.head())
print(f"Wrote: {SUBMISSION_PATH} (rows={len(df_submission)})")
