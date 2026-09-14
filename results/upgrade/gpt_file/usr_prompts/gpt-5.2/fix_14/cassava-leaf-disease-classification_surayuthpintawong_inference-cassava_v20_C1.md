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

2.7

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.8709579933514657

# 6. Current score

0.09342

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29559) has done: 'I fix the Albumentations v2 incompatibility by replacing the custom `ToTensor` transform with the official `albumentations.pytorch.ToTensorV2`, which provides the required `available_keys`. I also make checkpoint loading robust: if the specified `.pt` file is missing in `../input/`, the code fall back to using ImageNet-pretrained EfficientNet-B4 weights so the notebook can still run end-to-end and produce a valid `submission.csv`. Finally, I prevent the downstream `NameError` by ensuring inference always runs (even without the checkpoint) and by aligning the output rows to `sample_submission.csv` order to avoid any accidental mismatch.'
- What this solution (achieved 0.19581) has done: 'I fix the DataLoader crash by filtering out directory entries inside `test_images/` (the dataset currently tries to `imread()` a subfolder named `test_images`, causing the “reading folders” OSError). I also make image reading more robust by falling back to OpenCV if `skimage` can’t read a file, without changing the model or inference logic. To move your score toward the target, I ensure the checkpoint path is searched in the common Kaggle input locations for this dataset (so the intended trained weights load when available, instead of using an ImageNet backbone with a random 5-class head that yields ~0.29). The submission creation stays the same and is still aligned to `sample_submission.csv`.'
- What this solution (achieved 0.3651) has done: 'Your current score is far below the target, so the most likely issue is that you are *not actually using the trained checkpoint weights*, and/or your test-time preprocessing doesn’t match what the checkpoint expects (512-center-crop is a common mismatch). I make the checkpoint search more robust by recursively looking for the intended `efficientnet-b4-e10.pt` under the Kaggle input dataset tree, so the model uses the trained head instead of a random 5-class head. Then, as a minimal and safe accuracy improvement, I switch test preprocessing from `CenterCrop(512)` to the ImageNet-standard `Resize(512)->CenterCrop(380)` which matches EfficientNet-B4’s typical training/inference resolution and normalization, without changing the model or inference loop. Finally, I keep the submission aligned to `sample_submission.csv` and add a hard assertion on row count to catch silent mismatches that can tank accuracy.'
- What this solution (achieved 0.08632) has done: 'Your current score (0.3651) is far below the target (0.87096), so we should make a small, low-risk change that increases accuracy without changing the model or training logic. The most likely remaining mismatch is test-time preprocessing: if the checkpoint was trained with the common Cassava pipeline (random resized crop to 512 during training and *plain resize to 512 at inference*), then using a 380 center-crop at test time can throw away useful leaf context and hurt accuracy. I keep the same EfficientNet-B4 model and checkpoint loading, but change only the inference transform to `Resize(512,512)` (no crop) with the same normalization/tensor conversion, and I also set deterministic seeds to reduce run-to-run variance (this won’t lower accuracy, it just stabilizes it). Submission writing and alignment to `sample_submission.csv` remain identical.'
- What this solution (achieved 0.08595) has done: 'Your score (0.08632) is far below the target (0.87096), and the most likely cause is that the model is not actually using the trained 5-class head weights (checkpoint partially loads with `strict=False`, which can silently leave the classifier random). I keep the same EfficientNet-B4 architecture and the same inference loop, but make checkpoint loading *strict when possible* and explicitly error out/fallback only when shapes don’t match, so we don’t submit near-random predictions. I also add a tiny, safe improvement to test-time prediction by applying softmax before argmax (should be identical for argmax, but helps guard against any NaN/inf logits by normalizing) and enable `torch.cuda.amp.autocast` for numerical stability on GPU without changing semantics. Submission formatting and alignment to `sample_submission.csv` remains identical.'
- What this solution (achieved 0.08595) has done: 'Your score is extremely low for this competition, which is most consistent with an inference-time preprocessing mismatch (channel order) and/or the model checkpoint not being applied as intended. I keep your exact model and inference loop, but make the smallest high-impact fix: convert every loaded image to RGB explicitly (skimage often returns RGB but OpenCV returns BGR, and mixed behavior can destroy accuracy). I also harden checkpoint loading to handle the very common “state_dict nested under model / net / ema” patterns without changing semantics, so the intended trained weights are more likely to be used. These two changes should move accuracy upward toward your target while preserving your core approach and producing the same submission format.'
- What this solution (achieved 0.08595) has done: 'Your score is far below the target, which strongly suggests the test preprocessing still doesn’t match what the checkpoint expects, so predictions look close to random. I keep your exact model, checkpoint loading logic, and inference loop, but make one minimal, high-impact adjustment: switch inference normalization from ImageNet mean/std to the Cassava-competition common mean/std (trained weights for this dataset are often saved with that). To avoid harming performance if the checkpoint actually was trained with ImageNet normalization, I run inference twice (ImageNet-norm and Cassava-norm) and pick the one with higher accuracy on a small train-split validation set (no training, just forward passes), then use that transform for the test submission. This stays within your core approach (same architecture, no training changes) and should move accuracy upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.08595) has done: 'Your current score is far below the target, so we should make the smallest changes that plausibly fix “near-random” predictions without altering your model or inference loop. The most likely remaining issue is that the checkpoint’s state_dict keys don’t match torchvision EfficientNet-B4’s module names (common when training with `timm`), so your filtered non-strict load silently drops most weights and leaves the model effectively untrained. I add a minimal key-remapping step for the most common `timm`→`torchvision` EfficientNet naming differences, and I additionally hard-fail (fall back to ImageNet weights) if the mapped load still doesn’t actually load the classifier head. This keeps your architecture and pipeline intact, but should move accuracy sharply upward toward the target when the checkpoint is present and compatible.'
- What this solution (achieved 0.08595) has done: 'Your score is far below the target, which is most consistent with the checkpoint not being correctly applied (so the model behaves close to random) and/or a subtle preprocessing mismatch. I keep your exact model/inference pipeline, but make checkpoint loading more robust by (1) handling common `.pt` formats that store weights under `model.state_dict()` / `model.module.state_dict()`, (2) adding a minimal EfficientNet key-remap for the very common “timm with `act1`/`act2` names” case, and (3) hard-failing back to ImageNet weights unless we can confirm a meaningful fraction of backbone weights loaded (to avoid silently submitting random heads/backbones). This is a minimal change focused on making sure you actually use trained weights when present, which is the highest-likelihood path toward the target accuracy. Submission formatting and ordering remain unchanged.'
- What this solution (achieved 0.08595) has done: 'Your score is near-random, so the smallest likely root cause is that the checkpoint isn’t actually being loaded into the torchvision EfficientNet-B4 due to incompatible key naming (timm-trained checkpoints often don’t match torchvision’s module structure). I keep your architecture/inference loop intact, but strengthen the checkpoint compatibility by first trying to build the model via `torch.hub.load('rwightman/gen-efficientnet-pytorch', ...)` (timm-style EfficientNet) when a timm-style checkpoint is detected, and only fall back to torchvision if that fails. This is a targeted change to ensure your *trained* weights are used (the biggest lever toward your 0.87 target) while preserving the same preprocessing selection and submission alignment. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.08595) has done: 'Your score is near-random, so the smallest high-impact fix is to ensure the image tensor matches what EfficientNet expects: RGB, exactly 3 channels, and `uint8` before Albumentations normalization. Right now, `skimage` may return `float` images in `[0,1]` while you normalize with `max_pixel_value=255`, which can wreck inputs and collapse accuracy. I make a minimal, shared “read image as RGB uint8” function used by both datasets (no model/training loop changes), keeping your checkpoint logic and transform-selection logic intact. This should move accuracy sharply upward toward your target while still producing the same valid `submission.csv`.'
- What this solution (achieved 0.09342) has done: 'Your score is near-random, so the most likely issue is still “the checkpoint isn’t actually being used at inference time” even if a file is found, because (a) `torch.hub.load(...)` can fail offline on Kaggle and triggers a fallback to an untrained 5-class head, and (b) the current “filtered load” can report an OK ratio while still missing critical classifier/head weights. I make two minimal, high-impact changes: first, remove the network-dependent hub path and instead load into the existing torchvision EfficientNet-B4 by adapting common timm-style key patterns into torchvision-style keys; second, add a strict sanity check that the classifier weights are loaded (correct shape) and otherwise treat the checkpoint as unusable (falling back to ImageNet) to avoid silently submitting random outputs. This preserves your model family/inference loop and only touches checkpoint compatibility and validation of loading, which is directly tied to accuracy. The script still run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import warnings
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage import io

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import cv2

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

model_full_name = "efficientnet-b4-e10"
model_name = "efficientnet-b4"
folder_name = "effnetmodelv18"

TEST_DIR = "../input/cassava-leaf-disease-classification/test_images/"
SAMPLE_SUB_PATH = "../input/cassava-leaf-disease-classification/sample_submission.csv"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMG_DIR = "../input/cassava-leaf-disease-classification/train_images/"

CKPT_CANDIDATES = [
    os.path.join("../input", folder_name, model_full_name + ".pt"),
    os.path.join(
        "../input",
        "cassava-leaf-disease-classification",
        folder_name,
        model_full_name + ".pt",
    ),
    os.path.join(
        "../input", "cassava-leaf-disease-classification", model_full_name + ".pt"
    ),
]


def find_ckpt_recursive(search_root, filename):
    if not os.path.isdir(search_root):
        return None
    for root, dirs, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    return None


CKPT_PATH = None
for p in CKPT_CANDIDATES:
    if os.path.exists(p):
        CKPT_PATH = p
        break

if CKPT_PATH is None:
    CKPT_PATH = find_ckpt_recursive("../input", model_full_name + ".pt")

print("Using device:", device)
print("Test dir exists:", os.path.isdir(TEST_DIR), TEST_DIR)
print("Checkpoint found:", CKPT_PATH if CKPT_PATH is not None else "None")




## === cell 1
def make_transform(mean, std):
    return A.Compose(
        [
            A.Resize(height=512, width=512, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )


transform_imagenet = make_transform(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

transform_cassava = make_transform(
    mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
)




## === cell 2
def _read_image_rgb_uint8(img_path):
    image = None
    try:
        image = io.imread(img_path)
    except Exception:
        image = None

    if image is None:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError("Failed to read image: {}".format(img_path))
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = np.ascontiguousarray(image)

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                mx = float(np.nanmax(image)) if image.size else 0.0
                if mx <= 1.5:
                    image = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                else:
                    image = np.clip(image, 0.0, 255.0).round().astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        entries = sorted(os.listdir(root_dir))
        images = []
        for f in entries:
            fp = os.path.join(root_dir, f)
            if os.path.isfile(fp):
                images.append(f)
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = _read_image_rgb_uint8(img_path)

        if self.transform:
            out = self.transform(image=image)
            image = out["image"]

        return img_name, image


class TrainValDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"]
        y = int(row["label"])
        img_path = os.path.join(self.root_dir, img_name)

        image = _read_image_rgb_uint8(img_path)

        if self.transform:
            out = self.transform(image=image)
            image = out["image"]

        return image, y


print("Ready datasets/transforms.")



## === cell 3
if hasattr(torchvision.models, "EfficientNet_B4_Weights"):
    imagenet_weights = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
else:
    imagenet_weights = None


def _normalize_state_dict_keys(sd):
    new_state = {}
    for k, v in sd.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v
    return new_state


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "ema",
            "model_ema",
        ]:
            if key in ckpt_obj:
                v = ckpt_obj[key]
                if isinstance(v, dict):
                    for k2 in ["state_dict", "model_state_dict", "model"]:
                        if k2 in v and isinstance(v[k2], dict):
                            return v[k2]
                    return v
                if hasattr(v, "state_dict") and callable(v.state_dict):
                    try:
                        return v.state_dict()
                    except Exception:
                        pass
    if hasattr(ckpt_obj, "state_dict") and callable(ckpt_obj.state_dict):
        try:
            return ckpt_obj.state_dict()
        except Exception:
            pass
    return ckpt_obj


def _looks_like_timm_efficientnet(sd):
    if not isinstance(sd, dict) or len(sd) == 0:
        return False
    keys = list(sd.keys())
    return (
        any(k.startswith("conv_stem.") for k in keys)
        or any(k.startswith("bn1.") for k in keys)
        or any(k.startswith("blocks.") for k in keys)
        or any(k.startswith("conv_head.") for k in keys)
        or any(k in ("classifier.weight", "classifier.bias") for k in keys)
    )


def _build_torchvision_effnet_b4(weights):
    m = torchvision.models.efficientnet_b4(weights=weights)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, 5)
    return m


def _maybe_remap_timm_to_torchvision_efficientnet(sd):
    """
    Change rationale (score): avoid torch.hub/timm dependency (can fail offline) and
    increase chance the checkpoint actually loads into torchvision EfficientNet-B4,
    preventing near-random predictions.
    """
    remapped = {}
    for k, v in sd.items():
        nk = k

        nk = nk.replace("conv_stem.", "features.0.0.")
        nk = nk.replace("bn1.", "features.0.1.")
        nk = nk.replace("conv_head.", "features.8.0.")
        nk = nk.replace("bn2.", "features.8.1.")

        nk = nk.replace(".act1.", ".activation.")
        nk = nk.replace(".act2.", ".activation.")

        nk = nk.replace("classifier.weight", "classifier.1.weight")
        nk = nk.replace("classifier.bias", "classifier.1.bias")

        remapped[nk] = v
    return remapped


def _load_ckpt_strict_or_filtered(model, state_dict):
    try:
        model.load_state_dict(state_dict, strict=True)
        return True, "strict"
    except Exception as e:
        print("Strict load failed, attempting safe filtered load. Reason:", repr(e))
        model_sd = model.state_dict()
        filtered = {}
        dropped = []
        for k, v in state_dict.items():
            if (
                k in model_sd
                and hasattr(v, "shape")
                and hasattr(model_sd[k], "shape")
                and tuple(v.shape) == tuple(model_sd[k].shape)
            ):
                filtered[k] = v
            else:
                dropped.append(k)

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print("Loaded checkpoint FILTERED (non-strict). Dropped keys:", len(dropped))
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))

        model_keys = list(model_sd.keys())
        loaded_keys = list(filtered.keys())
        ratio = float(len(loaded_keys)) / float(max(1, len(model_keys)))
        print("Overall loaded key ratio:", ratio)

        return (ratio >= 0.60), "filtered"


def _classifier_loaded_ok(model, state_dict_loaded):
    """
    Change rationale (score): if classifier is not loaded, predictions are often near-random.
    We treat that as checkpoint-unusable to avoid submitting random outputs.
    """
    try:
        w_key = "classifier.1.weight"
        b_key = "classifier.1.bias"
        if w_key not in state_dict_loaded or b_key not in state_dict_loaded:
            return False
        w = state_dict_loaded[w_key]
        b = state_dict_loaded[b_key]
        return tuple(w.shape) == (5, model.classifier[1].in_features) and tuple(
            b.shape
        ) == (5,)
    except Exception:
        return False


use_ckpt = CKPT_PATH is not None

model = _build_torchvision_effnet_b4(weights=None).to(device)

if use_ckpt:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint format not understood (expected dict-like state_dict)."
        )
    state_dict = _normalize_state_dict_keys(state_dict)

    if _looks_like_timm_efficientnet(state_dict):
        print(
            "Detected timm-style EfficientNet checkpoint; remapping keys to torchvision format."
        )
        state_dict = _maybe_remap_timm_to_torchvision_efficientnet(state_dict)
    else:
        print("Checkpoint looks torchvision-style; using keys as-is.")

    ok, how = _load_ckpt_strict_or_filtered(model, state_dict)

    if (not ok) or (not _classifier_loaded_ok(model, state_dict)):
        print(
            "WARNING: checkpoint did not load reliably (or classifier missing). Using ImageNet weights + random head."
        )
        model = _build_torchvision_effnet_b4(weights=imagenet_weights).to(device)
        use_ckpt = False
    else:
        print("Loaded checkpoint into torchvision model. mode:", how)
else:
    print("Checkpoint not found in candidates or recursive search.")
    print("Falling back to ImageNet-pretrained EfficientNet-B4 (random 5-class head).")
    model = _build_torchvision_effnet_b4(weights=imagenet_weights).to(device)

model.eval()
print("Model ready.")




## === cell 4
def infer_logits(dataloader):
    all_logits = []
    all_targets = []
    use_amp = bool(use_cuda)
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device).float()
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(xb)
            else:
                out = model(xb)
            all_logits.append(out.float().cpu())
            all_targets.append(yb.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


chosen_transform = transform_imagenet
chosen_name = "imagenet"

if os.path.exists(TRAIN_CSV_PATH) and os.path.isdir(TRAIN_IMG_DIR) and use_ckpt:
    df = pd.read_csv(TRAIN_CSV_PATH)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = min(1024, max(256, int(0.08 * len(df))))
    val_df = df.iloc[:n_val].copy()

    val_ds_im = TrainValDataset(val_df, TRAIN_IMG_DIR, transform=transform_imagenet)
    val_ds_ca = TrainValDataset(val_df, TRAIN_IMG_DIR, transform=transform_cassava)

    val_loader_im = DataLoader(
        val_ds_im, batch_size=16, shuffle=False, num_workers=1, pin_memory=use_cuda
    )
    val_loader_ca = DataLoader(
        val_ds_ca, batch_size=16, shuffle=False, num_workers=1, pin_memory=use_cuda
    )

    logits_im, y = infer_logits(val_loader_im)
    acc_im = accuracy_from_logits(logits_im, y)

    logits_ca, y2 = infer_logits(val_loader_ca)
    acc_ca = accuracy_from_logits(logits_ca, y2)

    print("Validation acc (ImageNet norm):", acc_im)
    print("Validation acc (Cassava  norm):", acc_ca)

    if acc_ca > acc_im:
        chosen_transform = transform_cassava
        chosen_name = "cassava"
    else:
        chosen_transform = transform_imagenet
        chosen_name = "imagenet"
else:
    print(
        "Skipping val-based transform selection (missing train data or checkpoint). Using ImageNet norm."
    )

print("Chosen inference transform:", chosen_name)



## === cell 5
test_ds = TestDataset(root_dir=TEST_DIR, transform=chosen_transform)
testloader = DataLoader(
    test_ds, batch_size=4, shuffle=False, num_workers=1, pin_memory=use_cuda
)

print("Loaded test dataset with {} images".format(len(test_ds)))

names = []
predicted = []

use_amp = bool(use_cuda)

with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device)
        images_batch = images_batch.float()
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images_batch)
        else:
            logits = model(images_batch)

        probs = torch.softmax(logits.float(), dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(list(preds))

print("Inference done on {} images".format(len(names)))



## === cell 6
result = pd.DataFrame({"image_id": names, "label": predicted})

if os.path.exists(SAMPLE_SUB_PATH):
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    result = sample[["image_id"]].merge(result, on="image_id", how="left")
    result["label"] = result["label"].fillna(0).astype(int)

    assert len(result) == len(sample), "Submission row count mismatch: {} vs {}".format(
        len(result), len(sample)
    )
else:
    result["label"] = result["label"].astype(int)

result.to_csv("submission.csv", index=False)

print(result.head())
print("Wrote submission.csv with {} rows".format(len(result)))
print("Label value counts:\n", result["label"].value_counts(dropna=False).sort_index())
