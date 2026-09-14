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

3.9

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
seaborn==0.12.2
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

0.8933212450891508

# 6. Current score

0.60874

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing EfficientNet dependency by falling back to torchvision’s built-in EfficientNet-B7 when `efficientnet_pytorch` isn’t available, while keeping the rest of the ensemble logic intact. I also fix the Kaggle path detection so it always finds the provided `/kaggle/input/cassava-leaf-disease-classification` dataset and correctly lists test images, avoiding the current `FileNotFoundError`. Albumentations v2 API changes be addressed by updating `RandomResizedCrop`/`Rotate` usage to the new parameter names so transforms build successfully. Finally, I make the checkpoint loading and model head wiring robust (including EfficientNet key-mapping), and ensure the probability aggregation always produces a valid `(N, 5)` array so `submission.csv` is written correctly.'
- What this solution (achieved 0.11584) has done: 'Your low score is consistent with the model weights not being found/loaded, so you’re effectively submitting near-constant/random predictions (or the sample submission fallback). The minimal fix is to point `pretrained_models` to actual weight files that exist in this environment (your current `../input/densenet201-04-2019data/*.pth` and `../input/eb7m-seed70/*.pth` folders aren’t present in the provided data tree), while keeping the exact same ensemble/TTA/prediction logic. I also make the test image listing deterministic (sorted) to avoid any possible row-order mismatch with `sample_submission.csv`. Finally, I ensure we always align predictions to the official sample submission’s `image_id` order before writing `submission.csv`, which can materially affect accuracy if ordering ever differs.'
- What this solution (achieved 0.28774) has done: 'Your current score (0.11584) is far below the target (0.8933), and the most likely reason is that your script isn’t loading any real trained weights, so it falls back to essentially constant labels (the sample submission), which scores very poorly. The minimal, score-relevant fix is to stop scanning all of `/kaggle/input/**` for random `.pth/.pt` files and instead load the competition-provided trained model checkpoints if they exist; since none are in your provided tree, the next best minimal fix is to use ImageNet-pretrained weights for the same architectures (this keeps the exact same inference/ensemble/TTA pipeline but makes predictions non-random). Additionally, your ResNet/DenseNet wrappers currently create a new `fc`/`classifier` layer that is never loaded from checkpoint (so even if you had weights, the head would be random); we fix this by wiring the wrapper head to use the base model’s existing `fc/classifier` so checkpoint heads can load. Finally, we keep your sample_submission alignment merge (good) and ensure test image selection always matches sample_submission ordering.'
- What this solution (achieved 0.59828) has done: 'Your score is far below the target, and the dominant issue is that when no finetuned checkpoints exist, the current code uses ImageNet backbones but keeps a randomly-initialized 5-class head, which makes predictions nearly random (hence ~0.28). I keep your exact ensemble/TTA inference flow, but when using the ImageNet fallback I switch to using the pretrained models’ native 1000-class heads and then map those logits to 5 classes via a fixed, dataset-derived linear projection computed once from the train set (mean 1000-d logits per cassava class). This preserves the same architecture and forward pass, only adding a lightweight calibration/projection step aligned to the metric, and it stays within time limits by computing features on a small, stratified subset. I also fix the current bug where `build_torchvision_efficientnet_b7()` ignores `num_classes` and always loads ImageNet weights, and ensure submission alignment stays identical.'
- What this solution (achieved 0.58707) has done: 'Your current score is far below target, so we should cautiously improve accuracy without changing the core ensemble/TTA/inference logic. The biggest score-relevant bug is that your ImageNet fallback projection is built using *CenterCrop only* on training images, while several of your test-time transforms use `RandomResizedCrop` and `Rotate`, creating a train/test transform mismatch inside the projection mapping; I rebuild each projection using a small set of the *same* TTA transforms (averaged) so the 1000→5 mapping matches the inference distribution. I also fix a correctness hazard in `FinalLayerMixupModel.forward()` where `x.squeeze()` can drop the batch dimension for batch_size=1, and I make the projection-building deterministic by stratified sampling with a seed (same compute budget). These are minimal changes that keep the architecture and inference flow intact, but should move the score upward toward your target.'
- What this solution (achieved 0.53326) has done: 'I keep your exact ensemble + TTA inference flow, but fix two score-critical correctness issues that currently make the ImageNet fallback much weaker than it should be. First, I ensure the ImageNet-backbone projections are built and applied on the correct input resolution by adding a `Resize` before `CenterCrop/RandomResizedCrop`, because `CenterCrop(512)` on smaller images can fail or behave inconsistently and harms the learned 1000→5 mapping. Second, I correct the projection bias term computation (currently `b = -global_mean @ M` is mathematically inconsistent with the intended “centered logits” mapping), which should be `b = -(global_mean @ W)` so the projector output is properly centered. These are minimal changes that preserve your architecture, loss, loops, and submission semantics, but should move accuracy upward toward the target.'
- What this solution (achieved 0.63117) has done: 'We need to move your accuracy up toward 0.893 (current 0.533), so we should improve the *existing* ImageNet→cassava projection without changing your ensemble/TTA structure. The biggest score bottleneck is that the projection currently uses raw class-mean logits as `W` and a bias computed from a global mean; this isn’t a proper discriminative mapping and tends to underperform. I keep the same fallback idea (1000→5 projector) but compute `W,b` via a tiny, deterministic ridge-regression fit on the same small stratified subset you already sample (so runtime stays bounded), using the same TTA transforms for feature extraction. This preserves your architecture/inference semantics (still ImageNet heads + linear projector, same TTA averaging) while making the mapping materially more predictive, which should move the score upward toward the target band.'
- What this solution (achieved 0.60874) has done: 'I make two score-relevant fixes while keeping your ensemble/TTA + ImageNet→cassava linear projection core logic intact. First, the current projector is fit against one-hot labels via ridge regression, but inference applies `softmax(projected_logits)`; to better match accuracy, I fit the projector in logit-space by regressing to `log(class_priors + onehot)` (a minimal label-space change) and I L2-normalize ImageNet logits before fitting/applying (stabilizes across TTA/architectures without changing models). Second, I reduce avoidable projection noise by slightly increasing the deterministic per-class sample used to fit W,b (still small enough to stay within the 600s budget), which should move your score upward toward the 0.893 target band. All paths and submission alignment remain unchanged and the script still write `submission.csv`.'
- What this solution (achieved 0.67227) has done: 'Your current score (0.60874) is far below the target (0.8933), so the smallest safe way to move upward is to improve the existing ImageNet→cassava linear projection without changing your ensemble/TTA or model architectures. I keep your exact inference flow, but replace the current “log(onehot/prior)” ridge target with a standard ridge-regression classifier fit directly to one-hot labels (with an explicit intercept), which better matches argmax-accuracy after softmax. I also remove the feature-centering/bias coupling (mu/b) that can destabilize calibration across TTAs, and I slightly increase the per-class projection-fit sample size to reduce variance while staying within time limits. Submission ordering/alignment remains anchored to `sample_submission.csv`, and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.66517) has done: 'I make two minimal, score-relevant changes that keep your ensemble/TTA and ImageNet→cassava projection approach intact: (1) remove accidental nondeterminism from your TTA pipeline by fixing seeds per TTA transform (RandomResizedCrop/Rotate currently vary each run and also differ between projection-fit vs inference), and (2) make the projection fitting match inference by building the projector using the exact same set of TTA transforms you use at test time (currently some models use fewer TTAs when fitting than when predicting). These changes don’t alter architecture, loss, or inference semantics; they just align the learned 1000→5 mapping to what you actually feed the models at test time and stabilize results, which should move accuracy upward toward the target. All paths remain the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.60874) has done: 'Your current score (0.665) is far below the target (0.893), so we should improve the existing ImageNet→cassava projection accuracy without changing your ensemble/TTA structure or model architectures. The main score bottleneck is that the ridge projector is fit on raw logits while inference uses `softmax`, and the raw logits’ scale varies a lot across images/TTAs; we can keep the same linear projector but fit it on **log-softmax features** (more stable, closer to the decision space used for argmax) while preserving the same models and TTA loop. To reduce under/over-regularization without changing the method, we also make the ridge strength proportional to sample count (same closed-form solve), which tends to generalize better. Finally, we keep submission alignment to `sample_submission.csv` unchanged and still write `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import numpy as np
import glob




## === cell 1
def find_pretrained_models():
    patterns = [
        "../input/**/**/*.pth",
        "../input/**/**/*.pt",
        "/kaggle/input/**/**/*.pth",
        "/kaggle/input/**/**/*.pt",
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    found = [
        p for p in found if not p.endswith(".pth.tar") and not p.endswith(".pt.tar")
    ]
    return sorted(set(found))


pretrained_models = find_pretrained_models()

print(f"{len(pretrained_models)} model files found (pth/pt).")
if len(pretrained_models) > 0:
    print("\n".join(pretrained_models[:50]))
    if len(pretrained_models) > 50:
        print(f"... ({len(pretrained_models)-50} more)")



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms  # 学習済みモデル、画像変換
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import json
import time
import pickle


from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 3
import sys

_EFFNET_BACKEND = None
EfficientNet = None
try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet as _EfficientNetPyTorch

    EfficientNet = _EfficientNetPyTorch
    _EFFNET_BACKEND = "efficientnet_pytorch"
    print("Using efficientnet_pytorch backend.")
except Exception as e:
    _EFFNET_BACKEND = "torchvision"
    print(
        f"efficientnet_pytorch not found; will use torchvision efficientnet instead. ({type(e).__name__}: {e})"
    )



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
CANDIDATE_BASE_DIRS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
BASE_DIR = None
for p in CANDIDATE_BASE_DIRS:
    if os.path.isdir(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    BASE_DIR = "data"

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE")
print(f"KAGGLE_KERNEL_RUN_TYPE={run_type}")

if os.path.isdir(f"{BASE_DIR}/test_images"):
    TEST_PATH = f"{BASE_DIR}/test_images"
    test_files = sorted(os.listdir(TEST_PATH))
    print("Using test_images.")
elif os.path.isdir(f"{BASE_DIR}/train_images"):
    TEST_PATH = f"{BASE_DIR}/train_images"
    test_files = sorted(os.listdir(TEST_PATH))[:32]
    print("Using train_images (subset) because test_images not found.")
else:
    raise FileNotFoundError(
        f"Neither test_images nor train_images found under BASE_DIR={BASE_DIR}"
    )

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.Rotate(limit=(-30, 30), p=1.0),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.Rotate(limit=(-30, 30), p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 10
def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v
    return cleaned


def load_checkpoint_safely(model_or_wrapper, ckpt_path, strict=False):
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        state = _clean_state_dict_keys(state)
    missing, unexpected = model_or_wrapper.load_state_dict(state, strict=strict)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"load_state_dict(strict={strict}) missing={len(missing)} unexpected={len(unexpected)} for {os.path.basename(ckpt_path)}"
        )
    return missing, unexpected




## === cell 11
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.fc = model.fc

        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)

            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs  # torch.Size([64, 3, 256, 256])
        x2 = inputs[index]  # torch.Size([64, 3, 256, 256])

        x1 = self.convlayer(x1)  # torch.Size([64, 512, 1, 1])
        x2 = self.convlayer(x2)  # torch.Size([64, 512, 1, 1])

        mixed_x = lam * x1 + (1 - lam) * x2  # torch.Size([64, 512, 1, 1])
        mixed_x = torch.flatten(mixed_x, 1)  # torch.Size([64, 512])
        outputs = self.fc(mixed_x)  # torch.Size([64, 5])

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 12
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        self.fc = model.classifier

        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)

            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs  # torch.Size([12, 3, 512, 512])
        x2 = inputs[index]  # torch.Size([12, 3, 512, 512])

        x1 = self.convlayer(x1)  # torch.Size([12, 1920, 16, 16])
        x2 = self.convlayer(x2)  # torch.Size([12, 1920, 16, 16])

        x1 = self.AdaptiveAvgPool2d(x1)  # torch.Size([12, 1920, 1, 1])
        x2 = self.AdaptiveAvgPool2d(x2)  # torch.Size([12, 1920, 1, 1])

        mixed_x = lam * x1 + (1 - lam) * x2  # torch.Size([64, 1920, 1, 1])
        mixed_x = torch.flatten(mixed_x, 1)  # torch.Size([64, 1920])
        outputs = self.fc(mixed_x)  # torch.Size([64, 5])

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 13
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        self.criterion = criterion
        self.alpha = alpha

        if hasattr(model, "_fc"):
            num_ftrs = model._fc.in_features
            model._fc = nn.Linear(num_ftrs, num_classes)
            self.model = model
            self._backend = "efficientnet_pytorch"
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            else:
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            self.model = model
            self._backend = "torchvision"
        else:
            raise AttributeError(
                "Unsupported EfficientNet model structure (no _fc or classifier)."
            )

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        print("ここにきてはいけない")
        sys.exit()




## === cell 14
def build_torchvision_efficientnet_b7(num_classes, imagenet_head=True):
    m = torchvision.models.efficientnet_b7(
        weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1
    )
    if not imagenet_head:
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m




## === cell 15
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")  # (H, W, C) の numpy.ndarray
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR => RGB に変換
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        img = self.load_image(image_id)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, image_id




## === cell 16
class LogitProjector(nn.Module):
    def __init__(self, W, b=None, normalize_inputs=True, eps=1e-8):
        super().__init__()
        W = torch.as_tensor(W, dtype=torch.float32)
        self.register_buffer("W", W)  # (1000, 5)
        if b is None:
            b = torch.zeros((W.shape[1],), dtype=torch.float32)
        b = torch.as_tensor(b, dtype=torch.float32)
        self.register_buffer("b", b)

        self.normalize_inputs = normalize_inputs
        self.eps = eps

    def forward(self, logits_1000):
        if self.normalize_inputs:
            denom = torch.linalg.norm(
                logits_1000, ord=2, dim=1, keepdim=True
            ).clamp_min(self.eps)
            logits_1000 = logits_1000 / denom
        return logits_1000 @ self.W + self.b


def _build_projection_from_train(
    arch_name,
    train_csv_path,
    train_images_dir,
    transform_for_features,
    max_per_class=40,
    batch_size=16,
):
    df_train = pd.read_csv(train_csv_path)
    df_train["label"] = df_train["label"].astype(int)

    rng = np.random.RandomState(SEED)
    parts = []
    for c in range(num_classes):
        dfi = df_train[df_train["label"] == c]
        if len(dfi) > max_per_class:
            take_idx = rng.choice(len(dfi), size=max_per_class, replace=False)
            dfi = dfi.iloc[take_idx]
        parts.append(dfi)
    df_small = pd.concat(parts, axis=0).reset_index(drop=True)

    class TrainSmallDataset(data.Dataset):
        def __init__(self, df, transform=None):
            self.image_ids = df.image_id.tolist()
            self.labels = df.label.tolist()
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            image_id = self.image_ids[idx]
            label = int(self.labels[idx])
            img = cv2.imread(f"{train_images_dir}/{image_id}")
            if img is None:
                raise FileNotFoundError(
                    f"Failed to read image: {train_images_dir}/{image_id}"
                )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, label

    if arch_name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif arch_name == "densenet201":
        base = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    elif arch_name in ["efficientnet-b7", "efficientnetb7", "eb7"]:
        base = build_torchvision_efficientnet_b7(
            num_classes=num_classes, imagenet_head=True
        )
    else:
        raise ValueError(f"Unsupported arch for projection: {arch_name}")

    base.to(device)
    base.eval()
    torch.set_grad_enabled(False)

    ds = TrainSmallDataset(df_small, transform=transform_for_features)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    X_list = []
    y_list = []

    for xb, yb in tqdm(dl, desc=f"Build projection ({arch_name})"):
        xb = xb.to(device)
        logits = base(xb)  # (B, 1000)

        feats = torch.log_softmax(logits, dim=1)

        X_list.append(feats.detach().float().cpu().numpy())
        y_list.append(yb.detach().cpu().numpy().astype(int))

    X = np.concatenate(X_list, axis=0)  # (N, 1000)
    y = np.concatenate(y_list, axis=0)  # (N,)

    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = np.clip(X_norm, 1e-8, None)
    X = X / X_norm

    Y = np.eye(num_classes, dtype=np.float32)[y]  # (N, 5)

    X1 = np.concatenate(
        [X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1
    )  # (N,1001)

    lam0 = 3.0
    lam = lam0 * (X1.shape[0] / 200.0)

    XtX = X1.T @ X1  # (1001,1001)
    XtY = X1.T @ Y  # (1001,5)
    W1 = np.linalg.solve(
        XtX + lam * np.eye(XtX.shape[0], dtype=XtX.dtype), XtY
    )  # (1001,5)

    W = W1[:-1, :]  # (1000,5)
    b = W1[-1, :]  # (5,)

    del base
    torch.cuda.empty_cache()
    return W.astype(np.float32), b.astype(np.float32)


def _build_projection_from_train_tta(
    arch_name,
    train_csv_path,
    train_images_dir,
    transforms_for_features,
    max_per_class=35,
    batch_size=12,
):
    Ms = []
    bs = []
    for tid, tfm in enumerate(transforms_for_features):
        seed_everything(SEED + 1000 + tid)

        M, b = _build_projection_from_train(
            arch_name=arch_name,
            train_csv_path=train_csv_path,
            train_images_dir=train_images_dir,
            transform_for_features=tfm,
            max_per_class=max_per_class,
            batch_size=batch_size,
        )
        Ms.append(M)
        bs.append(b)
    M_mean = np.mean(np.stack(Ms, axis=0), axis=0)
    b_mean = np.mean(np.stack(bs, axis=0), axis=0)
    return M_mean, b_mean




## === cell 17
def predict_model(basename, net, dataloader, projector=None, tta_id=0):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    projector: optional LogitProjector mapping ImageNet-1000 logits to 5 classes
    """

    model_start_time = time.time()

    net.to(device)
    net.eval()  # 検証モード
    torch.set_grad_enabled(False)

    seed_everything(SEED + 2000 + int(tta_id))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")

        for inputs, image_ids in progress:
            inputs = inputs.to(device)

            outputs = net(inputs, False, "test")
            if projector is not None:
                outputs = projector(torch.log_softmax(outputs, dim=1))
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")

    return np.concatenate(probability, axis=0)




## === cell 18
probability = []

start_time = time.time()

sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
df_sample = pd.read_csv(sample_sub_path)
sample_image_ids = df_sample["image_id"].tolist()

df_test = pd.DataFrame({"image_id": sample_image_ids})
df_test["label"] = 1

USE_IMAGENET_FALLBACK = True

supported = []
for p in pretrained_models:
    b = os.path.splitext(os.path.basename(p))[0].lower()
    if any(
        k in b
        for k in [
            "resnet18",
            "resnet50",
            "resnet152",
            "resnext101",
            "densenet201",
            "efficientnet-b7",
            "efficientnetb7",
            "eb7",
        ]
    ):
        supported.append(p)
pretrained_models = supported
print(f"{len(pretrained_models)} supported model files after filtering.")

model_specs = []
if len(pretrained_models) > 0:
    for p in pretrained_models:
        model_specs.append(("ckpt", p))
else:
    if not USE_IMAGENET_FALLBACK:
        print("No pretrained models found; writing sample_submission.csv as fallback.")
        df_sub = df_sample.copy()
        df_sub.to_csv("submission.csv", index=False)
    else:
        model_specs = [
            ("imagenet", "resnet50"),
            ("imagenet", "densenet201"),
            ("imagenet", "efficientnet-b7"),
        ]
        print(
            "No finetuned checkpoints found; using ImageNet pretrained backbones:",
            model_specs,
        )

proj_cache = {}
proj_cache_path = "imagenet_to_cassava_projection.pkl"
if os.path.exists(proj_cache_path):
    try:
        with open(proj_cache_path, "rb") as f:
            proj_cache = pickle.load(f)
        print("Loaded projection cache:", list(proj_cache.keys()))
    except Exception as e:
        print("Failed to load projection cache; will rebuild.", type(e).__name__, e)
        proj_cache = {}

if "df_sub" not in globals():
    for spec_type, spec_val in model_specs:
        if spec_type == "ckpt":
            pretrained_model = spec_val
            basename = os.path.splitext(os.path.basename(pretrained_model))[0]
            basename_l = basename.lower()
        else:
            pretrained_model = None
            basename = spec_val
            basename_l = spec_val.lower()

        criterion = nn.CrossEntropyLoss()

        projector = None  # default: none for finetuned 5-class heads

        if "resnet18" in basename_l:
            MODEL_NAME = "resnet18"
            if spec_type == "imagenet":
                net_base = models.resnet18(
                    weights=models.ResNet18_Weights.IMAGENET1K_V1
                )
            else:
                net_base = models.resnet18(weights=None)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename_l:
            MODEL_NAME = "resnet50"
            if spec_type == "imagenet":
                net_base = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V2
                )
                if "resnet50" not in proj_cache:
                    M, b = _build_projection_from_train_tta(
                        "resnet50",
                        train_csv_path=f"{BASE_DIR}/train.csv",
                        train_images_dir=f"{BASE_DIR}/train_images",
                        transforms_for_features=transform["test"],
                        max_per_class=80,
                        batch_size=12,
                    )
                    proj_cache["resnet50"] = {"W": M, "b": b}
                projector = LogitProjector(
                    proj_cache["resnet50"]["W"], proj_cache["resnet50"]["b"]
                ).to(device)

                class ImageNetHeadWrapper(nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m

                    def forward(self, inputs, labels, phase):
                        return self.m(inputs)

                net = ImageNetHeadWrapper(net_base)
            else:
                net_base = models.resnet50(weights=None)
                net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename_l:
            MODEL_NAME = "resnet152"
            if spec_type == "imagenet":
                net_base = models.resnet152(
                    weights=models.ResNet152_Weights.IMAGENET1K_V2
                )
            else:
                net_base = models.resnet152(weights=None)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename_l:
            MODEL_NAME = "resnext101"
            if spec_type == "imagenet":
                net_base = models.resnext101_32x8d(
                    weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
                )
            else:
                net_base = models.resnext101_32x8d(weights=None)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename_l:
            MODEL_NAME = "densenet201"
            if spec_type == "imagenet":
                net_base = models.densenet201(
                    weights=models.DenseNet201_Weights.IMAGENET1K_V1
                )
                if "densenet201" not in proj_cache:
                    M, b = _build_projection_from_train_tta(
                        "densenet201",
                        train_csv_path=f"{BASE_DIR}/train.csv",
                        train_images_dir=f"{BASE_DIR}/train_images",
                        transforms_for_features=transform["test"],
                        max_per_class=80,
                        batch_size=10,
                    )
                    proj_cache["densenet201"] = {"W": M, "b": b}
                projector = LogitProjector(
                    proj_cache["densenet201"]["W"], proj_cache["densenet201"]["b"]
                ).to(device)

                class ImageNetHeadWrapper(nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m

                    def forward(self, inputs, labels, phase):
                        return self.m(inputs)

                net = ImageNetHeadWrapper(net_base)
            else:
                net_base = models.densenet201(weights=None)
                net = FinalLayerMixupModelDenseNet(
                    net_base, criterion, num_classes, False
                )
            BATCH_SIZE = 12
        elif (
            ("efficientnet-b7" in basename_l)
            or ("efficientnetb7" in basename_l)
            or ("eb7" in basename_l)
        ):
            MODEL_NAME = "efficientnet-b7"
            if spec_type == "imagenet":
                if _EFFNET_BACKEND == "efficientnet_pytorch":
                    net_base = EfficientNet.from_pretrained("efficientnet-b7")
                else:
                    net_base = build_torchvision_efficientnet_b7(
                        num_classes=num_classes, imagenet_head=True
                    )
                if "efficientnet-b7" not in proj_cache:
                    M, b = _build_projection_from_train_tta(
                        "efficientnet-b7",
                        train_csv_path=f"{BASE_DIR}/train.csv",
                        train_images_dir=f"{BASE_DIR}/train_images",
                        transforms_for_features=transform["test"],
                        max_per_class=70,
                        batch_size=8,
                    )
                    proj_cache["efficientnet-b7"] = {"W": M, "b": b}
                projector = LogitProjector(
                    proj_cache["efficientnet-b7"]["W"],
                    proj_cache["efficientnet-b7"]["b"],
                ).to(device)

                class ImageNetHeadWrapper(nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m

                    def forward(self, inputs, labels, phase):
                        return self.m(inputs)

                net = ImageNetHeadWrapper(net_base)
            else:
                if _EFFNET_BACKEND == "efficientnet_pytorch":
                    net_base = EfficientNet.from_name("efficientnet-b7")
                else:
                    net_base = build_torchvision_efficientnet_b7(
                        num_classes=num_classes, imagenet_head=False
                    )
                net = FinalLayerMixupModelEN(net_base, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported; skipping.")
            continue

        print(f"{basename}: {MODEL_NAME} ({spec_type})")

        if spec_type == "ckpt":
            try:
                if MODEL_NAME == "efficientnet-b7":
                    if hasattr(net, "model"):
                        load_checkpoint_safely(
                            net.model, pretrained_model, strict=False
                        )
                    else:
                        load_checkpoint_safely(net, pretrained_model, strict=False)
                else:
                    load_checkpoint_safely(net, pretrained_model, strict=False)
            except Exception as e:
                print(
                    f"Skipping {pretrained_model} due to load error: {type(e).__name__}: {e}"
                )
                del net
                torch.cuda.empty_cache()
                continue

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = {
                "test": TestDataset(df_test, transform=transform_),
            }
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                ),
            }

            proba = predict_model(
                basename, net, dataloader, projector=projector, tta_id=tid
            )
            probability.append(proba)

        del net
        if projector is not None:
            del projector
        torch.cuda.empty_cache()

    try:
        with open(proj_cache_path, "wb") as f:
            pickle.dump(proj_cache, f)
        print("Saved projection cache:", list(proj_cache.keys()))
    except Exception as e:
        print("Failed to save projection cache (non-fatal):", type(e).__name__, e)

    if len(probability) == 0:
        print("No models produced predictions; writing sample_submission.csv fallback.")
        df_sub = df_sample.copy()
        df_sub.to_csv("submission.csv", index=False)
    else:
        prob_arr = np.array(probability)
        if prob_arr.ndim != 3:
            raise ValueError(
                f"Unexpected probability array shape: {prob_arr.shape}. Expected 3D (K, N, C)."
            )
        df_test["mean"] = prob_arr.mean(axis=0).argmax(axis=1)

        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 19
if "df_sub" in globals():
    df_test = df_sub.copy()
else:
    if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
        df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
    else:
        df_test["label"] = df_test["mean"].astype(int)



## === cell 20
df_test.head()



## === cell 21
df_out = df_sample[["image_id"]].merge(
    df_test[["image_id", "label"]], on="image_id", how="left"
)
df_out["label"] = df_out["label"].fillna(0).astype(int)

df_out[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_out[["image_id", "label"]].shape)
print(df_out[["image_id", "label"]].head())
