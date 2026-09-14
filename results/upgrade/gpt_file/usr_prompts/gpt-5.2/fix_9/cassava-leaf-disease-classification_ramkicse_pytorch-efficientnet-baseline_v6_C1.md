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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8292535509217286

# 6. Current score

0.17564

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the TensorBoard `SummaryWriter` import/usage that’s crashing due to an incompatible tensorboard/protobuf stack in this environment, since it’s not required for inference/submission. I also fix the missing `efficientnet_pytorch` dependency by switching to `torchvision`’s built-in EfficientNet-B7 while preserving the same “EfficientNet-B7 + 5-class head + load weights if provided” core logic, with a safe fallback if the external weight file isn’t present/compatible. Albumentations v2 removed several old transforms, so I rewrite the import list to only use transforms that exist and keep the same augmentation intent (crop/flip/rotate/color jitter/normalize + ToTensorV2). Finally, I ensure the dataloaders are created and inference runs end-to-end, producing `submission.csv` with the exact required columns.'
- What this solution (achieved 0.61099) has done: 'Your current 0.61099 is far below the 0.82925 target, and the biggest likely cause is that inference is running with a randomly initialized EfficientNet-B7 whenever the external `weight.pt` doesn’t strictly match the torchvision model (common mismatch), which tanks accuracy. I keep your core approach (EfficientNet-B7 + 5-class linear head + argmax predictions) but make weight loading robust by accepting common checkpoint formats and loading with `strict=False` while verifying that classifier weights match, so you actually use the provided trained weights. I also fix a small but important split issue (`y` shape for StratifiedKFold) and align validation preprocessing to standard EfficientNet behavior (resize then center-crop) without changing semantics. These changes are minimal, inference-only, and should move the score substantially upward toward the target band while still producing the same `submission.csv` format.'
- What this solution (achieved 0.61099) has done: 'The current gap to the target is large (0.61099 vs 0.82925), and the most likely reason is that your inference model is not using the correct trained weights due to checkpoint key mismatches (e.g., head named `fc.*` or `classifier.*`, DataParallel prefixes, etc.), so it behaves close to random/weak. I keep your core logic (EfficientNet-B7 + 5-class head + argmax) but make weight loading robust by (1) selecting the correct EfficientNet implementation based on what the checkpoint contains and (2) mapping common head key patterns into the current model before `load_state_dict`. I also switch inference preprocessing to the canonical EfficientNet validation transform (resize to 256 then center crop to 224, same normalization), which is a minimal alignment change and often gives a noticeable accuracy lift without changing architecture/training. These changes are inference-only, keep the pipeline intact, and should move accuracy upward toward your target band while still producing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.82925), so we should increase performance with the smallest changes that keep the same “EfficientNet-B7 + 5-class head + argmax” core logic. The most likely cause is that inference is still using mostly-random weights because the checkpoint keys don’t match your model (common with `efficientnet_pytorch` vs `torchvision`, DataParallel prefixes, and head naming), so I make the weight-loading path more robust by remapping common EfficientNet(-PyTorch) keys into the torchvision EfficientNet-B7 naming and then loading non-strictly. I also set the torchvision model’s `weights=None` even during training for consistency with external weights usage (this doesn’t change your approach, it just avoids silently mixing ImageNet weights), and I print a small “loaded params” sanity summary so you can confirm weights truly loaded. These are minimal, inference-relevant changes that should move accuracy materially upward toward the target while keeping the same submission format and pipeline.'
- What this solution (achieved 0.17564) has done: 'Your current score is far below the target, so the smallest likely improvement is to ensure you are actually using meaningful pretrained weights at inference (instead of a mostly-random model due to checkpoint/key mismatch). I keep the same EfficientNet-B7 + 5-class head + argmax core logic, but make the checkpoint loader more compatible: handle common wrapper keys, DataParallel prefixes, and (crucially) map `efficientnet_pytorch` block naming into torchvision’s exact EfficientNet-B7 module paths using the model’s own `state_dict()` as a template. If the external checkpoint still can’t be applied, we fall back to ImageNet weights (still EfficientNet-B7, same head replacement) rather than `weights=None`, which should move accuracy upward while preserving evaluation semantics. These changes are inference-relevant, minimal, and still produce a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.20254) has done: 'Your score (0.17564) is far below the target (0.82925), so we should improve accuracy with the smallest change that fixes the most likely root cause: the model is probably running with an untrained/random 5-class head when the external checkpoint doesn’t load cleanly into the torchvision EfficientNet. I keep your EfficientNet-B7 + 5-class linear head + argmax inference logic, but make checkpoint loading truly compatible by (1) building the model as EfficientNet-B7 with ImageNet weights first, (2) remapping common checkpoint key patterns (DataParallel prefixes + `efficientnet_pytorch` naming) into torchvision keys, and (3) verifying that the classifier head weights actually changed; if not, we fall back to ImageNet backbone + random head (still better than fully random backbone) but we avoid silently “loading nothing”. This should move accuracy substantially upward toward the target without changing your training loop/architecture or submission semantics. I also keep I/O paths and submission format identical.'
- What this solution (achieved 0.17601) has done: 'Your current 0.20254 is far below the 0.82925 target, so we should increase accuracy with the smallest change that addresses the most likely root cause: the external `weight.pt` is not actually being loaded into the torchvision EfficientNet-B7 (key mismatches), so inference runs with an essentially untrained/random 5-class head/backbone mix. I keep your core logic (EfficientNet-B7 + 5-class linear head + argmax predictions) but make weight loading robust by (1) detecting and instantiating the correct EfficientNet implementation (`efficientnet_pytorch` if the checkpoint uses `_blocks/_conv_stem/_fc` keys; otherwise torchvision), and (2) remapping common head keys in both directions. I also ensure the inference model is built with `imagenet_fallback=False` so we don’t silently mix ImageNet weights with a cassava-trained head when the checkpoint is meant to fully define weights, while still falling back to ImageNet only if no checkpoint can be loaded at all. These are inference-only, minimal changes intended to move the score upward toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 0.17564) has done: 'Your current score (0.176) is far below the target (0.829), so the minimal, highest-impact fix is to ensure inference uses the actual cassava-trained checkpoint rather than silently falling back to an ImageNet backbone with a random 5-class head (which typically yields very low accuracy). I keep your exact core logic (EfficientNet-B7 + 5-class head + argmax) and only adjust the checkpoint-loading path to correctly instantiate EfficientNet-B7 with ImageNet weights first, then replace the head and load the checkpoint in a way that matches the common “ramki” weights format (usually based on `efficientnet_pytorch` with ImageNet init). I also make the “loaded_ok” criterion robust (don’t require an arbitrary number of changed tensors) and explicitly validate that the classifier head weights were loaded; if not, we fall back to a safer path. These changes are inference-only, keep paths/format unchanged, and should move accuracy substantially upward toward the target band.'

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap, json, warnings

warnings.filterwarnings("ignore")


def run_cmd(cmd):
    try:
        subprocess.check_call(cmd, shell=True)
    except Exception as e:
        print(f"[WARN] Command failed (continuing): {cmd}\n  -> {e}")




## === cell 1
run_cmd(
    "ls -la ../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar 2>/dev/null || true"
)
if os.path.exists(
    "../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar"
):
    run_cmd(
        "pip -q install ../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar"
    )



## === cell 2
run_cmd("ls -lrt ../input/ramki-cassava-weights/weight.pt 2>/dev/null || true")



## === cell 3
run_cmd("ls ../input/cassava-leaf-disease-classification 2>/dev/null || true")
run_cmd("ls /kaggle/input/cassava-leaf-disease-classification 2>/dev/null || true")



## === cell 4
CANDIDATE_BASES = [
    "../input/cassava-leaf-disease-classification/",
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
    "/kaggle/data/input/cassava-leaf-disease-classification/",
]
base_path = None
for p in CANDIDATE_BASES:
    if os.path.exists(os.path.join(p, "train.csv")):
        base_path = p
        break
if base_path is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset folder in expected locations."
    )
print("Using base_path:", base_path)



## === cell 5
Training = False



## === cell 6
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 7
import torchvision
from torchvision import models
from torchvision.models import EfficientNet_B7_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 9
if torch.cuda.is_available():
    torch.cuda.device_count()



## === cell 10
writer = None



## === cell 11
SEED = 42
N_FOLDS = 10
N_EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 5e-4
NUM_CLASSES = 5




## === cell 12
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)



## === cell 13
base_path = base_path



## === cell 14
train_path = os.path.join(base_path, "train_images/")
test_path = os.path.join(base_path, "test_images/")

train_csv = pd.read_csv(os.path.join(base_path, "train.csv"))
sample = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))



## === cell 15
train_csv.head()



## === cell 16
train_csv_disease = train_csv.label.map(
    {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy",
    }
)
diseases = train_csv_disease.value_counts()



## === cell 17
diseases



## === cell 18
try:
    diseases.plot.pie()
    plt.show()
except Exception as e:
    print("[WARN] plot failed:", e)



## === cell 19
assert (
    "image_id" in sample.columns and "label" in sample.columns
), "sample_submission.csv must have image_id and label columns"




## === cell 20
class MyDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx].label) if "label" in self.df.columns else 0
        p = self.df.iloc[idx].image_id

        p_path = (train_path if not self.test else test_path) + p

        image = Image.open(p_path).convert("RGB")
        image = np.array(image)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image, label




## === cell 21
transforms_train = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.333), p=1.0
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.5,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

transforms_valid = A.Compose(
    [
        A.Resize(height=256, width=256, p=1.0),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)



## === cell 22
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)



## === cell 23
train_csv.shape



## === cell 24
model_name = "efficientnet-b7"


def build_model(num_classes=5, imagenet_fallback=False):
    weights = EfficientNet_B7_Weights.DEFAULT if imagenet_fallback else None
    m = models.efficientnet_b7(weights=weights)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def _strip_prefix(state_dict, prefixes=("module.", "model.", "net.")):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _prefer_efficientnet_pytorch(state_dict):
    if not isinstance(state_dict, dict):
        return False
    keys = list(state_dict.keys())
    return any(
        k.startswith("_conv_stem.") or k.startswith("_fc.") or k.startswith("_blocks.")
        for k in keys
    )


def _build_effnet_pytorch_b7(num_classes=5, imagenet_init=True):
    """
    CHANGE (score improvement): Many public Cassava checkpoints (including "ramki" weight.pt)
    were trained starting from EfficientNet-PyTorch ImageNet weights. If we instantiate from
    scratch, any missing keys during non-strict loading leave random tensors and tank accuracy.
    Using from_pretrained() keeps the same architecture/training approach while ensuring a
    strong default for any missing weights.
    """
    try:
        from efficientnet_pytorch import EfficientNet
    except Exception as e:
        raise ImportError(
            "efficientnet_pytorch not available but checkpoint looks like efficientnet_pytorch."
        ) from e

    if imagenet_init:
        m = EfficientNet.from_pretrained("efficientnet-b7")
    else:
        m = EfficientNet.from_name("efficientnet-b7")

    in_features = m._fc.in_features
    m._fc = nn.Linear(in_features, num_classes)
    return m


def _remap_head_keys_for_effnet_pytorch(state_dict):
    """Map common head names to efficientnet_pytorch's _fc.*."""
    if not isinstance(state_dict, dict):
        return state_dict
    sd = dict(state_dict)

    if "classifier.1.weight" in sd and "_fc.weight" not in sd:
        sd["_fc.weight"] = sd["classifier.1.weight"]
    if "classifier.1.bias" in sd and "_fc.bias" not in sd:
        sd["_fc.bias"] = sd["classifier.1.bias"]

    if "fc.weight" in sd and "_fc.weight" not in sd:
        sd["_fc.weight"] = sd["fc.weight"]
    if "fc.bias" in sd and "_fc.bias" not in sd:
        sd["_fc.bias"] = sd["fc.bias"]

    if "classifier.weight" in sd and "_fc.weight" not in sd:
        sd["_fc.weight"] = sd["classifier.weight"]
    if "classifier.bias" in sd and "_fc.bias" not in sd:
        sd["_fc.bias"] = sd["classifier.bias"]

    return sd


def _remap_head_keys_for_torchvision_efficientnet(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    sd = dict(state_dict)

    if "fc.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd["fc.weight"]
    if "fc.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd["fc.bias"]

    if "classifier.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd["classifier.weight"]
    if "classifier.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd["classifier.bias"]

    if "_fc.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd["_fc.weight"]
    if "_fc.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd["_fc.bias"]

    return sd


def _load_state_with_report(model, state):
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    after = model.state_dict()

    changed = 0
    total = 0
    for k in before:
        if k in after:
            total += 1
            if not torch.equal(before[k], after[k].detach().cpu()):
                changed += 1

    head_changed = 0
    head_keys = []
    if "classifier.1.weight" in before:
        head_keys = ["classifier.1.weight", "classifier.1.bias"]
    elif "_fc.weight" in before:
        head_keys = ["_fc.weight", "_fc.bias"]

    for hk in head_keys:
        if hk in before and hk in after:
            if not torch.equal(before[hk], after[hk].detach().cpu()):
                head_changed += 1

    print(
        f"[INFO] load_state_dict changed_tensors={changed}/{total} head_changed={head_changed}/{len(head_keys)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    return changed, head_changed, missing, unexpected


weights_file = "../input/ramki-cassava-weights/weight.pt"
if not os.path.exists(weights_file):
    alt = "/kaggle/input/ramki-cassava-weights/weight.pt"
    if os.path.exists(alt):
        weights_file = alt

ckpt_state = None
ckpt_is_effnet_pytorch = False
if (not Training) and os.path.exists(weights_file):
    try:
        ckpt = torch.load(weights_file, map_location="cpu")
        ckpt_state = _strip_prefix(_extract_state_dict(ckpt))
        ckpt_is_effnet_pytorch = _prefer_efficientnet_pytorch(ckpt_state)
        print(
            f"[INFO] Detected checkpoint type: {'efficientnet_pytorch' if ckpt_is_effnet_pytorch else 'torchvision/other'}"
        )
    except Exception as e:
        print("[WARN] Could not read checkpoint to detect type:", repr(e))
        ckpt_state = None
        ckpt_is_effnet_pytorch = False

if not Training and ckpt_state is not None:
    if ckpt_is_effnet_pytorch:
        model = _build_effnet_pytorch_b7(num_classes=NUM_CLASSES, imagenet_init=True)
    else:
        model = build_model(num_classes=NUM_CLASSES, imagenet_fallback=False)
else:
    model = build_model(num_classes=NUM_CLASSES, imagenet_fallback=(not Training))

loaded_ok = False
if not Training and ckpt_state is not None:
    try:
        state = ckpt_state
        if ckpt_is_effnet_pytorch:
            state = _remap_head_keys_for_effnet_pytorch(state)
        else:
            state = _remap_head_keys_for_torchvision_efficientnet(state)

        changed, head_changed, missing, unexpected = _load_state_with_report(
            model, state
        )

        loaded_ok = head_changed >= 2
        print("Loaded weights (non-strict):", weights_file, "| loaded_ok:", loaded_ok)
    except Exception as e:
        print(
            "[WARN] Could not load weights into selected EfficientNet implementation. Reason:",
            repr(e),
        )
        loaded_ok = False

if (not Training) and (not loaded_ok):
    print(
        "[WARN] Proceeding without external cassava weights; falling back to ImageNet backbone + new 5-class head."
    )
    model = build_model(num_classes=NUM_CLASSES, imagenet_fallback=True)

model.to(device)
model.eval()



## === cell 25
trainset = MyDataset(train_csv, transforms=transforms_train, test=False)
train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

testset = MyDataset(sample, transforms=transforms_valid, test=True)
test_loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 26
len(train_loader)



## === cell 27
BATCH_SIZE




## === cell 28
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




## === cell 29
def train_model(model, epoch, dataloader_train, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        imgs_train, labels_train = (
            imgs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True).long(),
        )
        output_train = model(imgs_train)

        loss = criterion(output_train, labels_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accs.update(
            (output_train.argmax(1) == labels_train).sum().item() / imgs_train.size(0),
            imgs_train.size(0),
        )
        losses.update(loss.item(), imgs_train.size(0))
        tk.set_postfix(loss=losses.avg, acc=accs.avg)
    return losses.avg


def test_model(model, dataloader_valid, criterion):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid, labels_valid = (
                imgs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True).long(),
            )
            output_valid = model(imgs_valid)

            loss = criterion(output_valid, labels_valid)
            losses.update(loss.item(), imgs_valid.size(0))
            accs.update(
                (output_valid.argmax(1) == labels_valid).sum().item()
                / imgs_valid.size(0),
                imgs_valid.size(0),
            )
            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg




## === cell 30
X = train_csv.iloc[:, :-1]
y = train_csv["label"].values



## === cell 31
if Training:
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

        valid = train_csv.iloc[valid_idx].reset_index(drop=True)
        train = train_csv.iloc[train_idx].reset_index(drop=True)

        dataset_train = MyDataset(train, transforms=transforms_train)
        dataset_valid = MyDataset(valid, transforms=transforms_valid)

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=BATCH_SIZE,
            num_workers=2,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        dataloader_valid = DataLoader(
            dataset_valid,
            batch_size=BATCH_SIZE,
            num_workers=2,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = build_model(num_classes=NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True, min_lr=1e-5
        )

        best_acc = 0.0
        for epoch in range(N_EPOCHS):
            train_loss = train_model(
                model, epoch, dataloader_train, criterion, optimizer
            )
            val_loss, acc = test_model(model, dataloader_valid, criterion)
            scheduler.step(acc)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "weight.pt")

            print("current_val_acc:", acc, "best_val_acc:", best_acc)



## === cell 32
test_pred = []

model.eval()
with torch.no_grad():
    for i, (images, _) in enumerate(tqdm(test_loader, position=0, leave=True)):
        images = images.to(device, non_blocking=True)
        pred = model(images)
        pred = pred.argmax(1).cpu().numpy().astype(int)
        test_pred.extend(pred.tolist())

sample["label"] = test_pred
sample = sample[["image_id", "label"]]
sample.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample.shape)



## === cell 33
sample.head()



## === cell 34
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
