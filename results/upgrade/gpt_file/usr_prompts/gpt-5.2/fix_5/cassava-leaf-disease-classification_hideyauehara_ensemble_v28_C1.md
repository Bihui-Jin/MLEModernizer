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

0.8921124206708976

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the hard dependency on `efficientnet_pytorch` (not installed) by skipping EfficientNet checkpoints if any are present, so inference can proceed with the available torchvision models. I also fix the Kaggle path detection so it always finds `../input/cassava-leaf-disease-classification/test_images` in this environment, avoiding the local `data/train_images` FileNotFoundError. Next, I update the Albumentations `RandomResizedCrop` calls to the v2 API (requires `size=(H,W)`), which currently crashes. Finally, I make the ensemble aggregation robust and always write a valid `submission.csv` with the correct columns, even if no checkpoints are found (fallback to a safe baseline).'
- What this solution (achieved 0.05531) has done: 'Your current 0.05531 score is consistent with “no checkpoints found → fallback all-zeros labels” (or effectively random), so the smallest change that can move you toward the 0.892 target is to actually load usable pretrained .pth models by fixing the input path discovery for those model files. I keep your model wrappers, inference loop, TTA, and averaging exactly as-is, but expand the checkpoint globbing to search the real dataset/input directories available in this environment (including nested folders). I also add a deterministic safeguard for the Albumentations `RandomResizedCrop` v2 signature by explicitly setting `scale`/`ratio` so it behaves stably across versions. This should raise the score substantially (toward your target) without changing the core modeling logic.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is consistent with the code not finding any real `.pth` checkpoints, so it falls back to predicting a constant label (near-random accuracy). To move the score toward the 0.892 target with minimal logic changes, I (1) expand checkpoint discovery to include common Kaggle “Dataset” subfolders like `/kaggle/input/**/` (including `*.pt` as well), and (2) make test image path selection robust by checking both `test_images` and nested `cassava-leaf-disease-classification/test_images` under each candidate base. I also fix a subtle inference shape bug (`squeeze()` can drop the batch dimension when batch_size=1) that can silently ruin predictions for the last batch, without changing the model architecture. Everything else (models, TTA, averaging, submission format) stays the same.'
- What this solution (achieved 0.05531) has done: 'Your 0.05531 score strongly suggests your inference is effectively untrained because the loaded checkpoints’ classifier weights don’t match the instantiated model heads (you create new `fc`/`classifier` layers but then load a full state_dict, which silently fails or loads incorrectly depending on the saved keys). The smallest score-improving change (without altering architecture/loops/transforms) is to load checkpoint weights into the correct submodules: map common checkpoint key prefixes (`module.`, `model.`), and if the checkpoint contains only the backbone, load it into `convlayer`/`features` while separately loading `fc` when present. I also add an explicit “best-effort” loader that errors loudly if nothing was loaded (instead of producing near-random predictions), which should move accuracy sharply toward your target if the `.pth` files are valid cassava-trained weights. Everything else—model choices, TTA list, softmax-mean-argmax, and submission format—remains the same.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os
from pathlib import Path

CANDIDATE_CKPT_ROOTS = [
    "../input",  # standard Kaggle notebooks
    "/kaggle/input",  # standard Kaggle
    "/kaggle/data",  # this environment shows data under /kaggle/data
    "/kaggle/data/input",  # mirror
    "/kaggle/working",  # working dir mirror
]

ckpt_patterns = []
for root in CANDIDATE_CKPT_ROOTS:
    if os.path.exists(root):
        for ext in ["pth", "pt"]:
            ckpt_patterns += [
                str(Path(root) / f"*.{ext}"),
                str(Path(root) / f"*/*.{ext}"),
                str(Path(root) / f"*/*/*.{ext}"),
                str(Path(root) / f"*/*/*/*.{ext}"),
                str(Path(root) / f"*/*/*/*/*.{ext}"),
                str(Path(root) / f"*/*/*/*/*/*.{ext}"),
                str(Path(root) / f"*/*/*/*/*/*/*.{ext}"),
            ]

pretrained_models = []
for pat in ckpt_patterns:
    pretrained_models.extend(glob.glob(pat))

pretrained_models = sorted(list(dict.fromkeys(pretrained_models)))

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(pretrained_models[:50]))



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models

import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import random
import time
import sys

from tqdm import tqdm

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
EFFNET_AVAILABLE = False
EfficientNet = None
try:
    from efficientnet_pytorch import EfficientNet  # type: ignore

    EFFNET_AVAILABLE = True
except ModuleNotFoundError:
    EFFNET_AVAILABLE = False
    EfficientNet = None
    print(
        "Warning: efficientnet_pytorch not available. Any EfficientNet checkpoints will be skipped."
    )



## === cell 4
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
CANDIDATE_BASE_DIRS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/input/cassava-leaf-disease-classification",
]

BASE_DIR = None
TEST_PATH = None

for base in CANDIDATE_BASE_DIRS:
    if not os.path.exists(base):
        continue

    cand1 = os.path.join(base, "test_images")
    cand2 = os.path.join(base, "cassava-leaf-disease-classification", "test_images")
    if os.path.exists(cand1):
        BASE_DIR = base
        TEST_PATH = cand1
        break
    if os.path.exists(cand2):
        BASE_DIR = base
        TEST_PATH = cand2
        break

if BASE_DIR is None:
    BASE_DIR = "/kaggle/data/cassava-leaf-disease-classification"
    cand1 = os.path.join(BASE_DIR, "test_images")
    cand2 = os.path.join(BASE_DIR, "cassava-leaf-disease-classification", "test_images")
    TEST_PATH = cand1 if os.path.exists(cand1) else cand2

test_files = sorted([f for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg")])

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean = [0.430, 0.497, 0.313]
std = [0.238, 0.240, 0.228]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 10
pass




## === cell 11
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
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

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = torch.flatten(mixed_x, 1)
        outputs = self.fc(mixed_x)

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
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
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

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = torch.flatten(mixed_x, 1)
        outputs = self.fc(mixed_x)

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

        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)

        self.model = model
        self.criterion = criterion

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
pass




## === cell 15
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 16
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return obj


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


def load_checkpoint_into_wrapper(net, state, model_name, basename=""):
    state = _extract_state_dict(state)
    if not isinstance(state, dict):
        raise ValueError(f"{basename}: checkpoint is not a state_dict-like dict.")

    state = _strip_prefix(state)

    if model_name == "efficientnet-b7":
        missing, unexpected = net.model.load_state_dict(state, strict=False)
        return missing, unexpected

    if isinstance(net, FinalLayerMixupModel):
        if any(k.startswith("convlayer.") or k.startswith("fc.") for k in state.keys()):
            missing, unexpected = net.load_state_dict(state, strict=False)
            return missing, unexpected

        conv_sd = {}
        fc_sd = {}
        for k, v in state.items():
            if k.startswith("fc."):
                fc_sd[k.replace("fc.", "", 1)] = v
            else:
                conv_sd[k] = v

        missing1, unexpected1 = net.convlayer.load_state_dict(conv_sd, strict=False)
        missing2, unexpected2 = (
            net.fc.load_state_dict(fc_sd, strict=False) if len(fc_sd) else ([], [])
        )
        return list(missing1) + list(missing2), list(unexpected1) + list(unexpected2)

    if isinstance(net, FinalLayerMixupModelDenseNet):
        if any(k.startswith("convlayer.") or k.startswith("fc.") for k in state.keys()):
            missing, unexpected = net.load_state_dict(state, strict=False)
            return missing, unexpected

        feat_sd = {}
        fc_sd = {}
        for k, v in state.items():
            if k.startswith("features."):
                feat_sd[k.replace("features.", "", 1)] = v
            elif k.startswith("classifier."):
                fc_sd[k.replace("classifier.", "", 1)] = v
            elif k.startswith("fc."):
                fc_sd[k.replace("fc.", "", 1)] = v
            else:
                pass

        missing1, unexpected1 = net.convlayer.load_state_dict(feat_sd, strict=False)
        missing2, unexpected2 = (
            net.fc.load_state_dict(fc_sd, strict=False) if len(fc_sd) else ([], [])
        )
        return list(missing1) + list(missing2), list(unexpected1) + list(unexpected2)

    missing, unexpected = net.load_state_dict(state, strict=False)
    return missing, unexpected


def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")

        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            outputs = net(inputs, False, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 17
probability = []
start_time = time.time()

usable_models = []
for pretrained_model in pretrained_models:
    base = os.path.splitext(os.path.basename(pretrained_model))[0]
    if "efficientnet-b7" in base and not EFFNET_AVAILABLE:
        print(f"Skipping {base} (efficientnet_pytorch not available).")
        continue
    usable_models.append(pretrained_model)

for pretrained_model in usable_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        MODEL_NAME = "resnet18"
        net = models.resnet18(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 64
    elif "resnet50" in basename:
        MODEL_NAME = "resnet50"
        net = models.resnet50(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 32
    elif "resnet152" in basename:
        MODEL_NAME = "resnet152"
        net = models.resnet152(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 16
    elif "resnext101" in basename:
        MODEL_NAME = "resnext101"
        net = models.resnext101_32x8d(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 12
    elif "densenet201" in basename:
        MODEL_NAME = "densenet201"
        net = models.densenet201(weights=None)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
        BATCH_SIZE = 12
    elif "efficientnet-b7" in basename:
        MODEL_NAME = "efficientnet-b7"
        net = EfficientNet.from_name(MODEL_NAME)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 10
    else:
        print(f"{basename} is not supported. Skipping.")
        continue

    print(f"{basename}: {MODEL_NAME}")

    state = torch.load(pretrained_model, map_location="cpu")

    missing, unexpected = load_checkpoint_into_wrapper(
        net if MODEL_NAME != "efficientnet-b7" else net,
        state,
        MODEL_NAME,
        basename=basename,
    )
    if len(missing) > 0 and len(unexpected) > 0:
        fc_keys_present = any(
            ("fc" in k) or ("_fc" in k) or ("classifier" in k)
            for k in _strip_prefix(_extract_state_dict(state)).keys()
        )
        if not fc_keys_present:
            print(
                f"{basename}: checkpoint seems incompatible (no classifier keys). Skipping to avoid degrading score."
            )
            del net
            torch.cuda.empty_cache()
            continue

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,  # keep safe in Kaggle
                pin_memory=True,
            )
        }
        proba = predict_model(basename, net, dataloader)  # (N, 5)
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

if len(probability) == 0:
    print(
        "No usable checkpoints found; creating a valid baseline submission (all labels=0)."
    )
    df_test["mean"] = 0
else:
    prob_stack = np.stack(probability, axis=0)  # (M, N, 5)
    df_test["mean"] = prob_stack.mean(axis=0).argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 18
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    df_test["label"] = df_test["mean"].astype(int)



## === cell 19
df_test.head()



## === cell 20
submission = df_test[["image_id", "label"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
