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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8862194016319129

# 6. Current score

0.22496

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.22496) has done: 'I fix the missing checkpoint directory by adding a safe fallback that uses a pretrained timm ResNet-50 when the external Kaggle dataset with weights is not available, so inference can still run end-to-end and produce `submission.csv`. I also fix `predict()` so it doesn’t rely on dict `.values()` ordering (which can swap `image`/`label`) and so it works even when `ckpts` is empty (no zero-dim concatenation). Additionally, I update a couple of deprecated/invalid calls (`F.softmax(dim=...)`, `np.int`) that would break under the provided modern package stack. These changes keep the core architecture/inference approach intact (same model, same averaging logic when checkpoints exist) while ensuring a valid submission is always generated.'
- What this solution (achieved 0.22496) has done: 'Your current low score is because the fallback path uses an ImageNet-pretrained ResNet50 head that is randomly initialized for 5 cassava classes, so predictions are essentially random. To move the score toward your target with minimal semantic change, I keep the same ResNet50 architecture and inference flow, but (1) make checkpoint discovery robust to the common “nested competition folder” layout so your actual trained weights are found/loaded, and (2) add a safe strict=False fallback when loading state_dicts to handle common key-prefix mismatches (e.g., `module.`) without crashing or silently using random weights. If no checkpoints are found, the code still produce a valid submission as before. These changes are directly aimed at using the intended trained model weights, which should increase accuracy toward the target.'
- What this solution (achieved 0.22496) has done: 'Your current score is far below the target, so we should improve accuracy by ensuring inference actually uses your trained Cassava weights instead of the random 5-class head from the ImageNet fallback. I make checkpoint discovery robust by recursively searching common Kaggle input locations for `.pth/.pt/.bin` files and prioritizing the “best/val” checkpoints, then load them safely (handling both `state_dict` wrappers and `module.` prefixes) exactly as your existing averaging-inference already expects. This keeps your core model and prediction logic unchanged while greatly increasing the chance that the correct trained weights are used. If no checkpoints are found, it still fall back to ImageNet pretrained and produce a valid `submission.csv`.'
- What this solution (achieved 0.22496) has done: 'Your current score indicates inference is still mostly running on the fallback ImageNet-pretrained backbone with a random 5-class head (or not actually loading the intended fine-tuned weights). I make the smallest changes that increase the chance your real Cassava checkpoints are found and correctly loaded: (1) expand checkpoint search to include `/kaggle/input` and common “working” paths and (2) improve state_dict extraction for typical checkpoint wrappers (`model`, `state_dict`, `net`) while keeping your model, transforms, and prediction averaging identical. I also load checkpoints with `weights_only=False` (Torch 2.6 safety default can otherwise fail/silently skip) so we actually get tensors. If checkpoints are still not found, the script behave exactly as before and still write a valid `submission.csv`.'
- What this solution (achieved 0.22496) has done: 'Your score is far below the target, so the most likely issue is that inference is still not loading the intended fine-tuned Cassava checkpoints and is instead using the ImageNet fallback with a random 5-class head. I make minimal, score-relevant changes to (1) point `DATA_PATH` to the actual available dataset directory in this environment, (2) broaden checkpoint discovery to include the provided `/kaggle/data` and nested competition folders (where your weights are most likely located here), and (3) ensure checkpoint loading handles common wrappers/prefixes without silently failing. This preserves your model architecture and prediction averaging logic; it only increases the chance you actually use trained weights, which should move accuracy toward the target. The script still always produces a valid `submission.csv`.'
- What this solution (achieved 0.22496) has done: 'Your score is far below the target, so the most likely issue is that inference is still not actually loading the intended fine-tuned cassava checkpoints (and is falling back to an ImageNet backbone with a random 5-class head). I make minimal, score-relevant changes to (1) prioritize finding the correct checkpoint files by restricting search to likely weight folders and de-prioritizing irrelevant `.pth` files under the dataset tree, (2) improve checkpoint unwrapping/cleaning to handle common key mismatches (`module.`, `model.`/`backbone.` prefixes, and classifier naming) so the weights load into your existing `CassavaNet` without silently failing, and (3) add a one-time sanity check that at least one checkpoint successfully loads; otherwise it falls back as before and still writes `submission.csv`. Core model architecture, transforms, and the existing “average softmax across checkpoints” inference logic remain unchanged.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division
import random
import os
import sys
import warnings

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")



## === cell 1
extra_timm_path = "../input/pytorchimagemodelsmaster/pytorch-image-models-master"
if os.path.isdir(extra_timm_path) and extra_timm_path not in sys.path:
    sys.path.append(extra_timm_path)



## === cell 2
_CAND_DATA_PATHS = [
    "../input/cassava-leaf-disease-classification/",  # classic Kaggle
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
    "/kaggle/data/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/",
]
DATA_PATH = None
for _p in _CAND_DATA_PATHS:
    if os.path.isfile(os.path.join(_p, "sample_submission.csv")):
        DATA_PATH = _p
        break
if DATA_PATH is None:
    DATA_PATH = "../input/cassava-leaf-disease-classification/"

NUM_FOLDS = 5
bs = 16
EPOCHS = 10
sz = 448
SNAPMIX_ALPHA = 5.0
SNAPMIX_PCT = 0.5
GRAD_ACCUM_STEPS = 1
TIMM_MODEL = "resnet50"

print("Using DATA_PATH:", DATA_PATH)




## === cell 3
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 1234
seed_everything(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 4
class CassavaDataset(Dataset):
    """Cassava dataset."""

    def __init__(self, dataframe, root_dir, transforms=None):
        super().__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def get_img_bgr_to_rgb(self, path):
        im_bgr = cv2.imread(path)
        if im_bgr is None:
            raise FileNotFoundError("Could not read image at path: {}".format(path))
        im_rgb = im_bgr[:, :, ::-1]
        return im_rgb

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = self.get_img_bgr_to_rgb(img_name)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        label_val = self.dataframe.iloc[idx, 1] if self.dataframe.shape[1] > 1 else 0
        try:
            label_val = int(label_val)
        except Exception:
            label_val = 0

        sample = {
            "image": image,
            "label": torch.tensor(label_val, dtype=torch.long),
        }
        return sample




## === cell 5
def train_transforms():
    return Compose(
        [
            A.RandomResizedCrop(sz, sz),
            A.HorizontalFlip(p=0.5),
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


def valid_transforms():
    return Compose(
        [
            A.Resize(sz, sz),
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




## === cell 6
class CassavaNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = timm.create_model(TIMM_MODEL, pretrained=pretrained)
        n_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats




## === cell 7
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_spm(input, target, model):
    imgsize = (sz, sz)
    bs_ = input.size(0)
    with torch.no_grad():
        output, fms = model(input)
        clsw = model.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()

        clslogit = F.softmax(clsw.forward(poolfea), dim=1)
        logitlist = []
        for i in range(bs_):
            logitlist.append(clslogit[i, target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs_):
            evimap = out[i, target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
            outmaps = F.interpolate(
                outmaps, imgsize, mode="bilinear", align_corners=False
            )

        outmaps = outmaps.squeeze()

        for i in range(bs_):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum() + 1e-12

    return outmaps, clslogit


def snapmix(input, target, alpha, model=None):
    lam_a = torch.ones(input.size(0), device=input.device)
    lam_b = 1 - lam_a
    target_b = target.clone()

    wfmaps, _ = get_spm(input, target, model)
    bs_ = input.size(0)
    lam = np.random.beta(alpha, alpha)
    lam1 = np.random.beta(alpha, alpha)

    rand_index = torch.randperm(bs_, device=input.device)
    wfmaps_b = wfmaps[rand_index, :, :]
    target_b = target[rand_index]

    same_label = target == target_b
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

    area = (bby2 - bby1) * (bbx2 - bbx1)
    area1 = (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1)

    if area1 > 0 and area > 0:
        ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
        ncont = F.interpolate(
            ncont, size=(bbx2 - bbx1, bby2 - bby1), mode="bilinear", align_corners=True
        )
        input[:, :, bbx1:bbx2, bby1:bby2] = ncont
        lam_a = 1 - wfmaps[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (
            wfmaps.sum(2).sum(1) + 1e-8
        )
        lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(2).sum(1) / (
            wfmaps_b.sum(2).sum(1) + 1e-8
        )
        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]
        lam_box = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
        )
        lam_a[torch.isnan(lam_a)] = lam_box
        lam_b[torch.isnan(lam_b)] = 1 - lam_box

    return input, target, target_b, lam_a, lam_b




## === cell 8
class SnapMixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, criterion, outputs, ya, yb, lam_a, lam_b):
        loss_a = criterion(outputs, ya)
        loss_b = criterion(outputs, yb)
        loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        return loss




## === cell 9
def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum() if reduction == "sum" else loss
    )


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction="mean"):
        super().__init__()
        self.epsilon = float(epsilon)
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)




## === cell 10
model = CassavaNet(pretrained=False).to(device)




## === cell 11
def _clean_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    sd = state_dict

    if any(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    for pref in ("model.", "net.", "encoder."):
        if any(isinstance(k, str) and k.startswith(pref) for k in sd.keys()):
            sd = {k.replace(pref, "", 1): v for k, v in sd.items()}

    rename_map = {
        "fc.weight": "classifier.weight",
        "fc.bias": "classifier.bias",
        "head.weight": "classifier.weight",
        "head.bias": "classifier.bias",
        "classifier.fc.weight": "classifier.weight",
        "classifier.fc.bias": "classifier.bias",
    }
    if any(k in sd for k in rename_map.keys()):
        for k_old, k_new in rename_map.items():
            if k_old in sd and k_new not in sd:
                sd[k_new] = sd[k_old]

    if any(isinstance(k, str) and k.startswith("backbone.") for k in sd.keys()):
        backbone_prefix = "backbone."
        new_sd = dict(sd)
        for k, v in sd.items():
            if isinstance(k, str) and k.startswith(backbone_prefix):
                new_sd[k.replace(backbone_prefix, "backbone.", 1)] = v
        sd = new_sd

    return sd


def _unwrap_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for k in ("state_dict", "model", "net", "weights", "model_state_dict"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def _load_model_state(model, ckpt):
    state = _unwrap_checkpoint(ckpt)
    state = _clean_state_dict(state)
    if not isinstance(state, dict):
        return False
    try:
        model.load_state_dict(state, strict=True)
        return True
    except Exception:
        try:
            model.load_state_dict(state, strict=False)
            return True
        except Exception:
            return False


def predict(model, ckpts, dataloader):
    predict_list = []
    model.eval()

    use_ckpts = isinstance(ckpts, (list, tuple)) and len(ckpts) > 0

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader), desc="Predict"):
            images = data["image"].to(device, non_blocking=True)

            if use_ckpts:
                avg_preds = []
                for ckpt in ckpts:
                    ok = _load_model_state(model, ckpt)
                    if not ok:
                        continue
                    model.eval()
                    outputs, _ = model(images)
                    preds = F.softmax(outputs, dim=1).detach().cpu().numpy()
                    avg_preds.append(preds)

                if len(avg_preds) == 0:
                    outputs, _ = model(images)
                    batch_preds = F.softmax(outputs, dim=1).detach().cpu().numpy()
                else:
                    batch_preds = np.mean(np.stack(avg_preds, axis=0), axis=0)
            else:
                outputs, _ = model(images)
                batch_preds = F.softmax(outputs, dim=1).detach().cpu().numpy()

            predict_list.append(batch_preds)

    predict_arr = np.concatenate(predict_list, axis=0)
    return predict_arr.argmax(axis=1)




## === cell 12
test_df = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
test_dir = os.path.join(DATA_PATH, "test_images")

test_ds = CassavaDataset(
    dataframe=test_df, root_dir=test_dir, transforms=valid_transforms()
)

test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 13
def _find_checkpoint_files(base_dirs, exts=(".pth", ".pt", ".bin"), max_files=20):
    found = []
    seen = set()

    def _dir_ok(d):
        if not d or not os.path.isdir(d):
            return False
        ld = d.lower()
        if any(
            x in ld
            for x in (
                "/train_images",
                "/test_images",
                "/train_tfrecords",
                "/test_tfrecords",
            )
        ):
            return False
        return True

    for base in base_dirs:
        if not _dir_ok(base):
            continue
        for root, _, files in os.walk(base):
            if not _dir_ok(root):
                continue
            for fn in files:
                lf = fn.lower()
                if lf.endswith(exts):
                    fp = os.path.join(root, fn)
                    if fp not in seen:
                        seen.add(fp)
                        found.append(fp)

    def _priority_key(path):
        name = os.path.basename(path).lower()
        pth = path.lower()
        pri = 50

        if "best" in name:
            pri = 0
        elif "final" in name:
            pri = 1
        elif "val" in name or "valid" in name:
            pri = 2
        elif "fold" in name:
            pri = 3
        elif "epoch" in name:
            pri = 6
        else:
            pri = 10

        if any(
            x in pth
            for x in (
                "/weights",
                "/weight",
                "/ckpt",
                "/checkpoint",
                "/checkpoints",
                "/models",
                "/model",
            )
        ):
            pri -= 2

        if any(x in name for x in ("optimizer", "sched", "scheduler", "ema", "scaler")):
            pri += 20

        try:
            szb = os.path.getsize(path)
        except Exception:
            szb = -1
        return (pri, -szb, name)

    found = sorted(found, key=_priority_key)
    return found[:max_files]


ckpts = []

candidate_dirs = [
    "../input/cassavalblsmoothingresnet50",
    "../input/cassavalblsmoothingresnet50/cassavalblsmoothingresnet50",
    "../input/cassavalblsmoothingresnet50/weights",
    "../input/cassavalblsmoothingresnet50/checkpoints",
    "../input",
    "/kaggle/input",
    "/kaggle/working",
    "/kaggle/data",
    "/kaggle/data/input",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    DATA_PATH,
]

ckpt_files = _find_checkpoint_files(candidate_dirs, max_files=15)

for fpath in ckpt_files:
    try:
        ckpts.append(torch.load(fpath, map_location="cpu", weights_only=False))
    except TypeError:
        try:
            ckpts.append(torch.load(fpath, map_location="cpu"))
        except Exception:
            pass
    except Exception:
        pass

loaded_any = False
if len(ckpts) > 0:
    for ckpt in ckpts:
        if _load_model_state(model, ckpt):
            loaded_any = True
            break

if (len(ckpts) == 0) or (not loaded_any):
    model = CassavaNet(pretrained=True).to(device)
    ckpts = []  # ensure we don't average over unusable checkpoints

print("Found ckpts (loaded objects):", len(ckpts))
print("At least one ckpt successfully loaded into model:", loaded_any)
if len(ckpt_files) > 0:
    print("Top checkpoint file candidates:")
    for p in ckpt_files[:10]:
        print(" -", p)



## === cell 14
test_predict_list = predict(model, ckpts, test_dl)



## === cell 15
test_df["label"] = test_predict_list.astype(int)
test_df[["image_id", "label"]].to_csv("submission.csv", index=False)
print(test_df.head())
print("Wrote submission.csv with shape:", test_df[["image_id", "label"]].shape)
