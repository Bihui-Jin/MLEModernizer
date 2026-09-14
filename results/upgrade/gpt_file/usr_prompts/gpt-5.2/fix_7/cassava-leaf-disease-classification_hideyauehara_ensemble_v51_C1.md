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

0.896343306134784

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I first fix why you’re not getting a valid/meaningful Kaggle score: your `pretrained_models` glob points to `../input/...` folders that don’t exist in this environment, so the code falls back to a constant label baseline. Then I make the model discovery robust by searching the actual provided dataset tree under `/kaggle/input` and `/kaggle/data` for `.pth` files, without changing the inference/ensemble logic. Finally, I ensure we always read test image ids from `sample_submission.csv` (guaranteed correct order/rows) instead of listing a directory, preventing submission misalignment issues that can silently hurt accuracy.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.8963), and the most likely cause is that your submission is misaligned with the hidden test set because the code falls back to reading images from `train_images` when `test_images` isn’t found at the chosen `BASE_DIR`. I make the dataset path selection deterministic by always locating the real competition root that contains `sample_submission.csv` and `test_images`, and I always drive prediction order from `sample_submission.csv` (never from directory listing). I also filter the `.pth` discovery to only search inside the competition folder to avoid accidentally loading unrelated weights that produce garbage predictions. These changes keep your model/inference logic intact while fixing the main failure mode that can yield near-random accuracy.'
- What this solution (achieved 0.05531) has done: 'Your current score suggests you’re effectively submitting a near-constant or garbage predictor because no compatible `.pth` weights are actually being found/loaded, so the ensemble never uses real trained models. I make the model discovery deterministic by searching only within the competition dataset tree for `.pth` files and refusing to “accidentally” scan unrelated folders, which should surface the intended weights if they exist. I also fix a key mismatch bug when loading EfficientNet weights: your wrapper changes the classifier keys, so loading often fails silently/gets skipped; I map state-dict keys into the correct submodule with minimal logic while keeping the same architecture and inference. Finally, I ensure the predictions are always written to `label` (not left in `mean`) and remain aligned to `sample_submission.csv` order (which you already do).'
- What this solution (achieved 0.05531) has done: 'Your score (0.05531) is so far below the target that the most likely issue is still that you are effectively producing a near-constant baseline because no valid checkpoints are being loaded, or they’re not being loaded because the model-name matching is too strict. I keep the same ensemble + TTA inference logic, but (1) broaden `.pth` discovery to also include `.pt`/`.bin` and prioritize checkpoints near the competition folder, (2) make the basename→architecture routing robust to common naming patterns (e.g., `effnet_b7`, `efficientnetb7`, `resnext101_32x8d`) without changing architectures, and (3) make the state-dict loading more tolerant/compatible by mapping common key prefixes (e.g., `model.`, `net.`, `module.`) while still refusing clearly incompatible weights. These are minimal, score-relevant fixes intended to ensure you actually use real trained weights and therefore move accuracy sharply toward the target band, while still writing a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, and the most likely reason (given this code) is that you’re not actually using any trained cassava checkpoints, so the pipeline either falls back to label=0 or loads incompatible weights and produces essentially random outputs. I make the checkpoint discovery robust to the common Kaggle layout by also scanning `/kaggle/working` (where users often save weights) and a small set of likely subfolders, while still prioritizing checkpoints inside the competition tree. I also fix a TTA bug that can severely hurt accuracy: `RandomResizedCrop` is stochastic at inference, so I replace those two test-time transforms with deterministic `Resize+CenterCrop` variants (keeping the same overall “4 TTA passes” structure). Finally, I keep submission ordering strictly aligned to `sample_submission.csv` and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os


def find_model_files(base_dir: str):
    """
    Change (score-relevant, minimal):
    - Many Kaggle notebooks save trained checkpoints to /kaggle/working (output dir), not inside the input dataset tree.
      If we only search under BASE_DIR, we often find zero checkpoints => constant baseline => very low score.
    - We keep the same ensemble inference logic; we only improve discovery of the intended trained weights.
    - Still constrain search to a small set of likely roots to avoid picking up unrelated weights.
    """
    exts = ("*.pth", "*.pt", "*.bin")

    search_roots = []
    if base_dir and os.path.exists(base_dir):
        search_roots.append(base_dir)

    for d in [
        "/kaggle/working",
        "/kaggle/working/models",
        "/kaggle/working/weights",
        "/kaggle/working/checkpoints",
        "/kaggle/input",
        "/kaggle/data",
    ]:
        if os.path.exists(d):
            search_roots.append(d)

    candidates = []
    for root in search_roots:
        for ext in exts:
            candidates += glob.glob(os.path.join(root, "**", ext), recursive=True)

    def _rank(p):
        pl = p.lower()
        bonus = 0
        if "cassava-leaf-disease-classification" in pl:
            bonus -= 100
        if "/kaggle/working" in pl:
            bonus -= 50
        return (bonus, len(p), p)

    candidates = sorted(set(candidates), key=_rank)
    return candidates




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

from pathlib import Path
import random
import time
import sys

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
from torchvision.models import efficientnet_b7



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def detect_base_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "data/cassava-leaf-disease-classification",
        "input/cassava-leaf-disease-classification",
        "data",
        "input",
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "sample_submission.csv")) and os.path.exists(
            os.path.join(d, "test_images")
        ):
            return d
    for d in candidates:
        if os.path.exists(os.path.join(d, "sample_submission.csv")):
            return d
    return "data"


BASE_DIR = detect_base_dir()
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    raise FileNotFoundError(
        f"sample_submission.csv not found under BASE_DIR={BASE_DIR}"
    )

TEST_PATH = os.path.join(BASE_DIR, "test_images")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"test_images not found under BASE_DIR={BASE_DIR}")

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")



## === cell 7
df_test = pd.read_csv(SAMPLE_SUB_PATH)[["image_id", "label"]].copy()
print("Loaded sample_submission:", df_test.shape)

missing = 0
for fn in df_test["image_id"].head(20).tolist():
    if not os.path.exists(os.path.join(TEST_PATH, fn)):
        missing += 1
if missing > 0:
    raise FileNotFoundError(
        f"Some sample_submission image_ids do not exist in TEST_PATH={TEST_PATH}. Check BASE_DIR."
    )



## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 9
pass




## === cell 10
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
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
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 11
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
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
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
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
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 12
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()

        if (
            hasattr(model, "classifier")
            and isinstance(model.classifier, nn.Sequential)
            and len(model.classifier) > 1
        ):
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(
                "Unexpected EfficientNet model structure; cannot locate classifier layer."
            )

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




## === cell 13
pass




## === cell 14
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, image_id




## === cell 15
def predict_model(basename, net, dataloader):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    """
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

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = net(inputs, False, "test")
                probability.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")

    return np.concatenate(probability, axis=0)




## === cell 16
def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in state and isinstance(state[k], dict):
                return state[k]
    return state


def _strip_known_prefixes(sd, prefixes):
    if not isinstance(sd, dict):
        return sd
    out = sd
    changed = True
    while changed and isinstance(out, dict) and len(out) > 0:
        changed = False
        for p in prefixes:
            if all(k.startswith(p) for k in out.keys()):
                out = {k[len(p) :]: v for k, v in out.items()}
                changed = True
    return out


def _strip_module_prefix(sd):
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _try_load(net, model_name, state):
    """
    Change (score-relevant, minimal):
    - Robustly load checkpoints saved with prefixes like "model.", "net.", or "module.".
    - Keep strict loading as the first attempt (no semantics change when already compatible).
    """
    sd = _unwrap_state_dict(state)
    sd = _strip_module_prefix(sd)
    sd = _strip_known_prefixes(sd, prefixes=["model.", "net."])

    if not isinstance(sd, dict) or len(sd) == 0:
        return False

    try:
        net.load_state_dict(sd, strict=True)
        return True
    except RuntimeError:
        pass

    if model_name == "efficientnet-b7" and hasattr(net, "model"):
        try:
            net.model.load_state_dict(sd, strict=True)
            return True
        except RuntimeError:
            pass

        try:
            sd_pref = {("model." + k): v for k, v in sd.items()}
            net.load_state_dict(sd_pref, strict=True)
            return True
        except RuntimeError:
            pass

        try:
            sd_unpref = {
                k.replace("model.", "", 1): v
                for k, v in sd.items()
                if k.startswith("model.")
            }
            if len(sd_unpref) > 0:
                net.model.load_state_dict(sd_unpref, strict=True)
                return True
        except RuntimeError:
            pass

    missing_ok = ("fc.weight", "fc.bias", "classifier.1.weight", "classifier.1.bias")
    try:
        incompatible = net.load_state_dict(sd, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if len(unexpected) == 0 and (
            len(missing) == 0 or missing.issubset(set(missing_ok))
        ):
            return True
    except Exception:
        pass

    return False


def _infer_model_name_from_basename(basename: str):
    """
    Change (score-relevant, minimal):
    - Accept common filename patterns so we don't skip valid checkpoints purely due to naming.
    """
    b = basename.lower().replace("_", "").replace("-", "")
    if "resnet18" in b:
        return "resnet18"
    if "resnet50" in b:
        return "resnet50"
    if "resnet152" in b:
        return "resnet152"
    if "resnext101" in b or "resnext10132x8d" in b:
        return "resnext101"
    if "densenet201" in b:
        return "densenet201"
    if "efficientnetb7" in b or "effnetb7" in b:
        return "efficientnet-b7"
    return None


pretrained_models = find_model_files(BASE_DIR)

print(f"{len(pretrained_models)} model files found.")
print("\n".join(np.array(pretrained_models[:200], dtype=str)))
if len(pretrained_models) > 200:
    print(f"... (showing first 200 of {len(pretrained_models)})")

probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "WARNING: No pretrained model files found. Will output baseline label=0 for all images."
    )
    df_test["label"] = 0
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]

        criterion = nn.CrossEntropyLoss()

        MODEL_NAME = _infer_model_name_from_basename(basename)
        if MODEL_NAME is None:
            continue

        if MODEL_NAME == "resnet18":
            net = models.resnet18(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif MODEL_NAME == "resnet50":
            net = models.resnet50(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif MODEL_NAME == "resnet152":
            net = models.resnet152(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif MODEL_NAME == "resnext101":
            net = models.resnext101_32x8d(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif MODEL_NAME == "densenet201":
            net = models.densenet201(weights=None)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif MODEL_NAME == "efficientnet-b7":
            net = efficientnet_b7(weights=None)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            continue

        print(f"{basename}: {MODEL_NAME} -> loading {pretrained_model}")

        state = torch.load(pretrained_model, map_location="cpu")
        ok = _try_load(net, MODEL_NAME, state)
        if not ok:
            print(
                f"WARNING: Failed to load weights for {basename}. Skipping this checkpoint."
            )
            del net
            if device == "cuda":
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
                    pin_memory=(device == "cuda"),
                ),
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(probability) == 0:
        print(
            "WARNING: No supported models were loaded successfully. Falling back to label=0 baseline."
        )
        df_test["label"] = 0
    else:
        proba_mean = np.mean(np.stack(probability, axis=0), axis=0)
        df_test["mean"] = proba_mean.argmax(axis=1)
        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 17
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"].astype(int)



## === cell 18
df_test.head()



## === cell 19
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
