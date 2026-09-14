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
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.852825627077667

# 6. Current score

0.51719

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51719) has done: 'I fix the runtime failure by making the weight-loading step robust to missing `/kaggle/input/cassava-pytorch-starter-train` checkpoints and instead automatically load any matching fold checkpoints if they exist. If no checkpoints are found, the code fall back to a deterministic pretrained ResNet18 backbone (same model head and inference logic) so it can still run end-to-end and produce a valid `submission.csv`. I also fix dataset indexing to avoid `.loc[idx]` pitfalls and ensure the predicted labels are written as plain integer values aligned with `sample_submission.csv` order. These changes are minimal, unblock execution, and should increase score versus random output when the provided weights are absent.'
- What this solution (achieved 0.51719) has done: 'Your score is far below the target (0.51719 vs 0.8528), so we should improve accuracy with the smallest changes that keep your architecture and inference loop intact. The biggest issue is that the `resnet18` backbone is still using its original `fc` layer (1000-way ImageNet head) during the forward pass, and your custom `self.logit` never receives proper features—this makes predictions close to random. I minimally fix the model init to replace `backbone.fc` with `nn.Identity()` so your head is applied to pooled features as intended, and I also align normalization to ImageNet mean/std when using pretrained weights (fallback path), which typically boosts accuracy without changing core logic. Everything else (data loading, TTA/ensemble structure, submission writing) stays the same and it still produce `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'Your current score gap is large (0.51719 vs target 0.85283), so we should cautiously improve accuracy without changing the core model/inference logic. The main minimal gain comes from matching the test-time preprocessing to how the fold checkpoints were trained: when using external weights, we should use the same normalization those starters typically used (0.5/0.5), while keeping ImageNet normalization for the pretrained-resnet fallback. I also make the DataLoader deterministic and increase `num_workers` a bit for stability/speed (doesn’t change semantics). Everything else (ResNet18 backbone, custom head, fold averaging, softmax+argmax, submission format) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import glob
import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
DIR_WEIGHTS = "/kaggle/input/cassava-pytorch-starter-train"

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 64
SIZE = 256


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)




## === cell 2
class CassavaDataset(Dataset):
    def __init__(self, df, dataset="train", transforms=None):
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        image_src = f"{DIR_INPUT}/{self.dataset}_images/{image_id}"
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image




## === cell 3
class CassavaModel(nn.Module):
    def __init__(self, num_classes=5, backbone_pretrained=False):
        super().__init__()

        weights = (
            torchvision.models.ResNet18_Weights.DEFAULT if backbone_pretrained else None
        )
        self.backbone = torchvision.models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.25, self.training)
        x = self.logit(x)
        return x




## === cell 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CKPT_MEAN = (0.5, 0.5, 0.5)
CKPT_STD = (0.5, 0.5, 0.5)

transforms_test_imagenet = A.Compose(
    [
        A.Resize(height=SIZE, width=SIZE, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_test_ckpt = A.Compose(
    [
        A.Resize(height=SIZE, width=SIZE, p=1.0),
        A.Normalize(mean=CKPT_MEAN, std=CKPT_STD, p=1.0),
        ToTensorV2(p=1.0),
    ]
)



## === cell 5
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
submission_df["label"] = 0
submission_df.head()



## === cell 6
if submission_df.shape[0] == 1:
    submission_df = pd.DataFrame(
        [
            {"image_id": "2216849948.jpg", "label": 0},
            {"image_id": "2216849948.jpg", "label": 0},
        ]
    ).reset_index(drop=True)

submission_df.head()




## === cell 7
def find_checkpoints(dir_weights: str, n_folds: int):
    """
    Tries to locate fold checkpoints in the given directory.
    Supports both:
      - model_state_fold_{i}.pth
      - any *.pth that contains 'fold_{i}' in the name
    Returns a list of paths indexed by fold (None if missing).
    """
    paths_by_fold = [None] * n_folds
    if not os.path.isdir(dir_weights):
        return paths_by_fold

    for i in range(n_folds):
        p = os.path.join(dir_weights, f"model_state_fold_{i}.pth")
        if os.path.exists(p):
            paths_by_fold[i] = p

    if any(p is None for p in paths_by_fold):
        all_pths = glob.glob(os.path.join(dir_weights, "*.pth"))
        for p in all_pths:
            m = re.search(r"fold[_\-]?(\d+)", os.path.basename(p))
            if m:
                fi = int(m.group(1))
                if 0 <= fi < n_folds and paths_by_fold[fi] is None:
                    paths_by_fold[fi] = p

    return paths_by_fold


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
ckpt_paths = find_checkpoints(DIR_WEIGHTS, N_FOLDS)

use_folds = [i for i, p in enumerate(ckpt_paths) if p is not None]
use_external_weights = len(use_folds) > 0

transforms_test = (
    transforms_test_ckpt if use_external_weights else transforms_test_imagenet
)

g = torch.Generator()
g.manual_seed(SEED)

dataset_test = CassavaDataset(
    df=submission_df, dataset="test", transforms=transforms_test
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    generator=g,
)

submissions = None

if use_external_weights:
    fold_indices = use_folds
else:
    fold_indices = [0]  # single model fallback

for i_fold in fold_indices:
    model = CassavaModel(num_classes=5, backbone_pretrained=(not use_external_weights))
    model.to(device)

    if use_external_weights:
        checkpoint = torch.load(ckpt_paths[i_fold], map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    test_preds = None

    for batch in dataloader_test:
        images = batch.to(device, dtype=torch.float)

        with torch.no_grad():
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1).detach().cpu()

        if test_preds is None:
            test_preds = preds
        else:
            test_preds = torch.cat((test_preds, preds), dim=0)

    denom = len(fold_indices)
    if submissions is None:
        submissions = test_preds / denom
    else:
        submissions += test_preds / denom

submissions[:10]



## === cell 8
pred_labels = torch.argmax(submissions, dim=1).cpu().numpy().astype(int)
submission_df["label"] = pred_labels
submission_df.to_csv("submission.csv", index=False)
submission_df.head()
