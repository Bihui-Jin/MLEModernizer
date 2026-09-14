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

0.8871260199456029

# 6. Current score

0.24477

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28363) has done: 'I fix the runtime error by aligning the test preprocessing image size with what the torchvision ViT expects (224×224), which caused the forward-pass assertion. I also make the inference loop robust by ensuring `predictions` always gets filled and by concatenating safely only after the loop completes. Finally, I keep the model and weight-loading logic unchanged, and just ensure the pipeline runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.24477) has done: 'Your current score (0.28363) is far below the target (0.8871), and the most likely cause is a preprocessing mismatch with the weights you’re loading (e.g., the saved ViT was trained on 384px inputs, not 224px, and/or without the extra CenterCrop). I keep your model/weights and inference loop the same, but adjust only the test-time transforms to match common ViT-B/16 training/inference conventions: direct resize to the model’s expected image size and ImageNet normalization, removing the 512 center-crop that can destroy composition. I also auto-detect the correct input size (224 vs 384) by inspecting the ViT’s patch embedding / position embedding when possible, so the transform matches the loaded checkpoint without changing architecture. These minimal changes should legitimately increase accuracy toward your target while preserving core logic and producing the same `submission.csv` format.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import pandas as pd
import albumentations as albu
import matplotlib.pyplot as plt
import json
import seaborn as sns
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split
from albumentations.pytorch import ToTensorV2
import time
import datetime
import copy
import warnings

WHEEL_PATH = (
    "../input/vision-transformer/vision_transformer_pytorch-1.0.2-py2.py3-none-any.whl"
)
if os.path.exists(WHEEL_PATH):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", WHEEL_PATH]
        )
    except Exception:
        pass

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
def build_vit_model(num_classes: int = 5):
    """
    Returns a ViT-B/16 style model with a num_classes head.
    Prefer vision_transformer_pytorch if importable; otherwise use torchvision vit_b_16.
    """
    try:
        from vision_transformer_pytorch import VisionTransformer  # type: ignore

        m = VisionTransformer.from_name("ViT-B_16", num_classes=num_classes)
        model_type = "vision_transformer_pytorch.VisionTransformer"
        return m, model_type
    except Exception:
        m = models.vit_b_16(weights=None)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, num_classes)
        model_type = "torchvision.models.vit_b_16"
        return m, model_type


model, model_type = build_vit_model(num_classes=5)
print("Using model type:", model_type)

WEIGHTS_PATH = "../input/vitb16trained/ViT-B_16_trained.pt"
if os.path.exists(WEIGHTS_PATH):
    state = torch.load(WEIGHTS_PATH, map_location=torch.device("cpu"))
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    try:
        res = model.load_state_dict(state, strict=True)
        if hasattr(res, "missing_keys") and (res.missing_keys or res.unexpected_keys):
            print("Loaded weights with strict=True but got:")
            print("Missing keys:", res.missing_keys)
            print("Unexpected keys:", res.unexpected_keys)
    except Exception as ex:
        warnings.warn(
            f"Strict weight load failed ({ex}). Retrying with strict=False to allow minor key diffs."
        )
        res = model.load_state_dict(state, strict=False)
        if hasattr(res, "missing_keys"):
            print("Missing keys:", len(res.missing_keys))
            print("Unexpected keys:", len(res.unexpected_keys))
else:
    warnings.warn(
        f"Weight file not found at {WEIGHTS_PATH}. Model will run with random weights."
    )


def infer_vit_image_size(m) -> int:
    if hasattr(m, "image_size"):
        try:
            v = int(getattr(m, "image_size"))
            if v in (224, 384, 256, 512):
                return v
        except Exception:
            pass

    patch = 16
    if hasattr(m, "patch_size"):
        try:
            patch = int(getattr(m, "patch_size"))
        except Exception:
            pass
    if hasattr(m, "conv_proj") and hasattr(m.conv_proj, "kernel_size"):
        try:
            ks = m.conv_proj.kernel_size
            patch = int(ks[0] if isinstance(ks, (tuple, list)) else ks)
        except Exception:
            pass

    pos = None
    for attr in (
        "pos_embedding",
        "pos_embed",
        "position_embedding",
        "position_embeddings",
    ):
        if hasattr(m, attr):
            pos = getattr(m, attr)
            break
    try:
        if isinstance(pos, torch.Tensor):
            n = int(pos.shape[1])
            n_patches = n - 1 if n > 1 else n
            grid = int(round(np.sqrt(n_patches)))
            if grid * grid == n_patches:
                size = grid * patch
                if size in (224, 384, 256, 512):
                    return size
    except Exception:
        pass

    return 224


INFERRED_IMAGE_SIZE = infer_vit_image_size(model)
print("Inferred ViT input image size:", INFERRED_IMAGE_SIZE)




## === cell 2
class CassavaDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None
    ):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]["image_id"])
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if x is None:
            raise FileNotFoundError(f"Could not read image at path: {im_path}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.transforms:
            x = self.transforms(image=x)["image"]

        if self.train:
            y = int(self.df.iloc[index]["label"])
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.df)




## === cell 3
test_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
image_path = "../input/cassava-leaf-disease-classification/test_images/"

test_aug = albu.Compose(
    [
        albu.Resize(
            INFERRED_IMAGE_SIZE, INFERRED_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
        ),
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)

test_dataset = CassavaDataset(
    df=test_df, imfolder=image_path, train=False, transforms=test_aug
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    num_workers=4,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

predictions = []

for imgs in test_loader:
    imgs = imgs.to(device, non_blocking=True)
    with torch.no_grad():
        outputs = model(imgs)
        predicted = outputs.argmax(dim=1)
    predictions.append(predicted.detach().cpu().numpy())

if len(predictions) == 0:
    raise RuntimeError(
        "No predictions were generated. Check test_loader/dataset reading."
    )

preds = np.concatenate(predictions, axis=0).astype(np.int64)
assert len(preds) == len(
    test_df
), f"Pred length {len(preds)} != test_df length {len(test_df)}"

sub = test_df[["image_id"]].copy()
sub["label"] = preds
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Unique labels predicted:", np.unique(sub["label"], return_counts=True))
