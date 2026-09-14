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

0.26532

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28363) has done: 'I fix the runtime error by aligning the test preprocessing image size with what the torchvision ViT expects (224×224), which caused the forward-pass assertion. I also make the inference loop robust by ensuring `predictions` always gets filled and by concatenating safely only after the loop completes. Finally, I keep the model and weight-loading logic unchanged, and just ensure the pipeline runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.24477) has done: 'Your current score (0.28363) is far below the target (0.8871), and the most likely cause is a preprocessing mismatch with the weights you’re loading (e.g., the saved ViT was trained on 384px inputs, not 224px, and/or without the extra CenterCrop). I keep your model/weights and inference loop the same, but adjust only the test-time transforms to match common ViT-B/16 training/inference conventions: direct resize to the model’s expected image size and ImageNet normalization, removing the 512 center-crop that can destroy composition. I also auto-detect the correct input size (224 vs 384) by inspecting the ViT’s patch embedding / position embedding when possible, so the transform matches the loaded checkpoint without changing architecture. These minimal changes should legitimately increase accuracy toward your target while preserving core logic and producing the same `submission.csv` format.'
- What this solution (achieved 0.24477) has done: 'Your score is far below the target, so the most likely issue is that your loaded ViT checkpoint expects *different input normalization and/or resize policy* than the ImageNet mean/std you’re using, causing near-random predictions. I keep your model/weights and pure-inference approach unchanged, but make the test-time transform robust by (1) using the official torchvision ViT normalization when you’re using `torchvision.models.vit_b_16`, and (2) trying a small set of common resize+normalization variants and selecting the one with the best confidence profile (lowest average entropy) on the test set. This does not use labels and keeps evaluation semantics identical, but often fixes large performance drops due to preprocessing mismatch. The rest of the pipeline (dataset, dataloader, argmax predictions, submission format) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.26682) has done: 'Your current score is far below the target, so the most likely issue is that the loaded checkpoint is being fed inputs that don’t match the checkpoint’s *expected preprocessing* (especially interpolation/resize policy and normalization). I keep your model/weights and single-pass inference intact, but (1) switch to torchvision’s official ViT preprocessing (including the correct resize+center-crop policy) when you’re using `torchvision.models.vit_b_16`, and (2) expand the candidate transform set to include the most common ViT conventions while selecting via your existing unsupervised entropy criterion. I also make DataLoader settings deterministic/stable and ensure the submission aligns exactly to `sample_submission.csv` ordering (already mostly true) and uses the correct integer dtype.'
- What this solution (achieved 0.26532) has done: 'Your score is far below the target, so the most likely cause is still a mismatch between the checkpoint you load and the preprocessing you feed into the model, making predictions close to random. I keep your model, weights, and single-pass inference logic intact, but add one minimal, legitimate improvement: include the *native torchvision ViT_B_16 weight transforms* (when that model type is used) and make the candidate selection slightly more robust by ranking candidates using a combined “low entropy + balanced non-collapsed predictions” heuristic (still unsupervised, no labels). I also add a very small TTA option (original + horizontal flip averaged) as an additional candidate, without changing the model or training, which often yields a meaningful accuracy gain on this dataset. The pipeline still write a valid `submission.csv` with the required columns and ordering.'

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
import torchvision.transforms as tvt
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


def _get_torchvision_vit_weights_transform_or_none(image_size: int):
    """
    Change rationale (score): if the loaded model is torchvision ViT, its canonical preprocessing
    is encoded in the weights meta/transforms; using it can fix large accuracy drops from
    subtle resize/crop/interpolation differences.
    """
    try:
        w = models.ViT_B_16_Weights.IMAGENET1K_V1
        t = w.transforms()
        return t
    except Exception:
        return None


def _get_candidate_transforms(image_size: int, model_type: str):
    candidates = []

    candidates.append(
        (
            "resize_imagenet_norm_bilinear",
            albu.Compose(
                [
                    albu.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
                    albu.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ],
                p=1.0,
            ),
        )
    )

    candidates.append(
        (
            "resize_vit_norm_05_bilinear",
            albu.Compose(
                [
                    albu.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
                    albu.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ],
                p=1.0,
            ),
        )
    )

    resize_side = int(round(image_size * 256 / 224))  # 224->256, 384->439
    candidates.append(
        (
            f"resize{resize_side}_centercrop{image_size}_imagenet_norm_bilinear",
            albu.Compose(
                [
                    albu.Resize(
                        resize_side, resize_side, interpolation=cv2.INTER_LINEAR
                    ),
                    albu.CenterCrop(image_size, image_size),
                    albu.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ],
                p=1.0,
            ),
        )
    )

    candidates.append(
        (
            f"resize{resize_side}_centercrop{image_size}_vit_norm_05_bilinear",
            albu.Compose(
                [
                    albu.Resize(
                        resize_side, resize_side, interpolation=cv2.INTER_LINEAR
                    ),
                    albu.CenterCrop(image_size, image_size),
                    albu.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ],
                p=1.0,
            ),
        )
    )

    candidates.append(
        (
            "resize_imagenet_norm_bicubic",
            albu.Compose(
                [
                    albu.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
                    albu.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ],
                p=1.0,
            ),
        )
    )
    candidates.append(
        (
            f"resize{resize_side}_centercrop{image_size}_imagenet_norm_bicubic",
            albu.Compose(
                [
                    albu.Resize(
                        resize_side, resize_side, interpolation=cv2.INTER_CUBIC
                    ),
                    albu.CenterCrop(image_size, image_size),
                    albu.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ],
                p=1.0,
            ),
        )
    )

    if "torchvision.models.vit_b_16" in str(model_type):
        candidates.append(
            (
                f"tv_vit_like_resize{resize_side}_centercrop{image_size}_bicubic_imagenet_norm",
                albu.Compose(
                    [
                        albu.Resize(
                            resize_side, resize_side, interpolation=cv2.INTER_CUBIC
                        ),
                        albu.CenterCrop(image_size, image_size),
                        albu.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value=255.0,
                            p=1.0,
                        ),
                        ToTensorV2(),
                    ],
                    p=1.0,
                ),
            )
        )

        candidates.append(
            (
                f"tv_vit_like_resize{resize_side}_centercrop{image_size}_bicubic_imagenet_norm_TTA_hflip",
                albu.Compose(
                    [
                        albu.Resize(
                            resize_side, resize_side, interpolation=cv2.INTER_CUBIC
                        ),
                        albu.CenterCrop(image_size, image_size),
                        albu.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value=255.0,
                            p=1.0,
                        ),
                        ToTensorV2(),
                    ],
                    p=1.0,
                ),
            )
        )

    return candidates


def _mean_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1).clamp_min(1e-12)
    ent = -(p * torch.log(p)).sum(dim=1)  # [B]
    return ent.mean()


def _collapse_penalty_from_preds(preds: np.ndarray, num_classes: int = 5) -> float:
    """
    Change rationale (score): entropy-only selection can pick a confidently-wrong preprocessing
    that collapses into 1 class. We add a tiny penalty for highly imbalanced predictions to avoid
    obvious collapse modes while staying unsupervised.
    """
    counts = np.bincount(preds.astype(np.int64), minlength=num_classes).astype(
        np.float64
    )
    p = counts / max(1.0, counts.sum())
    eps = 1e-12
    ent = float(-(p * np.log(p + eps)).sum())  # max is log(num_classes)
    return float((np.log(num_classes) - ent) / np.log(num_classes))


@torch.no_grad()
def _predict_logits(model, loader, device):
    logits_list = []
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)
        logits_list.append(outputs.detach().cpu())
    return torch.cat(logits_list, dim=0)




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

candidates = _get_candidate_transforms(INFERRED_IMAGE_SIZE, model_type=model_type)

best_name = None
best_score = None
best_preds = None


def _seed_worker(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


g = torch.Generator()
g.manual_seed(42)

for name, test_aug in candidates:
    test_dataset = CassavaDataset(
        df=test_df, imfolder=image_path, train=False, transforms=test_aug
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=g,
    )

    if name.endswith("_TTA_hflip"):
        logits1 = _predict_logits(model, test_loader, device)

        base = test_aug
        try:
            tlist = []
            for t in base.transforms:
                if isinstance(t, ToTensorV2):
                    tlist.append(albu.HorizontalFlip(p=1.0))
                    tlist.append(t)
                else:
                    tlist.append(t)
            tta_aug = albu.Compose(tlist, p=1.0)
            test_dataset2 = CassavaDataset(
                df=test_df, imfolder=image_path, train=False, transforms=tta_aug
            )
            test_loader2 = DataLoader(
                test_dataset2,
                batch_size=32,
                num_workers=2,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=_seed_worker,
                generator=g,
            )
            logits2 = _predict_logits(model, test_loader2, device)
            logits = (logits1 + logits2) / 2.0
        except Exception:
            logits = logits1
    else:
        logits = _predict_logits(model, test_loader, device)

    preds = logits.argmax(dim=1).numpy().astype(np.int64)
    mean_ent = float(_mean_entropy_from_logits(logits).item())
    collapse_pen = _collapse_penalty_from_preds(preds, num_classes=5)

    combined = mean_ent + 0.25 * collapse_pen

    uniq, cnt = np.unique(preds, return_counts=True)
    print(
        f"Candidate '{name}': mean_entropy={mean_ent:.5f}, collapse_pen={collapse_pen:.5f}, combined={combined:.5f}, pred_counts={dict(zip(uniq.tolist(), cnt.tolist()))}"
    )

    if best_score is None or combined < best_score:
        best_score = combined
        best_name = name
        best_preds = preds

if best_preds is None:
    raise RuntimeError("No predictions were generated for any candidate transform.")

if len(best_preds) != len(test_df):
    raise RuntimeError(
        f"Pred length {len(best_preds)} != test_df length {len(test_df)}"
    )

sub = test_df[["image_id"]].copy()
sub["label"] = best_preds.astype(int)
sub.to_csv("submission.csv", index=False)

print("Selected transform:", best_name, "with combined score:", best_score)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Unique labels predicted:", np.unique(sub["label"], return_counts=True))
