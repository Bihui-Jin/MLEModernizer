# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models



## === cell 1
model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # speed; semantics unchanged for inference



## === cell 2
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

proto_aug = A.Compose(
    [
        A.LongestMaxSize(max_size=512, p=1.0),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0, p=1.0),
        A.CenterCrop(height=512, width=512, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



## === cell 3
use_ckpt = os.path.exists(model_path)

if use_ckpt:
    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Linear(2048, 5)
    model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    use_imagenet_mapping = False
else:
    warnings.warn(
        f"Checkpoint not found at {model_path}. Falling back to torchvision pretrained ResNeXt-50 ImageNet head "
        f"and using a deterministic 1000->5 mapping built from train.csv."
    )
    weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
    model = models.resnext50_32x4d(weights=weights)  # outputs 1000 ImageNet logits
    model.to(device)
    model.eval()
    use_imagenet_mapping = True



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
assert {"image_id", "label"}.issubset(
    sample_sub.columns
), "sample_submission.csv must have image_id,label columns"
if not os.path.isdir(test_images_path):
    raise FileNotFoundError(f"test_images_path not found: {test_images_path}")

tta_count = 10



## === cell 5
imagenet_W = None  # [1000,5]
imagenet_b = None  # [5]

_image_cache = {}


def _load_image_np(img_path: str) -> np.ndarray:
    cached = _image_cache.get(img_path, None)
    if cached is not None:
        return cached
    arr = np.array(Image.open(img_path).convert("RGB"))
    _image_cache[img_path] = arr
    return arr


def _apply_aug_to_tensor(image_np: np.ndarray, aug: A.Compose) -> torch.Tensor:
    image = aug(image=image_np)["image"]
    if image.ndim != 3:
        raise ValueError(f"Unexpected image ndim={image.ndim}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        chw = image
    else:
        chw = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(chw).float()


def _iter_batches(df: pd.DataFrame, bs: int):
    n = len(df)
    for i in range(0, n, bs):
        yield df.iloc[i : i + bs]


def _get_tta_logits_for_images(
    model_: torch.nn.Module,
    image_nps: list,
    aug: A.Compose,
    tta: int,
) -> torch.Tensor:
    b = len(image_nps)
    x_list = []
    for img_np in image_nps:
        for _ in range(tta):
            x_list.append(_apply_aug_to_tensor(img_np, aug))
    x = torch.stack(x_list, dim=0).to(device)  # [B*T,3,512,512]
    logits = model_(x).float()  # [B*T,C]
    logits = logits.view(b, tta, -1).mean(dim=1)  # [B,C]
    return logits


if use_imagenet_mapping:
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(
            f"train.csv not found at {train_csv_path}, required for fallback mapping."
        )
    train_df = pd.read_csv(train_csv_path)
    train_images_path = os.path.join(os.path.dirname(train_csv_path), "train_images")
    if not os.path.isdir(train_images_path):
        alt = "../input/cassava-leaf-disease-classification/train_images"
        if os.path.isdir(alt):
            train_images_path = alt
        else:
            raise FileNotFoundError(
                f"train_images folder not found (tried {train_images_path} and {alt})."
            )

    per_class_cap = 900
    sampled_ids = []
    for k in range(5):
        cls = train_df[train_df["label"] == k][["image_id"]].copy()
        if len(cls) == 0:
            continue
        take = min(per_class_cap, len(cls))
        cls = cls.sample(n=take, random_state=SEED).reset_index(drop=True)
        cls["label"] = k
        sampled_ids.append(cls)
    sampled_df = pd.concat(sampled_ids, axis=0).reset_index(drop=True)

    batch_size = (
        12 if torch.cuda.is_available() else 4
    )  # smaller because each batch expands by TTA

    X_chunks = []
    Y_chunks = []

    model.eval()
    with torch.inference_mode():
        for chunk in _iter_batches(sampled_df, batch_size):
            img_nps = []
            keep_labels = []
            for img_id, y in zip(chunk["image_id"].tolist(), chunk["label"].tolist()):
                img_path = os.path.join(train_images_path, img_id)
                if not os.path.exists(img_path):
                    continue
                img_nps.append(_load_image_np(img_path))
                keep_labels.append(int(y))
            if len(img_nps) == 0:
                continue

            logits_mean = (
                _get_tta_logits_for_images(model, img_nps, sub_aug, tta_count)
                .detach()
                .cpu()
            )  # [B,1000]
            X_chunks.append(logits_mean)

            y = torch.tensor(keep_labels, dtype=torch.long)
            y_oh = torch.nn.functional.one_hot(y, num_classes=5).float()  # [B,5]
            Y_chunks.append(y_oh)

    if len(X_chunks) == 0:
        raise RuntimeError(
            "Could not build fallback mapping: no training images were loaded."
        )

    X = torch.cat(X_chunks, dim=0)  # [N,1000]
    Y = torch.cat(Y_chunks, dim=0)  # [N,5]

    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xw = (X - X_mean) / X_std  # whitened features

    ones = torch.ones((Xw.shape[0], 1), dtype=Xw.dtype)
    Xa = torch.cat([Xw, ones], dim=1)  # [N,1001]

    lam = 1e-2
    I = torch.eye(Xa.shape[1], dtype=Xa.dtype)
    I[-1, -1] = 0.0  # do not regularize bias
    A_aug = torch.cat([Xa, (lam**0.5) * I], dim=0)  # [(N+1001),1001]
    Y_aug = torch.cat([Y, torch.zeros((Xa.shape[1], Y.shape[1]), dtype=Y.dtype)], dim=0)

    W_full = torch.linalg.lstsq(A_aug, Y_aug).solution  # [1001,5]

    Ww = W_full[:-1, :]  # [1000,5]
    bw = W_full[-1, :]  # [5]
    W_un = Ww / X_std.squeeze(0).unsqueeze(1)  # [1000,5]
    b_un = bw - (X_mean.squeeze(0) / X_std.squeeze(0)) @ Ww  # [5]

    imagenet_W = W_un.to(device)
    imagenet_b = b_un.to(device)



## === cell 6
predictions = []

model.eval()
with torch.inference_mode():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        image_np = _load_image_np(img_path)

        x_list = []
        for _ in range(tta_count):
            x_list.append(_apply_aug_to_tensor(image_np, sub_aug))
        x = torch.stack(x_list, dim=0).to(device)  # [T,3,512,512]

        outputs = model(x)  # [T,5] for ckpt, [T,1000] for fallback

        if use_imagenet_mapping:
            outputs = outputs @ imagenet_W + imagenet_b  # [T,5]

        image_pred = outputs.mean(dim=0, keepdim=True)  # [1,5]
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df = sub_df.merge(sample_sub[["image_id"]], on="image_id", how="right")
sub_df["label"] = sub_df["label"].fillna(0).astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print(f"Wrote submission.csv with shape={sub_df.shape}")
