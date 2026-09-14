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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageChops, ImageOps
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import (
    vit_h_14,
    ViT_H_14_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.isdir(
        os.path.join(p, "train_images")
    ):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset root. Expected train.csv + train_images under one of: "
        + ", ".join(DATA_ROOT_CANDIDATES)
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("Device:", device)




## === cell 1
def invert_square_pad(img: Image.Image) -> Image.Image:
    width, height = img.size
    img2 = ImageChops.offset(img, width // 2, height // 2)

    max_side = max(width, height)
    pad_left = (max_side - width) // 2
    pad_top = (max_side - height) // 2
    pad_right = (max_side - width) - pad_left
    pad_bottom = (max_side - height) - pad_top

    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return img2

    base = img2
    mirror_x = ImageOps.mirror(base)
    mirror_y = ImageOps.flip(base)
    mirror_xy = ImageOps.mirror(mirror_y)

    w, h = base.size
    canvas = Image.new("RGB", (w * 3, h * 3))
    tiles = [
        [mirror_xy, mirror_y, mirror_xy],
        [mirror_x, base, mirror_x],
        [mirror_xy, mirror_y, mirror_xy],
    ]
    for r in range(3):
        for c in range(3):
            canvas.paste(tiles[r][c], (c * w, r * h))

    left = w + (w - width) // 2 - pad_left
    top = h + (h - height) // 2 - pad_top
    right = left + width + pad_left + pad_right
    bottom = top + height + pad_top + pad_bottom

    padded = canvas.crop((left, top, right, bottom))
    return padded


torch_transforms_ResNet = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((518, 518)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

torch_transforms_EfficientNet = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((480, 480)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

torch_transforms_DenseNet = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## === cell 2
NUM_CLASSES = 5


def build_torch_model_backbone(name: str):
    if name == "vit_h_14":
        m = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, NUM_CLASSES)
        return m
    if name == "efficientnet_v2_l":
        m = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        return m
    if name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, NUM_CLASSES)
        return m
    if name == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, NUM_CLASSES)
        return m
    raise ValueError(f"Unknown backbone: {name}")


model1 = build_torch_model_backbone("densenet121").to(device).eval()
model2 = build_torch_model_backbone("resnet50").to(device).eval()
model3 = build_torch_model_backbone("vit_h_14").to(device).eval()
model4 = build_torch_model_backbone("efficientnet_v2_l").to(device).eval()

print("Models initialized (torchvision pretrained backbones).")




## === cell 3
class CassavaImageDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.has_label = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"]) if self.has_label else -1
        img = Image.open(os.path.join(self.image_dir, image_id)).convert("RGB")
        return image_id, img, y


def pil_collate_fn(batch):
    image_ids, imgs, ys = zip(*batch)
    ys = torch.tensor(ys, dtype=torch.long)
    return list(image_ids), list(imgs), ys


@torch.no_grad()
def _predict_features_batch_from_pil_list(pil_imgs, batch_size_gpu: int) -> np.ndarray:
    """
    Core logic preserved: for each image, compute softmax probabilities from 4 backbones,
    then concatenate [p1,p2,p3,p4] into a 20-dim feature vector.
    Speed fix: do it in batches per backbone to reduce Python overhead and GPU launch overhead.
    """
    n = len(pil_imgs)
    out = np.empty((n, NUM_CLASSES * 4), dtype=np.float32)

    for start in range(0, n, batch_size_gpu):
        end = min(start + batch_size_gpu, n)
        imgs = pil_imgs[start:end]

        x1 = torch.stack([torch_transforms_DenseNet(im) for im in imgs], dim=0)
        x2 = torch.stack([torch_transforms_ResNet(im) for im in imgs], dim=0)
        x3 = torch.stack([torch_transforms_VIT(im) for im in imgs], dim=0)
        x4 = torch.stack([torch_transforms_EfficientNet(im) for im in imgs], dim=0)

        if device.type == "cuda":
            x1 = x1.pin_memory().to(device, non_blocking=True)
            x2 = x2.pin_memory().to(device, non_blocking=True)
            x3 = x3.pin_memory().to(device, non_blocking=True)
            x4 = x4.pin_memory().to(device, non_blocking=True)
        else:
            x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)

        p1 = (
            F.softmax(model1(x1), dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        p2 = (
            F.softmax(model2(x2), dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        p3 = (
            F.softmax(model3(x3), dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        p4 = (
            F.softmax(model4(x4), dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        out[start:end, :] = np.concatenate([p1, p2, p3, p4], axis=1)

    return out


def extract_features_dataframe(
    df: pd.DataFrame, image_dir: str, *, loader_bs: int, gpu_bs: int
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    ds = CassavaImageDataset(df, image_dir)
    num_workers = min(4, os.cpu_count() or 1)
    dl = DataLoader(
        ds,
        batch_size=loader_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pil_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    X = np.empty((len(ds), NUM_CLASSES * 4), dtype=np.float32)
    y = None
    image_ids_all = [None] * len(ds)

    ofs = 0
    for b, (image_ids, imgs, ys) in enumerate(dl):
        feats = _predict_features_batch_from_pil_list(imgs, batch_size_gpu=gpu_bs)
        bs = feats.shape[0]
        X[ofs : ofs + bs] = feats
        image_ids_all[ofs : ofs + bs] = image_ids
        if (y is None) and ("label" in df.columns):
            y = np.empty((len(ds),), dtype=np.int64)
        if y is not None:
            y[ofs : ofs + bs] = ys.numpy()

        ofs += bs
        if (ofs % 200 == 0) or (ofs == len(ds)):
            print(f"Features: {ofs}/{len(ds)}", end="\r")
    print()
    return X, y, image_ids_all




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain columns: image_id, label")

MAX_META_TRAIN = 6000
if len(train_df) > MAX_META_TRAIN:
    meta_train_df, _ = train_test_split(
        train_df,
        train_size=MAX_META_TRAIN,
        stratify=train_df["label"],
        random_state=SEED,
    )
else:
    meta_train_df = train_df

meta_train_df = meta_train_df.reset_index(drop=True)
print("Meta-train size:", len(meta_train_df))

X_meta, y_meta, _ = extract_features_dataframe(
    meta_train_df, TRAIN_DIR, loader_bs=32, gpu_bs=16
)



## === cell 5
decision_tree = RandomForestClassifier(
    n_estimators=90,
    criterion="gini",
    max_depth=6,
    random_state=42,
    n_jobs=-1,
)
decision_tree.fit(X_meta, y_meta)
print("Meta-model fitted.")



## === cell 6
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Missing test_images directory at: {TEST_DIR}")

image_ids = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
print("Test images:", len(image_ids))

test_df = pd.DataFrame({"image_id": image_ids})
combined_output, _, image_ids_out = extract_features_dataframe(
    test_df, TEST_DIR, loader_bs=32, gpu_bs=16
)

if image_ids_out != image_ids:
    raise RuntimeError(
        "Image order mismatch during feature extraction; submission would misalign."
    )

prediction = decision_tree.predict(combined_output).astype(int)
print("Predictions:", prediction.shape, "unique:", np.unique(prediction))

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    if "image_id" in sample.columns and len(sample) == len(submission):
        submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
        if submission["label"].isna().any():
            raise RuntimeError(
                "Submission merge produced NaNs; check image_id alignment."
            )
        submission["label"] = submission["label"].astype(int)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
submission.head()
