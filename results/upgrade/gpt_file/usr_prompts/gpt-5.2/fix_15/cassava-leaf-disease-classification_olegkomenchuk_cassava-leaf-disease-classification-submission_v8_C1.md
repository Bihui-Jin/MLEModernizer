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

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from albumentations import Compose, Normalize, CenterCrop, LongestMaxSize, PadIfNeeded
from albumentations.pytorch import ToTensorV2

import torchvision



## === cell 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 2
class Config:
    cfg = {
        "batch_size": 32,
        "num_workers": 6,
        "image_size": (
            456,
            456,
        ),  # final crop (H, W); EfficientNet-B5 default crop is 456
        "resize_longer": 512,  # (kept for compatibility; not used when using weights transforms)
        "use_torchvision_weights_transforms": True,
        "crop_tta": "5crop",  # options: "1crop" or "5crop"
        "num_classes": 5,
        "model_path": "/kaggle/input/cassava-leaf-disease-classification/efficientnet_b0_ra-3dd342df.pth",
        "max_train_per_class_for_prototypes": None,
        "tta_mode": "4flip",  # options: "2flip" (orig+H) or "4flip" (orig+H+V+HV)
        "backbone": "efficientnet_b5",
        "prototypes_per_class": 6,
        "kmeans_iters": 20,
        "prototype_agg": "lse",  # options: "max" or "lse"
        "lse_temperature": 12.0,
        "use_class_priors": True,
        "class_prior_strength": 0.30,
        "class_prior_eps": 1e-6,
    }




## === cell 3
base_dir = Path("/kaggle/input/cassava-leaf-disease-classification")
test_img_dir = base_dir / "test_images"
train_img_dir = base_dir / "train_images"

test_df = pd.read_csv(base_dir / "sample_submission.csv")
train_df = pd.read_csv(base_dir / "train.csv")

assert test_img_dir.exists(), f"Test image directory not found: {test_img_dir}"
assert train_img_dir.exists(), f"Train image directory not found: {train_img_dir}"
assert {"image_id", "label"}.issubset(
    test_df.columns
), "sample_submission.csv must have image_id and label columns."
assert {"image_id", "label"}.issubset(
    train_df.columns
), "train.csv must have image_id and label columns."

print("train_df:", train_df.shape, "test_df:", test_df.shape)




## === cell 4
class CassavaDataset(Dataset):
    def __init__(
        self,
        df,
        image_size,
        augments=None,
        img_dir=None,
        with_label=False,
        image_id_col="image_id",
        label_col="label",
    ):
        self.with_label = with_label
        self.image_size = image_size  # (H, W) final crop
        self.augments = augments
        self.img_dir = Path(img_dir) if img_dir is not None else Path(".")
        self.image_id_col = image_id_col
        self.label_col = label_col

        self.image_ids = df[image_id_col].tolist()
        if self.with_label:
            self.labels = df[label_col].astype(int).tolist()
        else:
            self.labels = None

    def __getitem__(self, idx):
        img_path = self.img_dir / self.image_ids[idx]
        image = cv.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.augments:
            image = self.augments(image=image)["image"]

        out = {"X": image}
        if self.with_label:
            out["y"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return out

    def __len__(self):
        return len(self.image_ids)




## === cell 5
def _get_backbone_weights_and_preproc(backbone: str):
    if backbone == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        crop_size = 224
        resize_size = 256
    elif backbone == "efficientnet_b3":
        weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        crop_size = 300
        resize_size = 320
    elif backbone == "efficientnet_b5":
        weights = torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
        crop_size = 456
        resize_size = 488
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    mean = list(weights.transforms().mean)
    std = list(weights.transforms().std)
    return weights, (crop_size, crop_size), resize_size, mean, std


BACKBONE = Config.cfg.get("backbone", "efficientnet_b0")
WEIGHTS, TV_CROP_HW, TV_RESIZE, TV_MEAN, TV_STD = _get_backbone_weights_and_preproc(
    BACKBONE
)


class Augments:
    if Config.cfg.get("use_torchvision_weights_transforms", True):
        test_augments = Compose(
            [
                LongestMaxSize(max_size=TV_RESIZE, interpolation=cv.INTER_LINEAR),
                PadIfNeeded(
                    min_height=TV_CROP_HW[0],
                    min_width=TV_CROP_HW[1],
                    border_mode=cv.BORDER_REFLECT_101,
                    value=None,
                    p=1.0,
                ),
                CenterCrop(height=TV_CROP_HW[0], width=TV_CROP_HW[1]),
                Normalize(mean=TV_MEAN, std=TV_STD, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
        Config.cfg["image_size"] = TV_CROP_HW
    else:
        test_augments = Compose(
            [
                LongestMaxSize(
                    max_size=Config.cfg["resize_longer"], interpolation=cv.INTER_LINEAR
                ),
                PadIfNeeded(
                    min_height=Config.cfg["image_size"][0],
                    min_width=Config.cfg["image_size"][1],
                    border_mode=cv.BORDER_REFLECT_101,
                    value=None,
                    p=1.0,
                ),
                CenterCrop(
                    height=Config.cfg["image_size"][0],
                    width=Config.cfg["image_size"][1],
                ),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )




## === cell 6
test_dataset = CassavaDataset(
    df=test_df,
    image_size=Config.cfg["image_size"],
    augments=Augments.test_augments,
    img_dir=test_img_dir,
    with_label=False,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), len(test_dataloader)




## === cell 7
def efficientnet(num_classes: int, backbone: str = "efficientnet_b0"):
    if backbone == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
    elif backbone == "efficientnet_b3":
        model = torchvision.models.efficientnet_b3(
            weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
    elif backbone == "efficientnet_b5":
        model = torchvision.models.efficientnet_b5(
            weights=torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features=in_features, out_features=num_classes, bias=True
    )
    return model


model = efficientnet(
    Config.cfg["num_classes"], backbone=Config.cfg.get("backbone", "efficientnet_b0")
).to(device)



## === cell 8
ckpt_path = Config.cfg.get("model_path", None)
if ckpt_path is not None and os.path.exists(ckpt_path):
    print(
        f"NOTE: Found checkpoint at {ckpt_path}, but skipping load because backbone={Config.cfg.get('backbone')} "
        "and the file is likely not compatible; using torchvision ImageNet pretrained weights for embeddings."
    )
else:
    print(
        "Using torchvision ImageNet pretrained weights for embeddings (no external checkpoint load)."
    )

model = model.to(device)




## === cell 9
def _get_feature_extractor(m: nn.Module) -> nn.Module:
    return m.features


@torch.no_grad()
def _embed_batch(
    feat_extractor: nn.Module, model: nn.Module, X: torch.Tensor
) -> torch.Tensor:
    f = feat_extractor(X)
    f = model.avgpool(f)
    f = torch.flatten(f, 1)
    f = torch.nn.functional.normalize(f, dim=1)
    return f


def _five_crop(X: torch.Tensor, crop_h: int, crop_w: int) -> list[torch.Tensor]:
    B, C, H, W = X.shape
    if H < crop_h or W < crop_w:
        pad_h = max(0, crop_h - H)
        pad_w = max(0, crop_w - W)
        X = torch.nn.functional.pad(X, (0, pad_w, 0, pad_h), mode="reflect")
        B, C, H, W = X.shape

    y0, x0 = 0, 0
    y1, x1 = 0, W - crop_w
    y2, x2 = H - crop_h, 0
    y3, x3 = H - crop_h, W - crop_w
    yc, xc = (H - crop_h) // 2, (W - crop_w) // 2

    crops = [
        X[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w],
        X[:, :, y1 : y1 + crop_h, x1 : x1 + crop_w],
        X[:, :, y2 : y2 + crop_h, x2 : x2 + crop_w],
        X[:, :, y3 : y3 + crop_h, x3 : x3 + crop_w],
        X[:, :, yc : yc + crop_h, xc : xc + crop_w],
    ]
    return crops


@torch.no_grad()
def extract_embeddings_tta(
    dataloader: DataLoader, model: nn.Module
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model.eval()
    feats_all = []
    ys_all = []
    feat_extractor = _get_feature_extractor(model).to(device)

    tta_mode = Config.cfg.get("tta_mode", "2flip")
    crop_tta = Config.cfg.get("crop_tta", "1crop")
    crop_h, crop_w = Config.cfg["image_size"]

    for batch in dataloader:
        X = batch["X"].to(device, non_blocking=True)

        if crop_tta == "5crop":
            crop_views = _five_crop(X, crop_h=crop_h, crop_w=crop_w)
        else:
            crop_views = [X]

        f_views = []
        for Xc in crop_views:
            f_list = []
            f_list.append(_embed_batch(feat_extractor, model, Xc))
            X_h = torch.flip(Xc, dims=[3])
            f_list.append(_embed_batch(feat_extractor, model, X_h))

            if tta_mode == "4flip":
                X_v = torch.flip(Xc, dims=[2])
                f_list.append(_embed_batch(feat_extractor, model, X_v))
                X_hv = torch.flip(X_h, dims=[2])
                f_list.append(_embed_batch(feat_extractor, model, X_hv))

            feats = torch.stack(f_list, dim=0).mean(dim=0)
            feats = torch.nn.functional.normalize(feats, dim=1)
            f_views.append(feats)

        feats = torch.stack(f_views, dim=0).mean(dim=0)
        feats = torch.nn.functional.normalize(feats, dim=1)

        feats_all.append(feats.detach().cpu())
        if "y" in batch:
            ys_all.append(batch["y"].detach().cpu())

    feats_all = torch.cat(feats_all, dim=0)
    ys = torch.cat(ys_all, dim=0) if len(ys_all) else None
    return feats_all, ys


def _kmeans_cosine(
    x: torch.Tensor, k: int, iters: int = 25, seed: int = 42
) -> torch.Tensor:
    x = x.to(torch.float32).contiguous()
    n, d = x.shape
    if n == 0:
        return torch.zeros((k, d), dtype=torch.float32)
    if n <= k:
        centers = x.clone()
        if centers.shape[0] < k:
            pad = centers[
                torch.randint(
                    0,
                    centers.shape[0],
                    (k - centers.shape[0],),
                    generator=torch.Generator().manual_seed(seed),
                )
            ]
            centers = torch.cat([centers, pad], dim=0)
        return torch.nn.functional.normalize(centers, dim=1)

    g = torch.Generator().manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    centers = x[init_idx].clone()

    for _ in range(iters):
        sims = x @ centers.t()
        assign = sims.argmax(dim=1)
        new_centers = []
        for j in range(k):
            mask = assign == j
            if mask.any():
                c = x[mask].mean(dim=0, keepdim=True)
            else:
                ridx = torch.randint(0, n, (1,), generator=g)
                c = x[ridx].clone()
            new_centers.append(c)
        centers = torch.cat(new_centers, dim=0)
        centers = torch.nn.functional.normalize(centers, dim=1)
    return centers


if Config.cfg["max_train_per_class_for_prototypes"] is None:
    train_df_capped = train_df.copy()
else:
    train_df_capped = (
        train_df.groupby("label", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(len(x), Config.cfg["max_train_per_class_for_prototypes"]),
                random_state=42,
            )
        )
        .reset_index(drop=True)
    )

print("Using train samples for prototypes:", train_df_capped.shape)

train_dataset = CassavaDataset(
    df=train_df_capped,
    image_size=Config.cfg["image_size"],
    augments=Augments.test_augments,
    img_dir=train_img_dir,
    with_label=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

train_emb, train_y = extract_embeddings_tta(train_loader, model)

num_classes = Config.cfg["num_classes"]
k = int(Config.cfg["prototypes_per_class"])
iters = int(Config.cfg["kmeans_iters"])

prototypes_per_class = []
for c in range(num_classes):
    mask = train_y == c
    x_c = train_emb[mask]
    x_c = torch.nn.functional.normalize(x_c.to(torch.float32), dim=1)
    centers = _kmeans_cosine(x_c, k=k, iters=iters, seed=42 + c)
    centers = torch.nn.functional.normalize(centers.to(torch.float32), dim=1)
    prototypes_per_class.append(centers)

prototypes_per_class = torch.stack(prototypes_per_class, dim=0).contiguous()  # [C,K,D]
print(
    "Prototypes per class shape:",
    prototypes_per_class.shape,
    "dtype:",
    prototypes_per_class.dtype,
    "device:",
    prototypes_per_class.device,
)

if Config.cfg.get("use_class_priors", True):
    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values.astype(np.float64)
    )
    priors = (counts + Config.cfg["class_prior_eps"]) / (
        counts.sum() + num_classes * Config.cfg["class_prior_eps"]
    )
    log_priors = np.log(priors).astype(np.float32)
    log_priors_t = torch.from_numpy(log_priors)  # CPU float32
else:
    log_priors_t = None



## === cell 10
model.eval()

y_prediction = []
feat_extractor = model.features
tta_mode = Config.cfg.get("tta_mode", "2flip")
crop_tta = Config.cfg.get("crop_tta", "1crop")
crop_h, crop_w = Config.cfg["image_size"]

C, K, D = prototypes_per_class.shape
prototypes_flat = prototypes_per_class.view(C * K, D).contiguous()  # CPU float32

agg_mode = Config.cfg.get("prototype_agg", "max")
tau = float(Config.cfg.get("lse_temperature", 12.0))

prior_strength = float(Config.cfg.get("class_prior_strength", 0.0))
use_priors = (log_priors_t is not None) and (prior_strength != 0.0)

for batch in test_dataloader:
    with torch.no_grad():
        X_test = batch["X"].to(device, non_blocking=True)

        if crop_tta == "5crop":
            crop_views = _five_crop(X_test, crop_h=crop_h, crop_w=crop_w)
        else:
            crop_views = [X_test]

        feats_views = []
        for Xc in crop_views:
            f_list = []
            f_list.append(_embed_batch(feat_extractor, model, Xc))
            X_h = torch.flip(Xc, dims=[3])
            f_list.append(_embed_batch(feat_extractor, model, X_h))

            if tta_mode == "4flip":
                X_v = torch.flip(Xc, dims=[2])
                f_list.append(_embed_batch(feat_extractor, model, X_v))
                X_hv = torch.flip(X_h, dims=[2])
                f_list.append(_embed_batch(feat_extractor, model, X_hv))

            feats = torch.stack(f_list, dim=0).mean(dim=0)
            feats = torch.nn.functional.normalize(feats, dim=1)
            feats_views.append(feats)

        feats = torch.stack(feats_views, dim=0).mean(dim=0)
        feats = torch.nn.functional.normalize(feats, dim=1)  # [B, D]

        sims_all = (
            feats.detach().cpu().to(torch.float32) @ prototypes_flat.t()
        )  # [B, C*K]
        sims_all = sims_all.view(sims_all.shape[0], C, K)  # [B, C, K]

        if agg_mode == "lse":
            sims = torch.logsumexp(sims_all * tau, dim=2) / tau  # [B, C]
        else:
            sims = sims_all.max(dim=2).values  # [B, C]

        if use_priors:
            sims = sims + prior_strength * log_priors_t.view(1, -1)

        preds = sims.argmax(dim=1).numpy().tolist()
        y_prediction.extend(preds)

print(
    "Predictions:",
    len(y_prediction),
    "examples; unique labels:",
    sorted(set(y_prediction))[:10],
)



## === cell 11
submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": y_prediction}
)
assert (
    submission.shape[0] == test_df.shape[0]
), "Prediction length mismatch with sample submission."

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv")
