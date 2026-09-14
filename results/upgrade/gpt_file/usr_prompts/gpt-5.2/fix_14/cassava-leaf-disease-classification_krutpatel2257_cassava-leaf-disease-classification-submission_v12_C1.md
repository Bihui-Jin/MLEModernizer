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

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

model_path = "../input/rnwcnwcttacalr/model(7).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
if not os.path.exists(test_images_path):
    test_images_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"

if not os.path.exists(train_csv_path):
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
if not os.path.exists(train_images_path):
    train_images_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True




## === cell 1
using_imagenet_fallback = False

model = models.resnext50_32x4d(weights=None)
model.fc = nn.Linear(2048, 5)

if os.path.exists(model_path):
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
else:
    using_imagenet_fallback = True
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

model.to(device)
model.eval()




## === cell 2
sub_aug = albumentations.Compose(
    [
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

sub_aug_hflip = albumentations.Compose(
    [
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

train_aug_fallback = albumentations.Compose(
    [
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 3
def hwc_float_to_chw_tensor(x: np.ndarray) -> torch.Tensor:
    """
    albumentations outputs HWC float32 (already normalized).
    Convert explicitly to avoid unintended scaling.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def _stratified_split_df(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    train_parts = []
    val_parts = []
    for lbl, g in df.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_parts.append(df.loc[val_idx])
        train_parts.append(df.loc[train_idx])
    train_df = (
        pd.concat(train_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return train_df, val_df


@torch.no_grad()
def _build_val_feature_cache(
    feature_extractor: nn.Module,
    df_val: pd.DataFrame,
    img_dir: str,
    aug,
    device: torch.device,
    batch_size: int = 128,
):
    feature_extractor.eval()

    feats_all = []
    y_all = []

    batch_imgs = []
    batch_y = []

    for _, row in df_val.iterrows():
        img_path = os.path.join(img_dir, row.image_id)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = aug(image=img)["image"]
        batch_imgs.append(hwc_float_to_chw_tensor(img))
        batch_y.append(int(row.label))

        if len(batch_imgs) >= batch_size:
            x = torch.stack(batch_imgs, dim=0).to(device)
            feats = feature_extractor(x).flatten(1).detach().cpu()
            feats_all.append(feats)
            y_all.append(torch.tensor(batch_y, dtype=torch.long))
            batch_imgs, batch_y = [], []

    if len(batch_imgs) > 0:
        x = torch.stack(batch_imgs, dim=0).to(device)
        feats = feature_extractor(x).flatten(1).detach().cpu()
        feats_all.append(feats)
        y_all.append(torch.tensor(batch_y, dtype=torch.long))

    if len(feats_all) == 0:
        return None, None
    return torch.cat(feats_all, dim=0), torch.cat(y_all, dim=0)


@torch.no_grad()
def _eval_head_accuracy_cached(
    head: nn.Module,
    val_feats_cpu: torch.Tensor,
    val_y_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    head.eval()
    if val_feats_cpu is None or val_y_cpu is None or val_feats_cpu.numel() == 0:
        return 0.0

    correct = 0
    total = int(val_y_cpu.numel())

    head = head.to(device)
    for i in range(0, total, batch_size):
        xb = val_feats_cpu[i : i + batch_size].to(device)
        yb = val_y_cpu[i : i + batch_size].to(device)
        logits = head(xb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())

    return correct / max(1, total)


def _cache_epoch_augmented_tensors(
    df: pd.DataFrame,
    img_dir: str,
    aug,
    epoch_seed: int,
):
    rng = np.random.RandomState(epoch_seed)
    imgs = []
    ys = []
    for i in range(len(df)):
        row = df.iloc[i]
        img_path = os.path.join(img_dir, row.image_id)
        if not os.path.exists(img_path):
            continue
        np.random.seed(int(rng.randint(0, 2**31 - 1)))
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = aug(image=img)["image"]
        imgs.append(hwc_float_to_chw_tensor(img))
        ys.append(int(row.label))
    if len(imgs) == 0:
        return None, None
    return torch.stack(imgs, dim=0), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def _extract_features_from_tensor_dataset(
    feature_extractor: nn.Module,
    x_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
):
    feature_extractor.eval()
    n = int(x_cpu.shape[0])
    feats = []
    for i in range(0, n, batch_size):
        xb = x_cpu[i : i + batch_size].to(device, non_blocking=True)
        fb = feature_extractor(xb).flatten(1).detach().cpu()
        feats.append(fb)
    return torch.cat(feats, dim=0)


def fit_fallback_linear_head(
    model_imagenet: nn.Module,
    train_csv: str,
    train_img_dir: str,
    train_aug,
    val_aug,
    device: torch.device,
    max_steps: int = 1200,  # kept for API compatibility; unused in epoch-based loop below
    batch_size: int = 64,
    val_frac: float = 0.1,
    val_check_every: int = 200,  # kept for API compatibility; replaced by per-epoch checks
) -> nn.Module:
    backbone = model_imagenet
    backbone.eval()

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()

    head = nn.Linear(2048, 5).to(device)

    df_all = pd.read_csv(train_csv)
    df_all["label"] = df_all["label"].astype(int)
    df_train, df_val = _stratified_split_df(df_all, val_frac=val_frac, seed=42)

    counts = df_train["label"].value_counts().sort_index()
    counts = counts.reindex(range(5), fill_value=1)
    inv = 1.0 / counts.to_numpy(dtype=np.float32)
    weights = inv / inv.mean()
    class_weights = torch.tensor(weights, device=device, dtype=torch.float32)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)

    opt = torch.optim.SGD(head.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)

    steps_per_epoch = max(1, int(np.ceil(len(df_train) / batch_size)))

    epochs = 12
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=0.001
    )

    val_feats_cpu, val_y_cpu = _build_val_feature_cache(
        feature_extractor=feature_extractor,
        df_val=df_val,
        img_dir=train_img_dir,
        aug=val_aug,
        device=device,
        batch_size=128,
    )

    best_state = None
    best_acc = -1.0

    head.train()
    for ep in range(epochs):
        df_epoch = df_train.sample(frac=1.0, random_state=42 + ep).reset_index(
            drop=True
        )

        x_cpu, y_cpu = _cache_epoch_augmented_tensors(
            df=df_epoch,
            img_dir=train_img_dir,
            aug=train_aug,
            epoch_seed=42 + ep,
        )
        if x_cpu is None:
            continue

        feats_cpu = _extract_features_from_tensor_dataset(
            feature_extractor=feature_extractor,
            x_cpu=x_cpu,
            device=device,
            batch_size=256,
        )

        n = int(feats_cpu.shape[0])
        for i in range(0, n, batch_size):
            feats = feats_cpu[i : i + batch_size].to(device, non_blocking=True)
            y = y_cpu[i : i + batch_size].to(device, non_blocking=True)

            logits = head(feats)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

        acc = _eval_head_accuracy_cached(
            head=head,
            val_feats_cpu=val_feats_cpu,
            val_y_cpu=val_y_cpu,
            device=device,
            batch_size=4096,
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    return head




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

tta_augs = [sub_aug, sub_aug_hflip]
batch_size = 32

fallback_head = None
fallback_feature_extractor = None
if using_imagenet_fallback:
    fallback_head = fit_fallback_linear_head(
        model_imagenet=model,
        train_csv=train_csv_path,
        train_img_dir=train_images_path,
        train_aug=train_aug_fallback,
        val_aug=sub_aug,
        device=device,
        max_steps=1800,
        batch_size=64,
        val_frac=0.1,
        val_check_every=200,
    )
    fallback_feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    fallback_feature_extractor.eval()

pred_image_ids = []
pred_labels = []

with torch.no_grad():
    batch_imgs = []
    batch_ids = []

    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        base_img = Image.open(img_path).convert("RGB")
        base_img = np.array(base_img)

        batch_imgs.append(base_img)
        batch_ids.append(sample_row.image_id)

        if len(batch_imgs) >= batch_size:
            probs_sum = None
            for aug in tta_augs:
                aug_tensors = [
                    hwc_float_to_chw_tensor(aug(image=img)["image"])
                    for img in batch_imgs
                ]
                x = torch.stack(aug_tensors, dim=0).to(device, non_blocking=True)

                if using_imagenet_fallback:
                    feats = fallback_feature_extractor(x).flatten(1)
                    logits5 = fallback_head(feats)
                    probs = torch.softmax(logits5, dim=1)
                else:
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)

                probs_sum = probs if probs_sum is None else (probs_sum + probs)

            probs_avg = probs_sum / len(tta_augs)
            batch_pred = (
                torch.argmax(probs_avg, dim=1).detach().cpu().numpy().astype(int)
            )

            pred_image_ids.extend(batch_ids)
            pred_labels.extend(batch_pred.tolist())

            batch_imgs = []
            batch_ids = []

    if len(batch_imgs) > 0:
        probs_sum = None
        for aug in tta_augs:
            aug_tensors = [
                hwc_float_to_chw_tensor(aug(image=img)["image"]) for img in batch_imgs
            ]
            x = torch.stack(aug_tensors, dim=0).to(device, non_blocking=True)

            if using_imagenet_fallback:
                feats = fallback_feature_extractor(x).flatten(1)
                logits5 = fallback_head(feats)
                probs = torch.softmax(logits5, dim=1)
            else:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)

            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        probs_avg = probs_sum / len(tta_augs)
        batch_pred = torch.argmax(probs_avg, dim=1).detach().cpu().numpy().astype(int)

        pred_image_ids.extend(batch_ids)
        pred_labels.extend(batch_pred.tolist())

sub_df = pd.DataFrame({"image_id": pred_image_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print(
    "Fallback used (trained linear head on frozen ImageNet backbone):",
    using_imagenet_fallback,
)
