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

# 5. Target score

0.8420973103656694

# 6. Current score

0.65882

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11547) has done: 'I fix the missing model file issue by making the script automatically fall back to a torchvision ImageNet-pretrained SqueezeNet when the external `.pth` file isn’t available, so inference can run end-to-end and produce `submission.csv`. I also update the Albumentations pipeline to the v2 API (which caused your `RandomResizedCrop` validation error) while keeping the same augmentation intent and normalization. Finally, I fix the tensor preprocessing so the augmented output is correctly converted to a CHW float tensor on the right device, and ensure inference runs under `torch.no_grad()` to avoid unnecessary memory usage. These changes are required for correctness (runtime) and should yield a reasonable baseline accuracy without changing the overall modeling approach.'
- What this solution (achieved 0.73991) has done: 'Your current score is low because the fallback model uses an ImageNet-pretrained SqueezeNet feature extractor but a randomly initialized 5-class classifier head, so predictions are essentially random. To move the score toward your target with minimal change and without altering the overall approach, I freeze the pretrained feature extractor and “calibrate” only the final 5-class head using the provided `train.csv` + `train_images` for a short, deterministic fine-tune. I also make inference preprocessing consistent with training by using a deterministic resize/center-crop (no heavy augmentation) while keeping the same normalization and the same TTA loop structure (now using repeated deterministic transforms, which is equivalent but stable). This should significantly increase accuracy versus the random head while preserving the same model family, loss (cross-entropy), and inference semantics (argmax over logits, submission format unchanged).'
- What this solution (achieved 0.6506) has done: 'We keep your SqueezeNet + “train only the classifier head” approach, but make a few minimal changes that usually raise accuracy significantly without changing the core model/loop: (1) match SqueezeNet’s expected 224 input resolution (your 256 center-crop is slightly off), (2) use a stratified train/val split and train based on validation accuracy (still cross-entropy, still only head trainable) to avoid overfitting and to ensure the head is actually learning, (3) apply class-weighted cross-entropy to counter the known label imbalance in Cassava, and (4) make TTA actually stochastic (light, label-preserving crops/flip) so averaging helps instead of repeating identical transforms. These are small, safe adjustments aimed at moving your 0.73991 toward the 0.842 target without changing the overall method.'
- What this solution (achieved 0.65882) has done: 'Your current score (0.6506) is well below the target (0.8421), so we should increase accuracy with minimal, low-risk changes that keep the same SqueezeNet + “train only head” core logic. The biggest likely drag is train/infer mismatch: you train with deterministic center-crop but infer with heavy random crops, which can harm accuracy; I switch TTA to a light, label-preserving resize+center-crop with only horizontal flip so averaging helps instead of injecting too much noise. I also add a standard RandomResizedCrop during head-training (still same model/loss/loop) to improve robustness while keeping resolution at 224. Finally, I make inference much faster (batch DataLoader + precomputed flip pair) so we can safely increase TTA passes without risking timeout, which usually nudges accuracy upward toward your target.'

# 9. Code solution

## === cell 0
import os
import warnings
import random

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models



## === cell 1
model_path = "../input/sn-wc-aug-full/model(5).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"


def _resolve_path(p, alt):
    return alt if (not os.path.exists(p) and os.path.exists(alt)) else p


sample_sub_path = _resolve_path(
    sample_sub_path,
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
)
test_images_path = _resolve_path(
    test_images_path, "/kaggle/input/cassava-leaf-disease-classification/test_images"
)
train_csv_path = _resolve_path(
    train_csv_path, "/kaggle/input/cassava-leaf-disease-classification/train.csv"
)
train_images_path = _resolve_path(
    train_images_path, "/kaggle/input/cassava-leaf-disease-classification/train_images"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("sample_sub_path exists:", os.path.exists(sample_sub_path), sample_sub_path)
print("test_images_path exists:", os.path.exists(test_images_path), test_images_path)
print("train_csv_path exists:", os.path.exists(train_csv_path), train_csv_path)
print("train_images_path exists:", os.path.exists(train_images_path), train_images_path)
print("model_path exists:", os.path.exists(model_path), model_path)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 2
use_pretrained_backbone = not os.path.exists(model_path)

if use_pretrained_backbone:
    warnings.warn(
        f"Model weights not found at {model_path}. "
        "Falling back to torchvision ImageNet-pretrained SqueezeNet and training ONLY the final 5-class head "
        "on the provided train.csv/train_images to avoid a random head (which scores near chance)."
    )

try:
    if use_pretrained_backbone:
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    else:
        model = models.squeezenet1_0(weights=None)
except Exception:
    model = models.squeezenet1_0(pretrained=use_pretrained_backbone)

model.classifier[1] = nn.Conv2d(
    in_channels=512, out_channels=5, kernel_size=(1, 1), stride=(1, 1)
)
model = model.to(device)

if not use_pretrained_backbone:
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

model.eval()



## === cell 3
sub_aug = A.Compose(
    [
        A.Resize(height=256, width=256, p=1.0),
        A.CenterCrop(height=224, width=224, p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 4
class CassavaTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_dir, transform):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row.image_id)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        aug_img = self.transform(image=img_np)["image"]  # HWC float32 normalized
        x = torch.from_numpy(aug_img).permute(2, 0, 1).contiguous().float()
        y = int(row.label)
        return x, y


train_tfms = A.Compose(
    [
        A.RandomResizedCrop(
            size=(224, 224),
            scale=(0.75, 1.0),
            ratio=(0.90, 1.11),
            interpolation=1,  # cv2.INTER_LINEAR; keep simple and stable
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

tta_aug_center = A.Compose(
    [
        A.Resize(height=256, width=256, p=1.0),
        A.CenterCrop(height=224, width=224, p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

tta_aug_flip = A.Compose(
    [
        A.Resize(height=256, width=256, p=1.0),
        A.CenterCrop(height=224, width=224, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



## === cell 5
if use_pretrained_backbone:
    for p in model.features.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable_params))

    train_df = pd.read_csv(train_csv_path)

    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    val_frac = 0.1
    tr_parts, va_parts = [], []
    for lbl, grp in train_df.groupby("label"):
        g = grp.sample(frac=1.0, random_state=SEED)
        n_val = max(1, int(round(len(g) * val_frac)))
        va_parts.append(g.iloc[:n_val])
        tr_parts.append(g.iloc[n_val:])
    tr_df = (
        pd.concat(tr_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    )
    va_df = (
        pd.concat(va_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    )

    print("train/val sizes:", len(tr_df), len(va_df))

    ds_tr = CassavaTrainDataset(tr_df, train_images_path, train_tfms)
    ds_va = CassavaTrainDataset(va_df, train_images_path, sub_aug)

    batch_size = 64 if torch.cuda.is_available() else 16
    loader_tr = torch.utils.data.DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    loader_va = torch.utils.data.DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    counts = tr_df["label"].value_counts().sort_index()
    counts = counts.reindex(range(5), fill_value=1)
    weights = (counts.sum() / counts).astype(np.float32).values
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("class weights:", class_weights.detach().cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(trainable_params, lr=2e-3)

    best_val_acc = -1.0
    best_state = None

    model.train()
    epochs = 4
    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1).squeeze(-1)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(xb.size(0))

        train_acc = correct / max(1, total)

        model.eval()
        va_correct, va_total = 0, 0
        with torch.no_grad():
            for xb, yb in loader_va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb).squeeze(-1).squeeze(-1)
                pred = torch.argmax(logits, dim=1)
                va_correct += int((pred == yb).sum().item())
                va_total += int(xb.size(0))
        val_acc = va_correct / max(1, va_total)

        print(
            f"epoch {ep+1}/{epochs} - loss: {running_loss/max(1,total):.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }

        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    model.eval()
    print("best_val_acc:", best_val_acc)




## === cell 6
class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_dir, transform):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx].image_id
        img_path = os.path.join(self.images_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        aug_img = self.transform(image=img_np)["image"]
        x = torch.from_numpy(aug_img).permute(2, 0, 1).contiguous().float()
        return image_id, x


sample_sub = pd.read_csv(sample_sub_path)

test_bs = 128 if torch.cuda.is_available() else 16

ds_test_center = CassavaTestDataset(sample_sub, test_images_path, tta_aug_center)
ds_test_flip = CassavaTestDataset(sample_sub, test_images_path, tta_aug_flip)

loader_center = torch.utils.data.DataLoader(
    ds_test_center,
    batch_size=test_bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
loader_flip = torch.utils.data.DataLoader(
    ds_test_flip,
    batch_size=test_bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model.eval()

all_logits = None
all_ids = []

with torch.no_grad():
    logits_list = []
    for image_ids, xb in loader_center:
        xb = xb.to(device, non_blocking=True)
        out = model(xb).squeeze(-1).squeeze(-1)  # [B,5]
        logits_list.append(out.detach().cpu())
        all_ids.extend(list(image_ids))
    all_logits = torch.cat(logits_list, dim=0)

    logits_list = []
    ids2 = []
    for image_ids, xb in loader_flip:
        xb = xb.to(device, non_blocking=True)
        out = model(xb).squeeze(-1).squeeze(-1)
        logits_list.append(out.detach().cpu())
        ids2.extend(list(image_ids))
    logits_flip = torch.cat(logits_list, dim=0)

if all_ids != ids2:
    raise RuntimeError("Test loader order mismatch between center and flip TTA.")

avg_logits = (all_logits + logits_flip) / 2.0
pred_labels = torch.argmax(avg_logits, dim=1).numpy().astype(int)

sub_df = pd.DataFrame({"image_id": all_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
