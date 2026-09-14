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

# 5. Target score

0.8665760048352976

# 6. Current score

0.65321

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The original notebook depends on missing external packages/modules (`utils`, `model`, custom config and transform builders, and a weights file), so it can’t run in this Kaggle environment and never produces a valid-length submission. I remove those broken imports and replace them with a self-contained PyTorch inference pipeline that reads the provided Cassava image folders and writes `submission.csv` in the exact required format and length. To keep runtime under control without extra training, the solution uses a standard ImageNet-pretrained ResNet50 as a zero-shot baseline and outputs deterministic predictions aligned to `sample_submission.csv` ordering. This fixes all runtime errors and guarantees a valid `.csv` submission file is generated end-to-end.'
- What this solution (achieved 0.75897) has done: 'Your current score is extremely low because the random fixed projection from 1000 ImageNet logits to 5 cassava classes produces essentially arbitrary labels; the smallest legitimate improvement is to keep the same ResNet50 backbone and inference loop but replace the random projection with a proper 5-class head trained on the provided `train.csv` + `train_images`. To preserve core logic (still ResNet50 + cross-entropy) and stay within runtime, I freeze the backbone and train only the final FC layer for a small number of epochs, then run the same deterministic test-time inference and write `submission.csv` in the exact sample order. This should move accuracy substantially upward toward your target without changing the overall approach (single model, standard transforms, argmax labels). I also keep determinism and ensure paths are found robustly under the provided directory structure.'
- What this solution (achieved 0.6846) has done: 'Your current gap to the target is about 0.1076 (0.75897 → 0.86658), so we should improve accuracy while keeping your ResNet50 + “train only fc” approach intact. The biggest low-risk gain is to stop overfitting/instability from training on heavily augmented full data without any validation signal by adding a stratified train/val split, then training the same frozen-backbone fc head using the best validation checkpoint (no early stopping, fixed epochs). This preserves your model, loss, optimizer type, and overall loop semantics, but makes the selected weights much closer to what generalize to the test set. I also add standard class-weighted cross entropy (computed from train split) to address label imbalance with minimal code change and typically improves accuracy for this competition.'
- What this solution (achieved 0.65321) has done: 'Your current score (0.6846) is well below the target (0.8666), so we should improve generalization while preserving your exact core approach: ResNet50 backbone frozen + train only the final FC with cross-entropy and argmax inference. The biggest low-risk issue is that you’re selecting the “best checkpoint” based on raw val accuracy while training with class-weighted loss; this can over-favor rare classes and reduce overall accuracy, so we keep the weighted loss but select the checkpoint by unweighted val loss (more aligned with accuracy on a similarly distributed test set). Additionally, we (1) train the FC head on the full training set after the best epoch is identified (same model/loop, just a second pass) and (2) use lightweight test-time augmentation (horizontal flip averaged with original) during inference to nudge accuracy upward without changing the model or training semantics. These are minimal, targeted changes that typically move Cassava accuracy upward toward your target while staying within runtime and producing the same submission format.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input",  # fallback; may contain files directly
    "/kaggle/data",
]


def find_existing_path(candidates, must_contain=None):
    for p in candidates:
        if os.path.exists(p):
            if must_contain is None:
                return p
            ok = True
            for mc in must_contain:
                if not os.path.exists(os.path.join(p, mc)):
                    ok = False
                    break
            if ok:
                return p
    return None


DATA_ROOT = find_existing_path(
    DATA_ROOT_CANDIDATES, must_contain=["sample_submission.csv", "test_images"]
)
if DATA_ROOT is None:
    for base in DATA_ROOT_CANDIDATES:
        nested = os.path.join(base, "cassava-leaf-disease-classification")
        if os.path.exists(nested) and os.path.exists(
            os.path.join(nested, "sample_submission.csv")
        ):
            DATA_ROOT = nested
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset directory."
    )

TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

print("DATA_ROOT:", DATA_ROOT)
print(
    "TRAIN_DIR exists:",
    os.path.exists(TRAIN_DIR),
    "num_files:",
    (
        len([f for f in os.listdir(TRAIN_DIR) if f.endswith(".jpg")])
        if os.path.exists(TRAIN_DIR)
        else -1
    ),
)
print(
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
    "num_files:",
    len([f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")]),
)
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV_PATH))

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv columns"
print("sample_submission shape:", sample_sub.shape)

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert list(train_df.columns) == ["image_id", "label"], "Unexpected train.csv columns"
print("train.csv shape:", train_df.shape)
print("train label distribution:\n", train_df["label"].value_counts().sort_index())




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 3
class CassavaTrainDataset(Dataset):
    def __init__(self, train_dir, df, transform=None):
        self.train_dir = train_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        path = os.path.join(self.train_dir, image_id)
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class CassavaTestDataset(Dataset):
    def __init__(self, test_dir, image_ids, transform=None):
        self.test_dir = test_dir
        self.image_ids = list(image_ids)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.test_dir, image_id)
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, image_id


train_tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

val_tfms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

infer_tfms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def stratified_split(df, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in sorted(df["label"].unique()):
        idx = df.index[df["label"] == c].to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.append(idx[:n_val])
    val_idx = np.concatenate(val_idx)
    val_mask = df.index.isin(val_idx)
    return df.loc[~val_mask].reset_index(drop=True), df.loc[val_mask].reset_index(
        drop=True
    )


train_split_df, val_split_df = stratified_split(train_df, val_frac=0.15, seed=42)
print("train split:", train_split_df.shape, "val split:", val_split_df.shape)
print("val label distribution:\n", val_split_df["label"].value_counts().sort_index())

train_ds = CassavaTrainDataset(TRAIN_DIR, train_split_df, transform=train_tfms)
val_ds = CassavaTrainDataset(TRAIN_DIR, val_split_df, transform=val_tfms)
test_ds = CassavaTestDataset(
    TEST_DIR, sample_sub["image_id"].values, transform=infer_tfms
)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=64,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

assert len(test_ds) == len(
    sample_sub
), "Test dataset length must match sample_submission length"



## === cell 4
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 5)

for name, p in model.named_parameters():
    p.requires_grad = name.startswith("fc.")

model.to(device)

train_counts = train_split_df["label"].value_counts().sort_index()
counts = train_counts.reindex([0, 1, 2, 3, 4]).fillna(0).to_numpy(dtype=np.float64)
counts = np.maximum(counts, 1.0)
class_weights = counts.sum() / (5.0 * counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
val_criterion_unweighted = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.fc.parameters(), lr=3e-3, weight_decay=1e-2)

EPOCHS = 5

best_val_loss = float("inf")
best_state = None
best_epoch = -1

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    train_loss = running_loss / max(1, total)
    train_acc = correct / max(1, total)

    model.eval()
    v_correct = 0
    v_total = 0
    v_running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            v_correct += int((preds == labels).sum().item())
            v_total += int(labels.size(0))
            v_running_loss += float(
                val_criterion_unweighted(logits, labels).item()
            ) * labels.size(0)

    val_acc = v_correct / max(1, v_total)
    val_loss = v_running_loss / max(1, v_total)

    print(
        f"epoch {epoch+1}/{EPOCHS} - train loss(w): {train_loss:.4f} - train acc: {train_acc:.4f} - val loss: {val_loss:.4f} - val acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

print("best epoch by val loss:", best_epoch, "best val loss:", best_val_loss)

if best_state is not None:
    model.load_state_dict(best_state)

model.to(device)
model.eval()

full_train_ds = CassavaTrainDataset(TRAIN_DIR, train_df, transform=train_tfms)
full_train_loader = DataLoader(
    full_train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

full_counts = train_df["label"].value_counts().sort_index()
full_counts = full_counts.reindex([0, 1, 2, 3, 4]).fillna(0).to_numpy(dtype=np.float64)
full_counts = np.maximum(full_counts, 1.0)
full_class_weights = full_counts.sum() / (5.0 * full_counts)
full_class_weights = torch.tensor(
    full_class_weights, dtype=torch.float32, device=device
)
criterion_full = nn.CrossEntropyLoss(weight=full_class_weights)

optimizer_full = torch.optim.AdamW(model.fc.parameters(), lr=3e-3, weight_decay=1e-2)

model.train()
for epoch in range(max(1, best_epoch)):
    running_loss = 0.0
    total = 0
    correct = 0
    for imgs, labels in full_train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer_full.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion_full(logits, labels)
        loss.backward()
        optimizer_full.step()

        running_loss += float(loss.item()) * labels.size(0)
        total += int(labels.size(0))
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    print(
        f"full-train epoch {epoch+1}/{max(1, best_epoch)} - loss(w): {running_loss/max(1,total):.4f} - acc: {correct/max(1,total):.4f}"
    )

model.eval()


@torch.no_grad()
def predict_logits(x):
    return model(x)  # (B,5)




## === cell 5
preds_out = []
ids_out = []

with torch.no_grad():
    for batch_imgs, batch_ids in test_loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        logits1 = predict_logits(batch_imgs)
        logits2 = predict_logits(
            torch.flip(batch_imgs, dims=[3])
        )  # horizontal flip on width
        logits = (logits1 + logits2) / 2.0
        batch_pred = logits.argmax(dim=1).detach().cpu().numpy()
        preds_out.append(batch_pred)
        ids_out.extend(list(batch_ids))

preds_out = np.concatenate(preds_out, axis=0).astype(int)
assert len(ids_out) == len(sample_sub)
assert len(preds_out) == len(sample_sub)

sub = pd.DataFrame({"image_id": ids_out, "label": preds_out})

sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
assert sub["label"].notna().all()
sub["label"] = sub["label"].astype(int)

out_path = "./submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("submission shape:", sub.shape)



## === cell 6
check = pd.read_csv("./submission.csv")
assert list(check.columns) == ["image_id", "label"]
assert len(check) == len(sample_sub)
assert check["image_id"].nunique() == len(sample_sub)
assert check["label"].between(0, 4).all()
print("Submission file validated OK.")
