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
import json
import time
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from torchmetrics.functional import accuracy  # noqa: F401

from sklearn import metrics, model_selection, preprocessing  # noqa: F401
from PIL import Image

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    RandomCrop,
    MultiplicativeNoise,
)  # noqa: F401

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.set_num_interop_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Image.MAX_IMAGE_PIXELS = None
try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample submission at {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test images dir at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train images dir at {TRAIN_IMG_DIR}"




## === cell 2
class LitModel(pl.LightningModule):
    def __init__(self, classify, n_cls=5, pretrained=False, t_data=None, v_data=None):
        super().__init__()
        self.classify = classify
        self.n_cls = n_cls
        self.pre_trained = pretrained
        self.model = self.modified_model()

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = 0.1
        self.t_data = t_data
        self.v_data = v_data
        self.batch_size = 256

        self.logits = nn.Linear(512, self.n_cls)

    def forward(self, x):
        embeddings = self.model(x)
        if self.classify:
            logits = self.logits(embeddings)
            return logits
        else:
            return embeddings

    def modified_model(self):
        if self.pre_trained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
        )
        return model




## === cell 3
lit_model = LitModel(True, 5, pretrained=True)
lit_model.to(DEVICE)

for p in lit_model.model.parameters():
    p.requires_grad = False
for p in lit_model.logits.parameters():
    p.requires_grad = True




## === cell 4
class SWAResnet(pl.LightningModule):
    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model
        self.swa_model = AveragedModel(self.model)

    def forward(self, x):
        logit = self.swa_model(x)
        return logit




## === cell 5
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tfms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ]
)

val_tfms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

test_tfms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)




## === cell 6
class CassavaTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, tfms):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        x = self.tfms(img)
        y = int(row["label"])
        return x, y




## === cell 7
train_df = pd.read_csv(TRAIN_CSV_PATH)

trn_df, val_df = model_selection.train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"],
)

n_cls = 5
counts = (
    trn_df["label"]
    .value_counts()
    .reindex(range(n_cls), fill_value=0)
    .values.astype(np.float32)
)
counts = np.maximum(counts, 1.0)
inv = 1.0 / counts
cls_w = inv / inv.mean()
class_weights = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
lit_model.criterion = nn.CrossEntropyLoss(weight=class_weights)

train_ds = CassavaTrainDataset(trn_df, TRAIN_IMG_DIR, train_tfms)
val_ds = CassavaTrainDataset(val_df, TRAIN_IMG_DIR, val_tfms)

batch_size = 64 if DEVICE == "cuda" else 32

num_workers = min(4, (os.cpu_count() or 2))
common_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    **common_loader_kwargs,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    **common_loader_kwargs,
)



## === cell 8
base_lr_for_bs256 = 0.1
lr = base_lr_for_bs256 * (batch_size / 256.0)

optimizer = torch.optim.SGD(
    lit_model.logits.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)


def eval_acc(model_, loader_):
    model_.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader_:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model_(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


head_ckpt_paths = [
    "/kaggle/input/cassava-leaf-disease-classification/logits_head.pth",
    "/kaggle/input/logits_head.pth",
    "/kaggle/working/logits_head.pth",
]
loaded = False
for pth in head_ckpt_paths:
    if os.path.exists(pth):
        state = torch.load(pth, map_location="cpu")
        try:
            lit_model.logits.load_state_dict(state)
            loaded = True
            break
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
                sd2 = {
                    k.replace("logits.", ""): v
                    for k, v in sd.items()
                    if k.startswith("logits.")
                }
                if sd2:
                    lit_model.logits.load_state_dict(sd2)
                    loaded = True
                    break

best_head_path = "/kaggle/working/logits_head_best.pth"

if os.path.exists(best_head_path):
    try:
        lit_model.logits.load_state_dict(torch.load(best_head_path, map_location="cpu"))
        loaded = True
        print("Loaded cached best logits head weights.")
    except Exception:
        pass

if not loaded:
    lit_model.model.eval()
    lit_model.logits.train()

    epochs = 18
    start = time.time()

    best_va = -1.0
    best_state = None

    for ep in range(epochs):
        running_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                feats = lit_model.model(xb)
            logits = lit_model.logits(feats)

            loss = lit_model.criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        scheduler.step()

        lit_model.model.eval()
        lit_model.logits.eval()
        va = eval_acc(lit_model, val_loader)

        if va > best_va:
            best_va = va
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in lit_model.logits.state_dict().items()
            }

        lit_model.logits.train()

        elapsed = time.time() - start
        print(
            f"epoch={ep+1}/{epochs} train_loss={running_loss/max(1,n_seen):.4f} "
            f"val_acc={va:.4f} best_val_acc={best_va:.4f} elapsed={elapsed:.1f}s"
        )

    torch.save(lit_model.logits.state_dict(), "/kaggle/working/logits_head.pth")
    if best_state is not None:
        torch.save(best_state, best_head_path)
        lit_model.logits.load_state_dict(best_state)
else:
    print("Loaded cached logits head weights (best if available).")

lit_model.eval()



## === cell 9
final_model = lit_model
final_model.to(DEVICE)
final_model.eval()



## === cell 10
sample_submission_df = pd.read_csv(SAMPLE_SUB_PATH)


class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, img_dir, tfms):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        x = self.tfms(img)
        return img_id, x


test_ds = CassavaTestDataset(
    sample_submission_df["image_id"].tolist(), TEST_IMG_DIR, test_tfms
)
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=256 if DEVICE == "cuda" else 32,
    shuffle=False,
    **common_loader_kwargs,
)

predictions = []
image_id = []

with torch.no_grad():
    for ids, xb in test_loader:
        xb = xb.to(DEVICE, non_blocking=True)

        logits1 = final_model(xb)

        xb_flip = torch.flip(xb, dims=[3])  # flip width dimension (N,C,H,W)
        logits2 = final_model(xb_flip)

        logits = 0.5 * (logits1 + logits2)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()

        predictions.extend(pred)
        image_id.extend(list(ids))



## === cell 11
assert len(predictions) == len(sample_submission_df), (
    len(predictions),
    len(sample_submission_df),
)
assert len(image_id) == len(sample_submission_df), (
    len(image_id),
    len(sample_submission_df),
)

my_submission = pd.DataFrame({"image_id": image_id, "label": predictions})
my_submission["label"] = my_submission["label"].astype(int)

my_submission = (
    my_submission.set_index("image_id")
    .loc[sample_submission_df["image_id"]]
    .reset_index()
)

out_path = "submission.csv"
my_submission.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape={my_submission.shape}")
print(my_submission.head())
