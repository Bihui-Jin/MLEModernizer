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
import sys
import glob
import json
import time
import warnings

warnings.filterwarnings("ignore")



## === cell 1
INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
WORKING_DIR = "/kaggle/working"
MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")  # original notebook used this

TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

assert os.path.exists(INPUT_DIR), f"Missing competition input at {INPUT_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test images at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train images at {TRAIN_IMG_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample submission at {SAMPLE_SUB_PATH}"

os.makedirs(WORKING_DIR, exist_ok=True)



## === cell 2
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())



## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



## === cell 4
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
    midas.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform  # returns a torch Tensor on CPU
    HAS_MIDAS = True
    print("MiDaS loaded.")
except Exception as e:
    HAS_MIDAS = False
    print(
        "WARNING: MiDaS could not be loaded; falling back to identity masking. Error:",
        repr(e),
    )

    def transform(img_rgb_uint8):
        x = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0
        return x.unsqueeze(0)  # keep on CPU; caller moves to device




## === cell 5
def build_effnet_b1_num_classes(
    num_classes: int = 5, imagenet_backbone: bool = True
) -> nn.Module:
    """
    Keep the same core architecture (EfficientNet-B1).
    If no custom model.pth is available, use ImageNet weights and keep a 5-class head.
    """
    if imagenet_backbone:
        weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
    else:
        weights = None

    model = torchvision.models.efficientnet_b1(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_prefix_if_present(state_dict, prefix: str):
    if not isinstance(state_dict, dict):
        return state_dict
    if all(isinstance(k, str) and k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


found_model_path = None
candidate_paths = [
    MODEL_PATH,
    os.path.join(INPUT_DIR, "model.pth"),
    os.path.join("/kaggle/input", "cassava-leaf-disease-classification", "model.pth"),
    os.path.join("/kaggle/input", "pretrained", "for_kaggle", "model.pth"),
    os.path.join("/kaggle/input", "pretrained", "model.pth"),
]
for p in candidate_paths:
    if os.path.exists(p):
        found_model_path = p
        break

use_custom_ckpt = found_model_path is not None
model = build_effnet_b1_num_classes(5, imagenet_backbone=not use_custom_ckpt)

if use_custom_ckpt:
    print("Loading custom weights:", found_model_path)
    ckpt = torch.load(found_model_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _strip_prefix_if_present(state, "module.")
    state = _strip_prefix_if_present(state, "model.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("Warning: load_state_dict non-strict keys")
        print("  Missing:", missing[:20], "..." if len(missing) > 20 else "")
        print("  Unexpected:", unexpected[:20], "..." if len(unexpected) > 20 else "")
else:
    print(
        "WARNING: model.pth not found; will fine-tune ImageNet EfficientNet-B1 to learn a cassava 5-class head."
    )

model.to(device)



## === cell 6
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def process(image_rgb_uint8: np.ndarray) -> torch.Tensor:
    image = cv2.resize(image_rgb_uint8, (240, 240), interpolation=cv2.INTER_AREA)
    x = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
    x = (x - IMNET_MEAN) / IMNET_STD
    return x.to(device)


def apply_midas_mask(img_rgb_uint8: np.ndarray) -> np.ndarray:
    if not HAS_MIDAS:
        return img_rgb_uint8

    input_batch = transform(img_rgb_uint8)
    if isinstance(input_batch, torch.Tensor):
        input_batch = input_batch.to(device, non_blocking=True)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = (
            F.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb_uint8.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
    output = prediction.detach().cpu().numpy()

    mask = np.array(output > 4000, dtype=np.uint8)
    mask_3d = np.stack((mask, mask, mask), axis=2)
    masked_arr = np.where(mask_3d == 1, img_rgb_uint8, mask_3d).astype(np.uint8)
    return masked_arr




## === cell 7
train_df = pd.read_csv(TRAIN_CSV_PATH)
assert set(train_df.columns) == {"image_id", "label"}
train_df["path"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
train_df = train_df[train_df["path"].apply(os.path.exists)].reset_index(drop=True)

perm = np.random.RandomState(0).permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(trn_df), "Val size:", len(val_df))

counts = trn_df["label"].value_counts().sort_index()
counts = counts.reindex(range(5), fill_value=1)
class_weights = (counts.sum() / counts).values.astype(np.float32)
class_weights = class_weights / class_weights.mean()
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
print("Class weights:", class_weights)




## === cell 8
class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, use_mask: bool = True):
        self.df = df
        self.use_mask = use_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"])
        if img is None:
            img = np.zeros((240, 240, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.use_mask:
            img = apply_midas_mask(img)
        x = process(img).squeeze(0)  # [3,240,240]
        y = int(row["label"])
        return x, y




## === cell 9
BATCH_SIZE = 32 if torch.cuda.is_available() else 16
EPOCHS = (
    2 if not use_custom_ckpt else 1
)  # if already have custom ckpt, just a light touch
LR = 1e-4

NUM_WORKERS = 0
PIN_MEMORY = bool(torch.cuda.is_available()) and NUM_WORKERS > 0

train_loader = torch.utils.data.DataLoader(
    CassavaDataset(trn_df, use_mask=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

val_loader = torch.utils.data.DataLoader(
    CassavaDataset(val_df, use_mask=True),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)




## === cell 10
def evaluate_accuracy(model: nn.Module, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


t0 = time.time()
model.train()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    n = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * yb.size(0)
        n += yb.size(0)

    train_loss = running_loss / max(n, 1)
    val_acc = evaluate_accuracy(model, val_loader)
    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss={train_loss:.4f} - val_acc={val_acc:.4f}"
    )

print("Training time (s):", round(time.time() - t0, 2))
model.eval()

torch.save(model.state_dict(), MODEL_PATH)
print("Saved model to:", MODEL_PATH)



## === cell 11
files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*")))
assert len(files) > 0, f"No test images found in {TEST_IMG_DIR}"

names, labels = [], []

try:
    import tqdm

    iterator = tqdm.tqdm(files)
except Exception:
    iterator = files

for file in iterator:
    img = cv2.imread(file)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masked_arr = apply_midas_mask(img)
    topred = process(masked_arr)

    with torch.no_grad():
        out = model(topred)

    pred = int(torch.argmax(F.softmax(out, dim=1), dim=1).item())
    names.append(os.path.basename(file))
    labels.append(pred)

assert len(names) > 0, "No predictions were generated; check image loading."



## === cell 12
sub = pd.read_csv(SAMPLE_SUB_PATH)

pred_map = dict(zip(names, labels))
sub["label"] = sub["image_id"].map(pred_map).fillna(0).astype(int)

out_path = os.path.join(WORKING_DIR, "submission.csv")
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Unique image_ids:", sub["image_id"].nunique())

assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == 2676
