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
import sys
import copy
import datetime
import random
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

import cv2


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

DATA_ROOT = "../input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.exists(DATA_ROOT):
    alt_root = "/kaggle/input/cassava-leaf-disease-classification"
    if os.path.exists(alt_root):
        DATA_ROOT = alt_root
        TEST_DIR = os.path.join(DATA_ROOT, "test_images")
        TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
        TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
        SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing test directory: {TEST_DIR}"
assert os.path.exists(TRAIN_DIR), f"Missing train directory: {TRAIN_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
)
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model = model.to(device)

weights_path = "../input/wwwwww/weight_epoch_14.pth"


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(ckpt):
    """
    Why: Many Kaggle checkpoints are wrapped (Lightning, custom dicts).
    Extract the actual parameter dict without changing model/inference semantics.
    """
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema_state_dict",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        tensor_like = 0
        total = 0
        for k, v in ckpt.items():
            total += 1
            if isinstance(k, str) and torch.is_tensor(v):
                tensor_like += 1
        if total > 0 and tensor_like / total > 0.8:
            return ckpt
    return ckpt


def _filter_state_by_shape(state, model_state):
    """
    Why: If the found checkpoint is close but has extra keys or mismatched heads,
    we still want to load all matching layers (including fc when shapes match).
    This is safer than loading random weights and keeps architecture unchanged.
    """
    if not isinstance(state, dict):
        return state, [], list(model_state.keys())

    kept = {}
    dropped = []
    for k, v in state.items():
        if (
            k in model_state
            and torch.is_tensor(v)
            and torch.is_tensor(model_state[k])
            and v.shape == model_state[k].shape
        ):
            kept[k] = v
        else:
            dropped.append(k)
    missing = [k for k in model_state.keys() if k not in kept]
    return kept, dropped, missing


def find_weight_fallback():
    """
    Why: Current score indicates weights usually aren't found/loaded.
    Prefer resnet50-like cassava checkpoints by filename + approximate size.
    """
    direct = glob.glob("/kaggle/input/**/weight_epoch_14.pth", recursive=True)
    if direct:
        return direct[0]

    patterns = [
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
        "/kaggle/input/**/*.bin",
    ]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p, recursive=True))

    def score_path(p):
        b = os.path.basename(p).lower()
        s = 0

        if "cassava" in b:
            s += 8
        if "resnet50" in b or ("resnet" in b and "50" in b) or "r50" in b:
            s += 6
        if "epoch_14" in b or "epoch14" in b:
            s += 4
        if "best" in b:
            s += 4
        if "checkpoint" in b or "ckpt" in b:
            s += 1

        if "effnet" in b or "efficientnet" in b:
            s -= 6
        if "tf" in b or "tflite" in b or "keras" in b:
            s -= 8

        try:
            sz = os.path.getsize(p)
            if sz < 200_000:
                s -= 10
            elif 60_000_000 <= sz <= 140_000_000:
                s += 8
            elif 20_000_000 <= sz <= 400_000_000:
                s += 2
            else:
                s -= 1
        except Exception:
            pass

        return s

    scored = sorted(((score_path(p), p) for p in all_files), reverse=True)
    if scored:
        best_score, best_path = scored[0]
        if best_score >= 10:
            return best_path
    return None


loaded = False
loaded_full = False
loaded_filtered = False

if not os.path.exists(weights_path):
    fb = find_weight_fallback()
    if fb is not None:
        weights_path = fb

if os.path.exists(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _strip_module_prefix(state)

    model_state = model.state_dict()

    try:
        model.load_state_dict(state, strict=True)
        loaded = True
        loaded_full = True
    except Exception:
        filtered_state, dropped_keys, missing_keys = _filter_state_by_shape(
            state, model_state
        )
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        loaded = True
        loaded_filtered = True

    fc_w = model_state["fc.weight"].shape
    fc_b = model_state["fc.bias"].shape
    print(f"Loaded checkpoint: {weights_path}")
    print(f"Load mode: {'STRICT_FULL' if loaded_full else 'FILTERED_PARTIAL'}")
    if loaded_filtered and isinstance(state, dict):
        print(f"Filtered state_dict keys kept: {len(filtered_state)} / {len(state)}")
        print(
            f"load_state_dict reports Missing={len(missing)} Unexpected={len(unexpected)}"
        )
        fc_w_loaded = ("fc.weight" in filtered_state) and (
            filtered_state["fc.weight"].shape == fc_w
        )
        fc_b_loaded = ("fc.bias" in filtered_state) and (
            filtered_state["fc.bias"].shape == fc_b
        )
        print(f"fc loaded? weight={fc_w_loaded} bias={fc_b_loaded}")
        if len(filtered_state) < 50:
            print(
                "WARNING: Very few keys matched your model; checkpoint likely different architecture."
            )
else:
    print(
        f"Cassava weights not found at {weights_path}. Using ImageNet-pretrained ResNet50 backbone + random 5-class head."
    )

model.eval()



## === cell 2
img_size = 512
transforms = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].astype(str).tolist()
len(test_image_ids), test_image_ids[:3]



## === cell 3
train_df = pd.read_csv(TRAIN_CSV_PATH)
assert {"image_id", "label"}.issubset(train_df.columns)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)


class TrainSet(Dataset):
    def __init__(self, img_dir: str, df: pd.DataFrame, transforms):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        path = os.path.join(self.img_dir, fn)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = (
            self.transforms(img)
            if self.transforms is not None
            else torch.from_numpy(img)
        )
        return x, y


do_fc_train = not loaded_full

if do_fc_train:
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")
    model.train()

    rng = np.random.RandomState(42)
    idxs = np.arange(len(train_df))
    rng.shuffle(idxs)
    val_frac = 0.1
    val_n = int(round(len(idxs) * val_frac))
    val_idxs = idxs[:val_n]
    tr_idxs = idxs[val_n:]

    tr_df = train_df.iloc[tr_idxs].reset_index(drop=True)
    va_df = train_df.iloc[val_idxs].reset_index(drop=True)

    train_set = TrainSet(TRAIN_DIR, tr_df, transforms)
    val_set = TrainSet(TRAIN_DIR, va_df, transforms)

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # keep CUDA stability in Kaggle
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    class_counts = (
        train_df["label"].value_counts().reindex(range(5), fill_value=1).values
    )
    class_weights = (class_counts.sum() / class_counts).astype(np.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, device=device, dtype=torch.float32)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    epochs = 3
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            epoch_loss += float(loss.detach().cpu().item()) * bs
            n += bs

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = torch.as_tensor(yb, device=device, dtype=torch.long)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        val_acc = correct / max(1, total)
        avg_loss = epoch_loss / max(1, n)
        print(
            f"fc train ep={ep}/{epochs}: train_loss={avg_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(
        f"Selected best fc by val_acc={best_val_acc:.4f} (split {len(tr_df)}/{len(va_df)})"
    )
else:
    print(
        "Skipping fc training because a full checkpoint was loaded (assumed cassava-trained)."
    )

model.eval()




## === cell 4
class InferSet(Dataset):
    def __init__(self, img_dir: str, image_ids, transforms):
        self.img_dir = img_dir
        self.image_ids = list(image_ids)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        path = os.path.join(self.img_dir, fn)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = (
            self.transforms(img)
            if self.transforms is not None
            else torch.from_numpy(img)
        )
        return x, fn


infer_set = InferSet(TEST_DIR, test_image_ids, transforms)

infer_loader = DataLoader(
    infer_set,
    batch_size=32,
    shuffle=False,
    num_workers=0,  # keep CUDA stability in Kaggle
    pin_memory=torch.cuda.is_available(),
)

xb, fnb = next(iter(infer_loader))
xb.shape, fnb[:3]



## === cell 5
preds_all = []
fns_all = []

with torch.no_grad():
    model.eval()
    for xb, fns in infer_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
        preds_all.append(preds)
        fns_all.extend(list(fns))

preds_all = np.concatenate(preds_all, axis=0)
assert len(preds_all) == len(test_image_ids) == len(fns_all)

sub = pd.DataFrame({"image_id": fns_all, "label": preds_all})
sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
assert (
    sub["label"].notna().all()
), "Some predictions are missing after merge; check filenames."

out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
out_path, sub.shape, sub.head()



## === cell 6
check = pd.read_csv("./submission.csv")
print(check.columns.tolist(), len(check))
print(check.head())
assert check.columns.tolist() == ["image_id", "label"]
assert len(check) == len(
    sample_sub
), "Submission must match sample_submission row count."
print("submission.csv is ready.")
