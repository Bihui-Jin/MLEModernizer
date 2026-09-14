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
import glob
import random
import cv2
import numpy as np
import torch
import timm
import pandas as pd
import tqdm
from PIL import Image

torch.backends.cudnn.benchmark = True

DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("DEVICE:", DEVICE)
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print(
    "TEST_IMG_DIR exists:",
    os.path.exists(TEST_IMG_DIR),
    "num_files:",
    len(glob.glob(os.path.join(TEST_IMG_DIR, "*"))),
)
print(
    "TRAIN_IMG_DIR exists:",
    os.path.exists(TRAIN_IMG_DIR),
    "num_files:",
    len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*"))),
)




## === cell 1
def load_state_dict_flexible(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(unexpected) > 0:
        print(
            f"Warning: unexpected keys while loading {os.path.basename(path)} (showing up to 5): {unexpected[:5]}"
        )
    if len(missing) > 0:
        print(
            f"Warning: missing keys while loading {os.path.basename(path)} (showing up to 5): {missing[:5]}"
        )
    return model


def find_weight_file(preferred_substrings):
    candidates = []
    search_roots = ["../input", DATA_DIR]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        candidates.extend(glob.glob(os.path.join(root, "**", "*.pth"), recursive=True))
        candidates.extend(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))

    def score(p):
        lp = p.lower()
        s = 0
        for i, sub in enumerate(preferred_substrings):
            if sub in lp:
                s += 10 - i
        s -= lp.count(os.sep) * 0.01
        return s

    scored = [(score(p), p) for p in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])
    for sc, p in scored:
        lp = p.lower()
        if any(sub in lp for sub in preferred_substrings):
            return p
    return None


efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)
hrnet = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)

eff_w = find_weight_file(["cassava", "b5", "efficientnet", "tf_efficientnet_b5"])
hr_w = find_weight_file(
    ["cassava", "seresnext101", "se_resnext101", "32x4d", "resnext101"]
)

if eff_w is not None:
    print("Found efficientnet weights:", eff_w)
    efficient = load_state_dict_flexible(efficient, eff_w)
else:
    print(
        "No cassava efficientnet weights found; will train head on cassava train.csv to avoid random 5-class head."
    )

if hr_w is not None:
    print("Found seresnext101 weights:", hr_w)
    hrnet = load_state_dict_flexible(hrnet, hr_w)
else:
    print(
        "No cassava seresnext101 weights found; will train head on cassava train.csv to avoid random 5-class head."
    )

efficient.to(DEVICE).eval()
hrnet.to(DEVICE).eval()

print("Models have been loaded...\n")
print("efficient:", efficient.__class__.__name__)
print("hrnet:", hrnet.__class__.__name__)



## === cell 2
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

eff_cfg = resolve_data_config(efficient.default_cfg, model=efficient)
hr_cfg = resolve_data_config(hrnet.default_cfg, model=hrnet)

eff_tf = create_transform(**eff_cfg, is_training=False)
hr_tf = create_transform(**hr_cfg, is_training=False)

print("Efficient cfg:", eff_cfg)
print("HR cfg:", hr_cfg)


def _clahe_bgr(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


def _fallback_to_tensor_and_normalize(img_rgb_uint8, cfg):
    x = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0  # CHW
    mean = torch.tensor(cfg["mean"], dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(cfg["std"], dtype=x.dtype).view(3, 1, 1)
    x = (x - mean) / std
    return x


def to_tensor_via_timm_transform(image_bgr, transform, cfg, device=DEVICE):
    img = _clahe_bgr(image_bgr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    try:
        x = transform(pil_img)  # CHW float tensor normalized as required
    except TypeError:
        x = _fallback_to_tensor_and_normalize(img, cfg)

    x = x.unsqueeze(0).to(device, non_blocking=True)
    return x


eff_tf_train = create_transform(**eff_cfg, is_training=True)
hr_tf_train = create_transform(**hr_cfg, is_training=True)



## === cell 3
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

train_csv_path = os.path.join(DATA_DIR, "train.csv")
if not os.path.exists(train_csv_path):
    raise RuntimeError(f"Missing train.csv at {train_csv_path}")
train_df = pd.read_csv(train_csv_path)

val_frac = 0.1
train_idx, val_idx = [], []
for lbl, g in train_df.groupby("label"):
    idx = g.index.to_list()
    rng = np.random.RandomState(SEED + int(lbl))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_idx.extend(idx[:n_val])
    train_idx.extend(idx[n_val:])

train_split = train_df.loc[train_idx].reset_index(drop=True)
val_split = train_df.loc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(train_split), len(val_split))
print("Train label dist:\n", train_split["label"].value_counts().sort_index())
print("Val label dist:\n", val_split["label"].value_counts().sort_index())


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform, cfg):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        image_id = r["image_id"]
        y = int(r["label"])
        fp = os.path.join(self.img_dir, image_id)
        img = cv2.imread(fp)
        if img is None:
            img = np.zeros(
                (self.cfg["input_size"][1], self.cfg["input_size"][2], 3),
                dtype=np.uint8,
            )

        img = _clahe_bgr(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        try:
            x = self.transform(pil_img)
        except TypeError:
            x = _fallback_to_tensor_and_normalize(np.array(pil_img), self.cfg)

        return x, y


def freeze_backbone_train_head(model):
    for p in model.parameters():
        p.requires_grad = False
    clf = model.get_classifier()
    if isinstance(clf, nn.Module):
        for p in clf.parameters():
            p.requires_grad = True
    return model


need_train_eff = eff_w is None
need_train_hr = hr_w is None

if need_train_eff:
    efficient.train()
    freeze_backbone_train_head(efficient)

if need_train_hr:
    hrnet.train()
    freeze_backbone_train_head(hrnet)

BATCH_SIZE = 16 if torch.cuda.is_available() else 8
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

eff_train_loader = None
hr_train_loader = None
val_loader_eff = None
val_loader_hr = None

if need_train_eff:
    ds_tr_eff = CassavaDataset(train_split, TRAIN_IMG_DIR, eff_tf_train, eff_cfg)
    ds_va_eff = CassavaDataset(val_split, TRAIN_IMG_DIR, eff_tf, eff_cfg)
    eff_train_loader = DataLoader(
        ds_tr_eff,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader_eff = DataLoader(
        ds_va_eff,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

if need_train_hr:
    ds_tr_hr = CassavaDataset(train_split, TRAIN_IMG_DIR, hr_tf_train, hr_cfg)
    ds_va_hr = CassavaDataset(val_split, TRAIN_IMG_DIR, hr_tf, hr_cfg)
    hr_train_loader = DataLoader(
        ds_tr_hr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader_hr = DataLoader(
        ds_va_hr,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def train_head(model, train_loader, val_loader, epochs=2, lr=3e-3):
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for xb, yb in tqdm.tqdm(train_loader, desc=f"train ep{ep}", leave=False):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = torch.as_tensor(yb, device=DEVICE, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item()) * xb.size(0)
            tr_correct += int((out.argmax(dim=1) == yb).sum().item())
            tr_total += xb.size(0)

        model.eval()
        va_correct = 0
        va_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = torch.as_tensor(yb, device=DEVICE, dtype=torch.long)
                out = model(xb)
                va_correct += int((out.argmax(dim=1) == yb).sum().item())
                va_total += xb.size(0)

        print(
            f"ep {ep}: train_loss={tr_loss/max(1,tr_total):.4f} train_acc={tr_correct/max(1,tr_total):.4f} val_acc={va_correct/max(1,va_total):.4f}"
        )

    model.eval()
    return model


if need_train_eff:
    efficient = train_head(
        efficient, eff_train_loader, val_loader_eff, epochs=2, lr=3e-3
    )

if need_train_hr:
    hrnet = train_head(hrnet, hr_train_loader, val_loader_hr, epochs=2, lr=3e-3)

efficient.eval()
hrnet.eval()



## === cell 4
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    raise RuntimeError(f"Missing sample_submission.csv at {sample_path}")
sample = pd.read_csv(sample_path)
test_image_ids = sample["image_id"].tolist()

names, labels = [], []

with torch.no_grad():
    for image_id in tqdm.tqdm(test_image_ids, total=len(test_image_ids)):
        file = os.path.join(TEST_IMG_DIR, image_id)
        img = cv2.imread(file)
        if img is None:
            names.append(image_id)
            labels.append(0)
            continue

        x_hr = to_tensor_via_timm_transform(img, hr_tf, hr_cfg)
        x_eff = to_tensor_via_timm_transform(img, eff_tf, eff_cfg)

        hr_out = hrnet(x_hr)
        eff_out = efficient(x_eff)

        total = (hr_out + eff_out) / 2.0
        pred = int(torch.argmax(total, dim=1).item())

        names.append(image_id)
        labels.append(pred)

print(
    "Predictions made for:", len(names), "images (expected:", len(test_image_ids), ")"
)



## === cell 5
df = pd.DataFrame({"image_id": names, "label": labels}, columns=["image_id", "label"])

df = sample[["image_id"]].merge(df, on="image_id", how="left")
df["label"] = df["label"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {df.shape}")
print(df.head())
print("Label value counts:\n", df["label"].value_counts(dropna=False).sort_index())
