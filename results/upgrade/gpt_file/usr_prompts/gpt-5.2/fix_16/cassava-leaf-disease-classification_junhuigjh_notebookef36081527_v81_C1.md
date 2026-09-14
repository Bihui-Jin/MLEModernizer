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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
from torchvision.models import vit_h_14, densenet121
from torchvision.io import read_image, ImageReadMode

SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print("Warning: could not enable deterministic algorithms:", repr(e))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

try:
    torch.set_num_threads(min(8, os.cpu_count() or 2))
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
SAMPLE_SUB = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"

print("Environment ready. Device:", device)




## === cell 1
def invert_square_pad_tensor(img_t: torch.Tensor) -> torch.Tensor:
    c, h, w = img_t.shape
    dx = w // 2
    dy = h // 2
    if dx or dy:
        img_t = torch.roll(img_t, shifts=(-dy, -dx), dims=(1, 2))

    max_side = w if w >= h else h
    pad_w = max_side - w
    pad_h = max_side - h
    if pad_w or pad_h:
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        img_t = F.pad(img_t, (left, right, top, bottom), value=0)
    return img_t


_tfm_dn_tensor = v2.Compose(
    [
        v2.Resize((512, 512), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

_tfm_vit_tensor = v2.Compose(
    [
        v2.Resize((518, 518), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

_CAN_COMPILE = hasattr(torch, "compile")
_COMPILE_DATASET_FUNCS = False  # keep False for stability with num_workers>0

if _CAN_COMPILE and _COMPILE_DATASET_FUNCS:
    try:
        invert_square_pad_tensor = torch.compile(
            invert_square_pad_tensor, fullgraph=False, dynamic=True
        )
    except Exception:
        pass


class CassavaImageDataset(Dataset):
    def __init__(self, df, img_dir, tfm_dn, tfm_vit, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm_dn = tfm_dn
        self.tfm_vit = tfm_vit
        self.has_label = has_label

        self.image_ids = self.df["image_id"].to_numpy()
        if self.has_label:
            self.labels = self.df["label"].to_numpy(dtype=np.int64)
        else:
            self.labels = None

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.img_dir, image_id)

        img_t = read_image(path, mode=ImageReadMode.RGB)  # uint8, CxHxW

        x_dn = self.tfm_dn(img_t)

        img_sq = invert_square_pad_tensor(img_t)
        x_vit = self.tfm_vit(img_sq)

        if self.has_label:
            y = int(self.labels[idx])
            return image_id, x_dn, x_vit, y
        return image_id, x_dn, x_vit


torch_transforms_DN = _tfm_dn_tensor
torch_transforms_VIT = _tfm_vit_tensor

print("Transforms and dataset ready (tensor-only preprocessing).")



## === cell 2
NUM_CLASSES = 5

model1 = densenet121(weights="DEFAULT")
model1.to(device).eval()

model3 = vit_h_14(weights="DEFAULT")
model3.to(device).eval()

if device.type == "cuda":
    model1 = model1.to(memory_format=torch.channels_last)
    model3 = model3.to(memory_format=torch.channels_last)

if _CAN_COMPILE:
    try:
        model1 = torch.compile(model1, fullgraph=False, dynamic=True)
    except Exception:
        pass
    try:
        model3 = torch.compile(model3, fullgraph=False, dynamic=True)
    except Exception:
        pass

print("Models instantiated (pretrained backbones).")



## === cell 3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)

max_fit_samples = None
if (max_fit_samples is not None) and (len(train_df) > max_fit_samples):
    fit_df, _ = train_test_split(
        train_df,
        train_size=max_fit_samples,
        stratify=train_df["label"],
        random_state=SEED,
    )
else:
    fit_df = train_df.copy()

fit_ds = CassavaImageDataset(
    fit_df, TRAIN_IMG_DIR, torch_transforms_DN, torch_transforms_VIT, has_label=True
)

_cpu = os.cpu_count() or 2
_num_workers = min(12, _cpu) if device.type == "cuda" else min(8, _cpu)

_infer_bs = 96 if device.type == "cuda" else 16
_pin = device.type == "cuda"

_dl_kwargs = dict(
    batch_size=_infer_bs,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=_pin,
    persistent_workers=(_num_workers > 0),
)
if _num_workers > 0:
    _dl_kwargs["prefetch_factor"] = 6

fit_loader = DataLoader(fit_ds, **_dl_kwargs)


def _set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad = flag


dn_feat_dim = model1.classifier.in_features
dn_head = nn.Linear(dn_feat_dim, NUM_CLASSES).to(device)

vit_feat_dim = (
    model3.heads.head.in_features
    if hasattr(model3, "heads")
    else model3.head.in_features
)
vit_head = nn.Linear(vit_feat_dim, NUM_CLASSES).to(device)

_set_requires_grad(model1, False)
_set_requires_grad(model3, False)
_set_requires_grad(dn_head, True)
_set_requires_grad(vit_head, True)

ce = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(
    list(dn_head.parameters()) + list(vit_head.parameters()), lr=3e-4, weight_decay=1e-2
)

_vit_pos_embedding = model3.encoder.pos_embedding
_vit_dropout = model3.encoder.dropout
_vit_layers = model3.encoder.layers
_vit_ln = model3.encoder.ln
_vit_class_token = model3.class_token


def _densenet_feats(x_dn: torch.Tensor) -> torch.Tensor:
    f = model1.features(x_dn)
    f = F.relu(f, inplace=False)
    f = torch.nn.functional.adaptive_avg_pool2d(f, (1, 1))
    f = torch.flatten(f, 1)
    return f


def _vit_cls(x_vit: torch.Tensor) -> torch.Tensor:
    vit_feats = model3._process_input(x_vit)
    n = vit_feats.shape[0]
    batch_class_token = _vit_class_token.expand(n, -1, -1)
    vit_feats = torch.cat([batch_class_token, vit_feats], dim=1)
    vit_feats = vit_feats + _vit_pos_embedding
    vit_feats = _vit_dropout(vit_feats)
    vit_feats = _vit_layers(vit_feats)
    vit_feats = _vit_ln(vit_feats)
    return vit_feats[:, 0]


def cache_fit_backbone_features(dataloader, n_samples: int):
    if device.type == "cuda":
        dn_feats_all = torch.empty(
            (n_samples, dn_feat_dim), dtype=torch.float32, pin_memory=True
        )
        vit_cls_all = torch.empty(
            (n_samples, vit_feat_dim), dtype=torch.float32, pin_memory=True
        )
        ys_all = torch.empty((n_samples,), dtype=torch.int64, pin_memory=True)
    else:
        dn_feats_all = torch.empty((n_samples, dn_feat_dim), dtype=torch.float32)
        vit_cls_all = torch.empty((n_samples, vit_feat_dim), dtype=torch.float32)
        ys_all = torch.empty((n_samples,), dtype=torch.int64)

    write_pos = 0
    model1.eval()
    model3.eval()

    with torch.inference_mode():
        for _, x_dn, x_vit, y in dataloader:
            bs = x_dn.shape[0]

            if device.type == "cuda":
                x_dn = x_dn.to(
                    device, non_blocking=True, memory_format=torch.channels_last
                )
                x_vit = x_vit.to(device, non_blocking=True)
                y_t = (
                    y.to("cpu", non_blocking=True)
                    if torch.is_tensor(y)
                    else torch.as_tensor(y, dtype=torch.int64)
                )
            else:
                x_dn = x_dn.to(device)
                x_vit = x_vit.to(device)
                y_t = y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.int64)

            dn_feats = _densenet_feats(x_dn)
            vit_cls = _vit_cls(x_vit)

            dn_feats_all[write_pos : write_pos + bs].copy_(
                dn_feats.detach().to("cpu", non_blocking=_pin)
            )
            vit_cls_all[write_pos : write_pos + bs].copy_(
                vit_cls.detach().to("cpu", non_blocking=_pin)
            )
            ys_all[write_pos : write_pos + bs].copy_(y_t)

            write_pos += bs

    assert (
        write_pos == n_samples
    ), "Backbone feature cache did not fill expected sample count."
    return dn_feats_all, vit_cls_all, ys_all


try:
    fit_dn_feats_cpu, fit_vit_cls_cpu, fit_y_cpu = cache_fit_backbone_features(
        fit_loader, n_samples=len(fit_ds)
    )
except RuntimeError as e:
    print(
        "Warning: caching with num_workers failed; retrying with num_workers=0. Error was:",
        repr(e),
    )
    _num_workers = 0
    _dl_kwargs = dict(
        batch_size=_infer_bs,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=_pin,
        persistent_workers=False,
    )
    fit_loader = DataLoader(fit_ds, **_dl_kwargs)
    fit_dn_feats_cpu, fit_vit_cls_cpu, fit_y_cpu = cache_fit_backbone_features(
        fit_loader, n_samples=len(fit_ds)
    )


def train_linear_probes_from_cache(
    dn_feats_cpu, vit_cls_cpu, y_cpu, epochs: int = 3, batch_size: int = 256
):
    model1.eval()
    model3.eval()
    dn_head.train()
    vit_head.train()

    n = y_cpu.shape[0]
    for ep in range(epochs):
        total = 0
        correct_dn = 0
        correct_vit = 0

        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            dn_feats = dn_feats_cpu[i:j].to(device, non_blocking=_pin)
            vit_cls = vit_cls_cpu[i:j].to(device, non_blocking=_pin)
            y = y_cpu[i:j].to(device, non_blocking=_pin)

            logits_dn = dn_head(dn_feats)
            logits_vit = vit_head(vit_cls)
            loss = ce(logits_dn, y) + ce(logits_vit, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total += y.size(0)
            correct_dn += (logits_dn.argmax(1) == y).sum().item()
            correct_vit += (logits_vit.argmax(1) == y).sum().item()

        print(
            f"Probe epoch {ep+1}/{epochs} | "
            f"dn_acc={correct_dn/total:.4f} vit_acc={correct_vit/total:.4f}"
        )


train_linear_probes_from_cache(
    fit_dn_feats_cpu, fit_vit_cls_cpu, fit_y_cpu, epochs=3, batch_size=256
)


def extract_combined_logits_from_cache(
    dn_feats_cpu, vit_cls_cpu, y_cpu, batch_size: int = 512
):
    n = y_cpu.shape[0]
    feats = np.empty((n, NUM_CLASSES * 2), dtype=np.float32)

    dn_head.eval()
    vit_head.eval()

    write_pos = 0
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            dn_feats = dn_feats_cpu[i:j].to(device, non_blocking=_pin)
            vit_cls = vit_cls_cpu[i:j].to(device, non_blocking=_pin)

            logits_dn = dn_head(dn_feats)
            logits_vit = vit_head(vit_cls)
            combined = torch.cat([logits_dn, logits_vit], dim=1)

            bs = j - i
            feats[write_pos : write_pos + bs] = combined.detach().cpu().numpy()
            write_pos += bs

    assert (
        write_pos == n
    ), "Cached combined-logit extraction did not fill expected sample count."
    return feats, y_cpu.cpu().numpy() if torch.is_tensor(y_cpu) else np.asarray(y_cpu)


train_probs, train_labels = extract_combined_logits_from_cache(
    fit_dn_feats_cpu, fit_vit_cls_cpu, fit_y_cpu, batch_size=512
)

decision_tree = RandomForestClassifier(
    n_estimators=30,
    criterion="gini",
    max_depth=6,
    random_state=SEED,
    n_jobs=-1,
)
decision_tree.fit(train_probs, train_labels)

print("Meta-classifier trained on", len(train_labels), "samples.")



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["image_id"]].copy()

test_ds = CassavaImageDataset(
    test_df, TEST_IMG_DIR, torch_transforms_DN, torch_transforms_VIT, has_label=False
)

_dl_kwargs_test = dict(
    batch_size=_infer_bs,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=_pin,
    persistent_workers=(_num_workers > 0),
)
if _num_workers > 0:
    _dl_kwargs_test["prefetch_factor"] = 6

test_loader = DataLoader(test_ds, **_dl_kwargs_test)

combined_output = np.empty((len(test_ds), NUM_CLASSES * 2), dtype=np.float32)
image_ids_ordered = [None] * len(test_ds)

dn_head.eval()
vit_head.eval()

write_pos = 0
with torch.inference_mode():
    for image_id, x_dn, x_vit in test_loader:
        bs = x_dn.shape[0]
        if device.type == "cuda":
            x_dn = x_dn.to(device, non_blocking=True, memory_format=torch.channels_last)
            x_vit = x_vit.to(device, non_blocking=True)
        else:
            x_dn = x_dn.to(device)
            x_vit = x_vit.to(device)

        dn_feats = _densenet_feats(x_dn)
        logits_dn = dn_head(dn_feats)

        vit_cls = _vit_cls(x_vit)
        logits_vit = vit_head(vit_cls)

        combined = torch.cat([logits_dn, logits_vit], dim=1)

        end = write_pos + bs
        assert end <= len(test_ds), "Test write overflow."
        combined_output[write_pos:end] = combined.detach().cpu().numpy()
        image_ids_ordered[write_pos:end] = list(image_id)
        write_pos = end

assert write_pos == len(test_ds), "Did not fill all test predictions."
assert (
    len(image_ids_ordered) == combined_output.shape[0] == len(sample_sub)
), "Test alignment/length mismatch."

prediction = decision_tree.predict(combined_output).astype(int)

print("Test predictions ready:", prediction.shape)



## === cell 5
submission = pd.DataFrame({"image_id": image_ids_ordered, "label": prediction})
submission.to_csv("submission.csv", index=False)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows.")
