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
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import vit_h_14, densenet121

from PIL import Image, ImageOps, ImageChops

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
def invert_square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    dx = w // 2
    dy = h // 2

    if dx or dy:
        img = ImageChops.offset(img, -dx, -dy)

    max_side = w if w >= h else h
    pad_w = max_side - w
    pad_h = max_side - h
    if pad_w or pad_h:
        left = pad_w // 2
        top = pad_h // 2
        padding = (left, top, pad_w - left, pad_h - top)
        img = ImageOps.expand(img, border=padding, fill=None)

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((518, 518)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

torch_transforms_DN = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


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
        with Image.open(path) as im:
            img = im.convert("RGB")

        x_dn = self.tfm_dn(img)
        x_vit = self.tfm_vit(img)

        if self.has_label:
            y = int(self.labels[idx])
            return image_id, x_dn, x_vit, y
        return image_id, x_dn, x_vit


print("Transforms and dataset ready.")




## === cell 2
NUM_CLASSES = 5

model1 = densenet121(weights="DEFAULT")
model1.to(device).eval()

model3 = vit_h_14(weights="DEFAULT")
model3.to(device).eval()

if device.type == "cuda":
    model1 = model1.to(memory_format=torch.channels_last)
    model3 = model3.to(memory_format=torch.channels_last)

print("Models instantiated (pretrained backbones).")




## === cell 3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)

max_fit_samples = 2500
if len(train_df) > max_fit_samples:
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

_num_workers = min(4, (os.cpu_count() or 2))
_infer_bs = 32 if device.type == "cuda" else 16
_pin = device.type == "cuda"

fit_loader = DataLoader(
    fit_ds,
    batch_size=_infer_bs,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=_pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)


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


def _densenet_feats(x_dn: torch.Tensor) -> torch.Tensor:
    f = model1.features(x_dn)
    f = F.relu(f, inplace=False)
    f = torch.nn.functional.adaptive_avg_pool2d(f, (1, 1))
    f = torch.flatten(f, 1)
    return f


def _vit_cls(x_vit: torch.Tensor) -> torch.Tensor:
    vit_feats = model3._process_input(x_vit)
    n = vit_feats.shape[0]
    batch_class_token = model3.class_token.expand(n, -1, -1)
    vit_feats = torch.cat([batch_class_token, vit_feats], dim=1)
    vit_feats = vit_feats + model3.encoder.pos_embedding
    vit_feats = model3.encoder.dropout(vit_feats)
    vit_feats = model3.encoder.layers(vit_feats)
    vit_feats = model3.encoder.ln(vit_feats)
    return vit_feats[:, 0]


def train_linear_probes(dataloader, epochs: int = 3):
    model1.eval()
    model3.eval()
    dn_head.train()
    vit_head.train()
    for ep in range(epochs):
        total = 0
        correct_dn = 0
        correct_vit = 0
        for _, x_dn, x_vit, y in dataloader:
            if device.type == "cuda":
                x_dn = x_dn.to(
                    device, non_blocking=True, memory_format=torch.channels_last
                )
                x_vit = x_vit.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            else:
                x_dn = x_dn.to(device)
                x_vit = x_vit.to(device)
                y = y.to(device)

            with torch.no_grad():
                dn_feats = _densenet_feats(x_dn)
                vit_cls = _vit_cls(x_vit)

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


train_linear_probes(fit_loader, epochs=3)


def extract_combined_logits(dataloader, n_samples: int):
    feats = np.empty((n_samples, NUM_CLASSES * 2), dtype=np.float32)
    ys = np.empty((n_samples,), dtype=np.int64)

    dn_head.eval()
    vit_head.eval()
    write_pos = 0

    with torch.inference_mode():
        for _, x_dn, x_vit, y in dataloader:
            bs = x_dn.shape[0]
            if device.type == "cuda":
                x_dn = x_dn.to(
                    device, non_blocking=True, memory_format=torch.channels_last
                )
                x_vit = x_vit.to(device, non_blocking=True)
            else:
                x_dn = x_dn.to(device)
                x_vit = x_vit.to(device)

            dn_feats = _densenet_feats(x_dn)
            logits_dn = dn_head(dn_feats)

            vit_cls = _vit_cls(x_vit)
            logits_vit = vit_head(vit_cls)

            combined = torch.cat([logits_dn, logits_vit], dim=1)

            assert write_pos + bs <= n_samples, "Feature extraction write overflow."
            feats[write_pos : write_pos + bs] = combined.detach().cpu().numpy()

            ys_batch = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            ys[write_pos : write_pos + bs] = ys_batch
            write_pos += bs

    assert (
        write_pos == n_samples
    ), "Feature extraction did not fill expected sample count."
    return feats, ys


train_probs, train_labels = extract_combined_logits(fit_loader, n_samples=len(fit_ds))

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

test_loader = DataLoader(
    test_ds,
    batch_size=_infer_bs,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=_pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)

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

        assert write_pos + bs <= len(test_ds), "Test write overflow."
        combined_output[write_pos : write_pos + bs] = combined.detach().cpu().numpy()
        image_ids_ordered[write_pos : write_pos + bs] = list(image_id)
        write_pos += bs

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
