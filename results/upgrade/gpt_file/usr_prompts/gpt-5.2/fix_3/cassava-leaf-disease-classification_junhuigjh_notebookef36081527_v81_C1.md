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
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import vit_h_14, densenet121

from PIL import Image, ImageOps

SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    width, height = img.size
    dx = width // 2
    dy = height // 2

    if dx or dy:
        shifted = Image.new(img.mode, (width, height))
        shifted.paste(img.crop((dx, dy, width, height)), (0, 0))
        shifted.paste(img.crop((0, dy, dx, height)), (width - dx, 0))
        shifted.paste(img.crop((dx, 0, width, dy)), (0, height - dy))
        shifted.paste(img.crop((0, 0, dx, dy)), (width - dx, height - dy))
        img_pil = shifted
    else:
        img_pil = img

    max_side = width if width >= height else height
    pad_w = max_side - width
    pad_h = max_side - height
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

    return (
        ImageOps.expand(img_pil, border=padding, fill=None).convert("RGB")
        if (padding != (0, 0, 0, 0))
        else img_pil
    )


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
        img = Image.open(path).convert("RGB")
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
model1.classifier = nn.Linear(model1.classifier.in_features, NUM_CLASSES)
model1.to(device).eval()

model3 = vit_h_14(weights="DEFAULT")
if hasattr(model3, "heads") and hasattr(model3.heads, "head"):
    in_f = model3.heads.head.in_features
    model3.heads.head = nn.Linear(in_f, NUM_CLASSES)
else:
    model3.head = nn.Linear(model3.head.in_features, NUM_CLASSES)
model3.to(device).eval()

if device.type == "cuda":
    model1 = model1.to(memory_format=torch.channels_last)
    model3 = model3.to(memory_format=torch.channels_last)

print("Models instantiated (untrained heads).")




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

_num_workers = min(8, (os.cpu_count() or 2))
fit_loader = DataLoader(
    fit_ds,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)


def extract_combined_logits(dataloader):
    feats = []
    ys = []
    append_f = feats.append
    append_y = ys.append

    with torch.no_grad():
        for _, x_dn, x_vit, y in dataloader:
            if device.type == "cuda":
                x_dn = x_dn.to(
                    device, non_blocking=True, memory_format=torch.channels_last
                )
                x_vit = x_vit.to(device, non_blocking=True)
            else:
                x_dn = x_dn.to(device)
                x_vit = x_vit.to(device)

            logits_dn = model1(x_dn)
            logits_vit = model3(x_vit)

            combined = torch.cat([logits_dn, logits_vit], dim=1)
            append_f(combined.detach().cpu().numpy())
            append_y(y.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(ys, axis=0)


train_probs, train_labels = extract_combined_logits(fit_loader)

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
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

combined_output = []
image_ids_ordered = []
append_out = combined_output.append
extend_ids = image_ids_ordered.extend

with torch.no_grad():
    for image_id, x_dn, x_vit in test_loader:
        if device.type == "cuda":
            x_dn = x_dn.to(device, non_blocking=True, memory_format=torch.channels_last)
            x_vit = x_vit.to(device, non_blocking=True)
        else:
            x_dn = x_dn.to(device)
            x_vit = x_vit.to(device)

        logits_dn = model1(x_dn)
        logits_vit = model3(x_vit)
        combined = torch.cat([logits_dn, logits_vit], dim=1)

        append_out(combined.detach().cpu().numpy())
        extend_ids(list(image_id))

combined_output = np.concatenate(combined_output, axis=0)
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
