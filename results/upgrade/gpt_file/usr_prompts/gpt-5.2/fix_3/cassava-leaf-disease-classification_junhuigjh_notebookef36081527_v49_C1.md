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

# 5. Target score

0.8445149592021759

# 6. Current score

0.74066

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.74066) has done: 'I fix the immediate runtime blocker by removing the hard dependency on unavailable external model files and replacing it with a small, built-in Torchvision model so inference can run end-to-end in this Kaggle environment. I also ensure the prediction loop always fills `image_ids` and `prediction` to identical lengths (even if an image read fails) so the submission DataFrame can be created reliably. To nudge accuracy upward versus random guessing while keeping the approach simple and stable, I use an ImageNet-pretrained backbone and a deterministic, lightweight training of a linear classifier head on the provided `train.csv` images (core logic remains: single-model softmax → argmax labels). Finally, I write a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = f"{DATA_ROOT}/train.csv"
train_dir = f"{DATA_ROOT}/train_images"
test_dir = f"{DATA_ROOT}/test_images"

assert os.path.isfile(train_csv_path), f"Missing train.csv: {train_csv_path}"
assert os.path.isdir(train_dir), f"Missing train directory: {train_dir}"
assert os.path.isdir(test_dir), f"Missing test directory: {test_dir}"

test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
print("n_test_images:", len(test_images))

torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## === cell 1

NUM_CLASSES = 5

try:
    weights = models.ResNet50_Weights.DEFAULT
    backbone = models.resnet50(weights=weights)
except Exception:
    backbone = models.resnet50(pretrained=True)

backbone.fc = nn.Identity()
backbone = backbone.to(device).eval()

head = nn.Linear(2048, NUM_CLASSES).to(device)

print("Backbone + head ready.")



## === cell 2


def _load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = torch_transforms(img).unsqueeze(0)
    return x


@torch.no_grad()
def _extract_feat(x: torch.Tensor) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    feat = backbone(x)  # (1,2048)
    return feat


def _iter_train_rows(df: pd.DataFrame, root: str):
    for _, r in df.iterrows():
        yield os.path.join(root, r["image_id"]), int(r["label"])




## === cell 3

train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(int)

train_df = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
max_train = min(
    6000, len(train_df)
)  # keep small enough for time, large enough for signal
train_df_sub = train_df.iloc[:max_train].copy()
print("Training samples:", len(train_df_sub))

optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

head.train()
epochs = 2  # keep minimal to avoid long runtime; still improves over zero-shot
batch_feats = []
batch_labels = []
batch_size = 64


def _flush_batch(feats_list, labels_list):
    if not feats_list:
        return 0.0
    feats = torch.cat(feats_list, dim=0).to(device)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)
    logits = head(feats)
    loss = criterion(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


for ep in range(epochs):
    running = 0.0
    n_steps = 0
    batch_feats.clear()
    batch_labels.clear()
    for i, (img_path, y) in enumerate(
        _iter_train_rows(train_df_sub, train_dir), start=1
    ):
        try:
            x = _load_image_tensor(img_path)
            with torch.no_grad():
                feat = (
                    _extract_feat(x).detach().cpu()
                )  # keep CPU buffer to reduce GPU memory spikes
            batch_feats.append(feat)
            batch_labels.append(y)
        except Exception:
            continue

        if len(batch_labels) >= batch_size:
            loss = _flush_batch(batch_feats, batch_labels)
            running += loss
            n_steps += 1
            batch_feats.clear()
            batch_labels.clear()

        if i % 500 == 0:
            print(f"epoch {ep+1}/{epochs} step {i}/{len(train_df_sub)}", end="\r")

    loss = _flush_batch(batch_feats, batch_labels)
    if batch_labels:
        running += loss
        n_steps += 1

    avg_loss = running / max(1, n_steps)
    print(f"\nepoch {ep+1}/{epochs} avg_loss={avg_loss:.4f}")

head.eval()
print("Training done.")



## === cell 4

image_ids = []
prediction = []


@torch.no_grad()
def _predict_probs(x: torch.Tensor) -> np.ndarray:
    feat = _extract_feat(x)
    logits = head(feat)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()[0]


length = len(test_images)
for count, test_image in enumerate(test_images, start=1):
    image_ids.append(test_image)
    try:
        img_path = os.path.join(test_dir, test_image)
        x = _load_image_tensor(img_path)
        probs = _predict_probs(x)
        pred_label = int(np.argmax(probs))
    except Exception:
        pred_label = 0
    prediction.append(pred_label)

    if count % 50 == 0 or count == length:
        print(f"Count:{count}/{length}", end="\r")

print("\nDone. preds:", len(prediction), "ids:", len(image_ids))



## === cell 5

assert len(image_ids) == len(test_images), "image_ids length mismatch."
assert len(prediction) == len(test_images), "prediction length mismatch."

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
assert submission.shape[0] == len(test_images), "Submission row count mismatch."
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch."
submission["label"] = submission["label"].astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())



## === cell 6
print(submission.tail())
