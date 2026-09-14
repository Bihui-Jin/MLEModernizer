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

0.8618918102145663

# 6. Current score

0.12668

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14873) has done: 'I remove the broken dependencies on missing Kaggle Dataset code (the `utils`, `model`, `configs`, and custom transform modules) and replace them with a minimal, self-contained PyTorch + torchvision inference pipeline that can run in this environment. Since your current run never produced a valid submission, the priority is to generate a correctly sized `submission.csv` matching `sample_submission.csv` order and length. To keep “core logic” comparable (ResNet-style classifier), I use a standard `torchvision.models.resnet50` with a 5-class head and do single-model inference with Imagenet normalization. I also ensure deterministic ordering by reading image_ids from `sample_submission.csv` rather than relying on `os.listdir()`.'
- What this solution (achieved 0.12668) has done: 'Your current score is low because the model is an ImageNet-pretrained ResNet50 with a randomly initialized 5-class head, so predictions are essentially arbitrary for cassava labels. To move your score up toward the target while keeping the same core inference pipeline, I load a proper cassava-trained ResNet50 checkpoint (if available locally) and only fall back to your current “random head” behavior when no checkpoint exists. I also switch preprocessing resize to the standard ResNet input size (224) to match the pretrained feature statistics and reduce distribution shift, which is a minimal, architecture-preserving change. The rest of the logic (dataset, dataloader, argmax, submission format/order) stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from torchvision import models

DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

OUT_SUB_PATH = "./submission.csv"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
IM_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr_uint8(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img).float()


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, img_dir):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.img_dir, image_id)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        x = preprocess_bgr_uint8(img)
        return x, image_id


sample_df = pd.read_csv(SAMPLE_SUB_PATH)
assert "image_id" in sample_df.columns, "sample_submission.csv must contain image_id"
test_image_ids = sample_df["image_id"].tolist()

test_ds = CassavaTestDataset(test_image_ids, TEST_IMG_DIR)
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(test_ds), next(iter(test_loader))[0].shape



## === cell 2
num_classes = 5


def _try_find_ckpt():
    candidates = []
    for base in [
        "/kaggle/input",
        "../input",
        ".",
        "./checkpoints",
        "../working",
        "/kaggle/working",
    ]:
        if os.path.isdir(base):
            candidates.append(base)

    exts = (".pth", ".pt", ".bin")
    hits = []
    for base in candidates:
        for root, _, files in os.walk(base):
            for fn in files:
                lfn = fn.lower()
                if lfn.endswith(exts) and (
                    "cassava" in lfn
                    or "resnet" in lfn
                    or "model" in lfn
                    or "best" in lfn
                ):
                    hits.append(os.path.join(root, fn))

    def rank(p):
        lp = os.path.basename(p).lower()
        score = 0
        score += 5 if "best" in lp else 0
        score += 4 if "cassava" in lp else 0
        score += 3 if "resnet50" in lp else (2 if "resnet" in lp else 0)
        score += 2 if "fold" in lp else 0
        score += 1 if "final" in lp else 0
        return -score, len(lp)

    hits = sorted(hits, key=rank)
    return hits[0] if len(hits) else None


try:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
except Exception:
    model = models.resnet50(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, num_classes)

ckpt_path = _try_find_ckpt()
loaded_ckpt = False

if ckpt_path is not None and os.path.isfile(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = None
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            elif "model_state_dict" in ckpt and isinstance(
                ckpt["model_state_dict"], dict
            ):
                state = ckpt["model_state_dict"]
            else:
                if all(isinstance(k, str) for k in ckpt.keys()):
                    state = ckpt
        elif isinstance(ckpt, (list, tuple)):
            state = None

        if state is not None:
            new_state = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                new_state[nk] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            loaded_ckpt = True
            print("Loaded checkpoint:", ckpt_path)
            if missing:
                print("Missing keys (first 10):", missing[:10])
            if unexpected:
                print("Unexpected keys (first 10):", unexpected[:10])
    except Exception as e:
        print("Found checkpoint but failed to load:", ckpt_path)
        print("Load error:", repr(e))

if not loaded_ckpt:
    print(
        "No usable cassava checkpoint found; falling back to ImageNet backbone + random 5-class head (low score expected)."
    )

model = model.to(device)
model.eval()

sum(p.numel() for p in model.parameters()) / 1e6



## === cell 3
pred_labels = []
pred_image_ids = []

with torch.no_grad():
    for xb, image_ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        pred_labels.extend(preds)
        pred_image_ids.extend(list(image_ids))

assert (
    len(pred_image_ids) == len(test_image_ids) == len(pred_labels)
), "Prediction count must match sample submission length"
assert (
    pred_image_ids == test_image_ids
), "Image order mismatch; must match sample_submission.csv"

sub_df = pd.DataFrame({"image_id": pred_image_ids, "label": pred_labels})
sub_df.to_csv(OUT_SUB_PATH, index=False)

print(sub_df.head())
print("Wrote:", OUT_SUB_PATH, "rows:", len(sub_df))
print("Unique labels:", sorted(sub_df["label"].unique().tolist()))



## === cell 4
check_df = pd.read_csv(OUT_SUB_PATH)
assert list(check_df.columns) == ["image_id", "label"]
assert len(check_df) == len(sample_df)
check_df.head()
