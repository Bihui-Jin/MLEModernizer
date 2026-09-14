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

0.8499546690843155

# 6. Current score

0.09679

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09865) has done: 'The main blocker is that your notebook depends on a private “baseline-infer” code package (`data/`, `model/`, `configs/`, etc.) that is not present on this Kaggle runtime, so imports fail and later cells cascade into `NameError`s and an invalid/partial `submission.csv`. I replace those missing modules with an equivalent, minimal PyTorch inference pipeline using a standard `torchvision` ResNet-50 (same backbone family as your config name suggests), load ImageNet-pretrained weights for reasonable accuracy, and run deterministic test-time preprocessing. I also fix submission generation to exactly match `sample_submission.csv` ordering/length so Kaggle accepts it. This preserves the core intent (ResNet50 classifier + inference over test_images) while making it runnable end-to-end and producing a valid `submission.csv`.'
- What this solution (achieved 0.09679) has done: 'Your current score (0.09865) is far below the target (~0.85) because the model is effectively making near-random predictions: the ResNet50 classification head is randomly initialized since the referenced Cassava-trained weights file is missing. To move the score toward the target with minimal core-logic changes, I (1) load a proper Cassava finetuned checkpoint if it exists in the dataset tree (instead of only one hardcoded path), and (2) align preprocessing with standard ResNet inference by using torchvision’s official weights transform (resize/crop + normalization), which improves accuracy when using either ImageNet or finetuned weights. The model architecture (ResNet50 + linear head), inference loop, and submission semantics remain the same, but the input pipeline and checkpoint discovery are fixed to make the predictions meaningful. The script still runs end-to-end and writes a valid `/kaggle/working/submission.csv`.'

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

import cv2
from torchvision import models


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
SUB_PATH = "/kaggle/working/submission.csv"

assert os.path.isdir(TEST_DIR), f"Missing test_images dir: {TEST_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
NUM_CLASSES = 5

weights_enum = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights_enum)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)


def _find_checkpoint_candidates(search_roots):
    exts = (".pth", ".pt", ".bin")
    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    path = os.path.join(dirpath, fn)
                    try:
                        if (
                            os.path.getsize(path) < 200_000
                        ):  # <200KB unlikely to be ResNet50 weights
                            continue
                    except OSError:
                        continue
                    candidates.append(path)
    return candidates


def _score_ckpt_path(p):
    name = os.path.basename(p).lower()
    score = 0
    for key, w in [
        ("cassava", 10),
        ("resnet50", 8),
        ("baseline", 6),
        ("best", 6),
        ("fold", 4),
        ("epoch", 3),
        ("model", 2),
    ]:
        if key in name:
            score += w
    if "baseline" in p.lower():
        score += 3
    return score


preferred_hardcoded = "../input/baseline-weights/epoch_1.pth"
search_roots = ["/kaggle/input", "/kaggle/data/input"]
candidates = []

if os.path.isfile(preferred_hardcoded):
    candidates = [preferred_hardcoded]
else:
    candidates = _find_checkpoint_candidates(search_roots)
    candidates = sorted(candidates, key=_score_ckpt_path, reverse=True)

loaded_any = False
if len(candidates) > 0:
    for weights_path in candidates[:10]:
        try:
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]

            if isinstance(state, dict):
                new_state = {}
                for k, v in state.items():
                    nk = k
                    if nk.startswith("module."):
                        nk = nk[len("module.") :]
                    if nk.startswith("model."):
                        nk = nk[len("model.") :]
                    new_state[nk] = v
                state = new_state

            missing, unexpected = model.load_state_dict(state, strict=False)

            print("Loaded external weights:", weights_path)
            print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
            loaded_any = True
            break
        except Exception as e:
            continue

if not loaded_any:
    print(
        "No compatible external weights found; using ImageNet-pretrained backbone with randomly initialized head."
    )

model.eval()



## === cell 2
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

resnet_preprocess = weights_enum.transforms()


def preprocess_bgr_to_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.size == 0:
        raise ValueError("Empty image encountered.")
    img = torch.from_numpy(img_rgb)  # HWC, uint8
    img = img.permute(2, 0, 1).contiguous()  # CHW
    x = resnet_preprocess(img)
    return x




## === cell 3
class InferSet(Dataset):
    def __init__(self, image_ids, img_dir):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        path = os.path.join(self.img_dir, fn)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        x = preprocess_bgr_to_tensor(img)
        return x, fn


sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission columns"
image_ids = sample_sub["image_id"].tolist()

infer_ds = InferSet(image_ids=image_ids, img_dir=TEST_DIR)
infer_loader = DataLoader(
    infer_ds,
    batch_size=64,  # slightly larger for throughput; does not change semantics
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

xb, fnb = next(iter(infer_loader))
xb.shape, fnb[0]



## === cell 4
preds_all = []
ids_all = []

with torch.no_grad():
    model.eval()
    for xb, fns in infer_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
        preds_all.append(pred)
        ids_all.extend(list(fns))

preds_all = np.concatenate(preds_all, axis=0)
assert len(ids_all) == len(image_ids) == len(preds_all), "Prediction length mismatch"

sub = pd.DataFrame({"image_id": ids_all, "label": preds_all})
sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
assert sub["label"].isna().sum() == 0, "Some test ids missing predictions"
sub["label"] = sub["label"].astype(int)

sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH, "rows:", len(sub))
sub.head()



## === cell 5
check = pd.read_csv(SUB_PATH)
assert len(check) == len(sample_sub), "Invalid submission: wrong number of rows"
assert list(check.columns) == ["image_id", "label"], "Invalid submission: wrong columns"
assert check["label"].between(0, 4).all(), "Labels must be in [0,4]"
check.tail()
