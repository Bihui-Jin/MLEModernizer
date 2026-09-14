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

0.58744

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60015) has done: 'I fix the two blocking FileNotFoundErrors by removing dependencies on external Kaggle datasets (`/kaggle/input/train-tree/...` and `/kaggle/input/resnet50_70_512x512/...`) and instead train the same kind of DecisionTreeClassifier directly from the provided `train.csv` and `train_images`. To preserve the original core intent (tree on top of model probabilities), I generate per-image “probability-like” features using the existing ResNet50-head function (kept) and compute logits/probabilities from it; this keeps the pipeline structure (CNN → probs → decision tree) while making it self-contained. I also ensure `test_image_ids` is always defined from `sample_submission.csv`, and make submission writing robust (correct columns, row count, and `.csv` suffix). These changes are necessary for end-to-end execution and should yield a non-trivial accuracy score compared to the previous uniform-fallback behavior.'
- What this solution (achieved 0.59268) has done: 'Your current score is low because `resnet50(weights=None)` is untrained, so the “probability-like” features going into the decision tree are essentially noise. To move the accuracy up toward your target with minimal logic change, I keep the same pipeline (CNN → 5-dim probs → DecisionTreeClassifier) but switch the ResNet50 backbone to ImageNet pretrained weights so the 5-dim outputs become meaningful features. I also normalize with the standard ImageNet mean/std to match the pretrained model and keep everything else (tree hyperparameters, sampling, submission writing) unchanged for stability and runtime. These changes should significantly reduce the gap toward 0.8445 without changing your overall approach.'
- What this solution (achieved 0.66293) has done: 'Your current gap to the target is large (0.59268 → 0.84451), and the biggest issue is that your “CNN → 5-dim probs → DecisionTree” pipeline is using an untrained 5-class head (random fc layer), so the features going into the tree are mostly noise. To preserve the same core logic while improving accuracy, I keep ResNet50 ImageNet weights but change the extracted features to the pretrained penultimate layer (2048-dim) instead of the random 5-dim logits; the decision tree still consumes “CNN-derived features,” just more meaningful ones. I also add a very small amount of test-time augmentation (horizontal flip averaged with original) inside the same `predict_*` function to improve feature stability without changing the overall approach or adding any new training loops. Finally, I keep submission alignment identical to `sample_submission.csv` and ensure the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.58744) has done: 'Your current score (0.66293) is far below the target (0.84451), so we should improve accuracy with the smallest changes that keep the same core pipeline (ResNet50 feature extractor → DecisionTreeClassifier). The biggest lever here without changing architecture is to train the tree on more (and more representative) data: increase `MAX_TRAIN_FOR_TREE` while keeping runtime bounded by extracting features in efficient batches. I also make the train subset stratified by label (instead of pure random) so the tree sees all classes proportionally, and I switch the tree to `class_weight="balanced"` to reduce bias toward majority classes—still the same model type and training semantics. Finally, I keep submission alignment identical to `sample_submission.csv` and still write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from sklearn.tree import DecisionTreeClassifier

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
COMP_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{COMP_PATH}/train.csv"
TRAIN_DIR = f"{COMP_PATH}/train_images"
TEST_DIR = f"{COMP_PATH}/test_images"
SAMPLE_SUB_PATH = f"{COMP_PATH}/sample_submission.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv must contain 'image_id'.")
test_image_ids = sample_sub["image_id"].astype(str).tolist()

train_df = pd.read_csv(TRAIN_CSV_PATH)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain 'image_id' and 'label' columns.")
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR}")
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")

print("Train rows:", len(train_df), "Test rows:", len(test_image_ids))



## === cell 2
torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet50_feature_extractor() -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()
    return m


model2 = build_resnet50_feature_extractor().to(device)
model2.eval()


@torch.inference_mode()
def extract_feats_batch_torch(model: nn.Module, pil_imgs) -> np.ndarray:
    xs1 = torch.stack([torch_transforms(im) for im in pil_imgs], dim=0).to(device)
    xs2 = torch.stack(
        [torch_transforms(im.transpose(Image.FLIP_LEFT_RIGHT)) for im in pil_imgs],
        dim=0,
    ).to(device)
    f1 = model(xs1)
    f2 = model(xs2)
    feats = 0.5 * (f1 + f2)
    return feats.detach().cpu().numpy().astype(np.float32)




## === cell 3
MAX_TRAIN_FOR_TREE = 12000  # increased from 6000 to better approach target accuracy
BATCH_SIZE = 32

n_train_total = len(train_df)
use_n = min(MAX_TRAIN_FOR_TREE, n_train_total)

rng = np.random.RandomState(0)

label_groups = train_df.groupby("label").indices
all_labels = sorted(label_groups.keys())
counts = train_df["label"].value_counts().to_dict()

selected_idxs = []
for lab in all_labels:
    idxs_lab = np.array(label_groups[lab], dtype=np.int64)
    rng.shuffle(idxs_lab)
    take = int(round(use_n * (counts[lab] / n_train_total)))
    take = max(1, min(take, len(idxs_lab)))
    selected_idxs.append(idxs_lab[:take])

use_idxs = np.concatenate(selected_idxs)
rng.shuffle(use_idxs)
if len(use_idxs) > use_n:
    use_idxs = use_idxs[:use_n]
elif len(use_idxs) < use_n:
    remaining = np.setdiff1d(np.arange(n_train_total), use_idxs, assume_unique=False)
    rng.shuffle(remaining)
    need = use_n - len(use_idxs)
    use_idxs = np.concatenate([use_idxs, remaining[:need]])

use_n = len(use_idxs)

FEAT_DIM = 2048
train_feats = np.zeros((use_n, FEAT_DIM), dtype=np.float32)
train_labels = np.zeros((use_n,), dtype=np.int64)

bad_count = 0
paths = [os.path.join(TRAIN_DIR, train_df.iloc[i]["image_id"]) for i in use_idxs]
labels = [int(train_df.iloc[i]["label"]) for i in use_idxs]
train_labels[:] = np.array(labels, dtype=np.int64)

for start in range(0, use_n, BATCH_SIZE):
    end = min(use_n, start + BATCH_SIZE)
    batch_paths = paths[start:end]
    pil_imgs = []
    ok_mask = np.ones((end - start,), dtype=bool)

    for k, p in enumerate(batch_paths):
        try:
            pil_imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pil_imgs.append(Image.new("RGB", (512, 512), (0, 0, 0)))
            ok_mask[k] = False
            bad_count += 1

    feats = extract_feats_batch_torch(model2, pil_imgs)  # [B, 2048]
    feats[~ok_mask] = 0.0
    train_feats[start:end] = feats

    if end % 256 == 0 or end == use_n:
        print(f"Feature extract train: {end}/{use_n} (bad={bad_count})", end="\r")
print()

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=8,
    min_samples_split=12,
    class_weight="balanced",
    random_state=0,
)
decision_tree.fit(train_feats, train_labels)
print("Decision tree trained on:", train_feats.shape)



## === cell 4
image_ids = []
prediction = []

length = len(test_image_ids)
test_paths = [os.path.join(TEST_DIR, iid) for iid in test_image_ids]

for start in range(0, length, BATCH_SIZE):
    end = min(length, start + BATCH_SIZE)
    batch_ids = test_image_ids[start:end]
    batch_paths = test_paths[start:end]

    pil_imgs = []
    ok_mask = np.ones((end - start,), dtype=bool)
    for k, p in enumerate(batch_paths):
        try:
            pil_imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pil_imgs.append(Image.new("RGB", (512, 512), (0, 0, 0)))
            ok_mask[k] = False

    feats = extract_feats_batch_torch(model2, pil_imgs)
    feats[~ok_mask] = 0.0

    preds = decision_tree.predict(feats).astype(int).tolist()

    image_ids.extend(batch_ids)
    prediction.extend(preds)

    if end % 200 == 0 or end == length:
        print(f"Predict test: {end}/{length}", end="\r")
print()

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})

if submission.shape[0] != len(test_image_ids):
    raise RuntimeError("Submission row count does not match sample_submission length.")
if list(submission.columns) != ["image_id", "label"]:
    raise RuntimeError("Submission columns are incorrect.")

submission["label"] = submission["label"].astype(int).clip(0, 4)

submission.to_csv("submission.csv", index=False)
print(submission.shape)
print(submission.dtypes)
print(submission.iloc[:5])
