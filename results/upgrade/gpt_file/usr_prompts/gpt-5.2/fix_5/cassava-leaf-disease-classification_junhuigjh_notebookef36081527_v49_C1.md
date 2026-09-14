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

0.66293

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60015) has done: 'I fix the two blocking FileNotFoundErrors by removing dependencies on external Kaggle datasets (`/kaggle/input/train-tree/...` and `/kaggle/input/resnet50_70_512x512/...`) and instead train the same kind of DecisionTreeClassifier directly from the provided `train.csv` and `train_images`. To preserve the original core intent (tree on top of model probabilities), I generate per-image “probability-like” features using the existing ResNet50-head function (kept) and compute logits/probabilities from it; this keeps the pipeline structure (CNN → probs → decision tree) while making it self-contained. I also ensure `test_image_ids` is always defined from `sample_submission.csv`, and make submission writing robust (correct columns, row count, and `.csv` suffix). These changes are necessary for end-to-end execution and should yield a non-trivial accuracy score compared to the previous uniform-fallback behavior.'
- What this solution (achieved 0.59268) has done: 'Your current score is low because `resnet50(weights=None)` is untrained, so the “probability-like” features going into the decision tree are essentially noise. To move the accuracy up toward your target with minimal logic change, I keep the same pipeline (CNN → 5-dim probs → DecisionTreeClassifier) but switch the ResNet50 backbone to ImageNet pretrained weights so the 5-dim outputs become meaningful features. I also normalize with the standard ImageNet mean/std to match the pretrained model and keep everything else (tree hyperparameters, sampling, submission writing) unchanged for stability and runtime. These changes should significantly reduce the gap toward 0.8445 without changing your overall approach.'
- What this solution (achieved 0.66293) has done: 'Your current gap to the target is large (0.59268 → 0.84451), and the biggest issue is that your “CNN → 5-dim probs → DecisionTree” pipeline is using an untrained 5-class head (random fc layer), so the features going into the tree are mostly noise. To preserve the same core logic while improving accuracy, I keep ResNet50 ImageNet weights but change the extracted features to the pretrained penultimate layer (2048-dim) instead of the random 5-dim logits; the decision tree still consumes “CNN-derived features,” just more meaningful ones. I also add a very small amount of test-time augmentation (horizontal flip averaged with original) inside the same `predict_*` function to improve feature stability without changing the overall approach or adding any new training loops. Finally, I keep submission alignment identical to `sample_submission.csv` and ensure the script still runs end-to-end and writes `submission.csv`.'

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
def extract_feats_torch(model: nn.Module, pil_img: Image.Image) -> np.ndarray:
    x1 = torch_transforms(pil_img).unsqueeze(0).to(device)
    x2 = (
        torch_transforms(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
        .unsqueeze(0)
        .to(device)
    )
    f1 = model(x1)
    f2 = model(x2)
    feats = 0.5 * (f1 + f2)
    feats = feats.detach().cpu().numpy()[0].astype(np.float32)  # shape (2048,)
    return feats




## === cell 3
MAX_TRAIN_FOR_TREE = (
    6000  # chosen to keep runtime reasonable while producing a usable tree
)

n_train_total = len(train_df)
use_n = min(MAX_TRAIN_FOR_TREE, n_train_total)

rng = np.random.RandomState(0)
idxs = np.arange(n_train_total)
rng.shuffle(idxs)
use_idxs = idxs[:use_n]

FEAT_DIM = 2048
train_feats = np.zeros((use_n, FEAT_DIM), dtype=np.float32)
train_labels = np.zeros((use_n,), dtype=np.int64)

bad_count = 0
for j, ridx in enumerate(use_idxs, start=1):
    image_id = train_df.iloc[ridx]["image_id"]
    label = int(train_df.iloc[ridx]["label"])
    img_path = os.path.join(TRAIN_DIR, image_id)
    train_labels[j - 1] = label

    try:
        img = Image.open(img_path).convert("RGB")
        feats2 = extract_feats_torch(model2, img)  # shape (2048,)
    except Exception:
        feats2 = np.zeros(FEAT_DIM, dtype=np.float32)
        bad_count += 1

    train_feats[j - 1] = feats2

    if j % 200 == 0 or j == use_n:
        print(f"Feature extract train: {j}/{use_n} (bad={bad_count})", end="\r")
print()

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=8,
    min_samples_split=12,
    random_state=0,
)
decision_tree.fit(train_feats, train_labels)
print("Decision tree trained on:", train_feats.shape)



## === cell 4
image_ids = []
prediction = []

length = len(test_image_ids)
for idx, image_id in enumerate(test_image_ids, start=1):
    image_ids.append(image_id)
    img_path = os.path.join(TEST_DIR, image_id)

    try:
        img = Image.open(img_path).convert("RGB")
        feats2 = extract_feats_torch(model2, img)  # shape (2048,)
    except Exception:
        feats2 = np.zeros(FEAT_DIM, dtype=np.float32)

    pred_label = int(decision_tree.predict(feats2.reshape(1, -1))[0])
    prediction.append(pred_label)

    if idx % 100 == 0 or idx == length:
        print(f"Predict test: {idx}/{length}", end="\r")
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
