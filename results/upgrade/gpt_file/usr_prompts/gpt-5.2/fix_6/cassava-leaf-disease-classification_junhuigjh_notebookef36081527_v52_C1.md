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

0.8578120278029616

# 6. Current score

0.55344

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.68386) has done: 'I remove the hard dependency on missing external model files under `/kaggle/input/...` by falling back to a built-in torchvision ResNet50 feature extractor (same overall approach: extract per-image model features then train a DecisionTreeClassifier). I also fix the cell-order/state bug that caused `use_model1` to be undefined by ensuring it is always defined before `get_probs_for_image()` is used. To keep runtime under control while preserving the same “train on extracted features → predict test” semantics, I batch image loading and feature extraction and avoid TensorFlow entirely (since it’s not guaranteed in this environment). Finally, I ensure a valid `submission.csv` with the exact required columns is always written.'
- What this solution (achieved 0.6633) has done: 'Your current gap to the target is large (0.68386 → 0.8578, about +25%), so we should improve accuracy but keep the same “extract CNN features → train DecisionTree → predict” core pipeline. The biggest minimal win is fixing image preprocessing: ResNet50 pretrained weights expect a 224×224 crop-style input; your 512×512 resize distorts scale and wastes compute, hurting both accuracy and runtime. We switch to the standard ResNet50 validation transform (resize shorter side to 256 then center-crop 224) while keeping the same model, same classifier, and same training loop. We also add safe PIL file-handle closing (context managers) to avoid intermittent I/O issues without changing semantics.'
- What this solution (achieved 0.51158) has done: 'You’re currently far below the target (0.6633 vs 0.8578), so we need a modest, low-risk accuracy lift while keeping the same “ResNet50 feature extractor → DecisionTreeClassifier” pipeline. The biggest issue is that you train a single shallow tree on highly imbalanced classes; a minimal change that usually boosts accuracy substantially is to use `class_weight="balanced"` and allow a bit more tree capacity (`min_samples_leaf` + slightly deeper `max_depth`) without changing the modeling family. I also make feature extraction more stable by ensuring deterministic ordering and by using `torch.no_grad()`/`inference_mode()` already present while avoiding any accidental randomness; core feature extractor and training loop remain the same. The submission format and paths stay unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.55344) has done: 'Your current score (0.51158) is far below the target (0.85781), so we should improve accuracy while keeping the same “ResNet50 feature extractor → DecisionTreeClassifier” pipeline. The most impactful minimal fix is to use the ResNet50 feature extractor exactly as intended by the pretrained weights: apply the weights’ official preprocessing (already done) and, crucially, use the weights’ built-in inference-time normalization consistently and avoid any accidental CPU/GPU nondeterminism. Then we make a small, low-risk DecisionTree adjustment that usually improves generalization on this task without changing the model family: tune `max_depth`/`min_samples_leaf` slightly toward a better bias/variance point and add `min_samples_split`. Finally, we keep submission generation identical but ensure strict alignment and stable ordering.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.tree import DecisionTreeClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TRAIN_DIR = f"{DATA_ROOT}/train_images"
TEST_DIR = f"{DATA_ROOT}/test_images"
SAMPLE_SUB = f"{DATA_ROOT}/sample_submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

use_model1 = False
model1 = None

try:
    _weights = models.ResNet50_Weights.DEFAULT
    torch_transforms = _weights.transforms()
except Exception:
    torch_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )




## === cell 1
@torch.inference_mode()
def pil_to_torch_tensor(img_pil: Image.Image) -> torch.Tensor:
    return torch_transforms(img_pil)


def build_feature_extractor() -> nn.Module:
    try:
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
    except Exception as e:
        print(
            f"WARNING: Could not load pretrained ResNet50 weights; using random init. Root error: {repr(e)}"
        )
        backbone = models.resnet50(weights=None)

    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


model2 = build_feature_extractor()


def get_features_for_image(img_path: str) -> np.ndarray:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        x = pil_to_torch_tensor(img).unsqueeze(0).to(device)
    emb = model2(x).detach().float().cpu().numpy()[0].astype(np.float32)  # (2048,)
    if use_model1:
        p1 = np.array(model1.predict(None), dtype=np.float32)  # unreachable
    else:
        p1 = np.zeros((5,), dtype=np.float32)
    return np.concatenate([p1, emb], axis=0)  # (5 + 2048,)




## === cell 2
@torch.inference_mode()
def get_features_for_paths(paths, batch_size: int = 32) -> np.ndarray:
    feats = []
    n = len(paths)

    paths = list(paths)

    for start in range(0, n, batch_size):
        batch_paths = paths[start : start + batch_size]
        imgs = []
        for fp in batch_paths:
            with Image.open(fp) as img:
                img = img.convert("RGB")
                imgs.append(pil_to_torch_tensor(img))
        x = torch.stack(imgs, dim=0).to(device)  # (B,3,H,W) where H=W=224

        emb = model2(x).detach().float().cpu().numpy().astype(np.float32)  # (B,2048)
        p1 = np.zeros((emb.shape[0], 5), dtype=np.float32)
        feats.append(np.concatenate([p1, emb], axis=1))  # (B,2053)

        done = start + len(batch_paths)
        if done % 256 == 0 or done == n:
            print(f"Features: {done}/{n}", end="\r")
    print()
    return np.vstack(feats)


train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)

train_df["filepath"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_DIR, x))
train_df = train_df[train_df["filepath"].apply(os.path.exists)].reset_index(drop=True)

train_df = train_df.sort_values("image_id").reset_index(drop=True)

train_labels = train_df["label"].astype(int).to_numpy()
train_paths = train_df["filepath"].tolist()

train_features = get_features_for_paths(train_paths, batch_size=32)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=SEED,
)
decision_tree.fit(train_features, train_labels)



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
assert "image_id" in sample_sub.columns

test_image_ids = sample_sub["image_id"].tolist()
test_filepaths = [os.path.join(TEST_DIR, iid) for iid in test_image_ids]

missing = [fp for fp in test_filepaths if not os.path.exists(fp)]
assert len(missing) == 0, f"Missing {len(missing)} test images; example: {missing[0]}"

test_features = get_features_for_paths(test_filepaths, batch_size=32)

prediction = decision_tree.predict(test_features).astype(int)



## === cell 4
submission = pd.DataFrame({"image_id": test_image_ids, "label": prediction})
assert submission.shape[0] == len(sample_sub), "Submission row count mismatch"
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch"

submission.to_csv("submission.csv", index=False)
submission.head(10)
