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
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet121, resnet50
from torchvision.models import DenseNet121_Weights, ResNet50_Weights

from sklearn.tree import DecisionTreeClassifier

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_ROOT}/train.csv"
TRAIN_DIR = f"{DATA_ROOT}/train_images"
TEST_DIR = f"{DATA_ROOT}/test_images"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"

TREE_TRAIN_PATH = "/kaggle/input/train-tree/train_tree.csv"
MODEL1_PATH = "/kaggle/input/densenet/keras/default/1/DenseNet (1).keras"
MODEL2_PATH = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)

assert os.path.exists(TRAIN_CSV_PATH), f"Missing {TRAIN_CSV_PATH}"
assert os.path.exists(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"



## === cell 2
densenet_weights = DenseNet121_Weights.IMAGENET1K_V1
resnet_weights = ResNet50_Weights.IMAGENET1K_V2

dn_mean = list(densenet_weights.transforms().mean)
dn_std = list(densenet_weights.transforms().std)
rn_mean = list(resnet_weights.transforms().mean)
rn_std = list(resnet_weights.transforms().std)

torch_transform_224 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dn_mean, std=dn_std),
    ]
)

torch_transform_512 = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=rn_mean, std=rn_std),
    ]
)


def pil_to_batch_tensor(
    img: Image.Image, tfm: transforms.Compose, device: torch.device
) -> torch.Tensor:
    x = tfm(img).unsqueeze(0).to(device, non_blocking=True)
    return x




## === cell 3
def _safe_load_state_dict(
    model: nn.Module, ckpt_path: str, device: torch.device
) -> bool:
    """Try to load a PyTorch checkpoint into `model`. Returns True if loaded, False otherwise."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    try:
        obj = torch.load(ckpt_path, map_location=device)
        if isinstance(obj, nn.Module):
            model.load_state_dict(obj.state_dict(), strict=False)
            return True
        if isinstance(obj, dict):
            state = obj.get("state_dict", obj.get("model", obj))
            if isinstance(state, dict) and any(
                k.startswith("module.") for k in state.keys()
            ):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            if isinstance(state, dict):
                model.load_state_dict(state, strict=False)
                return True
        return False
    except Exception:
        return False


model1 = densenet121(weights=densenet_weights)
model2 = resnet50(weights=resnet_weights)

loaded1 = _safe_load_state_dict(model1, MODEL1_PATH, device)
loaded2 = _safe_load_state_dict(model2, MODEL2_PATH, device)

model1.to(device).eval()
model2.to(device).eval()

USE_PENULTIMATE_EMBEDS = True

print(f"Device: {device}")
print(f"Loaded external weights: model1={loaded1}, model2={loaded2}")
print(
    "Using CNN feature vectors (two outputs concatenated -> DecisionTree): "
    + (
        "penultimate pooled embeddings"
        if USE_PENULTIMATE_EMBEDS
        else "final 1000-d logits"
    )
)



## === cell 4
train_df = pd.read_csv(TRAIN_CSV_PATH)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(f"train.csv columns unexpected: {train_df.columns.tolist()}")


def _extract_feats_batch(x1: torch.Tensor, x2: torch.Tensor) -> np.ndarray:
    """
    Returns concatenated feature vectors.
    DenseNet121 penultimate pooled embedding: 1024-d
    ResNet50 penultimate pooled embedding: 2048-d
    Total: 3072-d
    """
    with torch.inference_mode():
        if USE_PENULTIMATE_EMBEDS:
            f1 = model1.features(x1)
            f1 = torch.relu(f1)
            f1 = torch.nn.functional.adaptive_avg_pool2d(f1, (1, 1))
            f1 = torch.flatten(f1, 1)

            f2 = model2.conv1(x2)
            f2 = model2.bn1(f2)
            f2 = model2.relu(f2)
            f2 = model2.maxpool(f2)
            f2 = model2.layer1(f2)
            f2 = model2.layer2(f2)
            f2 = model2.layer3(f2)
            f2 = model2.layer4(f2)
            f2 = model2.avgpool(f2)
            f2 = torch.flatten(f2, 1)
        else:
            f1 = model1(x1)
            f2 = model2(x2)

    f1 = f1.detach().cpu().numpy().astype(np.float32)
    f2 = f2.detach().cpu().numpy().astype(np.float32)
    return np.concatenate([f1, f2], axis=1)


def infer_combined_feats_for_ids(image_ids, root_dir, batch_size=16):
    feat_dim = 3072 if USE_PENULTIMATE_EMBEDS else 2000
    combined = np.zeros((len(image_ids), feat_dim), dtype=np.float32)
    missing = []

    for start in range(0, len(image_ids), batch_size):
        end = min(len(image_ids), start + batch_size)
        ids = image_ids[start:end]

        imgs1, imgs2, ok_idx = [], [], []
        for j, image_id in enumerate(ids):
            p = os.path.join(root_dir, image_id)
            if not os.path.exists(p):
                missing.append(image_id)
                continue
            img = Image.open(p).convert("RGB")
            imgs1.append(torch_transform_224(img))
            imgs2.append(torch_transform_512(img))
            ok_idx.append(j)

        if not ok_idx:
            continue

        x1 = torch.stack(imgs1, dim=0).to(device, non_blocking=True)
        x2 = torch.stack(imgs2, dim=0).to(device, non_blocking=True)

        concat = _extract_feats_batch(x1, x2)

        for local_pos, j in enumerate(ok_idx):
            combined[start + j, :] = concat[local_pos]

        if (end % 256) == 0 or end == len(image_ids):
            print(f"Extracted train features: {end}/{len(image_ids)}", end="\r")

    print()
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} train images under {root_dir}. Example: {missing[:5]}"
        )
    return combined


labels = train_df["label"].astype(int).to_numpy()
train_image_ids = train_df["image_id"].tolist()
train_feats_all = infer_combined_feats_for_ids(
    train_image_ids, TRAIN_DIR, batch_size=16
)

feat_mean = train_feats_all.mean(axis=0, keepdims=True).astype(np.float32)
feat_std = train_feats_all.std(axis=0, keepdims=True).astype(np.float32)
feat_std = np.where(feat_std < 1e-6, 1.0, feat_std).astype(np.float32)
train_feats_all = (train_feats_all - feat_mean) / feat_std

decision_tree = DecisionTreeClassifier(
    criterion="gini", max_depth=8, min_samples_split=12, random_state=42
)
decision_tree.fit(train_feats_all, labels)

print("Trained DecisionTree on generated CNN features:", train_feats_all.shape)



## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission missing 'image_id'. Columns: {sample_sub.columns.tolist()}"
    )

image_ids = sample_sub["image_id"].tolist()


def infer_combined_feats_test(image_ids, root_dir, batch_size=16):
    feat_dim = 3072 if USE_PENULTIMATE_EMBEDS else 2000
    combined = np.zeros((len(image_ids), feat_dim), dtype=np.float32)
    missing = []

    for start in range(0, len(image_ids), batch_size):
        end = min(len(image_ids), start + batch_size)
        ids = image_ids[start:end]

        imgs1, imgs2, ok_idx = [], [], []
        for j, image_id in enumerate(ids):
            p = os.path.join(root_dir, image_id)
            if not os.path.exists(p):
                missing.append(image_id)
                continue
            img = Image.open(p).convert("RGB")
            imgs1.append(torch_transform_224(img))
            imgs2.append(torch_transform_512(img))
            ok_idx.append(j)

        if not ok_idx:
            continue

        x1 = torch.stack(imgs1, dim=0).to(device, non_blocking=True)
        x2 = torch.stack(imgs2, dim=0).to(device, non_blocking=True)

        concat = _extract_feats_batch(x1, x2)

        for local_pos, j in enumerate(ok_idx):
            combined[start + j, :] = concat[local_pos]

        if (end % 256) == 0 or end == len(image_ids):
            print(f"Extracted test features: {end}/{len(image_ids)}", end="\r")

    print()
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} test images under {root_dir}. Example: {missing[:5]}"
        )
    return combined


combined_feats_test = infer_combined_feats_test(image_ids, TEST_DIR, batch_size=16)
combined_feats_test = (combined_feats_test - feat_mean) / feat_std



## === cell 6
prediction = decision_tree.predict(combined_feats_test).astype(int)

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission = submission[["image_id", "label"]]
submission["label"] = submission["label"].astype(int)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Label counts:\n", submission["label"].value_counts().sort_index())
