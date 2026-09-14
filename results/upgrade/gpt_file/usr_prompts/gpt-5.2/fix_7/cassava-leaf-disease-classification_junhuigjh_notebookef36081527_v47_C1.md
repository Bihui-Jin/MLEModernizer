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

0.8248715624055606

# 6. Current score

0.67526

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60277) has done: 'I make the script robust to missing external Kaggle-dataset artifacts by (1) removing the hard dependency on `train_tree.csv` and the external model checkpoints, and instead training the same DecisionTree meta-model using out-of-fold probabilities computed from the two torchvision models. I also fix the CUDA/CPU dtype/device mismatch by ensuring both models and inputs are on the same device. Finally, I guarantee a valid `submission.csv` is always written by aligning predictions to `sample_submission.csv` ordering and using the required columns and types. These changes preserve the core approach (two CNN probability vectors concatenated → DecisionTreeClassifier) while making it run end-to-end in the provided environment.'
- What this solution (achieved 0.61099) has done: 'Your current score is low mainly because both CNNs are effectively untrained (`weights=None` and the provided external checkpoints don’t exist), so the DecisionTree is learning on near-random softmax outputs. To move accuracy upward toward the 0.8249 target with minimal semantic change, I keep the exact same pipeline (DenseNet+ResNet → softmax probs concat → DecisionTree) but load strong default ImageNet pretrained weights for both backbones when external weights are missing. I also switch normalization to the standard ImageNet mean/std to match those pretrained weights (still the same “resize → tensor → normalize” feature extraction logic). Everything else (fold logic, tree hyperparams, submission alignment/format) remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.679) has done: 'Your score is held back because both CNN heads are randomly initialized for 5-way cassava classes, so the “probability features” given to the DecisionTree are mostly uninformative. To move accuracy up toward the 0.8249 target while preserving the same core pipeline (two CNNs → softmax probs concat → DecisionTree), I (1) use the ImageNet backbones strictly as feature extractors by outputting their native 1000-logit probabilities, and (2) train the same DecisionTree on the concatenated 2000-dim probability vectors. This is a minimal semantic adjustment (still “CNN probability vectors concatenated → DecisionTree”), but it makes the features meaningful without any extra training loops or new models. I also keep submission alignment unchanged and ensure `submission.csv` is written exactly in the required format.'
- What this solution (achieved 0.67526) has done: 'Your current pipeline is likely underperforming because it uses ImageNet 1000-class *softmax probabilities* as features; those are highly compressed and discard useful information for transfer. To move accuracy up toward the 0.8249 target while preserving the exact core approach (two CNN outputs concatenated → DecisionTree), I switch the features from softmax probabilities to the *pre-softmax logits* (still “CNN output vectors”, just before normalization), which is a minimal semantic tweak but typically much more informative. I also ensure inference uses `torch.inference_mode()` (no semantic change) and keep submission alignment/format identical. No model architecture, loss, or training loop changes are introduced.'
- What this solution (achieved 0.67526) has done: 'Your current score is far below the target (0.675 vs 0.8249), so we should make a small but meaningful improvement without changing the core pipeline (two pretrained CNN outputs concatenated → DecisionTree). The biggest limiting factor is that the tree is being trained on high-dimensional, uncalibrated logits with no normalization; DecisionTrees are very sensitive to feature scaling/offsets, so a simple standardization of the concatenated CNN outputs (fit on train, apply to test) usually improves accuracy while keeping the exact same modeling approach. I also make the DenseNet and ResNet input preprocessing match their official pretrained-weight transforms (still “resize→tensor→normalize”, just the correct normalization per backbone) to improve feature quality. Everything else (same models, no training loops, same DecisionTree, same submission alignment/format) remains intact and it still write `submission.csv`.'

# 9. Code solution

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

USE_LOGITS_FEATURES = True

print(f"Device: {device}")
print(f"Loaded external weights: model1={loaded1}, model2={loaded2}")
print(
    "Using ImageNet pretrained 1000-dim outputs as features for each model; "
    + ("logits (pre-softmax)" if USE_LOGITS_FEATURES else "softmax probabilities")
    + " -> concatenated -> DecisionTree."
)



## === cell 4
train_df = pd.read_csv(TRAIN_CSV_PATH)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(f"train.csv columns unexpected: {train_df.columns.tolist()}")


def infer_combined_feats_for_ids(image_ids, root_dir, batch_size=16):
    combined = np.zeros((len(image_ids), 2000), dtype=np.float32)
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

        with torch.inference_mode():
            out1 = model1(x1)
            out2 = model2(x2)

            if USE_LOGITS_FEATURES:
                f1 = out1.detach().cpu().numpy().astype(np.float32)
                f2 = out2.detach().cpu().numpy().astype(np.float32)
            else:
                f1 = (
                    torch.softmax(out1, dim=1).detach().cpu().numpy().astype(np.float32)
                )
                f2 = (
                    torch.softmax(out2, dim=1).detach().cpu().numpy().astype(np.float32)
                )

        concat = np.concatenate([f1, f2], axis=1)
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
    combined = np.zeros((len(image_ids), 2000), dtype=np.float32)
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

        with torch.inference_mode():
            out1 = model1(x1)
            out2 = model2(x2)

            if USE_LOGITS_FEATURES:
                f1 = out1.detach().cpu().numpy().astype(np.float32)
                f2 = out2.detach().cpu().numpy().astype(np.float32)
            else:
                f1 = (
                    torch.softmax(out1, dim=1).detach().cpu().numpy().astype(np.float32)
                )
                f2 = (
                    torch.softmax(out2, dim=1).detach().cpu().numpy().astype(np.float32)
                )

        concat = np.concatenate([f1, f2], axis=1)
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
