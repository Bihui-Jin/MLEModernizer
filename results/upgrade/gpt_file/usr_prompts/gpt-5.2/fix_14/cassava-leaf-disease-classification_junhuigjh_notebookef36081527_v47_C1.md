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

0.74178

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60277) has done: 'I make the script robust to missing external Kaggle-dataset artifacts by (1) removing the hard dependency on `train_tree.csv` and the external model checkpoints, and instead training the same DecisionTree meta-model using out-of-fold probabilities computed from the two torchvision models. I also fix the CUDA/CPU dtype/device mismatch by ensuring both models and inputs are on the same device. Finally, I guarantee a valid `submission.csv` is always written by aligning predictions to `sample_submission.csv` ordering and using the required columns and types. These changes preserve the core approach (two CNN probability vectors concatenated → DecisionTreeClassifier) while making it run end-to-end in the provided environment.'
- What this solution (achieved 0.61099) has done: 'Your current score is low mainly because both CNNs are effectively untrained (`weights=None` and the provided external checkpoints don’t exist), so the DecisionTree is learning on near-random softmax outputs. To move accuracy upward toward the 0.8249 target with minimal semantic change, I keep the exact same pipeline (DenseNet+ResNet → softmax probs concat → DecisionTree) but load strong default ImageNet pretrained weights for both backbones when external weights are missing. I also switch normalization to the standard ImageNet mean/std to match those pretrained weights (still the same “resize → tensor → normalize” feature extraction logic). Everything else (fold logic, tree hyperparams, submission alignment/format) remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.679) has done: 'Your score is held back because both CNN heads are randomly initialized for 5-way cassava classes, so the “probability features” given to the DecisionTree are mostly uninformative. To move accuracy up toward the 0.8249 target while preserving the same core pipeline (two CNNs → softmax probs concat → DecisionTree), I (1) use the ImageNet backbones strictly as feature extractors by outputting their native 1000-logit probabilities, and (2) train the same DecisionTree on the concatenated 2000-dim probability vectors. This is a minimal semantic adjustment (still “CNN probability vectors concatenated → DecisionTree”), but it makes the features meaningful without any extra training loops or new models. I also keep submission alignment unchanged and ensure `submission.csv` is written exactly in the required format.'
- What this solution (achieved 0.67526) has done: 'Your current pipeline is likely underperforming because it uses ImageNet 1000-class *softmax probabilities* as features; those are highly compressed and discard useful information for transfer. To move accuracy up toward the 0.8249 target while preserving the exact core approach (two CNN outputs concatenated → DecisionTree), I switch the features from softmax probabilities to the *pre-softmax logits* (still “CNN output vectors”, just before normalization), which is a minimal semantic tweak but typically much more informative. I also ensure inference uses `torch.inference_mode()` (no semantic change) and keep submission alignment/format identical. No model architecture, loss, or training loop changes are introduced.'
- What this solution (achieved 0.67526) has done: 'Your current score is far below the target (0.675 vs 0.8249), so we should make a small but meaningful improvement without changing the core pipeline (two pretrained CNN outputs concatenated → DecisionTree). The biggest limiting factor is that the tree is being trained on high-dimensional, uncalibrated logits with no normalization; DecisionTrees are very sensitive to feature scaling/offsets, so a simple standardization of the concatenated CNN outputs (fit on train, apply to test) usually improves accuracy while keeping the exact same modeling approach. I also make the DenseNet and ResNet input preprocessing match their official pretrained-weight transforms (still “resize→tensor→normalize”, just the correct normalization per backbone) to improve feature quality. Everything else (same models, no training loops, same DecisionTree, same submission alignment/format) remains intact and it still write `submission.csv`.'
- What this solution (achieved 0.679) has done: 'The timeout is dominated by per-image PIL decoding + two separate resize/normalize passes, plus Python-loop overhead while building batches; the CNN forward passes are comparatively efficient on GPU but starved by the input pipeline. I keep the exact same feature extraction (two pretrained CNN penultimate embeddings concatenated, z-score, DecisionTree) but speed it up by (1) decoding each image once, resizing once to 512 and deriving the 224 view from it (same interpolation), (2) using a PyTorch `Dataset`/`DataLoader` with multiple workers + pinned memory to parallelize CPU I/O/resize, and (3) preallocating numpy outputs and avoiding per-sample Python assignment loops. I also enable `torch.backends.cudnn.benchmark` for faster fixed-size convs (no accuracy change) and reuse the same loader logic for train/test.'
- What this solution (achieved 0.66442) has done: 'Your current pipeline is already executing end-to-end and the main gap to the 0.8249 target is likely feature quality rather than the DecisionTree itself. To move accuracy upward with minimal semantic change, I keep the exact same “two pretrained CNN penultimate embeddings concatenated → z-score → DecisionTree” approach, but fix a subtle preprocessing mismatch: both backbones’ pretrained weights expect their own canonical resize/crop pipeline rather than a raw forced square resize. I implement each model’s official `weights.transforms()` (which includes proper interpolation/resize/crop and normalization) while still decoding each image only once, so the features better match what the backbones were trained on. This is a small, targeted change that typically improves transfer performance without changing the model architecture, loss, or training loop, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.67975) has done: 'Your current gap to the target is large (0.66442 → 0.82487), and the most likely bottleneck in this exact “CNN embeddings → z-score → DecisionTree” pipeline is that a single DecisionTree is too weak/unstable for 3072-d continuous embeddings. To improve accuracy while preserving the same training approach (tree-based classifier on concatenated CNN features), I replace `DecisionTreeClassifier` with `ExtraTreesClassifier` (still a tree ensemble, same semantics: non-linear splits on the same features). I keep the same feature extraction, preprocessing (`weights.transforms()`), standardization, and submission alignment/format; only the tree model is strengthened with minimal, conservative hyperparameters and a fixed random seed. This should move the score upward toward the target without changing the CNN side or any evaluation semantics, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.73617) has done: 'To move your accuracy upward toward the 0.8249 target without changing the core “two pretrained CNN embeddings → standardize → tree ensemble” pipeline, I make the smallest reliable improvement on the tree side: use out-of-bag (OOB) evaluation to automatically pick between `bootstrap=False` and `bootstrap=True` (a calibration/regularization tweak that often improves generalization). This keeps the exact same feature extraction, standardization, and ExtraTrees model class; we just choose the bootstrap setting based on the training data itself (no extra data, no metric mismatch). I also add `class_weight="balanced_subsample"` which is a minimal, legitimate way to reduce bias from label imbalance and typically improves accuracy for cassava. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.74066) has done: 'Your current score (0.73617) is below the target (0.82487), so we should make a small, low-risk generalization improvement while keeping the exact same pipeline: two pretrained CNN penultimate embeddings → standardize → ExtraTrees → predict. The main minimal lever here is to make the OOB-based bootstrap selection actually meaningful (your hard threshold likely prevents selection) and to slightly regularize/strengthen the ExtraTrees in a controlled way (more trees + a bit more depth) to close part of the gap without changing any core modeling semantics. I keep feature extraction, preprocessing (`weights.transforms()`), standardization, and submission alignment identical, and only adjust the tree selection logic and a couple of conservative tree hyperparameters. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.74178) has done: 'To move your score upward toward the 0.8249 target without changing the core pipeline (two pretrained CNN penultimate embeddings → standardize → ExtraTrees → predict), I make a single, low-risk generalization improvement: add a tiny amount of deterministic Gaussian noise to the standardized training features before fitting the tree ensemble. This acts like feature-space jitter/regularization for axis-aligned tree splits and often improves test accuracy for transfer embeddings, while keeping the exact same model class, loss/metric semantics, and inference path. I also make the OOB selection logic actually choose the better of the two fits (bootstrap True/False) based on OOB when available, instead of always picking bootstrap=True. Everything else (feature extraction, transforms, standardization, submission alignment/format) stays the same and it still writes a valid `submission.csv`.'

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
from sklearn.ensemble import ExtraTreesClassifier

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = (
    True  # faster conv algorithm selection for static shapes
)

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

dn_preprocess = densenet_weights.transforms()
rn_preprocess = resnet_weights.transforms()

_RESAMPLE = Image.BILINEAR




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
    "Using CNN feature vectors (two outputs concatenated -> Tree model): "
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

    f1 = f1.detach().cpu().numpy().astype(np.float32, copy=False)
    f2 = f2.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.concatenate([f1, f2], axis=1)


class _CassavaImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, root_dir: str):
        self.image_ids = list(image_ids)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        p = os.path.join(self.root_dir, image_id)
        img = Image.open(p).convert("RGB")

        x1 = dn_preprocess(img)  # DenseNet expected preprocessing
        x2 = rn_preprocess(img)  # ResNet expected preprocessing
        return idx, x1, x2, image_id


def _infer_combined_feats_dataloader(image_ids, root_dir, batch_size=32):
    feat_dim = 3072 if USE_PENULTIMATE_EMBEDS else 2000
    combined = np.zeros((len(image_ids), feat_dim), dtype=np.float32)

    ds = _CassavaImageDataset(image_ids, root_dir)

    num_workers = min(4, (os.cpu_count() or 2))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    missing = []
    for idxs, x1, x2, image_id_batch in loader:
        if x1 is None or x2 is None:
            missing.extend(list(image_id_batch))
            continue

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        concat = _extract_feats_batch(x1, x2)
        combined[idxs.numpy(), :] = concat

        end = int(idxs.max().item()) + 1
        if (end % 256) == 0 or end >= len(image_ids):
            print(f"Extracted features: {end}/{len(image_ids)}", end="\r")

    print()
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images under {root_dir}. Example: {missing[:5]}"
        )
    return combined


labels = train_df["label"].astype(int).to_numpy()
train_image_ids = train_df["image_id"].tolist()

train_feats_all = _infer_combined_feats_dataloader(
    train_image_ids, TRAIN_DIR, batch_size=32
)

feat_mean = train_feats_all.mean(axis=0, keepdims=True).astype(np.float32)
feat_std = train_feats_all.std(axis=0, keepdims=True).astype(np.float32)
feat_std = np.where(feat_std < 1e-6, 1.0, feat_std).astype(np.float32)
train_feats_all = (train_feats_all - feat_mean) / feat_std


def _fit_extratrees_with_oob_selection(X, y, base_kwargs):
    m_no = ExtraTreesClassifier(
        **{**base_kwargs, "bootstrap": False, "oob_score": False}
    )
    m_no.fit(X, y)
    train_acc_no = float(m_no.score(X, y))

    m_oob = ExtraTreesClassifier(
        **{**base_kwargs, "bootstrap": True, "oob_score": True}
    )
    m_oob.fit(X, y)
    oob = getattr(m_oob, "oob_score_", None)
    train_acc_oob = float(m_oob.score(X, y))

    print(f"ExtraTrees train acc bootstrap=False: {train_acc_no:.6f}")
    print(f"ExtraTrees train acc bootstrap=True : {train_acc_oob:.6f}")
    print(f"ExtraTrees OOB score (bootstrap=True): {oob}")

    if oob is not None:
        if oob >= 0.0:
            if oob >= (train_acc_no - 0.02):
                print(
                    "Selected model: bootstrap=True (OOB indicates comparable generalization)"
                )
                return m_oob
            else:
                print(
                    "Selected model: bootstrap=False (OOB indicates bootstrap may underfit)"
                )
                return m_no

    print("Selected model: bootstrap=False (no OOB available)")
    return m_no


base_tree_kwargs = dict(
    n_estimators=800,
    criterion="gini",
    max_depth=22,
    min_samples_split=12,
    min_samples_leaf=4,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample",  # keep as-is
)

_rng = np.random.RandomState(42)
JITTER_STD = 0.01  # small; keeps semantics intact while adding regularization
train_feats_fit = train_feats_all + (
    _rng.normal(0.0, JITTER_STD, size=train_feats_all.shape).astype(np.float32)
)

tree_model = _fit_extratrees_with_oob_selection(
    train_feats_fit, labels, base_tree_kwargs
)

print("Trained tree model on generated CNN features:", train_feats_all.shape)




## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission missing 'image_id'. Columns: {sample_sub.columns.tolist()}"
    )

image_ids = sample_sub["image_id"].tolist()

combined_feats_test = _infer_combined_feats_dataloader(
    image_ids, TEST_DIR, batch_size=32
)
combined_feats_test = (combined_feats_test - feat_mean) / feat_std




## === cell 6
prediction = tree_model.predict(combined_feats_test).astype(int)

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission = submission[["image_id", "label"]]
submission["label"] = submission["label"].astype(int)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Label counts:\n", submission["label"].value_counts().sort_index())
