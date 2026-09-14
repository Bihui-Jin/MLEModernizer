# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8691447567240859

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.60277) has done: 'I fix the runtime crash by resizing inputs for `vit_b_16` to its required `image_size` (224) instead of 518, which currently triggers the assertion error. I also correct a subtle dataset bug where the label check uses `"label" in row` (checks values, not index), which can silently mis-handle labels; this is score-neutral but improves correctness. Finally, I keep the rest of the pipeline identical so it runs end-to-end and writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.63901) has done: 'I fix the crash by making the ViT feature extractor compatible with the installed torchvision `vit_b_16` implementation (it doesn’t have `ln`; the final norm is `encoder.ln`). I keep the rest of the pipeline the same (same models, same feature concatenation, same DecisionTree training/prediction), only adjusting the ViT embedding function to be robust across torchvision versions. I also remove the accidental duplicate placeholder definition of `vit_b16_embed` to avoid confusion and ensure the correct function is used. This should run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.63901) has done: 'You’re currently training the DecisionTree on raw concatenated CNN logits + ResNet/ViT embeddings where the feature scales are wildly different, so the tree’s split decisions are dominated by whichever block has larger numeric ranges. To move accuracy up toward your target with minimal disruption, I add a deterministic feature standardization step (fit on train features, apply to test features) while keeping the exact same models, embeddings, and DecisionTree hyperparameters. This preserves the core logic (same feature extractors + same classifier) but improves the tree’s ability to use all feature blocks. I also ensure the scaler uses float32 for speed and stable memory use and keep the submission writing identical.'
- What this solution (achieved 0.63789) has done: 'You’re currently training the DecisionTree on features extracted with `model1` left at random initialization, which injects mostly-noise logits into the concatenated feature vector and drags accuracy down; the smallest meaningful improvement toward your target is to load ImageNet weights for `model1` by swapping it to a pretrained `resnet18` classifier head while keeping the exact same “logits as features” idea and the same downstream DecisionTree training. To preserve the core pipeline, we still concatenate `[model1_logits, resnet18_embed, vit_embed]`, keep the same scaler + DecisionTree hyperparameters, and keep the same submission writing and ordering logic. This change is directly relevant to score (better features), should remain within runtime, and doesn’t alter the training approach (still pure feature extraction + tree). I also set `model1` to `.eval()` with `weights` and keep input size/normalization exactly as you already use for the 512 branch.'
- What this solution (achieved 0.64013) has done: 'I make two minimal, directly score-relevant fixes while keeping your exact “pretrained feature extractors → concatenate → StandardScaler → DecisionTree” pipeline intact. First, `model1` currently has a randomly initialized 5-class head, so its logits are mostly noise; I replace its head with `nn.Identity()` so you use the pretrained 1000-d ImageNet logits as features (same idea: “logits as features”, but now meaningful). Second, the ViT embedding path can output different tensors depending on torchvision version; I force it to always return the pooled class-token embedding (consistent 768-d feature) instead of sometimes returning the full token sequence. These are small changes expected to raise accuracy toward your target without changing the training approach or adding new models.'
- What this solution (achieved 0.63789) has done: 'Your current gap to the target is large (0.64013 → 0.86914), so we need a meaningful but still minimal change that preserves your core “pretrained feature extractors → concatenate → StandardScaler → DecisionTree” pipeline. The smallest high-impact fix is to make `model1` and `model2` consume the ImageNet-preprocessing they were trained with (currently you normalize with ImageNet mean/std but you skip the required `ResNet18_Weights` resize/crop pipeline, and you also upscale to 512 which is off-distribution). I switch the 512 branch to the official ResNet18 weights transforms (224px crop + proper interpolation/antialias), while keeping the ViT branch exactly as-is and keeping the same feature extraction, scaler, DecisionTree hyperparameters, and submission writing. This should improve feature quality substantially and move accuracy upward toward your target without changing your overall approach.'
- What this solution (achieved 0.4787) has done: 'Your current score (0.63789) is far below the target (0.86914), so we need a meaningful accuracy lift while keeping your exact pipeline (fixed pretrained feature extractors → concatenate → StandardScaler → DecisionTree) intact. The biggest win with minimal disruption is to add simple, deterministic augmentation only for the training feature extraction (horizontal flip + light color jitter), while keeping the test extraction unchanged; this improves feature robustness without changing the model/learner. I also switch the DecisionTree’s `class_weight` to `"balanced"` to better handle cassava’s label imbalance, which often improves accuracy for this competition without changing the training approach. Finally, I keep the same submission ordering logic and ensure the file is written as `submission.csv`.'
- What this solution (achieved 0.46413) has done: 'Your current score (0.4787) is far below the target (0.8691), so we need a real accuracy lift while preserving your core pipeline (fixed pretrained feature extractors → concatenate → StandardScaler → DecisionTree). The most impactful minimal fix is to stop using augmented views to extract *training* features, because those random transforms inject label-noise into the feature table that a DecisionTree overfits badly; we keep augmentation off for feature extraction to make train/test features match. To still address class imbalance without changing the learning approach, we also use deterministic sample weights computed from label frequencies (instead of `class_weight="balanced"` inside the tree, which can interact oddly with tree split criteria) while keeping the same tree type and hyperparameters. Everything else (models, embeddings, concatenation, scaling, submission ordering/format) remains the same and the script still writes `submission.csv`.'
- What this solution (achieved 0.50934) has done: 'You’re far below the target (0.464 → 0.869, higher-is-better), so we need a meaningful accuracy lift without changing your core pipeline (fixed pretrained feature extractors → concatenate → StandardScaler → DecisionTree). The biggest issue is that you’re extracting features on the full training set without any validation, so the DecisionTree is likely overfitting and then generalizing poorly to test; we add a deterministic stratified holdout split and tune only `max_depth` (keeping the same model type and training approach) to pick a depth that generalizes better. This stays within your semantics (still a single DecisionTree on the same scaled concatenated features) and doesn’t add new models or change feature extraction. Finally, we refit on the full training set using the selected depth and generate `submission.csv` exactly as before.'
- What this solution (achieved 0.50934) has done: 'We keep your exact pipeline (fixed pretrained feature extractors → concatenate → StandardScaler → DecisionTree) and only make minimal changes that are directly likely to improve generalization toward the target accuracy. The main upgrade is to select `max_depth` using a small stratified K-fold CV instead of a single holdout split, which reduces selection noise without changing the model type or training semantics. We also add a tiny, deterministic tie-break rule (prefer shallower depth when CV scores are equal within a very small epsilon) to avoid overfitting and stabilize the chosen depth. Everything else (feature extraction, scaling, sample weighting, submission ordering/format) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.52877) has done: 'I keep your exact “pretrained extractors → concatenate → StandardScaler → DecisionTree” pipeline, but make two minimal changes that usually improve generalization (and thus accuracy) without changing the overall approach. First, I make the ResNet/Vit preprocessing match their official ImageNet weight transforms (instead of mixing a ResNet-specific crop/resize with a manual ViT normalization), which directly improves feature quality. Second, I slightly extend the `max_depth` search grid (still only tuning depth) so the CV-selected depth can land closer to the sweet spot for this feature set. Everything else (models, embeddings, scaling, sample weighting, submission ordering/format) remains the same and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, vit_b_16
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

NUM_CLASSES = 5

_resnet_weights = ResNet18_Weights.IMAGENET1K_V1
_vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
torch_transforms_512 = _resnet_weights.transforms()
torch_transforms_vit = _vit_weights.transforms()

torch_transforms_512_train_aug = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02
        ),
        torch_transforms_512,
    ]
)

torch_transforms_vit_train_aug = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02
        ),
        torch_transforms_vit,
    ]
)




## === cell 1
def build_model1():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval()
    return m


def build_model2():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.eval()
    return m


def build_model3():
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m.eval()
    return m


model1 = build_model1().to(device).eval()
model2 = build_model2().to(device).eval()
model3 = build_model3().to(device).eval()


@torch.no_grad()
def resnet18_embed(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def vit_b16_embed(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(m, "forward_features"):
        feats = m.forward_features(x)
        if isinstance(feats, torch.Tensor):
            if feats.ndim == 3:
                return feats[:, 0]
            return feats

    x = m._process_input(x)  # [B, num_patches, hidden_dim]
    n = x.shape[0]
    batch_class_token = m.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = m.encoder(x)

    if hasattr(m, "encoder") and hasattr(m.encoder, "ln"):
        x = m.encoder.ln(x)
    elif hasattr(m, "ln"):
        x = m.ln(x)

    return x[:, 0]




## === cell 2
class CassavaDualTransformDataset(Dataset):
    def __init__(self, df, img_dir, transform_512, transform_vit):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.t512 = transform_512
        self.tv = transform_vit

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        path = os.path.join(self.img_dir, image_id)

        img = Image.open(path).convert("RGB")
        x512 = self.t512(img)
        xvit = self.tv(img)

        y = row["label"] if "label" in self.df.columns else -1
        return image_id, x512, xvit, int(y)


def _make_loader(ds, batch_size):
    num_workers = 2
    pin = torch.cuda.is_available()
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    if num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    return DataLoader(ds, **kwargs)


def extract_features(df, img_dir, batch_size=32, train_mode=False):
    t512 = torch_transforms_512
    tvit = torch_transforms_vit

    ds = CassavaDualTransformDataset(df, img_dir, t512, tvit)
    loader = _make_loader(ds, batch_size=batch_size)

    out_dim = 1000 + 512 + 768
    n = len(ds)
    feats = np.empty((n, out_dim), dtype=np.float32)
    image_ids = [None] * n

    offset = 0
    with torch.inference_mode():
        for ids, x512, xvit, _ in loader:
            b = len(ids)
            image_ids[offset : offset + b] = list(ids)

            x512 = x512.to(device, non_blocking=True)
            xvit = xvit.to(device, non_blocking=True)

            p1 = (
                model1(x512).detach().cpu().numpy().astype(np.float32, copy=False)
            )  # [B, 1000]
            p2 = (
                resnet18_embed(model2, x512)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )  # [B, 512]
            p3 = (
                vit_b16_embed(model3, xvit)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )  # [B, 768]

            feats[offset : offset + b, :] = np.concatenate([p1, p2, p3], axis=1)
            offset += b

    return image_ids, feats




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)

train_ids, train_feats = extract_features(
    train_df[["image_id", "label"]], TRAIN_IMG_DIR, batch_size=32, train_mode=True
)
train_labels = train_df["label"].to_numpy()

if len(train_feats) != len(train_labels):
    raise RuntimeError(
        f"Feature/label length mismatch: feats={len(train_feats)} labels={len(train_labels)}"
    )


def _acc(y_true, y_pred):
    return float((y_true == y_pred).mean())


depth_grid = [5, 7, 9, 11, 13, 15, 17]
min_leaf_grid = [1, 2, 4, 8]
min_split_grid = [4, 9, 16]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

folds = list(skf.split(train_feats, train_labels))

fold_cache = []
for tr_idx, va_idx in folds:
    X_tr = train_feats[tr_idx]
    y_tr = train_labels[tr_idx]
    X_va = train_feats[va_idx]
    y_va = train_labels[va_idx]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32, copy=False)
    X_va_s = scaler.transform(X_va).astype(np.float32, copy=False)

    counts_tr = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64)
    class_w_tr = (len(y_tr) / (NUM_CLASSES * np.maximum(counts_tr, 1.0))).astype(
        np.float64
    )
    sample_weight_tr = class_w_tr[y_tr].astype(np.float64, copy=False)

    fold_cache.append((X_tr_s, y_tr, sample_weight_tr, X_va_s, y_va))

best_params = None
best_cv_acc = -1.0
eps = 1e-6

for depth in depth_grid:
    for min_leaf in min_leaf_grid:
        for min_split in min_split_grid:
            if min_split < 2 * min_leaf:
                continue

            fold_accs = []
            for X_tr_s, y_tr, sample_weight_tr, X_va_s, y_va in fold_cache:
                dt = DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=depth,
                    min_samples_split=min_split,
                    min_samples_leaf=min_leaf,
                    random_state=SEED,
                    class_weight=None,
                )
                dt.fit(X_tr_s, y_tr, sample_weight=sample_weight_tr)
                pred_va = dt.predict(X_va_s).astype(int, copy=False)
                fold_accs.append(_acc(y_va, pred_va))

            cv_acc = float(np.mean(fold_accs))

            if best_params is None:
                take = True
            else:
                prev_depth, prev_leaf, prev_split = best_params
                take = (cv_acc > best_cv_acc + eps) or (
                    abs(cv_acc - best_cv_acc) <= eps
                    and (
                        depth < prev_depth
                        or (depth == prev_depth and min_leaf > prev_leaf)
                        or (
                            depth == prev_depth
                            and min_leaf == prev_leaf
                            and min_split > prev_split
                        )
                    )
                )

            if take:
                best_cv_acc = cv_acc
                best_params = (depth, min_leaf, min_split)

best_depth, best_min_leaf, best_min_split = best_params
print(
    f"CV selection: max_depth={best_depth}, min_samples_leaf={best_min_leaf}, "
    f"min_samples_split={best_min_split} with cv_acc={best_cv_acc:.4f}"
)

scaler = StandardScaler(with_mean=True, with_std=True)
train_feats_scaled = scaler.fit_transform(train_feats).astype(np.float32, copy=False)

counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
class_w = (len(train_labels) / (NUM_CLASSES * np.maximum(counts, 1.0))).astype(
    np.float64
)
sample_weight = class_w[train_labels].astype(np.float64, copy=False)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_depth,
    min_samples_split=best_min_split,
    min_samples_leaf=best_min_leaf,
    random_state=SEED,
    class_weight=None,
)
decision_tree.fit(train_feats_scaled, train_labels, sample_weight=sample_weight)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1671639808.py in <cell line: 0>()
      1 train_df = pd.read_csv(TRAIN_CSV)
      2 
----> 3 train_ids, train_feats = extract_features(
      4     train_df[["image_id", "label"]], TRAIN_IMG_DIR, batch_size=32, train_mode=True
      5 )

/tmp/ipykernel_55/1725145677.py in extract_features(df, img_dir, batch_size, train_mode)
     81             )  # [B, 768]
     82 
---> 83             feats[offset : offset + b, :] = np.concatenate([p1, p2, p3], axis=1)
     84             offset += b
     85 

ValueError: could not broadcast input array from shape (32,1792) into shape (32,2280)

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["image_id"]].copy()
test_df["label"] = -1  # placeholder for dataset API

test_ids, test_feats = extract_features(
    test_df[["image_id", "label"]], TEST_IMG_DIR, batch_size=32, train_mode=False
)

id_to_row = {img_id: i for i, img_id in enumerate(test_ids)}
order_idx = np.fromiter(
    (id_to_row.get(i, -1) for i in sample_sub["image_id"].values), dtype=np.int64
)
if np.any(order_idx < 0):
    missing = sample_sub["image_id"].values[order_idx < 0].tolist()
    raise FileNotFoundError(
        f"Missing {len(missing)} test images on disk. First few: {missing[:5]}"
    )

test_feats_ordered = test_feats[order_idx]
test_feats_scaled = scaler.transform(test_feats_ordered).astype(np.float32, copy=False)

prediction = decision_tree.predict(test_feats_scaled).astype(int, copy=False)

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": prediction}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(
    f"Wrote submission.csv with shape={submission.shape} to {os.path.abspath('submission.csv')}"
)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3418977474.py in <cell line: 0>()
      3 test_df["label"] = -1  # placeholder for dataset API
      4 
----> 5 test_ids, test_feats = extract_features(
      6     test_df[["image_id", "label"]], TEST_IMG_DIR, batch_size=32, train_mode=False
      7 )

/tmp/ipykernel_55/1725145677.py in extract_features(df, img_dir, batch_size, train_mode)
     81             )  # [B, 768]
     82 
---> 83             feats[offset : offset + b, :] = np.concatenate([p1, p2, p3], axis=1)
     84             offset += b
     85 

ValueError: could not broadcast input array from shape (32,1792) into shape (32,2280)
