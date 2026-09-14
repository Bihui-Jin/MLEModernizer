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

0.71674

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60015) has done: 'I fix the two blocking FileNotFoundErrors by removing dependencies on external Kaggle datasets (`/kaggle/input/train-tree/...` and `/kaggle/input/resnet50_70_512x512/...`) and instead train the same kind of DecisionTreeClassifier directly from the provided `train.csv` and `train_images`. To preserve the original core intent (tree on top of model probabilities), I generate per-image “probability-like” features using the existing ResNet50-head function (kept) and compute logits/probabilities from it; this keeps the pipeline structure (CNN → probs → decision tree) while making it self-contained. I also ensure `test_image_ids` is always defined from `sample_submission.csv`, and make submission writing robust (correct columns, row count, and `.csv` suffix). These changes are necessary for end-to-end execution and should yield a non-trivial accuracy score compared to the previous uniform-fallback behavior.'
- What this solution (achieved 0.59268) has done: 'Your current score is low because `resnet50(weights=None)` is untrained, so the “probability-like” features going into the decision tree are essentially noise. To move the accuracy up toward your target with minimal logic change, I keep the same pipeline (CNN → 5-dim probs → DecisionTreeClassifier) but switch the ResNet50 backbone to ImageNet pretrained weights so the 5-dim outputs become meaningful features. I also normalize with the standard ImageNet mean/std to match the pretrained model and keep everything else (tree hyperparameters, sampling, submission writing) unchanged for stability and runtime. These changes should significantly reduce the gap toward 0.8445 without changing your overall approach.'
- What this solution (achieved 0.66293) has done: 'Your current gap to the target is large (0.59268 → 0.84451), and the biggest issue is that your “CNN → 5-dim probs → DecisionTree” pipeline is using an untrained 5-class head (random fc layer), so the features going into the tree are mostly noise. To preserve the same core logic while improving accuracy, I keep ResNet50 ImageNet weights but change the extracted features to the pretrained penultimate layer (2048-dim) instead of the random 5-dim logits; the decision tree still consumes “CNN-derived features,” just more meaningful ones. I also add a very small amount of test-time augmentation (horizontal flip averaged with original) inside the same `predict_*` function to improve feature stability without changing the overall approach or adding any new training loops. Finally, I keep submission alignment identical to `sample_submission.csv` and ensure the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.58744) has done: 'Your current score (0.66293) is far below the target (0.84451), so we should improve accuracy with the smallest changes that keep the same core pipeline (ResNet50 feature extractor → DecisionTreeClassifier). The biggest lever here without changing architecture is to train the tree on more (and more representative) data: increase `MAX_TRAIN_FOR_TREE` while keeping runtime bounded by extracting features in efficient batches. I also make the train subset stratified by label (instead of pure random) so the tree sees all classes proportionally, and I switch the tree to `class_weight="balanced"` to reduce bias toward majority classes—still the same model type and training semantics. Finally, I keep submission alignment identical to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.74439) has done: 'Your current score is far below the target, so we should improve accuracy while preserving the same core pipeline (ImageNet-pretrained ResNet50 feature extractor → DecisionTreeClassifier). The largest safe gain here is to switch the tree to a RandomForestClassifier (same “tree family” learner consuming the same extracted features) to reduce overfitting/variance that a single shallow tree has on high-dimensional embeddings. To keep runtime within 600s, we keep feature extraction identical and limit the forest size/depth conservatively while using all CPU cores. We also keep the exact same submission alignment to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.72085) has done: 'Your current score (0.74439) is below the target (0.84451), so we should improve accuracy with minimal changes while keeping the same core pipeline (ImageNet-pretrained ResNet50 embeddings → RandomForestClassifier). The biggest low-risk gain is to (1) apply L2-normalization to the 2048-d ResNet features before training/prediction (helps tree ensembles on high-dimensional embeddings), and (2) slightly strengthen the forest while staying within time (more trees, `max_features="sqrt"`, and a bit more depth) to reduce variance. I also ensure the exact same normalization is applied to both train and test features and keep all I/O paths and submission alignment unchanged. These changes keep the same overall model family and inference semantics but typically improve accuracy on this task.'
- What this solution (achieved 0.70329) has done: 'Your current score (0.72085) is well below the target (0.84451), so we should push accuracy up with the smallest changes that keep the same pipeline (ImageNet-pretrained ResNet50 embeddings → RandomForestClassifier → submission). The main low-risk gain is to make the forest a bit stronger and more stable by increasing trees and enabling out-of-bag scoring (no change to training loops or data usage), while also removing the test-time flip augmentation which can hurt when the learned decision boundaries don’t benefit from flip invariance with a tree ensemble. I’m also switching feature extraction to use the model’s recommended preprocessing for the selected weights to better match ImageNet normalization (same semantics, just correct config), and keeping L2-normalization as-is. These are minimal, runtime-safe changes that typically move accuracy upward on this task without changing your overall approach or I/O.'
- What this solution (achieved 0.713) has done: 'I fix the immediate runtime errors by making the ResNet50 preprocessing robust to missing `weights.meta["mean"/"std"]` in this torchvision build, falling back to standard ImageNet mean/std (score-neutral but unblocks execution). This also restores the definition of `extract_feats_batch_torch` (it wasn’t defined because cell 2 crashed), which fixes the downstream `NameError`s in training and inference. I keep the same pipeline (ImageNet-pretrained ResNet50 → 2048-d features → L2 normalize → RandomForest) and only adjust the preprocessing retrieval to be compatible with the Kaggle environment. The script run end-to-end and write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.70777) has done: 'Your current score (0.713) is well below the target (0.8445), so we should improve accuracy with the smallest changes that keep the same core pipeline (ImageNet-pretrained ResNet50 embeddings → L2 normalize → RandomForest). The main likely issue is that your ResNet50 transform doesn’t follow the weights’ exact recommended resize/crop behavior; switching to `weights.transforms()` keeps the same “pretrained ResNet50 features” logic but makes the embeddings better aligned to ImageNet training, typically yielding a noticeable accuracy lift. I also slightly increase the training set size (within time) because the forest benefits from more labeled embeddings and this doesn’t change the training approach. Finally, I keep submission alignment identical to `sample_submission.csv` and ensure the CSV is written correctly.'
- What this solution (achieved 0.71674) has done: 'I keep your exact pipeline (ImageNet-pretrained ResNet50 embeddings → L2 normalize → RandomForest → submission) and make two small, score-relevant adjustments that typically lift accuracy without changing semantics. First, I switch from training on a stratified subset (16k) to using the full training set (18,721) since feature extraction is already batched and the forest benefits from more labeled embeddings. Second, I add a very light, deterministic test-time augmentation (original + horizontal flip averaged in feature space) which often improves robustness on cassava leaves while keeping the same model and inference approach. Everything else (paths, transforms via `weights.transforms()`, L2 normalization, forest hyperparameters, and submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.72272) has done: 'Your current score (0.71674) is well below the target (0.84451), so we should improve accuracy with the smallest changes that keep the exact same pipeline (ImageNet-pretrained ResNet50 embeddings → L2 normalize → RandomForest → submission). The most likely bottleneck is that a default RandomForest is relatively weak for this kind of embedding classification; switching to `ExtraTreesClassifier` keeps the same “tree ensemble on fixed features” core logic but typically yields a clear accuracy bump with minimal code change and similar runtime. I also extract train features with the same light TTA you already use at test time (original + hflip averaged) to reduce train/test feature mismatch without changing the model family or adding training loops. All paths, transforms, batching, L2 normalization, and the submission format remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.72272) has done: 'We keep your exact pipeline (ImageNet-pretrained ResNet50 embeddings → L2 normalize → ExtraTrees → submission) and make only small, score-relevant adjustments that typically improve accuracy without changing the overall approach. The main change is to align feature extraction with the pretrained weights’ recommended preprocessing even when `weights.transforms()` is unavailable: use the standard ImageNet resize(256)+center-crop(224) fallback instead of resizing to 512x512, which can distort features and hurt downstream tree performance. In addition, we enable simple feature caching to `working/` so you can safely increase model strength without risking timeouts (no semantic change; it just avoids recomputation on reruns). Everything else (TTA hflip, L2 norm, ExtraTrees training, submission alignment/format) remains the same.'
- What this solution (achieved 0.71674) has done: 'Your current score (0.72272) is well below the target (0.84451), so we should nudge accuracy upward with minimal, low-risk changes that keep the same core pipeline (ImageNet-pretrained ResNet50 embeddings → L2 normalize → ExtraTrees → submission). The most impactful small fix is to reduce train/test feature mismatch by using the *same deterministic test-time preprocessing* but adding *light train-time augmentation* (random resized crop + horizontal flip) during feature extraction for the training set only—this preserves the “fixed feature extractor → tree model” approach while improving generalization. To avoid changing the model family/logic, ExtraTrees stays the same; we just average a couple augmented views per training image to get more robust embeddings. We also version the train feature cache filename to prevent accidentally reusing old (non-augmented) cached features.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

torch.manual_seed(0)
np.random.seed(0)

ImageFile.LOAD_TRUNCATED_IMAGES = True



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
weights = models.ResNet50_Weights.IMAGENET1K_V2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    torch_transforms_eval = weights.transforms()
except Exception:
    IMNET_MEAN_FALLBACK = (0.485, 0.456, 0.406)
    IMNET_STD_FALLBACK = (0.229, 0.224, 0.225)
    torch_transforms_eval = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMNET_MEAN_FALLBACK, std=IMNET_STD_FALLBACK),
        ]
    )

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
try:
    if hasattr(torch_transforms_eval, "transforms"):
        for t in torch_transforms_eval.transforms:
            if isinstance(t, transforms.Normalize):
                IMNET_MEAN = tuple(float(x) for x in t.mean)
                IMNET_STD = tuple(float(x) for x in t.std)
                break
except Exception:
    pass

torch_transforms_train_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
    ]
)


def build_resnet50_feature_extractor() -> nn.Module:
    m = models.resnet50(weights=weights)
    m.fc = nn.Identity()
    return m


model2 = build_resnet50_feature_extractor().to(device)
model2.eval()


@torch.inference_mode()
def extract_feats_batch_torch(model: nn.Module, pil_imgs, tfm) -> np.ndarray:
    xs = torch.stack([tfm(im) for im in pil_imgs], dim=0).to(device, non_blocking=True)
    feats = model(xs)
    return feats.detach().cpu().numpy().astype(np.float32)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


@torch.inference_mode()
def extract_feats_batch_torch_tta_hflip_eval(model: nn.Module, pil_imgs) -> np.ndarray:
    xs = torch.stack([torch_transforms_eval(im) for im in pil_imgs], dim=0).to(
        device, non_blocking=True
    )
    feats1 = model(xs)
    feats2 = model(torch.flip(xs, dims=[3]))  # horizontal flip in tensor space
    feats = 0.5 * (feats1 + feats2)
    return feats.detach().cpu().numpy().astype(np.float32)


@torch.inference_mode()
def extract_feats_batch_torch_train_aug_avg(
    model: nn.Module, pil_imgs, n_views: int = 2
) -> np.ndarray:
    acc = None
    for _ in range(n_views):
        feats = extract_feats_batch_torch(model, pil_imgs, torch_transforms_train_aug)
        acc = feats if acc is None else (acc + feats)
    return (acc / float(n_views)).astype(np.float32)




## === cell 3
MAX_TRAIN_FOR_TREE = len(train_df)
BATCH_SIZE = 64

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

os.makedirs("/kaggle/working", exist_ok=True)

TRAIN_CACHE_VERSION = "v2_trainaug2views"
train_cache_path = (
    f"/kaggle/working/train_feats_{use_n}_resnet50v2_{TRAIN_CACHE_VERSION}_l2.npy"
)
train_label_cache_path = f"/kaggle/working/train_labels_{use_n}.npy"

if os.path.exists(train_cache_path) and os.path.exists(train_label_cache_path):
    train_feats = np.load(train_cache_path)
    train_labels = np.load(train_label_cache_path)
    print("Loaded cached train features:", train_feats.shape)
else:
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
                pil_imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                ok_mask[k] = False
                bad_count += 1

        feats = extract_feats_batch_torch_train_aug_avg(
            model2, pil_imgs, n_views=2
        )  # [B, 2048]
        feats[~ok_mask] = 0.0
        feats = l2_normalize_rows(feats)
        train_feats[start:end] = feats

        if end % 256 == 0 or end == use_n:
            print(f"Feature extract train: {end}/{use_n} (bad={bad_count})", end="\r")
    print()

    np.save(train_cache_path, train_feats)
    np.save(train_label_cache_path, train_labels)
    print("Saved cached train features:", train_feats.shape)

forest = ExtraTreesClassifier(
    n_estimators=900,
    max_depth=24,
    min_samples_split=6,
    max_features="sqrt",
    class_weight="balanced",
    random_state=0,
    n_jobs=-1,
    bootstrap=False,
)
forest.fit(train_feats, train_labels)
print("Tree ensemble trained on:", train_feats.shape)



## === cell 4
image_ids = []
prediction = []

length = len(test_image_ids)
test_paths = [os.path.join(TEST_DIR, iid) for iid in test_image_ids]

test_cache_path = f"/kaggle/working/test_feats_{length}_resnet50v2_tta_l2.npy"

if os.path.exists(test_cache_path):
    test_feats_all = np.load(test_cache_path)
    print("Loaded cached test features:", test_feats_all.shape)
else:
    test_feats_all = np.zeros((length, FEAT_DIM), dtype=np.float32)
    for start in range(0, length, BATCH_SIZE):
        end = min(length, start + BATCH_SIZE)
        batch_paths = test_paths[start:end]

        pil_imgs = []
        ok_mask = np.ones((end - start,), dtype=bool)
        for k, p in enumerate(batch_paths):
            try:
                pil_imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                pil_imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                ok_mask[k] = False

        feats = extract_feats_batch_torch_tta_hflip_eval(model2, pil_imgs)
        feats[~ok_mask] = 0.0
        feats = l2_normalize_rows(feats)
        test_feats_all[start:end] = feats

        if end % 200 == 0 or end == length:
            print(f"Feature extract test: {end}/{length}", end="\r")
    print()
    np.save(test_cache_path, test_feats_all)
    print("Saved cached test features:", test_feats_all.shape)

preds = forest.predict(test_feats_all).astype(int).tolist()
image_ids = list(test_image_ids)
prediction = preds

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
