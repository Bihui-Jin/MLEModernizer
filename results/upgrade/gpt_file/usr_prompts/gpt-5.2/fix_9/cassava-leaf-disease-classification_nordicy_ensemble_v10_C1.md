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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

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

0.8774554245995769

# 6. Current score

0.53737

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'Your script didn’t yield a Kaggle score mainly because it’s likely failing before creating a valid `submission.csv` due to missing input model files (and also has a path typo: `casava-aug` vs `cassava`). I make the smallest changes to (1) reliably locate the dataset directory, (2) load model weights only if they exist (otherwise fall back to the base randomly-initialized models so a submission is always produced), and (3) guarantee the submission row order exactly matches `sample_submission.csv` (avoids silent misalignment). These are execution- and correctness-critical changes and should move you from “no score” to a valid score; if the weights are present, the score should also improve versus random outputs. I not change your model architectures, transforms, or ensembling logic beyond these safeguards.'
- What this solution (achieved 0.05531) has done: 'Your score is extremely low because all models are being run with `weights=None` and your checkpoints are almost certainly not being found/loaded (so predictions are effectively random). The smallest change that should move accuracy sharply upward toward your target is to (1) robustly discover and load the `.pth` checkpoints anywhere under `/kaggle/input` (instead of hardcoded, likely-wrong dataset names), and (2) ensure inference preprocessing matches what these torchvision backbones expect (simple ImageNet normalize + resize, without CLAHE that can shift the distribution if the model wasn’t trained with it). I keep your ensemble logic, architectures (ResNet50 + EfficientNetV2-S), and inference loop intact—only making loading and preprocessing corrections. The script still always produces a valid `submission.csv` in the correct row order.'
- What this solution (achieved 0.11024) has done: 'Your current score is far below target because the script still isn’t actually using any trained weights (so predictions are effectively random), and it also drops the ResNet branch entirely despite defining ensemble weights. I make the smallest changes to (1) robustly find *any* plausible cassava checkpoints under `/kaggle/input` and load them (including common key patterns like `model_state_dict`), (2) ensure preprocessing matches the model family (EfficientNet 384, ResNet 224) without changing model architectures, and (3) restore the intended weighted ensemble by combining ResNet and EfficientNet probabilities. These changes keep your inference-only approach and architectures intact but should move accuracy sharply upward toward the target if any real checkpoints exist in the environment.'
- What this solution (achieved 0.11024) has done: 'Your score is near-random because the script is almost certainly loading the wrong (or no) checkpoints: the checkpoint finder doesn’t ensure the file actually matches the model’s final layer shape (5 classes), and it can also reuse the same checkpoint for multiple EfficientNet “folds” due to broad keyword matching. I make minimal, inference-only changes to (1) validate checkpoints by checking that they contain a 5-class classifier head compatible with each architecture, (2) ensure the three EfficientNet models load three distinct checkpoints when available, and (3) print what was actually loaded so you can confirm you’re not running random weights. This keeps your exact model architectures, transforms, and ensemble logic intact, but should move accuracy sharply upward toward the target if any real cassava-trained checkpoints exist in `/kaggle/input`. The script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.10762) has done: 'Your score is still near-random because the inference code almost certainly isn’t finding any real cassava-trained checkpoints under `/kaggle/input`, so it runs with randomly initialized weights. I make the smallest changes to (1) broaden and prioritize checkpoint discovery specifically for this competition (including common directory names like `working/` and `output/`, plus “fold”/“checkpoint” naming), (2) add a strict compatibility check that the checkpoint’s *head tensors* match 5 classes **and** actually get loaded (so we don’t silently accept junk), and (3) prevent reusing the same checkpoint across all three EfficientNet models by picking distinct compatible files deterministically. This preserves your architectures, transforms, inference loops, and ensemble math, but should move accuracy sharply upward toward your target if any valid checkpoints exist in the environment, while still always producing `submission.csv`.'
- What this solution (achieved 0.64611) has done: 'The timeout is driven primarily by (1) training four large CNNs from scratch for a full epoch each and (2) very slow image loading/augmentation with `num_workers=0`, plus extra overhead from repeatedly scanning `/kaggle/input` for checkpoints and from dict-based reordering of test predictions. The optimizations below keep the exact same models, losses, epochs, and inference logic, but reduce constant factors: enable fast multi-worker DataLoaders with deterministic seeding, use pinned memory + non-blocking transfers, enable cuDNN benchmarking (fixed input sizes), avoid repeated expensive checkpoint directory walks, and remove unnecessary dict/stack reordering by collecting outputs in-order. These changes are provably equivalent in evaluation semantics (same data, same forward passes, same averaging/weights) and target only runtime bottlenecks.'
- What this solution (achieved 0.53737) has done: 'Your current gap to the target is large (0.64611 vs 0.87746), so we should improve accuracy while keeping your training+ensemble core logic intact. The biggest accuracy limiter here is that EfficientNetV2-S is being trained/inferred with a 384 resize that doesn’t match the default architecture’s expected 384? (it commonly performs best at 384, but torchvision’s v2_s is typically trained at 384 while your head/dropout is very strong), while ResNet50 is trained from scratch for only 1 epoch on 90% of data—so it underfits and the ensemble is dominated by weak models. With minimal semantic changes, we (1) fix reproducibility + class imbalance by adding weighted CrossEntropy (same loss family) and (2) ensure the LR schedule is stable by adding a cosine anneal over the single epoch (no early stopping, same epochs), which typically boosts 1-epoch-from-scratch performance without changing architecture. We also (3) enable AMP only for speed (keeps math close; outputs may differ slightly but evaluation semantics are unchanged) so we can afford a slightly larger batch for EfficientNet training within time, improving optimization per epoch.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # keep performance; semantics unchanged



## === cell 2
num_tta = 5  # kept (not used); preserve original semantics



## === cell 3
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset root in candidates: {DATA_ROOT_CANDIDATES}"
    )

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_image_dir = os.path.join(DATA_ROOT, "train_images")
test_image_dir = os.path.join(DATA_ROOT, "test_images")

print("DATA_ROOT:", DATA_ROOT)
print("train_csv_path exists:", os.path.exists(train_csv_path))
print("train_image_dir exists:", os.path.exists(train_image_dir))
print("test_image_dir exists:", os.path.exists(test_image_dir))



## === cell 4
test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 5
efficientnet_transforms = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

resnet_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

efficientnet_train_transforms = A.Compose(
    [
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

resnet_train_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name


class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self._image_ids = self.df["image_id"].tolist()
        self._labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self._image_ids[idx]
        y = int(self._labels[idx])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, torch.tensor(y, dtype=torch.long)




## === cell 7
def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

_DEFAULT_WORKERS = 4

test_dataset_eff = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)
test_loader_eff = DataLoader(
    test_dataset_eff,
    batch_size=32,
    shuffle=False,
    num_workers=_DEFAULT_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_DEFAULT_WORKERS > 0),
    prefetch_factor=2 if _DEFAULT_WORKERS > 0 else None,
    worker_init_fn=seed_worker if _DEFAULT_WORKERS > 0 else None,
    generator=g,
)

test_dataset_res = CassavaTestDataset(
    test_df, test_image_dir, transform=resnet_transforms
)
test_loader_res = DataLoader(
    test_dataset_res,
    batch_size=32,
    shuffle=False,
    num_workers=_DEFAULT_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_DEFAULT_WORKERS > 0),
    prefetch_factor=2 if _DEFAULT_WORKERS > 0 else None,
    worker_init_fn=seed_worker if _DEFAULT_WORKERS > 0 else None,
    generator=g,
)



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 9
def find_checkpoints_any(
    keywords_any=None, keywords_all=None, search_root="/kaggle/input"
):
    if keywords_any is None:
        keywords_any = []
    if keywords_all is None:
        keywords_all = []
    keywords_any = [k.lower() for k in keywords_any]
    keywords_all = [k.lower() for k in keywords_all]

    matches = []
    for root, _, files in os.walk(search_root):
        for fn in files:
            if not fn.lower().endswith((".pth", ".pt", ".bin")):
                continue
            fn_l = fn.lower()
            if keywords_all and not all(k in fn_l for k in keywords_all):
                continue
            if keywords_any and not any(k in fn_l for k in keywords_any):
                continue
            matches.append(os.path.join(root, fn))

    def _score(p):
        pl = p.lower()
        bonus = 0
        if "best" in pl:
            bonus -= 20
        if "final" in pl:
            bonus -= 10
        if "checkpoint" in pl or "ckpt" in pl:
            bonus -= 5
        if "fold" in pl:
            bonus -= 2
        if "epoch" in pl:
            bonus -= 1
        depth = p.count(os.sep)
        return (bonus, depth, len(p), p)

    matches = sorted(matches, key=_score)
    return matches


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _clean_state_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("net."):
            k = k[len("net.") :]
        new_state[k] = v
    return new_state


def ckpt_looks_compatible(ckpt_path, arch, device, num_classes=5):
    try:
        state = torch.load(ckpt_path, map_location=device)
        state = _extract_state_dict(state)
        if not isinstance(state, dict):
            return False
        state = _clean_state_keys(state)

        if arch == "resnet50":
            w = state.get("fc.weight", None)
            b = state.get("fc.bias", None)
            if w is None or b is None:
                return False
            if hasattr(w, "shape") and int(w.shape[0]) != num_classes:
                return False
            if hasattr(b, "shape") and int(b.shape[0]) != num_classes:
                return False
            return True

        if arch == "efficientnet_v2_s":
            w = state.get("classifier.1.weight", None)
            b = state.get("classifier.1.bias", None)
            if w is None or b is None:
                return False
            if hasattr(w, "shape") and int(w.shape[0]) != num_classes:
                return False
            if hasattr(b, "shape") and int(b.shape[0]) != num_classes:
                return False
            return True

        return False
    except Exception:
        return False


def safe_load_state_dict(model, ckpt_path, device):
    if ckpt_path is None:
        print("No checkpoint path provided; using randomly initialized weights.")
        return False
    if not os.path.exists(ckpt_path):
        print(
            f"Checkpoint not found: {ckpt_path} -> using randomly initialized weights."
        )
        return False

    state = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(state)

    if not isinstance(state, dict):
        print(f"Checkpoint format not understood for: {ckpt_path} -> skipping.")
        return False

    state = _clean_state_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected keys (first 10): {unexpected[:10]}")
    head_missing = any(
        k in missing
        for k in [
            "fc.weight",
            "fc.bias",
            "classifier.1.weight",
            "classifier.1.bias",
        ]
    )
    if head_missing:
        print(
            "  WARNING: classifier head weights were missing -> treating as NOT loaded."
        )
        return False
    return True


def pick_first_compatible(candidates, arch, device, exclude=None):
    exclude = set(exclude or [])
    for p in candidates:
        if p in exclude:
            continue
        if ckpt_looks_compatible(p, arch=arch, device=device, num_classes=5):
            return p
    return None


CHECKPOINT_SEARCH_ROOTS = [
    "/kaggle/input",
    "/kaggle/working",
    "/kaggle/data",
]

_FIND_CACHE = {}


def find_ckpts_multi_root(keywords_any=None, keywords_all=None):
    key = (tuple(sorted((keywords_any or []))), tuple(sorted((keywords_all or []))))
    out = []
    for r in CHECKPOINT_SEARCH_ROOTS:
        if not os.path.exists(r):
            continue
        cache_key = (r, key)
        if cache_key in _FIND_CACHE:
            out.extend(_FIND_CACHE[cache_key])
        else:
            res = find_checkpoints_any(
                keywords_any=keywords_any, keywords_all=keywords_all, search_root=r
            )
            _FIND_CACHE[cache_key] = res
            out.extend(res)

    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq




## === cell 10
from sklearn.model_selection import StratifiedShuffleSplit

train_df_full = pd.read_csv(train_csv_path)
labels = train_df_full["label"].astype(int).values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
train_idx, val_idx = next(sss.split(train_df_full, labels))
train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(train_df), "Val size:", len(val_df))
print(
    "Train label dist:",
    train_df["label"].value_counts(normalize=True).round(3).to_dict(),
)
print(
    "Val label dist:", val_df["label"].value_counts(normalize=True).round(3).to_dict()
)

_counts = train_df["label"].value_counts().sort_index()
_counts = _counts.reindex(range(5), fill_value=1)
class_weights = 1.0 / _counts.values.astype(np.float32)
class_weights = class_weights / class_weights.mean()
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
print("Class weights:", {i: float(w) for i, w in enumerate(class_weights)})


def train_one_model(
    model,
    arch_name,
    train_loader,
    val_loader,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path="best.pth",
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_acc = -1.0
    best_state = None

    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            scheduler.step()

            tr_loss += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        tr_loss /= max(1, tr_n)
        val_loss /= max(1, total)
        val_acc = correct / max(1, total)
        print(
            f"[{arch_name}] epoch {ep}/{epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        torch.save(best_state, out_path)
        print(
            f"[{arch_name}] saved best checkpoint to: {out_path}  best_val_acc={best_acc:.4f}"
        )

    return best_acc




## === cell 11
train_ds_eff = CassavaTrainDataset(
    train_df, train_image_dir, transform=efficientnet_train_transforms
)
val_ds_eff = CassavaTrainDataset(
    val_df, train_image_dir, transform=efficientnet_transforms
)

train_dl_eff = DataLoader(
    train_ds_eff,
    batch_size=20,
    shuffle=True,
    num_workers=_DEFAULT_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_DEFAULT_WORKERS > 0),
    prefetch_factor=2 if _DEFAULT_WORKERS > 0 else None,
    worker_init_fn=seed_worker if _DEFAULT_WORKERS > 0 else None,
    generator=g,
)
val_dl_eff = DataLoader(
    val_ds_eff,
    batch_size=32,
    shuffle=False,
    num_workers=_DEFAULT_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_DEFAULT_WORKERS > 0),
    prefetch_factor=2 if _DEFAULT_WORKERS > 0 else None,
    worker_init_fn=seed_worker if _DEFAULT_WORKERS > 0 else None,
    generator=g,
)

train_ds_res = CassavaTrainDataset(
    train_df, train_image_dir, transform=resnet_train_transforms
)
val_ds_res = CassavaTrainDataset(val_df, train_image_dir, transform=resnet_transforms)

train_dl_res = DataLoader(
    train_ds_res,
    batch_size=32,
    shuffle=True,
    num_workers=_DEFAULT_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_DEFAULT_WORKERS > 0),
    prefetch_factor=2 if _DEFAULT_WORKERS > 0 else None,
    worker_init_fn=seed_worker if _DEFAULT_WORKERS > 0 else None,
    generator=g,
)
val_dl_res = DataLoader(
    val_ds_res,
    batch_size=64,
    shuffle=False,
    num_workers=_DEFAULT_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_DEFAULT_WORKERS > 0),
    prefetch_factor=2 if _DEFAULT_WORKERS > 0 else None,
    worker_init_fn=seed_worker if _DEFAULT_WORKERS > 0 else None,
    generator=g,
)



## === cell 12
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

resnet_ckpt_local = "/kaggle/working/resnet50_best.pth"
eff1_ckpt_local = "/kaggle/working/effv2s_1_best.pth"
eff7_ckpt_local = "/kaggle/working/effv2s_7_best.pth"
eff8_ckpt_local = "/kaggle/working/effv2s_8_best.pth"

_ = train_one_model(
    resnet_model,
    "resnet50",
    train_dl_res,
    val_dl_res,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path=resnet_ckpt_local,
)

_ = train_one_model(
    efficientnet_model_1,
    "effv2s_1",
    train_dl_eff,
    val_dl_eff,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path=eff1_ckpt_local,
)

torch.manual_seed(SEED + 7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED + 7)
_ = train_one_model(
    efficientnet_model_7,
    "effv2s_7",
    train_dl_eff,
    val_dl_eff,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path=eff7_ckpt_local,
)

torch.manual_seed(SEED + 8)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED + 8)
_ = train_one_model(
    efficientnet_model_8,
    "effv2s_8",
    train_dl_eff,
    val_dl_eff,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path=eff8_ckpt_local,
)



## === cell 13
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_candidates = [resnet_ckpt_local]
resnet_candidates += find_ckpts_multi_root(
    keywords_any=["resnet", "resnet50"], keywords_all=["cassava"]
)
resnet_candidates += find_ckpts_multi_root(
    keywords_any=["resnet", "resnet50", "cassava"], keywords_all=[]
)
resnet_candidates += find_ckpts_multi_root(
    keywords_any=["resnet", "resnet50"], keywords_all=[]
)

resnet_ckpt = pick_first_compatible(resnet_candidates, arch="resnet50", device=device)
if resnet_ckpt is None:
    print("No compatible ResNet50 (5-class) checkpoint found; ResNet will be random.")
loaded_resnet = safe_load_state_dict(resnet_model, resnet_ckpt, device)
resnet_model = resnet_model.to(device)
resnet_model.eval()

efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff1_candidates = [eff1_ckpt_local]
eff1_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "efficientnetv2", "v2s", "eff"],
    keywords_all=["cassava"],
)
eff1_candidates += find_ckpts_multi_root(
    keywords_any=["eff", "fold", "best", "cassava"], keywords_all=[]
)
eff1_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "eff"], keywords_all=[]
)

eff1_ckpt = pick_first_compatible(
    eff1_candidates, arch="efficientnet_v2_s", device=device
)
if eff1_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_1; it will be random."
    )
loaded_eff1 = safe_load_state_dict(efficientnet_model_1, eff1_ckpt, device)
efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()

efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff7_candidates = [eff7_ckpt_local]
eff7_candidates += find_ckpts_multi_root(
    keywords_any=["eff", "fold", "best", "checkpoint", "ckpt", "cassava"],
    keywords_all=[],
)
eff7_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "efficientnetv2", "v2s", "eff"],
    keywords_all=["cassava"],
)
eff7_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "eff"], keywords_all=[]
)

eff7_ckpt = pick_first_compatible(
    eff7_candidates, arch="efficientnet_v2_s", device=device, exclude=[eff1_ckpt]
)
if eff7_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_7; it will be random."
    )
loaded_eff7 = safe_load_state_dict(efficientnet_model_7, eff7_ckpt, device)
efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()

efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff8_candidates = [eff8_ckpt_local]
eff8_candidates += find_ckpts_multi_root(
    keywords_any=["eff", "fold", "best", "checkpoint", "ckpt", "cassava"],
    keywords_all=[],
)
eff8_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "efficientnetv2", "v2s", "eff"],
    keywords_all=["cassava"],
)
eff8_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "eff"], keywords_all=[]
)

eff8_ckpt = pick_first_compatible(
    eff8_candidates,
    arch="efficientnet_v2_s",
    device=device,
    exclude=[eff1_ckpt, eff7_ckpt],
)
if eff8_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_8; it will be random."
    )
loaded_eff8 = safe_load_state_dict(efficientnet_model_8, eff8_ckpt, device)
efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()

weight_efficientnet = 0.7
weight_resnet = 0.3

print("Using checkpoints:")
print("  ResNet50:", resnet_ckpt, "loaded:", loaded_resnet)
print("  EfficientNetV2-S #1:", eff1_ckpt, "loaded:", loaded_eff1)
print("  EfficientNetV2-S #7:", eff7_ckpt, "loaded:", loaded_eff7)
print("  EfficientNetV2-S #8:", eff8_ckpt, "loaded:", loaded_eff8)

probs_eff_total = []
use_amp_inf = torch.cuda.is_available()
with torch.no_grad():
    for images, _img_names in test_loader_eff:
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp_inf):
            outputs_efficientnet1 = efficientnet_model_1(images)
            probs_efficientnet1 = F.softmax(outputs_efficientnet1, dim=1)

            outputs_efficientnet7 = efficientnet_model_7(images)
            probs_efficientnet7 = F.softmax(outputs_efficientnet7, dim=1)

            outputs_efficientnet8 = efficientnet_model_8(images)
            probs_efficientnet8 = F.softmax(outputs_efficientnet8, dim=1)

            combined_probs_eff = (
                probs_efficientnet1 + probs_efficientnet7 + probs_efficientnet8
            ) / 3.0

        probs_eff_total.append(combined_probs_eff.detach().cpu())

probs_eff_ordered = torch.cat(probs_eff_total, dim=0).numpy()

probs_res_total = []
with torch.no_grad():
    for images, _img_names in test_loader_res:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp_inf):
            outputs_resnet = resnet_model(images)
            probs_resnet = F.softmax(outputs_resnet, dim=1)
        probs_res_total.append(probs_resnet.detach().cpu())

probs_res_ordered = torch.cat(probs_res_total, dim=0).numpy()

probs_ens = weight_efficientnet * probs_eff_ordered + weight_resnet * probs_res_ordered
ordered_labels = probs_ens.argmax(axis=1).astype(int).tolist()

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": ordered_labels}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())
print("Rows:", len(submission_df), "Expected:", len(test_df))
print("Unique predicted labels:", submission_df["label"].value_counts().to_dict())
