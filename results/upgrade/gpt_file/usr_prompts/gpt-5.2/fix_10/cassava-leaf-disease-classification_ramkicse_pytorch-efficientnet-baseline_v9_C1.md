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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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

0.8434572378362043

# 6. Current score

0.22347

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.22347) has done: 'I first fix the missing weights path by making weight-loading robust: it look for the file if present, otherwise it fall back to a torchvision ImageNet-pretrained EfficientNet-B7 (same architecture) so inference can still run end-to-end and yield a valid submission. Next, I fix the device/type mismatch that caused CUDA inputs to be fed into a CPU model by ensuring the model is moved to the same device (and that loaded state_dict tensors are mapped correctly). Finally, I keep the rest of your inference pipeline intact and guarantee that `submission.csv` is always written with the required `image_id,label` columns and correct row count.'
- What this solution (achieved 0.22347) has done: 'Your low score is consistent with the fallback to ImageNet-pretrained weights (not cassava-trained), so the smallest high-impact improvement is to correctly locate and load an actual cassava fine-tuned checkpoint if it exists anywhere under `/kaggle/input` (without changing the model or inference logic). I add a lightweight checkpoint auto-discovery routine that searches for `.pth/.pt` files matching common patterns (including your original filename) and loads the best candidate, otherwise it keeps your current ImageNet fallback. I also make the weight loading a bit more robust to common key prefixes (`model.`, `net.`) so that a found checkpoint actually applies to the model. Everything else (EfficientNet-B7, transforms, argmax prediction, submission format) remains unchanged.'
- What this solution (achieved 0.22347) has done: 'Your current score is low because the pipeline is (likely) still falling back to ImageNet-pretrained weights or is loading a checkpoint that doesn’t actually match the EfficientNet-B7 classifier head, leaving it effectively random for cassava labels. I make the checkpoint discovery/load more robust and *only* accept checkpoints that contain actual EfficientNet-B7 tensor keys and a compatible classifier head (5 classes), otherwise keep searching instead of silently using a bad file. I also tighten normalization to the ImageNet mean/std used by torchvision EfficientNet weights (this preserves the same inference logic, but aligns preprocessing to the backbone’s expected distribution and typically improves accuracy with minimal risk). Finally, I keep everything else (model, argmax, dataloader, submission schema) unchanged and still guarantee `submission.csv` is written.'
- What this solution (achieved 0.22347) has done: 'Your current score (0.22347) is far below the target (0.84346), and the most likely cause is that you’re still effectively running with ImageNet weights (or loading a non-cassava checkpoint with a mismatched head), which produces near-random cassava labels. I keep your EfficientNet-B7 + argmax inference pipeline unchanged, but make checkpoint discovery/load stricter and more compatible by (1) searching more likely cassava locations first, (2) correctly handling common checkpoint formats (including checkpoints that store `classifier.1.{weight,bias}` under different names), and (3) rebuilding the model head to 5 classes before loading, while refusing partial/mismatched cassava heads that would degrade predictions. I also align preprocessing with the weights actually used: if a torchvision EfficientNet-B7 weight is used, we use its recommended transforms mean/std; otherwise we keep your ImageNet mean/std (same values) but ensure interpolation/resize match. This is a minimal change focused on getting the intended fine-tuned cassava weights actually loaded, which should move accuracy substantially toward the target.'
- What this solution (achieved 0.22347) has done: 'Your current score is far below target, so the most likely issue is that the EfficientNet-B7 classifier head weights are not being loaded (because `strict=False` can silently leave the head randomly initialized, which yields near-random labels). I keep your exact model (EfficientNet-B7) and inference flow, but make weight loading *strict where it matters*: require backbone + classifier to load cleanly when a cassava checkpoint is found, and only fall back to ImageNet weights if the checkpoint can’t correctly load a 5-class head. I also improve checkpoint compatibility by supporting common cassava checkpoint head key patterns (e.g., `fc.*` or `classifier.*`) by remapping them to `classifier.1.*` without changing architecture. This is a minimal change aimed specifically at ensuring the intended fine-tuned weights are actually used, which should move accuracy substantially toward the target.'
- What this solution (achieved 0.22347) has done: 'Your current score is far below the target, and the most likely reason is that inference is still running with ImageNet-pretrained weights because no valid cassava-finetuned checkpoint is actually being found/loaded. I keep your EfficientNet-B7 + argmax inference pipeline unchanged, but make checkpoint discovery more robust by also searching `/kaggle/working` (where you may have saved a trained model) and by prioritizing checkpoints that explicitly contain a 5-class head. I also tighten the loading acceptance criterion to ensure the classifier head truly loads (instead of silently remaining randomly initialized), otherwise it continues searching rather than immediately falling back. These changes are minimal and directly aimed at moving accuracy up toward your target by ensuring the intended finetuned weights are used.'
- What this solution (achieved 0.22347) has done: 'Your score is far below target, so the most likely issue is that you’re still falling back to ImageNet EfficientNet-B7 weights (or loading a non-cassava/mismatched checkpoint), which yields near-random cassava labels. I keep your EfficientNet-B7 + argmax inference pipeline intact, but (1) broaden and prioritize checkpoint discovery to include common Kaggle cassava notebook checkpoints and `.pth` files anywhere under `/kaggle/input`, and (2) tighten acceptance so we only stop when we find a checkpoint that clearly contains a 5-class head (or can be cleanly remapped to it) rather than silently accepting partial loads. I also make the head-key remapping cover more common naming patterns (e.g., `head.*`, `classifier.fc.*`) so genuine cassava checkpoints actually load into `classifier.1.*`. These are minimal, directly score-relevant changes aimed at getting the intended fine-tuned weights loaded, which should move accuracy substantially toward your target.'
- What this solution (achieved 0.22347) has done: 'Your current score is far below the target, so the smallest change likely to move accuracy up is to ensure we actually load a real cassava-finetuned EfficientNet checkpoint (not silently fall back to ImageNet). I keep your EfficientNet-B7 model and argmax inference unchanged, but make checkpoint loading stricter for the classifier head: we only accept a checkpoint if it loads the 5-class head weights successfully, otherwise we keep searching rather than accepting a partial load. I also expand head-key remapping to cover more common cassava checkpoint naming patterns (e.g., `classifier.0/1`, `head.classifier`, `fc`, `last_linear`) so genuine checkpoints can be applied without changing architecture. Finally, I keep submission writing identical and still guaranteed.'

# 9. Code solution

## === cell 0
import os

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification/"
WEIGHT_FILE = "/kaggle/input/ramki-cassava-weights/weight-at-epoch-61-acc-0.85794.pth"

assert os.path.exists(BASE_PATH), f"BASE_PATH not found: {BASE_PATH}"

TRAINING = False

import torch


def _iter_checkpoint_candidates(search_roots: list[str]):
    preferred_substrings = [
        "cassava",
        "leaf",
        "disease",
        "classification",
        "cbsd",
        "cgm",
        "cbb",
        "cmd",
        "efficientnet",
        "effnet",
        "b7",
        "tf_efficientnet",
        "ramki",
        "weight-at-epoch-61",
        "epoch",
        "acc",
        "best",
        "model",
        "checkpoint",
        "ckpt",
        "fold",
        "final",
    ]
    for search_root in search_roots:
        if not search_root or not os.path.isdir(search_root):
            continue
        for root, _, files in os.walk(search_root):
            for fn in files:
                lfn = fn.lower()
                if not (
                    lfn.endswith(".pth") or lfn.endswith(".pt") or lfn.endswith(".ckpt")
                ):
                    continue
                full = os.path.join(root, fn)
                low_full = full.lower()

                score = 0
                for s in preferred_substrings:
                    if s in low_full:
                        score += 1

                try:
                    sz = os.path.getsize(full)
                except Exception:
                    sz = 0
                if sz >= 5_000_000:
                    score += 2
                if sz >= 50_000_000:
                    score += 1

                depth_penalty = full.count(os.sep)
                yield (score, -depth_penalty, full)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    return None


def _clean_state_dict_keys(state: dict) -> dict:
    new_state = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        new_state[nk] = v
    return new_state


def _looks_like_efficientnet_state_dict(state: dict) -> bool:
    if not isinstance(state, dict) or len(state) == 0:
        return False
    keys = list(state.keys())
    has_features = any(k.startswith("features.") for k in keys) or any(
        "features" in k for k in keys
    )
    has_classifier = (
        any(k.startswith("classifier.") for k in keys)
        or any("classifier" in k for k in keys)
        or any(k.startswith("head.") for k in keys)
        or any("last_linear" in k for k in keys)
    )
    return has_features and has_classifier


def _classifier_out_features_if_present(state: dict):
    for w_key in (
        "classifier.1.weight",
        "classifier.weight",
        "fc.weight",
        "head.weight",
        "head.fc.weight",
        "classifier.fc.weight",
        "last_linear.weight",
    ):
        if (
            w_key in state
            and hasattr(state[w_key], "shape")
            and len(state[w_key].shape) == 2
        ):
            return int(state[w_key].shape[0])
    return None


def _try_load_state_dict(path: str) -> dict | None:
    try:
        obj = torch.load(path, map_location="cpu")
        state = _extract_state_dict(obj)
        if not isinstance(state, dict):
            return None
        state = _clean_state_dict_keys(state)
        return state
    except Exception:
        return None


def _find_best_checkpoint(preferred_path: str, num_classes: int = 5) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        state = _try_load_state_dict(preferred_path)
        if state is not None and _looks_like_efficientnet_state_dict(state):
            outc = _classifier_out_features_if_present(state)
            if (outc is None) or (outc == num_classes):
                return preferred_path

    search_roots = [
        "/kaggle/input/ramki-cassava-weights",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/working",
        "/kaggle/input",
    ]

    candidates = list(_iter_checkpoint_candidates(search_roots))
    if not candidates:
        return None

    candidates.sort(reverse=True)

    for _, __, path in candidates[:1200]:
        state = _try_load_state_dict(path)
        if state is None or (not _looks_like_efficientnet_state_dict(state)):
            continue
        outc = _classifier_out_features_if_present(state)
        if outc == num_classes:
            return path

    for _, __, path in candidates[:1200]:
        state = _try_load_state_dict(path)
        if state is None or (not _looks_like_efficientnet_state_dict(state)):
            continue
        outc = _classifier_out_features_if_present(state)
        if outc is None:
            return path

    return None


FOUND_WEIGHT_FILE = _find_best_checkpoint(WEIGHT_FILE, num_classes=5)
WEIGHTS_AVAILABLE = FOUND_WEIGHT_FILE is not None

if not WEIGHTS_AVAILABLE:
    print(
        "Warning: no compatible cassava EfficientNet checkpoint found under /kaggle/input or /kaggle/working. "
        "Falling back to torchvision ImageNet-pretrained EfficientNet-B7 weights for inference."
    )
else:
    if FOUND_WEIGHT_FILE != WEIGHT_FILE:
        print("Info: discovered compatible checkpoint:", FOUND_WEIGHT_FILE)
    else:
        print("Info: using preferred compatible checkpoint:", FOUND_WEIGHT_FILE)



## === cell 1
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import models



## === cell 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 3
SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001
NUM_CLASSES = 5



## === cell 4
import random


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)



## === cell 5
train_path = os.path.join(BASE_PATH, "train_images/")
test_path = os.path.join(BASE_PATH, "test_images/")

train_csv = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
sample = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

assert {"image_id", "label"}.issubset(train_csv.columns)
assert {"image_id", "label"}.issubset(sample.columns)
assert os.path.isdir(train_path), f"train_path not found: {train_path}"
assert os.path.isdir(test_path), f"test_path not found: {test_path}"

train_csv.head()



## === cell 6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transforms_train = A.Compose(
    [
        A.RandomResizedCrop(size=(300, 300), p=1.0),
        A.Rotate(limit=20, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

transforms_valid = A.Compose(
    [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
        ToTensorV2(p=1.0),
    ]
)




## === cell 7
class CasavaDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.test:
            label = 0
        else:
            label = int(self.df.iloc[idx].label)

        p = self.df.iloc[idx].image_id
        p_path = os.path.join(test_path if self.test else train_path, p)

        image = Image.open(p_path).convert("RGB")
        image = np.array(image)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image, label




## === cell 8
def build_efficientnet_b7(
    num_classes: int = 5, use_imagenet_weights: bool = False
) -> nn.Module:
    if use_imagenet_weights:
        weights = models.EfficientNet_B7_Weights.DEFAULT
        model = models.efficientnet_b7(weights=weights)
    else:
        model = models.efficientnet_b7(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


model_name = "efficientnet-b7"
model = build_efficientnet_b7(
    num_classes=NUM_CLASSES, use_imagenet_weights=not WEIGHTS_AVAILABLE
)


def _remap_head_keys_to_torchvision_efficientnet(state: dict) -> dict:
    remapped = dict(state)

    candidates = [
        ("fc.weight", "classifier.1.weight"),
        ("fc.bias", "classifier.1.bias"),
        ("classifier.weight", "classifier.1.weight"),
        ("classifier.bias", "classifier.1.bias"),
        ("head.weight", "classifier.1.weight"),
        ("head.bias", "classifier.1.bias"),
        ("head.fc.weight", "classifier.1.weight"),
        ("head.fc.bias", "classifier.1.bias"),
        ("classifier.fc.weight", "classifier.1.weight"),
        ("classifier.fc.bias", "classifier.1.bias"),
        ("last_linear.weight", "classifier.1.weight"),
        ("last_linear.bias", "classifier.1.bias"),
        ("head.classifier.weight", "classifier.1.weight"),
        ("head.classifier.bias", "classifier.1.bias"),
        ("head.fc.weight", "classifier.1.weight"),
        ("head.fc.bias", "classifier.1.bias"),
        ("classifier.0.weight", "classifier.1.weight"),
        ("classifier.0.bias", "classifier.1.bias"),
    ]

    for src, dst in candidates:
        if src in remapped and dst not in remapped:
            remapped[dst] = remapped[src]

    return remapped


def _head_loaded(missing_keys) -> bool:
    ms = set(missing_keys)
    return ("classifier.1.weight" not in ms) and ("classifier.1.bias" not in ms)


if WEIGHTS_AVAILABLE:
    obj = torch.load(FOUND_WEIGHT_FILE, map_location="cpu")
    state = _extract_state_dict(obj)
    if not isinstance(state, dict):
        raise RuntimeError(
            f"Checkpoint at {FOUND_WEIGHT_FILE} did not contain a valid state_dict-like mapping."
        )
    state = _clean_state_dict_keys(state)
    state = _remap_head_keys_to_torchvision_efficientnet(state)

    outc = _classifier_out_features_if_present(state)
    if outc is not None and outc != NUM_CLASSES:
        print(
            f"Warning: found checkpoint but classifier out_features={outc} != {NUM_CLASSES}. Falling back to ImageNet weights."
        )
        WEIGHTS_AVAILABLE = False
        model = build_efficientnet_b7(
            num_classes=NUM_CLASSES, use_imagenet_weights=True
        )
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)

        if not _head_loaded(missing):
            print(
                "Warning: checkpoint did not load 5-class classifier head (classifier.1.* missing). "
                "Will ignore this checkpoint and fall back to ImageNet weights to avoid near-random predictions."
            )
            WEIGHTS_AVAILABLE = False
            model = build_efficientnet_b7(
                num_classes=NUM_CLASSES, use_imagenet_weights=True
            )
        else:
            if len(unexpected) > 0:
                print(
                    "Warning: unexpected keys while loading weights:", unexpected[:10]
                )
            if len(missing) > 0:
                print("Warning: missing keys while loading weights:", missing[:10])

model = model.to(device)
model.eval()

if WEIGHTS_AVAILABLE:
    print("model loaded:", model_name, "from", FOUND_WEIGHT_FILE)
else:
    print(
        "model loaded:",
        model_name,
        "with torchvision ImageNet pretrained backbone (fallback)",
    )



## === cell 9
testset = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
)

len(test_loader), len(sample)



## === cell 10
test_pred = []

model.eval()
with torch.no_grad():
    for images, _ in tqdm(test_loader, total=len(test_loader), position=0, leave=True):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(1).detach().cpu().numpy().astype(int)
        test_pred.extend(pred.tolist())

assert len(test_pred) == len(
    sample
), "Prediction count mismatch with sample_submission rows."

submission = sample.copy()
submission["label"] = test_pred
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 11
import os

assert os.path.exists("submission.csv")
print("Wrote:", os.path.abspath("submission.csv"))
print("Rows:", sum(1 for _ in open("submission.csv")) - 1)
print(pd.read_csv("submission.csv").head())
