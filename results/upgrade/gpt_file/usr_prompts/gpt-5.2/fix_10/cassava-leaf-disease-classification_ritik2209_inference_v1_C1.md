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

0.7943487458446661

# 6. Current score

0.25598

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing `efficientnet_pytorch` dependency by removing the external wheel install and switching to the built-in `torchvision` EfficientNet-B3 with an equivalent classifier head, while keeping the same inference-only approach. I also fix the dataset bug where `train_path` is referenced but never defined (even if we only run test), make label handling robust for test rows, and ensure the model weights load correctly on CPU/GPU. Finally, I keep the same preprocessing semantics (resize→centercrop→normalize) and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.22272) has done: 'I fix the immediate runtime failure by locating the pretrained weights file dynamically under `/kaggle/input` (since `/kaggle/input/aefficientnet/weights.pt` doesn’t exist) and loading it safely. To move accuracy up toward the target, I also ensure the model uses the same ImageNet preprocessing expected by EfficientNet-B3 and run inference with a larger batch size for speed (score-neutral) while keeping the exact same architecture and inference-only approach. Finally, I keep the submission formatting and row order identical to `sample_submission.csv` so Kaggle accepts it.'
- What this solution (achieved 0.25598) has done: 'Your current 0.22272 score is far below the 0.7943 target (higher-is-better), so we should improve accuracy with the smallest change that preserves your inference-only EfficientNet-B3 core logic. The biggest issue is that your inference preprocessing (Resize→CenterCrop(300)) does not match EfficientNet-B3’s expected ImageNet eval preprocessing (Resize to 320 then CenterCrop to 300 is nonstandard), which can severely hurt accuracy. I switch to torchvision’s official `EfficientNet_B3_Weights` eval transform (center-crop size 300) to align normalization, resize/crop policy, and interpolation with the pretrained weights you’re using, while keeping the exact same model and inference loop. I also make the checkpoint selection slightly safer by preferring filenames that contain `b3` and by selecting the best-matching file, without changing how weights are loaded.'
- What this solution (achieved 0.25598) has done: 'Your current score (0.25598) is far below the target (0.79435), so we need a meaningful accuracy increase while keeping your core EfficientNet-B3 inference pipeline intact. The biggest issue is that your model’s classifier is replaced to 5 classes, but you often fail to load a compatible cassava fine-tuned checkpoint; with ImageNet weights + random 5-class head, accuracy be near-random. I make the checkpoint discovery stricter and more cassava-specific (prefer filenames/paths that indicate cassava and efficientnet-b3 and penalize irrelevant .pt files), and I also add robust handling for common checkpoint formats (including nested keys like `model_state_dict`, and automatic mapping for torchvision EfficientNet classifier keys). This keeps the same architecture and inference loop, but greatly increases the chance you actually load the intended fine-tuned weights, which should move the score substantially toward the target.'
- What this solution (achieved 0.25598) has done: 'Your current score (0.25598) is far below the 0.79435 target, so we need a real accuracy lift while keeping the same EfficientNet-B3 inference-only pipeline. The most likely reason for near-random accuracy is that the discovered checkpoint isn’t actually a cassava fine-tuned EfficientNet-B3, or it loads in a way that leaves the 5-class head uninitialized. I make checkpoint discovery more cassava-competition-specific (prefer paths under this dataset folder and filenames hinting 5-class cassava) and add safer state-dict normalization (including handling common `.ckpt`/Lightning formats and ignoring obviously incompatible classifier shapes). These are minimal changes that preserve your model/loop/transforms but materially increase the chance you load the intended fine-tuned weights, moving accuracy toward the target.'
- What this solution (achieved 0.25598) has done: 'We need to move accuracy up toward the 0.794 target, and your current 0.25598 strongly suggests the fine-tuned cassava checkpoint is still not being loaded (so you’re effectively doing near-random with a 5-class head). The minimal, core-logic-preserving fix is to (1) stop scanning all of `/kaggle/input` (which can pick irrelevant weights) and instead search only within the competition dataset directory first, and (2) load the checkpoint in a way that guarantees the classifier head is loaded when shapes match (and only drop it when it truly mismatches). I also make the key remapping cover the common Lightning `model.classifier.*` naming so a correct EfficientNet-B3 cassava checkpoint actually maps onto `classifier.1.*`. These changes keep the same EfficientNet-B3 architecture, transforms, and inference loop, but should materially increase the chance you’re using the intended 5-class cassava weights, pushing score toward the target band.'
- What this solution (achieved 0.25598) has done: 'Your score (0.25598) is far below the target (0.79435), so we should increase accuracy with the smallest change that preserves your EfficientNet-B3 inference-only pipeline. The most likely cause is still that the cassava fine-tuned checkpoint is not being found/loaded correctly, leaving you with an ImageNet backbone plus a random 5-class head; to fix this without changing the model/loop, we (1) constrain checkpoint search to likely competition/working locations first and (2) strengthen key remapping to cover common Lightning-style `model.classifier.*` / `classifier.1.*` vs `classifier.0/1.*` variants. We also stop dropping classifier tensors unless they truly mismatch shape, so when a correct 5-class head exists it actually loads. These changes keep architecture, transforms, and inference semantics intact, but substantially increase the chance you are using the intended fine-tuned cassava weights, moving accuracy toward the target band.'
- What this solution (achieved 0.25598) has done: 'Your current accuracy (0.25598) is far below the target (0.79435), so we need to ensure you’re actually using the intended cassava fine-tuned EfficientNet-B3 weights rather than an ImageNet backbone with a random 5-class head. The minimal, core-logic-preserving move is to (1) stop searching broadly first and instead explicitly search for checkpoints inside `/kaggle/input/aefficientnet/` (your original path hint) and the competition dataset folder, and (2) strengthen state-dict key normalization so common EfficientNet naming variants map correctly to torchvision’s `classifier.1.*` (including `classifier.fc.*`, `model.classifier.*`, and Lightning wrappers). Finally, we only drop classifier tensors when they truly mismatch shape, so a correct 5-class head loads when available, which should move accuracy substantially toward the target without changing architecture, transforms, or inference semantics.'
- What this solution (achieved 0.25598) has done: 'Your current accuracy is far below the target, which strongly suggests the fine-tuned cassava checkpoint still isn’t being loaded into the correct keys (especially the classifier head), leaving you near-random. I keep the same EfficientNet-B3 inference-only pipeline, but make the checkpoint loading more compatible by (1) expanding key remapping to cover the most common EfficientNet/Lightning variants for `classifier` and `_fc`, and (2) loading the checkpoint *after* remapping with a controlled fallback that tries `strict=True` first (when shapes match) to avoid silently missing lots of keys. I also narrow checkpoint search priority to the competition dataset/working directories before broader `/kaggle/input`, reducing the chance of picking an unrelated weights file. These are minimal changes intended to materially increase the probability that the intended 5-class cassava weights are actually used, pushing the score toward your 0.794 target.'

# 9. Code solution

## === cell 0
import os
import glob
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_path = f"{DATA_ROOT}/test_images/"
train_path = (
    f"{DATA_ROOT}/train_images/"  # used by Dataset even if test=False elsewhere
)

sample = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
assert {"image_id", "label"}.issubset(sample.columns)



## === cell 2
weights_obj = None
try:
    weights_obj = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
    model_transfer = torchvision.models.efficientnet_b3(weights=weights_obj)
    _using_imagenet = True
except Exception:
    model_transfer = torchvision.models.efficientnet_b3(weights=None)
    _using_imagenet = False

in_features = model_transfer.classifier[-1].in_features
model_transfer.classifier[-1] = nn.Linear(in_features, 5, bias=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_transfer = model_transfer.to(device)

print("Using ImageNet init:", _using_imagenet)




## === cell 3
def _find_checkpoint():
    search_roots = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/working",
        "/kaggle/input/aefficientnet",
        "/kaggle/input/cassava-leaf-disease-classification/**",
        "/kaggle/input",
    ]

    patterns = []
    for root in search_roots:
        patterns.extend(
            [
                f"{root}/*.pt",
                f"{root}/*.pth",
                f"{root}/*.bin",
                f"{root}/*.ckpt",
                f"{root}/**/*.pt",
                f"{root}/**/*.pth",
                f"{root}/**/*.bin",
                f"{root}/**/*.ckpt",
            ]
        )

    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))

    seen = set()
    deduped = []
    for p in candidates:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    candidates = deduped

    filtered = []
    for p in candidates:
        full = p.lower()
        name = os.path.basename(p).lower()

        if any(
            x in name
            for x in ["optimizer", "sched", "scheduler", "history", "log", "events"]
        ):
            continue
        if any(x in full for x in ["/wandb/", "/runs/", "/logs/"]):
            continue
        try:
            if os.path.getsize(p) < 800_000:  # <0.8MB
                continue
        except OSError:
            continue

        filtered.append(p)

    candidates = filtered
    if not candidates:
        return None

    def score_path(p: str) -> tuple:
        full = p.lower()
        name = os.path.basename(p).lower()
        s = 0

        if full.startswith("/kaggle/input/cassava-leaf-disease-classification/"):
            s += 420
        if "/cassava-leaf-disease-classification/" in full:
            s += 220
        if full.startswith("/kaggle/working/"):
            s += 260
        if full.startswith("/kaggle/input/aefficientnet/"):
            s += 180

        for k in ["cassava", "leaf", "disease", "cldc", "cassavaleaf"]:
            if k in full:
                s += 55

        for k in ["efficientnet", "effnet"]:
            if k in full:
                s += 45
        if re.search(r"\bb3\b", full) or "efficientnet_b3" in full or "effb3" in full:
            s += 45

        for k in [
            "best",
            "final",
            "fold",
            "checkpoint",
            "ckpt",
            "weights",
            "model",
            "finetune",
            "finetuned",
        ]:
            if k in name:
                s += 12

        for k in ["imagenet", "in1k", "openimages", "coco", "places"]:
            if k in full:
                s -= 14

        for k in [
            "5class",
            "5-class",
            "cassava5",
            "cldc5",
            "num_classes5",
            "num-classes-5",
        ]:
            if k in full:
                s += 30

        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0

        return (-s, -size, len(p), p)

    candidates = sorted(candidates, key=score_path)
    return candidates[0]


def _extract_state_dict(state):
    if not isinstance(state, dict):
        return state

    for k in ["state_dict", "model_state_dict", "model", "net", "network", "weights"]:
        if k in state and isinstance(state[k], dict):
            return state[k]

    for k in ["ema_state_dict", "ema", "model_ema"]:
        if k in state and isinstance(state[k], dict):
            inner = state[k]
            if "state_dict" in inner and isinstance(inner["state_dict"], dict):
                return inner["state_dict"]
            return inner

    return state


def _remap_keys_for_torchvision_efficientnet(sd: dict) -> dict:
    new_state = {}
    for k, v in sd.items():
        nk = k

        for prefix in (
            "model.",
            "module.",
            "net.",
            "model_transfer.",
            "network.",
            "backbone.",
            "student.",
            "teacher.",
            "ema_model.",
        ):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]

        if nk.startswith("model."):
            nk = nk[len("model.") :]

        head_weight_aliases = {
            "_fc.weight",
            "fc.weight",
            "head.weight",
            "head.fc.weight",
            "classifier.weight",
            "classifier.fc.weight",
            "classifier.1.weight",
            "model.classifier.1.weight",
        }
        head_bias_aliases = {
            "_fc.bias",
            "fc.bias",
            "head.bias",
            "head.fc.bias",
            "classifier.bias",
            "classifier.fc.bias",
            "classifier.1.bias",
            "model.classifier.1.bias",
        }

        if nk in head_weight_aliases:
            nk = "classifier.1.weight"
        elif nk in head_bias_aliases:
            nk = "classifier.1.bias"

        if nk == "classifier.0.weight":
            nk = "classifier.1.weight"
        elif nk == "classifier.0.bias":
            nk = "classifier.1.bias"

        if nk.startswith("head.fc."):
            nk = nk.replace("head.fc.", "classifier.1.")
        if nk.startswith("head."):
            if nk == "head.weight":
                nk = "classifier.1.weight"
            elif nk == "head.bias":
                nk = "classifier.1.bias"

        if nk.startswith("classifier.fc."):
            nk = nk.replace("classifier.fc.", "classifier.1.")

        new_state[nk] = v
    return new_state


def _drop_incompatible_classifier_tensors(sd: dict, model: torch.nn.Module) -> dict:
    model_sd = model.state_dict()
    cleaned = {}
    dropped = 0
    for k, v in sd.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) != tuple(model_sd[k].shape):
                if "classifier" in k:
                    dropped += 1
                    continue
        cleaned[k] = v
    if dropped > 0:
        print(
            f"Dropped {dropped} incompatible classifier tensors (wrong #classes head)."
        )
    return cleaned


weights_path = "/kaggle/input/aefficientnet/weights.pt"
if not os.path.exists(weights_path):
    found = _find_checkpoint()
    if found is not None:
        weights_path = found
    else:
        weights_path = None

loaded_any = False
if weights_path is not None and os.path.exists(weights_path):
    print("Loading checkpoint:", weights_path)
    state = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(state)

    if isinstance(state, dict):
        state = _remap_keys_for_torchvision_efficientnet(state)
        state = _drop_incompatible_classifier_tensors(state, model_transfer)

        try:
            model_transfer.load_state_dict(state, strict=True)
            print("Loaded weights with strict=True (fully compatible).")
        except Exception as e:
            missing, unexpected = model_transfer.load_state_dict(state, strict=False)
            print("Strict load failed, used strict=False. Reason:", repr(e))
            print(
                f"Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
            if any("classifier" in k for k in missing):
                print(
                    "Note: classifier keys missing; backbone may still be loaded (better than random head)."
                )
        loaded_any = True
else:
    print(
        "No external checkpoint found; using current model weights as-is (ImageNet init may still be used)."
    )




## === cell 4
class LeafDataset(Dataset):
    def __init__(self, dataframe, transform=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.image_id.values[idx]

        if not self.test:
            label = int(self.df.label.values[idx])
            p_path = os.path.join(train_path, p)
        else:
            label = 0
            p_path = os.path.join(test_path, p)

        image = cv2.imread(p_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {p_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)




## === cell 5
if weights_obj is not None:
    test_transforms = weights_obj.transforms()
else:
    test_transforms = transforms.Compose(
        [
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

testset = LeafDataset(sample, transform=test_transforms, test=True)
testLoader = DataLoader(
    testset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
model_transfer.eval()
test_pred = []

with torch.no_grad():
    for datatest, _ in testLoader:
        datatest = datatest.to(device, non_blocking=True)
        logits = model_transfer(datatest)
        pred = logits.argmax(1).detach().cpu().numpy().astype(int)
        test_pred.extend(pred.tolist())

sample["label"] = test_pred
sample.to_csv("submission.csv", index=False)

print(sample.head())
print("Wrote submission.csv with shape:", sample.shape)
print("Loaded any external checkpoint:", loaded_any)
print("Checkpoint path used:", weights_path)



## === cell 7
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample)
assert sub["image_id"].iloc[0] == sample["image_id"].iloc[0]
print(sub.head())
print("Submission OK:", sub.shape)
