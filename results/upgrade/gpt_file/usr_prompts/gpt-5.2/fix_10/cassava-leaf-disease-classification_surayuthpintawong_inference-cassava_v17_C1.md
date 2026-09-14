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

2.7

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.8750377757630704

# 6. Current score

0.11883

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the notebook-only `%cd` magics and fix the missing `efficientnet_pytorch` dependency by switching to the built-in `torchvision.models.efficientnet_b4` while keeping the same EfficientNet-B4 architecture and 5-class head. I also fix the Albumentations API break (`A.Flip` no longer exists) by using `A.HorizontalFlip`, and correct the custom `ToTensor` so it works with Albumentations’ dict interface and returns a proper float tensor. Finally, I make model weight loading robust (CPU/GPU map_location and common checkpoint formats) and ensure we always write a valid `submission.csv` aligned to `sample_submission.csv` image order.'
- What this solution (achieved 0.13677) has done: 'I fix the Albumentations runtime error by making the custom tensor transform compatible with Albumentations v2 (it must accept/return a dict and expose `available_keys`). Then I fix the missing-weights crash by searching for the checkpoint in common Kaggle input locations (including the dataset folder you actually have) and only failing with a clear message if nothing is found. Finally, I ensure the dataloader/model cells run in order and always write a valid `submission.csv` aligned to `sample_submission.csv`. These changes are correctness/stability fixes; they don’t change the model architecture or inference semantics beyond making the pipeline actually run.'
- What this solution (achieved 0.09342) has done: 'I fix two execution blockers: (1) the test image directory contains a nested `test_images/` folder, so the dataset currently tries to “read a folder as an image”; I robustly collect only actual image files (jpg/png/jpeg) recursively and keep names consistent with `sample_submission.csv`. (2) your run crashes when the expected checkpoint isn’t present; to ensure an end-to-end runnable pipeline that produces a valid `submission.csv`, I fall back to ImageNet pretrained EfficientNet-B4 weights if the competition checkpoint cannot be found (this should also improve the score versus random initialization, moving it toward your target). All other logic (EfficientNet-B4 head with 5 classes, transforms, inference, submission alignment) is kept the same.'
- What this solution (achieved 0.11286) has done: 'Your current score is extremely low mainly because the code is almost certainly running the ImageNet-pretrained fallback (random 5-class head), and even when a cassava checkpoint is found it may not actually be loaded into the correct keys due to common prefix/name mismatches. To move the accuracy up toward your target with minimal changes, I (1) expand checkpoint discovery to include common Kaggle EfficientNet cassava filenames/locations, (2) make state_dict key “cleaning” more robust for torchvision EfficientNet (handling `features.`/`classifier.`/`_fc` style keys) while still preserving the same model architecture, and (3) use a deterministic test-time transform (remove flip randomness) so predictions are stable and not degraded by random augmentation at inference. This keeps your core approach intact (EfficientNet-B4, single forward pass argmax, same preprocessing family) and should substantially increase score if a real cassava-trained checkpoint exists in your inputs. The submission writing/alignment logic remains the same.'
- What this solution (achieved 0.12668) has done: 'Your current score is far below the target, which strongly suggests the pipeline is still not using a real cassava-finetuned checkpoint (so the 5-class head is effectively random). To move accuracy up with minimal changes, I (1) add robust checkpoint *selection* (prefer files that look like cassava/5-class EfficientNet checkpoints and de-prioritize generic ImageNet/embeddings), (2) expand state_dict key cleaning to handle common EfficientNet naming patterns (including `encoder.`/`backbone.`/`model.model.` and EfficientNet-PyTorch `_conv_stem/_bn0/_fc` style), and (3) add a safe classifier-size check so we only load classifier weights when they match 5 classes (otherwise keep the existing randomly initialized 5-class head instead of partially mismatching). This keeps the same architecture (torchvision EfficientNet-B4 + Linear(…,5)), the same preprocessing family, and the same argmax inference, but makes it much more likely you actually load the intended competition weights and therefore move the score toward your target.'
- What this solution (achieved 0.23879) has done: 'Your score is far below the target, which strongly indicates the cassava fine-tuned checkpoint still isn’t being loaded correctly (so you’re effectively using an ImageNet backbone with a random 5-class head). I make a minimal, score-directed change to (1) expand key-cleaning so EfficientNet-PyTorch/timm-style checkpoints map correctly onto torchvision EfficientNet-B4 (including `features.*` vs `conv_stem/_bn0/_blocks/_conv_head/_bn1` naming and `head.fc`/`_fc` mapping), and (2) add a “strict-enough” load that only accepts a checkpoint when most backbone tensors actually match (otherwise we explicitly fall back to ImageNet to avoid silently-bad partial loads). This preserves the same architecture, transforms, and argmax inference; it only improves the probability that the intended 5-class cassava weights are actually applied. It still always write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.4503) has done: 'Your current score (0.23879) is far below the target (0.8750), so we should cautiously increase accuracy without changing the core approach (EfficientNet-B4 + argmax). The biggest likely issue is still “bad weights” (ImageNet fallback or partially/incorrectly loaded cassava checkpoint), plus a test-time preprocessing mismatch: using `CenterCrop(512,512)` fail to match typical EfficientNet training/inference pipelines and can discard important leaf regions. I (1) make checkpoint loading stricter in a *good* way by preferring checkpoints that actually contain a 5-class classifier and by choosing the candidate with the highest overlap of tensor shapes with the current model, and (2) change inference preprocessing to a deterministic `Resize(512,512)` (no randomness) to better match common cassava EfficientNet setups while preserving the same model and prediction semantics. These are minimal, score-directed changes that should move accuracy materially toward the target if any real cassava checkpoint exists in your inputs; the script still always produces a valid `submission.csv`.'
- What this solution (achieved 0.0867) has done: 'Your current score (0.4503) is far below the target (0.8750), so we should make the smallest changes that legitimately increase accuracy without changing the core model/inference semantics (EfficientNet-B4 + single-pass argmax). The biggest likely issue is still a preprocessing mismatch: this model family is typically trained/evaluated with EfficientNet’s own normalization (not ImageNet’s), so switching to the correct mean/std is a minimal but often high-impact fix. To avoid unintended score drops, I keep the same resize and deterministic pipeline, only adjusting `A.Normalize` to EfficientNet defaults and making the tensor conversion explicitly scale-safe (still identical semantics: normalized float tensor). The rest (checkpoint selection/loading, dataloader, argmax predictions, and submission alignment) is unchanged.'
- What this solution (achieved 0.11883) has done: 'Your current score (0.0867) is far below the target (0.8750), which strongly suggests you are still running with a random 5-class head (either due to failing to find a real cassava checkpoint or rejecting it via the overlap threshold). The smallest score-directed fix is to make checkpoint discovery actually find the common Kaggle cassava EfficientNet checkpoints stored as `.pth`/`.bin` but also frequently as `.ckpt`, and to relax the “overlap >= 50” gate slightly so we don’t incorrectly discard a valid cassava checkpoint due to minor key-name differences. I keep the exact same model (torchvision EfficientNet-B4 with a 5-class Linear head), transforms, and argmax inference; the only behavior change is “load the right weights more often”. The submission writing and alignment to `sample_submission.csv` stays identical.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import glob
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage import io

import albumentations as A

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

model_full_name = "efficientnet-b4-e0"
model_name = "efficientnet-b4"
folder_name = "effnetmodelb44"
WEIGHTS_PATH = os.path.join("/kaggle/input", folder_name, model_full_name + ".pt")

print("device:", device)
print(
    "TEST_DIR exists:",
    os.path.isdir(TEST_DIR),
    "n_entries:",
    (len(os.listdir(TEST_DIR)) if os.path.isdir(TEST_DIR) else 0),
)
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Initial WEIGHTS_PATH:", WEIGHTS_PATH)




## === cell 1
class ToTensorV2Compatible(object):
    available_keys = {"image"}

    def __call__(self, force_apply=False, **data):
        image = data.get("image", None)
        if image is None:
            return data
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        data["image"] = torch.from_numpy(image).float()
        return data




## === cell 2
def list_image_files(root_dir):
    """
    Bugfix: Kaggle dataset sometimes has nested `test_images/test_images/`.
    If we just os.listdir(root_dir), we may include subfolders and crash when reading them as images.
    We only return actual image files (recursively), but keep only the basename for submission alignment.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root_dir, "**", "*" + ext), recursive=True))
        paths.extend(
            glob.glob(os.path.join(root_dir, "**", "*" + ext.upper()), recursive=True)
        )
    paths = sorted(set(paths))
    return paths


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = list_image_files(root_dir)
        if len(self.image_paths) == 0:
            entries = [os.path.join(root_dir, x) for x in sorted(os.listdir(root_dir))]
            self.image_paths = [
                p
                for p in entries
                if os.path.isfile(p)
                and p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

        self.images = [os.path.basename(p) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        img_name = self.images[idx]
        image = io.imread(img_path)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        if self.transform:
            out = self.transform(image=image)
            image = out["image"]

        return img_name, image




## === cell 3
EFFNET_MEAN = (0.5, 0.5, 0.5)
EFFNET_STD = (0.5, 0.5, 0.5)

transform = A.Compose(
    [
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(
            mean=EFFNET_MEAN,
            std=EFFNET_STD,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2Compatible(),
    ]
)

test_ds = TestDataset(root_dir=TEST_DIR, transform=transform)
testloader = DataLoader(
    test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=use_cuda
)

print("Test dataset size:", len(test_ds))
print("First 5 test images (basenames):", test_ds.images[:5])



## === cell 4
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def clean_state_dict_keys_for_torchvision_effnet(sd):
    """
    Key cleaning only (no model change): improves odds a cassava-finetuned EfficientNet checkpoint
    maps correctly onto torchvision EfficientNet-B4.
    """
    clean_sd = {}
    for k, v in sd.items():
        nk = k

        for pref in [
            "module.",
            "model.",
            "model.model.",
            "net.",
            "network.",
            "backbone.",
            "encoder.",
        ]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]

        nk = nk.replace("_conv_stem.", "features.0.0.")
        nk = nk.replace("_bn0.", "features.0.1.")
        nk = nk.replace("_conv_head.", "features.7.0.")
        nk = nk.replace("_bn1.", "features.7.1.")

        if nk.startswith("_blocks."):
            nk = nk.replace("_blocks.", "features.1.", 1)

        nk = nk.replace("conv_stem.", "features.0.0.")
        nk = nk.replace("bn1.", "features.7.1.")
        nk = nk.replace("conv_head.", "features.7.0.")
        if nk.startswith("blocks."):
            nk = nk.replace("blocks.", "features.1.", 1)

        if nk in ["_fc.weight", "fc.weight", "head.fc.weight", "classifier.weight"]:
            nk = "classifier.1.weight"
        elif nk in ["_fc.bias", "fc.bias", "head.fc.bias", "classifier.bias"]:
            nk = "classifier.1.bias"

        clean_sd[nk] = v
    return clean_sd


def drop_mismatched_classifier(clean_sd, model):
    w_key = "classifier.1.weight"
    b_key = "classifier.1.bias"
    if w_key in clean_sd and hasattr(model, "classifier"):
        target_w = model.classifier[1].weight
        if tuple(clean_sd[w_key].shape) != tuple(target_w.shape):
            clean_sd.pop(w_key, None)
            clean_sd.pop(b_key, None)
    return clean_sd


def state_dict_has_5class_head(clean_sd):
    w_key = "classifier.1.weight"
    if w_key not in clean_sd:
        return False
    w = clean_sd[w_key]
    return isinstance(w, torch.Tensor) and (w.ndim == 2) and (w.shape[0] == 5)


def overlap_score_by_shape(clean_sd, model_sd):
    """
    Score-relevant: pick the checkpoint that best matches the current model by (key, shape) overlap.
    This avoids accidentally selecting unrelated files and improves chances of loading real cassava weights.
    """
    score = 0
    for k, v in clean_sd.items():
        if k in model_sd:
            try:
                if tuple(v.shape) == tuple(model_sd[k].shape):
                    score += 1
            except Exception:
                pass
    return score


def find_best_checkpoint(initial_path, data_root):
    """
    Score-relevant change:
      - Include `.ckpt` files (common for PyTorch Lightning Kaggle training outputs).
      - Relax the hard overlap gate slightly so we don't discard valid cassava checkpoints due to minor key diffs.
    """
    if os.path.exists(initial_path):
        return initial_path

    candidates = []
    for base in ["/kaggle/input", data_root]:
        candidates += glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)
        candidates += glob.glob(os.path.join(base, "**", "*.pth"), recursive=True)
        candidates += glob.glob(os.path.join(base, "**", "*.bin"), recursive=True)
        candidates += glob.glob(os.path.join(base, "**", "*.ckpt"), recursive=True)
    candidates = sorted(set(candidates))

    if not candidates:
        return None

    ref_model = efficientnet_b4(weights=None)
    in_features = ref_model.classifier[1].in_features
    ref_model.classifier[1] = nn.Linear(in_features, 5)
    ref_sd = ref_model.state_dict()

    best = None
    best_tuple = None  # (has_5class, overlap, name_score)
    pos_kw = [
        "cassava",
        "leaf",
        "disease",
        "effnet",
        "efficientnet",
        "b4",
        "fold",
        "best",
        "final",
        "ckpt",
        "checkpoint",
    ]
    neg_kw = [
        "imagenet",
        "pretrain",
        "embedding",
        "repr",
        "feature",
        "optimizer",
        "sched",
        "scheduler",
        "ema",
        "onnx",
        "tflite",
    ]

    def name_score(p):
        name = os.path.basename(p).lower()
        s = 0
        for kw in pos_kw:
            if kw in name:
                s += 1
        for kw in neg_kw:
            if kw in name:
                s -= 1
        return s

    for p in candidates:
        try:
            ckpt = torch.load(p, map_location="cpu")
            sd = extract_state_dict(ckpt)
            if not isinstance(sd, dict):
                continue
            clean_sd = clean_state_dict_keys_for_torchvision_effnet(sd)

            has5 = state_dict_has_5class_head(clean_sd)
            ov = overlap_score_by_shape(clean_sd, ref_sd)
            ns = name_score(p)

            t = (1 if has5 else 0, ov, ns)
            if (best_tuple is None) or (t > best_tuple):
                best_tuple = t
                best = p
        except Exception:
            continue

    if best is None:
        return None

    if best_tuple[1] < 30:
        return None
    return best


def backbone_load_ok(missing, unexpected):
    missing_backbone = [k for k in missing if not k.startswith("classifier.")]
    return len(missing_backbone) <= 40 and len(unexpected) <= 200


resolved = find_best_checkpoint(WEIGHTS_PATH, DATA_ROOT)
use_imagenet_fallback = resolved is None

if use_imagenet_fallback:
    print(
        "WARNING: No compatible cassava fine-tuned checkpoint found with sufficient tensor overlap.\n"
        "Falling back to torchvision ImageNet pretrained EfficientNet-B4 weights to ensure an end-to-end run."
    )
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
else:
    WEIGHTS_PATH = resolved
    print("Using WEIGHTS_PATH:", WEIGHTS_PATH)
    model = efficientnet_b4(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)

if not use_imagenet_fallback:
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    sd = extract_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise ValueError(
            "Loaded checkpoint is not a state_dict/dict-like object: %r" % type(sd)
        )

    clean_sd = clean_state_dict_keys_for_torchvision_effnet(sd)
    clean_sd = drop_mismatched_classifier(clean_sd, model)

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(
        "Loaded weights. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )

    if not backbone_load_ok(missing, unexpected):
        print(
            "WARNING: Checkpoint appears incompatible with torchvision EfficientNet-B4 (too many missing backbone keys). "
            "Falling back to ImageNet pretrained backbone for stability."
        )
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 5)
        model = model.to(device)

model.eval()
model = model.to(device)



## === cell 5
names = []
predicted = []

with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device, non_blocking=True).float()
        output = model(images_batch)
        pred = torch.argmax(output, dim=1).detach().cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(pred.tolist())

print("Preds:", len(predicted), "Names:", len(names))



## === cell 6
sample = pd.read_csv(SAMPLE_SUB_PATH)

pred_map = dict(zip(names, predicted))
sample["label"] = sample["image_id"].map(pred_map)

sample["label"] = sample["label"].fillna(0).astype(int)

sample.to_csv("submission.csv", index=False)

print(sample.head())
print("Wrote submission.csv with shape:", sample.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Unique predicted labels:", sorted(sample["label"].unique().tolist()))
