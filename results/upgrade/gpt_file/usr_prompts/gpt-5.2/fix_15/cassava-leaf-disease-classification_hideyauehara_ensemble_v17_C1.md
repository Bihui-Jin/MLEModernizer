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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.8909035962526443

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I remove the unavailable `efficientnet_pytorch` dependency by using `torchvision.models.efficientnet_b7` while keeping the same “efficientnet-b7” branch and classifier replacement behavior. I make the dataset path detection robust to Kaggle’s `/kaggle/input/...` structure so `test_files` is always defined and points to the real test set in Batch runs. I fix Albumentations v2 API errors by switching `RandomResizedCrop` to the v2-compatible signature and keep the same TTA intent. Finally, I fix the ensemble aggregation shape bug and ensure `submission.csv` is always written with the correct `image_id,label` columns.'
- What this solution (achieved 0.11584) has done: 'Your score is extremely low because the EfficientNet-B7 wrapper currently replaces the classifier inside `FinalLayerMixupModelEN`, but the loaded checkpoint likely already contains a trained classifier head; this mismatch makes predictions effectively random. I make the smallest fix: do not modify EfficientNet’s classifier during inference, and load the checkpoint with `strict=False` only for the EfficientNet branch to tolerate minor key differences while still loading the trained weights. I also stop passing the dummy `labels=False` into the model by allowing the forward to accept `labels=None` for test, keeping the same evaluation semantics. These changes preserve the same model family/architecture and TTA ensemble logic, but should move accuracy much closer to your target by correctly using the pretrained head.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the most likely issue is that the EfficientNet-B7 branch is not matching the checkpoint’s saved head: you currently wrap the torchvision EfficientNet directly but never replace its classifier to output 5 classes, so the loaded weights won’t map correctly and predictions become near-random. I make the smallest architecture-consistent fix: set `net.classifier[1] = nn.Linear(..., 5)` before wrapping, and load with `strict=True` for EfficientNet so we only proceed when the checkpoint matches the expected structure. I also ensure EfficientNet uses the correct forward signature (call `net(inputs, phase="test")` rather than passing positional args) to avoid accidental argument mismatches. These changes preserve your ensemble/TTA logic and should move accuracy substantially toward the target by correctly using the trained classifier head.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.8909), so the likely issue is that the checkpoint keys don’t match the wrapped model’s parameter names, causing weights not to load (or to load into the wrong places) and making predictions near-random. I keep the same model choices, wrappers, and TTA/ensemble averaging, but add a minimal “state_dict key normalization” step that strips common prefixes (`module.`, `_orig_mod.`, `model.`, `net.`, `encoder.`) before `load_state_dict`, and I fail fast if a suspiciously large fraction of parameters are missing/unexpected. I also fix deterministic settings (your seed function sets `deterministic=True` but also `benchmark=True`, which conflicts) to stabilize inference behavior without changing the approach. These are minimal, inference-only fixes aimed at actually using the trained weights correctly, which should move accuracy substantially toward your target.'
- What this solution (achieved 0.11584) has done: 'The current score is near-random, which usually happens when checkpoints aren’t actually being loaded into the same parameter names/shapes as the defined model (especially for the EfficientNet-B7 wrapper where the saved checkpoint may have used a different wrapper/head naming). I make the smallest inference-only change to robustly load checkpoints by mapping common EfficientNet head keys (e.g., `fc.*` saved by a wrapper) onto torchvision EfficientNet’s `classifier.1.*` when applicable, while keeping the exact same architecture and prediction/TTA logic. I also make the key-mismatch “fail fast” check more informative and ensure we always use the correct test set in Batch runs (no accidental 32-image debug slicing when `KAGGLE_KERNEL_RUN_TYPE` is unset). These changes are directly aimed at using the trained weights correctly, which should increase accuracy substantially toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, which strongly suggests the checkpoints are not actually being loaded (or are being loaded into the wrong parameter names), so inference is effectively using randomly initialized weights. I make the smallest inference-only change that increases the chance of correct weight loading by (1) removing the hard failure on key mismatch fraction (so we can fall back to a compatible load) and (2) making the loader try `strict=True` first, then automatically retry with `strict=False` after normalizing/remapping keys, while printing a concise summary so you can verify loads are sane. I also force the EfficientNet-B7 wrapper to load into the underlying `net.model` (the actual EfficientNet module) if the checkpoint appears to target it, which avoids common wrapper-prefix mismatches without changing the model architecture or prediction logic. These changes are directly targeted at getting the trained weights applied correctly, which should move accuracy substantially toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the most likely cause is still that the checkpoint weights are not being applied to the *actual* module that owns the parameters (especially for EfficientNet where the wrapper adds a `model.` prefix). I make a minimal, inference-only change in the checkpoint loader to automatically detect whether the checkpoint targets `net.model` (the wrapped backbone) and load into that submodule when appropriate, while keeping the same architecture, TTA, and averaging logic. I also make EfficientNet’s head remapping explicitly handle torchvision EfficientNet (`classifier.1.*`) and the common wrapper naming (`fc.*`) more reliably, so the 5-class head weights land in the right place. These changes are directly aimed at correctly using your trained weights and should move accuracy substantially upward toward your target.'
- What this solution (achieved 0.11584) has done: 'I make two minimal, score-critical fixes that commonly cause near-random accuracy in this exact setup: (1) ensure EfficientNet-B7 uses the *same pooling behavior as torchvision’s default forward* (your current wrapper bypasses `avgpool`, which breaks the checkpoint), and (2) make the checkpoint loader explicitly remap EfficientNet head keys `fc.*` → `classifier.1.*` **before** loading so the 5-class head weights land correctly. These changes keep the same model family, same TTA/ensemble averaging, same loss, and same inference loop, but should move your score substantially upward toward the target by actually applying the trained weights correctly. I also keep the submission writing logic unchanged and still produce `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the most likely cause is still that many checkpoints are not actually loading into the intended submodule/parameter names (especially for the EfficientNet-B7 branch and wrapper prefixes), leaving the model effectively untrained at inference. I make a minimal, inference-only improvement to the checkpoint loading: detect and remap common EfficientNet backbone prefixes (`features.*`, `model.features.*`) and head prefixes (`classifier.*`, `fc.*`) so they land on torchvision EfficientNet-B7’s expected keys, and prefer loading into the correct target module (net vs net.model) based on overlap. I also make the test image root resolution slightly more robust to the duplicated `test_images/test_images` directory variant, because predicting on the wrong folder can also crater the score. Core model definitions, TTA list, and inference loop remain unchanged; we only adjust key mapping and path detection to make the existing trained weights actually used.'
- What this solution (achieved 0.11584) has done: 'Your current score is near-random, so the smallest likely score-critical issue is still that some checkpoints are not being loaded into the *actual* parameter names of your wrapped models (especially the ResNet/DenseNet wrappers that introduce a new `fc` head). I add a minimal, model-specific key remap in the loader so that common saved keys like `fc.*` correctly map onto your wrapper heads (`fc.*` for ResNet wrapper, and `fc.*` → `fc.*` for DenseNet wrapper), and also remap `classifier.*` → `fc.*` when the checkpoint was saved from a different wrapper. I also make the `squeeze()` in the wrappers batch-safe by using `flatten(1)` so inference doesn’t accidentally drop the batch dimension for the last small batch (this is a correctness fix, not a model change). These changes keep the same architectures, TTA list, and averaging logic, but should substantially increase accuracy toward your target by ensuring you’re actually using trained weights and producing correct logits shapes.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, which strongly suggests many checkpoints are still not loading into the *actual* parameters used at inference (especially for the ResNet/DenseNet wrappers where the trained head might be saved as `model.fc.*` / `classifier.*` while your wrapper expects `fc.*`). I make the smallest inference-only fix: enhance the checkpoint key remapping to explicitly handle wrapper naming (`convlayer.*` + `fc.*`) and map common saved head keys onto your wrapper heads when shapes match, then retry strict load before falling back to non-strict. This keeps the same architectures, TTA transforms, and averaging logic, but should move accuracy substantially upward by ensuring the trained weights (particularly the final classification head) actually get applied. I also add a concise load diagnostic that reports the overlap ratio to confirm loads are sane without changing any training/inference semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is near-random and far below the target, so the most likely issue is that at least some checkpoints are loading but producing systematically wrong class IDs due to label-index mapping mismatch (common in this competition: many models were trained with swapped labels like 1↔2). I keep your entire model/ensemble/TTA inference logic unchanged and add a minimal, inference-only calibration step that searches a small set of safe class-permutation candidates and chooses the one that best matches the training distribution (a proxy when test labels are unavailable). This doesn’t change the core model architecture or training approach; it only remaps the final predicted argmax labels before writing the submission, which is directly score-relevant for accuracy. I also add a lightweight diagnostic to confirm we are using the real test set (2676 images) and that probabilities sum correctly, without altering outputs otherwise.'
- What this solution (achieved 0.11584) has done: 'The current score is near-random, which strongly suggests inference is not using the intended (trained) classifier head weights, most likely due to a head/key mismatch in checkpoint loading (especially for EfficientNet-B7 and the custom wrappers). I make the smallest, inference-only change to the checkpoint loader: after normalizing prefixes, additionally remap common head keys **both ways** (`fc.*` ↔ `classifier.*`/`classifier.1.*`) and choose the best remap by matching tensor shapes against the current model’s `state_dict`. This preserves your model architectures, TTA, and averaging logic, but should substantially increase accuracy by ensuring the final 5-class head is actually loaded instead of left randomly initialized. I also add a strict sanity check: if the head is still not loaded (shape-match not found), we warn clearly, but we still produce a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the most likely issue is that the checkpoint weights are not being applied to the correct parameters, especially the final classification head. I keep your exact model wrappers, TTA list, and averaging logic, but make the checkpoint loader choose the best key-remap by matching tensor *shapes* against the current model `state_dict` (this is a minimal inference-only change that makes head/backbone weights actually load). I also add an explicit EfficientNet head/backbone remap (`fc.*`/`classifier.*`/`classifier.1.*`) using shape checks, and I fail over cleanly without changing predictions when a remap doesn’t apply. This should move accuracy substantially upward toward your target while preserving your core approach and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = glob.glob(f"../input/eb7m-seed70/*.pth")

print(f"{len(pretrained_models)} models found.")
print("\n".join(np.sort(pretrained_models)))



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
import random
import time

from tqdm import tqdm

import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)



## === cell 3
from torchvision.models import efficientnet_b7



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def _resolve_base_dir():
    candidates = [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "data/cassava-leaf-disease-classification",
        "data",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "."


KRT = os.getenv("KAGGLE_KERNEL_RUN_TYPE")
BASE_DIR = _resolve_base_dir()

candidate_tests = [
    f"{BASE_DIR}/test_images",
    f"{BASE_DIR}/test_images/test_images",
    f"{BASE_DIR}/cassava-leaf-disease-classification/test_images",
    f"{BASE_DIR}/cassava-leaf-disease-classification/test_images/test_images",
]

TEST_PATH = None
for c in candidate_tests:
    if os.path.isdir(c):
        TEST_PATH = c
        break

if TEST_PATH is not None:
    test_files = sorted(os.listdir(TEST_PATH))
    print("Detected test_images; using full test set.")
else:
    if KRT == "Interactive":
        print("Test run in Kaggle environment (Interactive).")
        TEST_PATH = f"{BASE_DIR}/train_images"
        if not os.path.isdir(TEST_PATH):
            TEST_PATH = f"{BASE_DIR}/cassava-leaf-disease-classification/train_images"
        if os.path.isdir(f"{TEST_PATH}/train_images"):
            TEST_PATH = f"{TEST_PATH}/train_images"
        test_files = sorted(os.listdir(TEST_PATH))[:32]
    else:
        print("No test_images found; falling back to train_images (debug-like).")
        TEST_PATH = f"{BASE_DIR}/train_images"
        if not os.path.isdir(TEST_PATH):
            TEST_PATH = f"{BASE_DIR}/cassava-leaf-disease-classification/train_images"
        if os.path.isdir(f"{TEST_PATH}/train_images"):
            TEST_PATH = f"{TEST_PATH}/train_images"
        test_files = sorted(os.listdir(TEST_PATH))[:32]

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")
assert (
    len(test_files) > 0
), "No images found in TEST_PATH; please verify dataset is mounted."

if TEST_PATH is not None and len(test_files) not in (2676, 32):
    print(
        f"[warn] Unexpected number of images in TEST_PATH: {len(test_files)} (expected 2676 on real test)."
    )



## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 10
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = torch.flatten(mixed_x, 1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## === cell 11
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = torch.flatten(mixed_x, 1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## === cell 12
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels=None, phase="test"):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        print("ここにきてはいけない")
        raise SystemExit(1)




## === cell 13
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        if img is None:
            raise FileNotFoundError(f"Could not read image: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 14
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            outputs = net(inputs, labels=None, phase="test")
            proba = torch.softmax(outputs, dim=1)
            probability.append(proba.cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    out = np.concatenate(probability, axis=0)

    if not np.isfinite(out).all():
        bad = np.sum(~np.isfinite(out))
        print(
            f"[warn] {basename}: non-finite probabilities detected: {bad}. Replacing with zeros."
        )
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out




## === cell 15
def _extract_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        return state["state_dict"]
    return state


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ("module.", "_orig_mod.", "model.", "net.", "encoder.")
    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _remap_wrapper_convlayer_keys_if_needed(net, state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    net_sd = net.state_dict()
    net_keys = set(net_sd.keys())
    new_state = dict(state_dict)

    if "convlayer.0.weight" in net_keys:
        for k in list(state_dict.keys()):
            if k in net_keys:
                continue
            tk = f"convlayer.{k}"
            if tk in net_keys and tuple(state_dict[k].shape) == tuple(net_sd[tk].shape):
                new_state[tk] = state_dict[k]

    for src_prefix in ("backbone", "encoder", "base_model"):
        for k in list(state_dict.keys()):
            if k.startswith(src_prefix + "."):
                rest = k[len(src_prefix) + 1 :]
                tk = f"convlayer.{rest}"
                if tk in net_keys and tuple(state_dict[k].shape) == tuple(
                    net_sd[tk].shape
                ):
                    new_state[tk] = state_dict[k]

    return new_state


def _remap_model_specific_keys_if_needed(net, state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    net_sd = net.state_dict()
    net_keys = set(net_sd.keys())
    new_state = dict(state_dict)

    def _map_head(src_w, src_b, dst_w, dst_b):
        sw = new_state.get(src_w, None)
        sb = new_state.get(src_b, None)
        if sw is None or sb is None:
            return
        if dst_w in net_keys and dst_b in net_keys:
            if tuple(sw.shape) == tuple(net_sd[dst_w].shape) and tuple(
                sb.shape
            ) == tuple(net_sd[dst_b].shape):
                new_state[dst_w] = sw
                new_state[dst_b] = sb

    if "fc.weight" in net_keys and "fc.bias" in net_keys:
        for src in (
            "classifier",
            "head",
            "last_linear",
            "logits",
            "output",
            "model.fc",
            "net.fc",
        ):
            _map_head(f"{src}.weight", f"{src}.bias", "fc.weight", "fc.bias")
        for src in ("classifier.1", "classifier"):
            _map_head(f"{src}.weight", f"{src}.bias", "fc.weight", "fc.bias")

    if "classifier.1.weight" in net_keys and "classifier.1.bias" in net_keys:
        for src in (
            "fc",
            "head",
            "classifier",
            "model.fc",
            "net.fc",
            "model.classifier.1",
            "net.classifier.1",
        ):
            _map_head(
                f"{src}.weight",
                f"{src}.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            )
        _map_head(
            "classifier.weight",
            "classifier.bias",
            "classifier.1.weight",
            "classifier.1.bias",
        )

    return new_state


def _remap_efficientnet_backbone_prefixes_if_needed(net, state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    net_sd = net.state_dict()
    net_keys = set(net_sd.keys())
    new_state = dict(state_dict)

    def _try_map_prefix(src_prefix, tgt_prefix):
        for k in list(new_state.keys()):
            if k.startswith(src_prefix + "."):
                tk = tgt_prefix + k[len(src_prefix) :]
                if tk in net_keys and tuple(new_state[k].shape) == tuple(
                    net_sd[tk].shape
                ):
                    new_state[tk] = new_state[k]

    _try_map_prefix("backbone.features", "features")
    _try_map_prefix("encoder.features", "features")
    _try_map_prefix("model.features", "features")
    _try_map_prefix("net.features", "features")

    return new_state


def _choose_load_target_module(net, state_dict):
    if not isinstance(state_dict, dict):
        return net
    sd_keys = list(state_dict.keys())
    if len(sd_keys) == 0:
        return net

    if hasattr(net, "model") and isinstance(net.model, nn.Module):
        net_model_keys = set(net.model.state_dict().keys())
        net_keys = set(net.state_dict().keys())
        overlap_model = sum(1 for k in sd_keys if k in net_model_keys)
        overlap_net = sum(1 for k in sd_keys if k in net_keys)
        if overlap_model > overlap_net * 1.2 and overlap_model > 50:
            return net.model
    return net


def _overlap_ratio(module, state_dict):
    if not isinstance(state_dict, dict):
        return 0.0
    mk = set(module.state_dict().keys())
    if len(mk) == 0:
        return 0.0
    return sum(1 for k in state_dict.keys() if k in mk) / float(len(mk))


def _shape_guided_best_remap(state_dict, target_module):
    if not isinstance(state_dict, dict):
        return state_dict

    tgt_sd = target_module.state_dict()
    tgt_keys = set(tgt_sd.keys())

    def _score(sd):
        m = 0
        for k, v in sd.items():
            if (
                k in tgt_keys
                and hasattr(v, "shape")
                and tuple(v.shape) == tuple(tgt_sd[k].shape)
            ):
                m += 1
        return m

    candidates = []

    base = dict(state_dict)
    candidates.append(base)

    def _rename_prefix(sd, src_prefix, dst_prefix):
        out = dict(sd)
        for k in list(sd.keys()):
            if k.startswith(src_prefix):
                nk = dst_prefix + k[len(src_prefix) :]
                out[nk] = sd[k]
        return out

    candidates.append(_rename_prefix(base, "fc.", "classifier.1."))
    candidates.append(_rename_prefix(base, "classifier.1.", "fc."))

    candidates.append(_rename_prefix(base, "classifier.", "fc."))
    candidates.append(_rename_prefix(base, "fc.", "classifier."))

    candidates.append(_rename_prefix(base, "model.classifier.1.", "classifier.1."))
    candidates.append(_rename_prefix(base, "model.fc.", "fc."))

    best = candidates[0]
    best_score = _score(best)
    for sd in candidates[1:]:
        sc = _score(sd)
        if sc > best_score:
            best, best_score = sd, sc
    return best


def _load_with_fallbacks(net, raw_state, basename=""):
    state = _extract_state_dict(raw_state)

    state_norm = _normalize_state_dict_keys(state)
    state_norm = _remap_wrapper_convlayer_keys_if_needed(net, state_norm)
    state_norm = _remap_model_specific_keys_if_needed(net, state_norm)
    state_norm = _remap_efficientnet_backbone_prefixes_if_needed(net, state_norm)

    load_target = _choose_load_target_module(net, state_norm)

    state_norm = _shape_guided_best_remap(state_norm, load_target)

    print(
        f"[diag] {basename}: overlap(net)={_overlap_ratio(net, state_norm):.3f} "
        f"overlap(target)={_overlap_ratio(load_target, state_norm):.3f}"
    )

    try:
        missing, unexpected = load_target.load_state_dict(state_norm, strict=True)
        return missing, unexpected, "normalized_bestremap_strict"
    except Exception:
        pass

    try:
        missing, unexpected = load_target.load_state_dict(state_norm, strict=False)
        return missing, unexpected, "normalized_bestremap_nonstrict"
    except Exception:
        pass

    load_target2 = _choose_load_target_module(net, state)
    try:
        missing, unexpected = load_target2.load_state_dict(state, strict=False)
        return missing, unexpected, "direct_nonstrict"
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint for {basename}: {e}")




## === cell 16
if len(pretrained_models) == 0:
    print("No pretrained models found; falling back to sample_submission.")
    sample_path = f"{BASE_DIR}/sample_submission.csv"
    if not os.path.exists(sample_path):
        sample_path = (
            f"{BASE_DIR}/cassava-leaf-disease-classification/sample_submission.csv"
        )
    df_test = pd.read_csv(sample_path)
else:
    probability = []
    start_time = time.time()

    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]

        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net = models.resnet18(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            net = models.resnet50(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            net = models.resnet152(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            net = models.densenet201(weights=None)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            net = efficientnet_b7(weights=None)
            in_features = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(in_features, num_classes)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported.")
            raise SystemExit(1)

        print(f"{basename}: {MODEL_NAME}")

        raw_state = torch.load(pretrained_model, map_location="cpu")
        missing, unexpected, mode = _load_with_fallbacks(
            net, raw_state, basename=basename
        )
        print(
            f"[load] {basename}: mode={mode} missing={len(missing)} unexpected={len(unexpected)}"
        )

        sd_keys = set(net.state_dict().keys())
        head_keys = []
        if "fc.weight" in sd_keys:
            head_keys.append("fc.weight")
        if "classifier.1.weight" in sd_keys:
            head_keys.append("classifier.1.weight")
        head_missing = [k for k in head_keys if k in missing]
        if len(head_missing) > 0:
            print(
                f"[warn] {basename}: head weights appear missing: {head_missing} -> predictions may be near-random."
            )

        total_params = len(list(net.state_dict().keys()))
        mismatch_frac = (
            ((len(missing) + len(unexpected)) / total_params)
            if total_params > 0
            else 1.0
        )
        if total_params > 0 and mismatch_frac > 0.50:
            print(
                f"[warn] Very high mismatch fraction for {basename}: "
                f"missing={len(missing)} unexpected={len(unexpected)} total={total_params} frac={mismatch_frac:.3f}"
            )
            print(f"       Example missing: {missing[:5]}")
            print(f"       Example unexpected: {unexpected[:5]}")

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = {"test": TestDataset(df_test, transform=transform_)}
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=min(8, os.cpu_count() or 1),
                    pin_memory=(device == "cuda"),
                )
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    proba_mean = np.stack(probability, axis=0).mean(axis=0)

    train_csv_path = f"{BASE_DIR}/train.csv"
    if not os.path.exists(train_csv_path):
        train_csv_path = f"{BASE_DIR}/cassava-leaf-disease-classification/train.csv"
    train_df = pd.read_csv(train_csv_path)
    train_dist = (
        train_df["label"]
        .value_counts(normalize=True)
        .reindex(range(num_classes), fill_value=0.0)
        .values
    )

    def _js_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

    candidates = []
    candidates.append(np.arange(num_classes, dtype=int))  # identity
    candidates.append(np.array([0, 2, 1, 3, 4], dtype=int))  # swap 1<->2
    candidates.append(np.array([0, 1, 2, 4, 3], dtype=int))  # swap 3<->4
    candidates.append(np.array([0, 2, 1, 4, 3], dtype=int))  # swap 1<->2 and 3<->4

    best_perm = candidates[0]
    best_score = float("inf")
    base_pred = proba_mean.argmax(axis=1)
    for perm in candidates:
        mapped = perm[base_pred]
        pred_dist = np.bincount(mapped, minlength=num_classes).astype(np.float64)
        pred_dist = pred_dist / max(pred_dist.sum(), 1.0)
        js = _js_divergence(pred_dist, train_dist)
        if js < best_score:
            best_score = js
            best_perm = perm

    print(
        f"[calib] chosen label permutation: {best_perm.tolist()} (JS={best_score:.6f})"
    )

    mapped_pred = best_perm[base_pred]
    df_test["mean"] = mapped_pred.astype(int)

    print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 17
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    sample_path = f"{BASE_DIR}/sample_submission.csv"
    if not os.path.exists(sample_path):
        sample_path = (
            f"{BASE_DIR}/cassava-leaf-disease-classification/sample_submission.csv"
        )
    df_test = pd.read_csv(sample_path)
else:
    if "mean" in df_test.columns:
        df_test["label"] = df_test["mean"].astype(int)



## === cell 18
df_test



## === cell 19
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
