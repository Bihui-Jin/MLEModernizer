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
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
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

0.852825627077667

# 6. Current score

0.52093

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51719) has done: 'I fix the runtime failure by making the weight-loading step robust to missing `/kaggle/input/cassava-pytorch-starter-train` checkpoints and instead automatically load any matching fold checkpoints if they exist. If no checkpoints are found, the code fall back to a deterministic pretrained ResNet18 backbone (same model head and inference logic) so it can still run end-to-end and produce a valid `submission.csv`. I also fix dataset indexing to avoid `.loc[idx]` pitfalls and ensure the predicted labels are written as plain integer values aligned with `sample_submission.csv` order. These changes are minimal, unblock execution, and should increase score versus random output when the provided weights are absent.'
- What this solution (achieved 0.51719) has done: 'Your score is far below the target (0.51719 vs 0.8528), so we should improve accuracy with the smallest changes that keep your architecture and inference loop intact. The biggest issue is that the `resnet18` backbone is still using its original `fc` layer (1000-way ImageNet head) during the forward pass, and your custom `self.logit` never receives proper features—this makes predictions close to random. I minimally fix the model init to replace `backbone.fc` with `nn.Identity()` so your head is applied to pooled features as intended, and I also align normalization to ImageNet mean/std when using pretrained weights (fallback path), which typically boosts accuracy without changing core logic. Everything else (data loading, TTA/ensemble structure, submission writing) stays the same and it still produce `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'Your current score gap is large (0.51719 vs target 0.85283), so we should cautiously improve accuracy without changing the core model/inference logic. The main minimal gain comes from matching the test-time preprocessing to how the fold checkpoints were trained: when using external weights, we should use the same normalization those starters typically used (0.5/0.5), while keeping ImageNet normalization for the pretrained-resnet fallback. I also make the DataLoader deterministic and increase `num_workers` a bit for stability/speed (doesn’t change semantics). Everything else (ResNet18 backbone, custom head, fold averaging, softmax+argmax, submission format) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'Your current score (0.517) is far below the target (0.853), so the most likely issue is that inference preprocessing doesn’t match what the fold checkpoints expect, which can collapse accuracy even with correct weights. I keep the model and inference loop identical, but add an automatic checkpoint-normalization detection step that tries a small set of common mean/std pairs and picks the one that best matches checkpoint behavior by measuring prediction confidence (lower average entropy) on a small subset of test images. This keeps evaluation semantics the same (still softmax+argmax, same folds, same architecture) and should move the score upward toward the target when external weights exist. If no external weights exist, the pretrained-ResNet fallback path remains unchanged and still produces a valid `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'Your current score (0.517) is far below the target (0.853), so we should make a small, low-risk accuracy improvement without changing the model architecture or inference semantics. The most likely issue is a preprocessing mismatch: these cassava fold checkpoints are commonly trained with center-crop (after resize) at inference, and only resizing can noticeably hurt accuracy. I keep your normalization auto-detection and fold-averaging exactly as-is, but change the test transforms to `Resize -> CenterCrop -> Normalize`, which is a minimal adjustment that typically moves accuracy upward for this competition. The rest of the pipeline (checkpoint loading, ResNet18+custom head, softmax+argmax, submission writing) remains unchanged and still outputs a valid `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'Your score gap is large (0.51719 vs 0.85283), so the most likely issue is that inference preprocessing is still mismatched to how the external fold checkpoints were trained, causing predictions to collapse. I keep your ResNet18+custom head, fold-averaging, and softmax→argmax semantics unchanged, but (1) extend the normalization auto-detection with the two most common “torchvision fine-tune” norms (ImageNet mean/std with inputs scaled to [0,1] is already there, plus the common 0.485/0.456/0.406 with std 0.229/0.224/0.225 is there; we add the “0.5 mean with 0.25 std” and “0.485 mean with 0.25 std” variants often seen in Cassava starters), and (2) add a minimal resize policy variant (Resize->CenterCrop already; we also try ResizeLongestSide+Pad->CenterCrop style via Albumentations `LongestMaxSize`+`PadIfNeeded`) during calibration only, selecting the combo that yields lowest entropy. This doesn’t change your core inference loop; it only chooses a better test transform when external checkpoints exist, which should move accuracy upward toward the target band while still writing a valid `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'Your score is far below the target (0.51719 vs 0.85283), so we should improve accuracy with very small, low-risk changes that preserve your ResNet18+linear-head inference semantics. The biggest likely issue in your current pipeline is that your Albumentations `Normalize` is being applied to uint8 images without explicitly scaling to `[0,1]`, which can badly break both ImageNet and ckpt normalizations and collapse accuracy; we fix this by forcing `max_pixel_value=255.0`. To better match common Cassava starter checkpoints without changing core logic, we also expand the normalization candidates with a couple of frequently used “torchvision fine-tune” variants. Everything else (architecture, fold averaging, softmax→argmax, entropy-based selection, submission format) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'Your current score (0.517) is far below the target (0.853), so we should make a small change that preserves your model/inference logic but fixes a likely preprocessing mismatch that can collapse accuracy. The biggest minimal fix is to ensure your inference images are in the same scale (0–1) that your Albumentations `Normalize(mean, std)` candidates assume; while `Normalize` *usually* divides by `max_pixel_value`, being explicit about float conversion before normalization prevents silent dtype/scale edge cases across OpenCV/Albumentations versions. I also make the entropy-calibration step use the same `num_workers`/`pin_memory` setup as the main loader for consistency (no semantic change, just stability). Everything else (ResNet18 backbone + custom head, fold averaging, softmax→argmax, submission format/paths) remains intact and still writes `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'Your current score (0.517) is far below the target (0.853), so we should make a minimal change that plausibly fixes a major accuracy killer without altering your model/inference semantics. The most likely issue is that your checkpoint loader assumes `strict=True` and a specific key layout; if the external weights exist but were saved under `state_dict`, `model`, or with a `module.` prefix, your code silently falls back to the pretrained ResNet path (or fails), yielding much lower accuracy than expected. I add a robust-but-still-safe state-dict extraction + optional `module.` prefix stripping (still strict on actual parameter shapes), so the intended fold checkpoints are actually used when present. This keeps the same architecture, transforms selection, fold averaging, and softmax→argmax submission logic intact, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.51719) has done: 'I keep your ResNet18+custom head and fold-averaging inference exactly the same, but fix a likely major accuracy killer: using plain black padding in the `LongestMaxSize+PadIfNeeded` candidate can distort inputs compared to how these Cassava checkpoints were trained. I change the padding to “reflect” (a common, safer default) and use the same padding mode consistently in both calibration and final inference when that transform is selected, which can improve checkpoint compatibility without changing model logic. I also make the checkpoint loader accept the common “nested under `model.module.`” key pattern by stripping either `module.` or `model.` prefixes, so external fold weights (if present) are actually applied instead of falling back. These are minimal, targeted changes aimed at moving accuracy upward toward your 0.8528 target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.52093) has done: 'Your score gap is large (0.51719 vs 0.85283), so we should improve accuracy with minimal, low-risk changes that keep your ResNet18+head and fold-averaging inference intact. The most likely remaining score killer is that your external checkpoints may not be getting loaded (or partially loaded) due to key mismatches, and your current code would either error or silently fall back to the pretrained backbone path, both of which can severely limit accuracy. I make checkpoint loading robust to common key-prefix/layout variants while still enforcing shape correctness, and I also ensure the inference transform uses the same image interpolation setting as typical Cassava starters (bicubic) without changing your resize/crop logic. Everything else (architecture, softmax→argmax, entropy-based transform selection, submission schema/paths) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.52093) has done: 'Your current score is far below the target, so the highest-leverage minimal change is to ensure you actually load and use the external fold checkpoints when they exist. Right now `extract_state_dict()` strips the `backbone.` prefix, which typically prevents backbone weights from matching your model keys (your model expects `backbone.conv1.*`, etc.), causing the strict loader to fail and the code to silently fall back to the pretrained-only path (low accuracy). I stop stripping `backbone.` and make the loader explicitly handle the common case where a checkpoint was saved from a plain torchvision resnet (no `backbone.` prefix) by auto-adding `backbone.` during remap. This keeps the same model, transforms, fold-averaging, and softmax→argmax semantics, but should move the score sharply upward toward your target if the external weights are present.'
- What this solution (achieved 0.52093) has done: 'I make a minimal but high-impact fix in the checkpoint loader so it doesn’t wrongly reject usable external fold checkpoints due to harmless missing keys (e.g., `backbone.fc.*` when your model uses `Identity`, or optional buffers), which currently forces a fallback to pretrained-only behavior and caps accuracy around ~0.52. The change keeps your exact model architecture, transforms, fold-averaging, and softmax→argmax semantics, but loads checkpoints with `strict=False` after remapping (still enforcing shape compatibility for all loaded tensors), so the intended weights are actually used. I also keep your entropy-based transform selection intact, just making it use the same improved loader. This should move accuracy upward substantially toward the 0.8528 target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.52093) has done: 'The current score (0.52093) is far below the target (0.85283), so we should make a minimal change that increases accuracy by ensuring the external fold checkpoints actually load instead of partially loading and behaving like an untrained head. Right now `load_checkpoint_strict()` accepts cases where the backbone loads but the classifier head (`logit.*`) is missing; this typically collapses predictions and limits accuracy. I (1) make checkpoint loading explicitly require that `logit.weight` and `logit.bias` are loaded with correct shapes, and (2) if not present, remap common head key names (`fc.*`, `classifier.*`, `head.*`, `linear.*`) and also support a common pattern where the checkpoint saved the whole torchvision resnet `fc.*` while your model uses `logit.*`. Everything else (model architecture, transforms, fold averaging, softmax→argmax, and submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved 0.52093) has done: 'Your current score (0.52093) is far below the target (0.85283), so we should make the smallest changes that plausibly restore the intended strong checkpoint inference. The highest-likelihood remaining issue is that you’re using `sample_submission.csv` (test list) for the entropy-based transform selection, which is a weak proxy and can select a bad preprocessing; instead, calibrate transforms on a small labeled *train* subset by picking the transform that yields the best accuracy for the loaded checkpoint (same model, same softmax→argmax semantics). This does not change the model architecture or the final inference loop; it only chooses a better test preprocessing when external weights exist. I also make the calibration use a deterministic, stratified subset from `train.csv` to avoid class-imbalance bias and keep runtime within limits, and keep the fallback behavior unchanged when no external checkpoints are found.'

# 9. Code solution

## === cell 0
import os
import re
import glob
import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
DIR_WEIGHTS = "/kaggle/input/cassava-pytorch-starter-train"

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 64
SIZE = 256


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)




## === cell 2
class CassavaDataset(Dataset):
    def __init__(self, df, dataset="train", transforms=None):
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        image_src = f"{DIR_INPUT}/{self.dataset}_images/{image_id}"
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image




## === cell 3
class CassavaModel(nn.Module):
    def __init__(self, num_classes=5, backbone_pretrained=False):
        super().__init__()

        weights = (
            torchvision.models.ResNet18_Weights.DEFAULT if backbone_pretrained else None
        )
        self.backbone = torchvision.models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.25, self.training)
        x = self.logit(x)
        return x




## === cell 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CKPT_MEAN = (0.5, 0.5, 0.5)
CKPT_STD = (0.5, 0.5, 0.5)

CANDIDATE_NORMS = [
    ("ckpt_0.5", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ("imagenet", IMAGENET_MEAN, IMAGENET_STD),
    ("none_0_1", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ("mean0.5_std0.25", (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    ("imagenet_std0.25", IMAGENET_MEAN, (0.25, 0.25, 0.25)),
    ("imagenet_mean_only", IMAGENET_MEAN, (1.0, 1.0, 1.0)),
    ("ckpt_mean_only", (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
]


def make_test_transforms_resize_center(mean, std):
    return A.Compose(
        [
            A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
            A.ToFloat(max_value=255.0, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


def make_test_transforms_longestpad_center(mean, std):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=SIZE, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.PadIfNeeded(
                min_height=SIZE,
                min_width=SIZE,
                border_mode=cv2.BORDER_REFLECT_101,
                value=None,
                p=1.0,
            ),
            A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
            A.ToFloat(max_value=255.0, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


transforms_test_imagenet = make_test_transforms_resize_center(
    IMAGENET_MEAN, IMAGENET_STD
)
transforms_test_ckpt = make_test_transforms_resize_center(CKPT_MEAN, CKPT_STD)



## === cell 5
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
submission_df["label"] = 0
submission_df.head()



## === cell 6
if submission_df.shape[0] == 1:
    submission_df = pd.DataFrame(
        [
            {"image_id": "2216849948.jpg", "label": 0},
            {"image_id": "2216849948.jpg", "label": 0},
        ]
    ).reset_index(drop=True)

submission_df.head()




## === cell 7
def find_checkpoints(dir_weights: str, n_folds: int):
    """
    Tries to locate fold checkpoints in the given directory.
    Supports both:
      - model_state_fold_{i}.pth
      - any *.pth that contains 'fold_{i}' in the name
    Returns a list of paths indexed by fold (None if missing).
    """
    paths_by_fold = [None] * n_folds
    if not os.path.isdir(dir_weights):
        return paths_by_fold

    for i in range(n_folds):
        p = os.path.join(dir_weights, f"model_state_fold_{i}.pth")
        if os.path.exists(p):
            paths_by_fold[i] = p

    if any(p is None for p in paths_by_fold):
        all_pths = glob.glob(os.path.join(dir_weights, "*.pth"))
        for p in all_pths:
            m = re.search(r"fold[_\-]?(\d+)", os.path.basename(p))
            if m:
                fi = int(m.group(1))
                if 0 <= fi < n_folds and paths_by_fold[fi] is None:
                    paths_by_fold[fi] = p

    return paths_by_fold


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["model_state_dict", "state_dict", "model", "net", "weights"]:
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                ckpt_obj = v
                break

        if (
            isinstance(ckpt_obj, dict)
            and "state" in ckpt_obj
            and isinstance(ckpt_obj["state"], dict)
        ):
            for k in ["state_dict", "model_state_dict", "model", "net"]:
                v = ckpt_obj["state"].get(k, None)
                if isinstance(v, dict):
                    ckpt_obj = v
                    break

    if not isinstance(ckpt_obj, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict-like object.")

    cleaned = {}
    for k, v in ckpt_obj.items():
        nk = k
        for pref in ["module.", "model.", "net."]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v

    return cleaned


def load_checkpoint_strict(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(ckpt)

    model_sd = model.state_dict()
    remapped = {}

    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            remapped[k] = v
            continue

        bk = "backbone." + k
        if bk in model_sd and model_sd[bk].shape == v.shape:
            remapped[bk] = v
            continue

        head_prefix_map = [
            ("fc.", "logit."),
            ("classifier.", "logit."),
            ("head.", "logit."),
            ("linear.", "logit."),
        ]
        for src_pref, dst_pref in head_prefix_map:
            if k.startswith(src_pref):
                alt = dst_pref + k[len(src_pref) :]
                if alt in model_sd and model_sd[alt].shape == v.shape:
                    remapped[alt] = v
                    break
        else:
            if k.startswith("logit."):
                alt = "fc." + k[len("logit.") :]
                if alt in model_sd and model_sd[alt].shape == v.shape:
                    remapped[alt] = v
                    continue

    incompatible = []
    for k, v in remapped.items():
        if k not in model_sd or model_sd[k].shape != v.shape:
            incompatible.append(k)
    if incompatible:
        raise RuntimeError(f"Incompatible keys after remap: {incompatible[:10]}")

    msg = model.load_state_dict(remapped, strict=False)

    required_head = ["logit.weight", "logit.bias"]
    missing_required = []
    for rk in required_head:
        if rk not in remapped:
            missing_required.append(rk)
    if missing_required:
        raise RuntimeError(
            f"Checkpoint did not load required head keys {missing_required}; "
            f"this would severely hurt accuracy. Example missing_keys: {list(msg.missing_keys)[:10]}"
        )

    n_loaded = len(remapped)
    if n_loaded < 20:
        raise RuntimeError(
            f"Too few parameters loaded from checkpoint ({n_loaded}). Missing: {list(msg.missing_keys)[:10]}"
        )

    return model


def stratified_sample_df(df, label_col="label", n_per_class=64, seed=42):
    rng = np.random.RandomState(seed)
    parts = []
    for lab, g in df.groupby(label_col):
        g = g.sample(n=min(len(g), n_per_class), random_state=rng)
        parts.append(g)
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def acc_for_transform_on_train(
    ckpt_path, tfm, device, df_train_small, n_samples=None, batch_size=64
):
    if n_samples is not None:
        df_use = df_train_small.iloc[: min(len(df_train_small), n_samples)].reset_index(
            drop=True
        )
    else:
        df_use = df_train_small.reset_index(drop=True)

    ds = CassavaDataset(df=df_use, dataset="train", transforms=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    y_true = df_use["label"].values.astype(int)

    model = CassavaModel(num_classes=5, backbone_pretrained=False)
    model.to(device)
    load_checkpoint_strict(model, ckpt_path=ckpt_path, device=device)
    model.eval()

    preds_all = []
    with torch.no_grad():
        for batch in dl:
            images = batch.to(device, dtype=torch.float)
            logits = model(images)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).detach().cpu()
            preds_all.append(preds)
    y_pred = torch.cat(preds_all, dim=0).numpy().astype(int)

    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0
    return float((y_true[:n] == y_pred[:n]).mean())


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
ckpt_paths = find_checkpoints(DIR_WEIGHTS, N_FOLDS)

use_folds = [i for i, p in enumerate(ckpt_paths) if p is not None]
use_external_weights = len(use_folds) > 0

transforms_test = transforms_test_imagenet

if use_external_weights:
    calib_fold = use_folds[0]
    calib_path = ckpt_paths[calib_fold]

    train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
    train_small = stratified_sample_df(
        train_df, label_col="label", n_per_class=64, seed=SEED
    )

    candidates = []
    for name, mean, std in CANDIDATE_NORMS:
        candidates.append(
            (f"{name}__resize_center", make_test_transforms_resize_center(mean, std))
        )
        candidates.append(
            (
                f"{name}__longestpad_center",
                make_test_transforms_longestpad_center(mean, std),
            )
        )

    best = None
    for cand_name, tfm in candidates:
        try:
            acc = acc_for_transform_on_train(
                ckpt_path=calib_path,
                tfm=tfm,
                device=device,
                df_train_small=train_small,
                n_samples=None,
                batch_size=min(BATCH_SIZE, 64),
            )
            if (best is None) or (acc > best["acc"]):
                best = {"name": cand_name, "tfm": tfm, "acc": acc}
        except Exception:
            continue

    if best is None:
        transforms_test = transforms_test_ckpt
    else:
        transforms_test = best["tfm"]

g = torch.Generator()
g.manual_seed(SEED)

dataset_test = CassavaDataset(
    df=submission_df, dataset="test", transforms=transforms_test
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    generator=g,
)

submissions = None

if use_external_weights:
    fold_indices = use_folds
else:
    fold_indices = [0]  # single model fallback

for i_fold in fold_indices:
    model = CassavaModel(num_classes=5, backbone_pretrained=(not use_external_weights))
    model.to(device)

    if use_external_weights:
        load_checkpoint_strict(model, ckpt_path=ckpt_paths[i_fold], device=device)

    model.eval()
    test_preds = None

    for batch in dataloader_test:
        images = batch.to(device, dtype=torch.float)

        with torch.no_grad():
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1).detach().cpu()

        if test_preds is None:
            test_preds = preds
        else:
            test_preds = torch.cat((test_preds, preds), dim=0)

    denom = len(fold_indices)
    if submissions is None:
        submissions = test_preds / denom
    else:
        submissions += test_preds / denom

submissions[:10]



## === cell 8
pred_labels = torch.argmax(submissions, dim=1).cpu().numpy().astype(int)
submission_df["label"] = pred_labels
submission_df.to_csv("submission.csv", index=False)
submission_df.head()
