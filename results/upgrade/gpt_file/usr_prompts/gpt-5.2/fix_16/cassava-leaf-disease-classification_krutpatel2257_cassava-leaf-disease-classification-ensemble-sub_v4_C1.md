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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
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

0.8952855847688124

# 6. Current score

0.18498

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the unavailable `efficientnet_pytorch` dependency and replace it with `torchvision.models.efficientnet_b4`, which preserves the intended EfficientNet-B4 backbone while fixing the import error. I also make the model weight paths robust by searching under `/kaggle/input` for the referenced `.pth` files (or gracefully falling back to untrained weights if they truly aren’t present), ensuring the notebook runs end-to-end and always writes `submission.csv`. I update the Albumentations `RandomResizedCrop` call to the v2 API (`size=(H, W)`) to fix the validation error, and I ensure consistent device placement and `torch.no_grad()` during inference to avoid runtime issues. Finally, I keep the original inference logic (TTA, weighting, softmax+argmax) intact so the score behavior is unchanged aside from necessary compatibility fixes.'
- What this solution (achieved 0.61061) has done: 'Your current 0.61099 is far below the 0.8953 target, so we should make a small change that plausibly increases accuracy without changing the core model/ensemble/TTA logic. The biggest issue is that you’re doing *random* augmentations at test time (RandomResizedCrop + rotations), which can distort leaves and hurt accuracy; we switch to a standard deterministic validation-style resize/center-crop normalization for TTA (and keep flips/transpose as the only stochastic TTA). We also batch the 10 TTA forward passes per image into a single tensor to reduce overhead and keep runtime safely under the limit, without changing the inference semantics. Everything else (two backbones, weight loading, 0.6/0.4 weighting, softmax+argmax, submission format/pathing) stays the same.'
- What this solution (achieved 0.10688) has done: 'Your current score (0.61061) is far below the target (0.89529), so we should make a small, high-impact correction that improves accuracy without changing the ensemble/TTA/training semantics. The main likely issue is an incorrect input normalization: Albumentations `Normalize` already outputs float tensors in [0,1] normalized by mean/std, but you then apply `transforms.ToTensor()` which divides by 255 again, badly scaling inputs and crushing accuracy. I replace `ToTensor()` with a small converter that turns the already-normalized HWC float image into a CHW torch tensor without rescaling. Everything else (models, weights loading, 0.6/0.4 blending, 10x TTA, softmax+argmax, submission format) remains unchanged.'
- What this solution (achieved 0.11323) has done: 'Your 0.10688 score strongly suggests the model is effectively running with random/unloaded weights (or mismatched checkpoint keys), so the most direct way toward the 0.8953 target is to (1) robustly locate the provided `.pth` files under `/kaggle/input`, (2) load them with stricter/safer key handling and clear diagnostics so you don’t silently fall back to random init, and (3) keep the exact same ensemble + TTA + softmax/argmax logic otherwise. I also ensure the input tensor is `float32` on-device (to match checkpoint expectations) and that we don’t accidentally proceed if both weight files are missing. These are minimal changes that preserve your architecture and inference semantics but should materially increase accuracy if the intended weights are present.'
- What this solution (achieved 0.23543) has done: 'I fix the immediate runtime failure by removing the hard abort when checkpoints can’t be found, and instead robustly search for the referenced weight filenames (and, if still missing, fall back to ImageNet weights to avoid near-random predictions). I also correct a small logic bug in `load_weights_safely` where an `elif` branch is unreachable, and add clearer diagnostics about what actually loaded. These changes preserve your ensemble/TTA inference logic (same models, same 0.6/0.4 blend, same 10x TTA, same softmax+argmax), but should move the score substantially upward versus random-init behavior when the custom `.pth` files aren’t present. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.09417) has done: 'Your current score (0.23543) is far below the 0.8953 target, so the most likely issue is that the intended competition-trained checkpoints are still not being loaded (so you’re effectively using ImageNet/random heads). I make a minimal, score-relevant fix: after loading a checkpoint, verify that a meaningful fraction of parameters actually changed; if not, treat it as “not loaded” and fall back to the best available alternative. Additionally, EfficientNet-B4 in `torchvision` expects 380×380 inputs, so I change only the inference resize/crop from 512 to 380 to better match the backbone’s intended input scaling without changing your ensemble/TTA logic. Everything else—models, heads, 0.6/0.4 blending, 10x TTA, and softmax+argmax—remains the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the most likely score-killer is that neither competition-trained checkpoint is actually being found/loaded (so you’re effectively predicting with mismatched/random heads). I make a minimal, score-relevant fix by resolving those weight paths robustly: instead of only searching by basename, we also search by the *dataset directory name* (the part after `../input/`) so Kaggle’s mounted paths are matched correctly. I also keep the same models/ensemble/TTA and add a small guard that, if custom checkpoints are missing, we avoid using ImageNet weights with a randomly-initialized 5-class head (which tends to be confidently wrong) and instead use the provided `sample_submission` label distribution as a safe prior fallback; this should lift accuracy from ~0.09 toward the target when checkpoints are unavailable. The submission format and file path (`submission.csv`) remain unchanged.'
- What this solution (achieved 0.1136) has done: 'Your current score (0.11584) is far below the target (0.8953), so the most likely issue is that the intended competition-trained checkpoints are still not being loaded (or their keys aren’t being mapped onto your modified 5-class heads), leaving you with essentially random/incorrect heads. I make a minimal, score-relevant change to the weight-loading: explicitly handle common “wrapped” checkpoints and, crucially, auto-remap classifier keys when a checkpoint was saved before `torchvision`’s layer naming differences (e.g., `classifier.1.*` vs `classifier.0.*`), without changing the architecture or inference logic. I also remove the “label-prior fallback” (which caps accuracy near the class frequency, ~0.11) and instead always run inference with the best-available weights (custom if loadable; otherwise ImageNet backbone + random head as a last resort), because your target requires the actual trained head to be loaded. Finally, I keep the same ensemble/TTA/softmax+argmax semantics and still write a valid `submission.csv`.'
- What this solution (achieved 0.10575) has done: 'Your score (0.1136) is far below the target (0.8953), which strongly indicates the custom 5-class checkpoints still aren’t being correctly applied to the modified torchvision models (so you’re effectively predicting with a random head). I make a minimal, score-relevant change: when loading checkpoints for both ResNeXt and EfficientNet, automatically remap common classifier/fc key patterns (e.g., `fc.*` vs `classifier.*`, `classifier.1.*` vs `classifier.0.*`, and `model.fc.*` prefixes) and select the candidate state_dict that yields the best key match to the current model. I also add a small (non-invasive) diagnostic that prints whether the final layer weights actually changed after loading, to avoid silently proceeding with random heads. The model architectures, ensemble weighting (0.6/0.4), TTA count, and softmax+argmax submission semantics remain unchanged.'
- What this solution (achieved 0.1136) has done: 'Your very low accuracy (0.10575 vs target 0.8953) is most consistent with “custom checkpoints still not actually loading”, so the smallest score-moving change is to make weight loading robust to common checkpoint formats (including `{'model': ...}` and DataParallel prefixes) *without changing the model architecture or inference logic*. I adjust `load_weights_safely` to: (1) try a wider set of unwrap keys, (2) strip prefixes per-key (not only if *all* keys share it), and (3) handle the frequent case where the checkpoint contains a full wrapper state_dict including optimizer/epoch etc. Then I keep your exact ensemble (0.6/0.4), TTA count (10), augmentations, and softmax+argmax submission semantics unchanged, only adding a hard diagnostic print of how many tensors actually matched/loaded so you can confirm it’s no longer silently falling back.'
- What this solution (achieved 0.08408) has done: 'Your score (0.1136) is far below the target (0.8953), and given your current code, the most likely cause is that the custom trained checkpoints are still not being found/loaded, so you’re effectively using ImageNet backbones with a random 5-class head. I make a minimal, score-relevant change to `resolve_path` so it can also locate checkpoints that are stored as `.pth` files *inside competition datasets’ “output” folders* (a common Kaggle packaging pattern), by searching for files with the same basename anywhere under `/kaggle/input`. I also add a strict guard: if neither custom checkpoint loads successfully, fail fast instead of producing a low-accuracy submission—this prevents wasting submissions and makes it obvious when the real issue is missing weights (and once weights are found, accuracy should jump toward your target without changing inference logic). No changes to model architectures, ensemble weights, TTA count, augmentations, or softmax/argmax semantics.'
- What this solution (achieved 0.10762) has done: 'I remove the hard fail-fast that aborts execution when custom `.pth` checkpoints aren’t found, because it prevents generating any valid submission and causes the current runtime error. Instead, the code continue with the existing initialization (ImageNet weights when available) and still write `submission.csv` in the correct format, so you can at least submit and get a measurable score. I also make the weight-path resolver accept Kaggle-style `../input/...` paths by mapping them to `/kaggle/input/...`, which is a minimal, score-relevant fix that can allow the intended checkpoints to be found/loaded without changing your model or inference logic. All model architectures, ensemble weighting (0.6/0.4), TTA count, and softmax+argmax submission semantics remain unchanged.'
- What this solution (achieved 0.53363) has done: 'Your current score (0.10762) is far below the target (0.8953), which most strongly indicates the ensemble is effectively untrained for 5 classes (custom checkpoints not being loaded), causing near-random predictions. The smallest score-moving change is to make checkpoint resolution more robust by also matching files by **basename** and by **dataset folder name** even when the given path contains characters like parentheses, and to additionally search for common alternate extensions/filenames in the referenced dataset folders. I also make the loader accept checkpoints saved as full training dicts (e.g., `{'model': ..., 'state_dict': ...}`) and ensure we don’t accidentally reject a valid load just because the sampled “delta ratio” is small (which can happen if many layers are frozen/identical); instead we prioritize key+shape matches and head compatibility. Core model architectures, ensemble weighting (0.6/0.4), TTA count (10), augmentations, and softmax+argmax submission semantics are kept intact, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.27578) has done: 'Your score is far below the target, and the most likely remaining “score killer” is that your current TTA pipeline still uses random transpose+flips, which can *decrease* accuracy when you average logits (especially if the model wasn’t trained with those invariances). I make the smallest score-relevant change: keep the exact same ensemble, weights, softmax/argmax, and 10-pass averaging, but switch test-time augmentation to a deterministic 10-view scheme (original + HFlip + VFlip + HVFlip, repeated to reach 10) so inference is stable and typically closer to the validation-style preprocessing. I also fix one small semantic issue: apply softmax per-model before averaging (probability-level ensembling), which is a minimal post-processing change that often improves accuracy without changing models or training. Everything else (paths, weight-loading logic, EfficientNet-B4 + ResNeXt50, 0.6/0.4 weighting, submission format) stays the same and still produces `submission.csv`.'
- What this solution (achieved 0.18498) has done: 'Your current gap to the target is large (0.27578 vs 0.89529), so we need a small but high-impact fix that keeps your ensemble/TTA logic intact. The most likely score-killer is that your custom checkpoints are still not being applied because the loader rejects them with a hard “matchcount < 50” rule that can easily fail depending on checkpoint format (e.g., EMA-only, partial, different key names) even when the classifier head actually loads. I relax that rejection criterion to accept loads when the **final 5-class head weights changed** and at least a minimal number of tensors match, which preserves the same models/inference but makes it far more likely the trained heads are used. I also ensure the input tensor is contiguous (minor stability) and keep everything else (models, weights search, deterministic 10-view TTA, prob-ensemble then mean, submission writing) identical.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
from torchvision import models

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v3/eff_epoch_11.pth"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"




## === cell 2
def resolve_path(preferred_path: str):
    """
    Score-relevant robustness:
    If custom checkpoints are not found, accuracy collapses. Make path resolution robust to:
      - ../input -> /kaggle/input mapping
      - dataset-folder + inner path matching
      - basename search anywhere under /kaggle/input
      - common alternate extensions for checkpoints (pt/pth/bin)
    """
    if not preferred_path:
        return None

    norm = preferred_path.replace("\\", "/").strip()

    if norm.startswith("../input/"):
        norm = "/kaggle/input/" + norm[len("../input/") :]
    elif norm.startswith("/kaggle/working/../input/"):
        norm = "/kaggle/input/" + norm[len("/kaggle/working/../input/") :]
    elif norm.startswith("/kaggle/input/"):
        pass

    if os.path.exists(norm):
        return norm

    parts = [p for p in norm.split("/") if p not in ("", ".")]

    if "input" in parts:
        i = parts.index("input")
        if i + 1 < len(parts):
            dataset = parts[i + 1]
            inner = "/".join(parts[i + 2 :]) if (i + 2) < len(parts) else ""
            if dataset:
                if inner:
                    cand = os.path.join("/kaggle/input", dataset, inner)
                    if os.path.exists(cand):
                        return cand

                base = os.path.basename(norm)
                if base:
                    cands = glob.glob(
                        f"/kaggle/input/{dataset}/**/{base}", recursive=True
                    )
                    if cands:
                        cands = sorted(cands, key=lambda p: (len(p), p))
                        return cands[0]

                base = os.path.basename(norm)
                root, ext = os.path.splitext(base)
                if root:
                    for ex in [".pth", ".pt", ".bin"]:
                        cands = glob.glob(
                            f"/kaggle/input/{dataset}/**/{root}{ex}", recursive=True
                        )
                        if cands:
                            cands = sorted(cands, key=lambda p: (len(p), p))
                            return cands[0]

    base = os.path.basename(norm)
    if base:
        candidates = glob.glob(f"/kaggle/input/**/{base}", recursive=True)
        if candidates:
            candidates = sorted(candidates, key=lambda p: (len(p), p))
            return candidates[0]

        root, ext = os.path.splitext(base)
        if root:
            for ex in [".pth", ".pt", ".bin"]:
                candidates = glob.glob(f"/kaggle/input/**/{root}{ex}", recursive=True)
                if candidates:
                    candidates = sorted(candidates, key=lambda p: (len(p), p))
                    return candidates[0]

    return None


resnet_model_path_resolved = resolve_path(resnet_model_path)
effnet_model_path_resolved = resolve_path(effnet_model_path)

print("Resolved resnet weights:", resnet_model_path_resolved)
print("Resolved effnet weights:", effnet_model_path_resolved)

if not os.path.exists(sample_sub_path):
    sample_sub_path_alt = resolve_path(sample_sub_path)
    if sample_sub_path_alt:
        sample_sub_path = sample_sub_path_alt

if not os.path.exists(test_images_path):
    test_images_path_alt = resolve_path(test_images_path)
    if test_images_path_alt:
        test_images_path = test_images_path_alt
    else:
        dirs = glob.glob("/kaggle/input/**/test_images", recursive=True)
        if dirs:
            test_images_path = sorted(dirs, key=lambda p: (len(p), p))[0]

print("sample_sub_path:", sample_sub_path)
print("test_images_path:", test_images_path)
assert os.path.exists(
    sample_sub_path
), f"sample_submission.csv not found at {sample_sub_path}"
assert os.path.isdir(
    test_images_path
), f"test_images directory not found at {test_images_path}"



## === cell 3
use_resnet_pretrained = resnet_model_path_resolved is None
use_effnet_pretrained = effnet_model_path_resolved is None

resnet_weights = (
    models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if use_resnet_pretrained else None
)
effnet_weights = (
    models.EfficientNet_B4_Weights.IMAGENET1K_V1 if use_effnet_pretrained else None
)

resnet_model = models.resnext50_32x4d(weights=resnet_weights)
resnet_model.fc = nn.Linear(2048, 5)
resnet_model.to(DEVICE)

effnet_model = models.efficientnet_b4(weights=effnet_weights)
in_features = effnet_model.classifier[-1].in_features
effnet_model.classifier[-1] = nn.Linear(in_features, 5)
effnet_model.to(DEVICE)


def _strip_prefix_if_present_allkeys(state_dict, prefix: str):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(isinstance(k, str) and k.startswith(prefix) for k in keys):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _strip_prefix_per_key(state_dict, prefixes=("module.", "model.", "net.")):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        if isinstance(nk, str):
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
        out[nk] = v
    return out


def _state_dict_delta_ratio(
    before_sd: dict, after_sd: dict, sample_k: int = 50
) -> float:
    common = [
        k
        for k in before_sd.keys()
        if k in after_sd
        and torch.is_tensor(before_sd[k])
        and torch.is_tensor(after_sd[k])
    ]
    if not common:
        return 0.0
    common = common[: min(sample_k, len(common))]
    deltas = []
    for k in common:
        b = before_sd[k].detach().float().cpu()
        a = after_sd[k].detach().float().cpu()
        denom = b.abs().mean().item() + 1e-8
        deltas.append((a - b).abs().mean().item() / denom)
    return float(np.mean(deltas)) if deltas else 0.0


def _unwrap_checkpoint_to_state_dict(ckpt):
    """
    Score-relevant:
    Many Kaggle checkpoints are wrapped (epoch/optimizer/etc). Collect likely state_dicts.
    """
    candidates = []
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema",
            "ema_state_dict",
            "student",
            "teacher",
            "model_ema",
            "params",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                candidates.append(ckpt[k])

        for k in list(ckpt.keys()):
            v = ckpt.get(k)
            if isinstance(v, dict):
                for kk in ["state_dict", "model_state_dict", "model", "net"]:
                    if kk in v and isinstance(v[kk], dict):
                        candidates.append(v[kk])

        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            candidates.append(ckpt)

    uniq = []
    seen = set()
    for c in candidates:
        if id(c) not in seen:
            uniq.append(c)
            seen.add(id(c))
    return uniq


def _remap_effnet_classifier_keys(sd: dict, model: nn.Module) -> dict:
    if not isinstance(sd, dict) or not sd:
        return sd

    model_keys = set(model.state_dict().keys())
    has_ckpt_1 = ("classifier.1.weight" in sd) or ("classifier.1.bias" in sd)
    has_ckpt_0 = ("classifier.0.weight" in sd) or ("classifier.0.bias" in sd)

    model_has_1 = ("classifier.1.weight" in model_keys) or (
        "classifier.1.bias" in model_keys
    )
    model_has_0 = ("classifier.0.weight" in model_keys) or (
        "classifier.0.bias" in model_keys
    )

    if has_ckpt_1 and model_has_0 and (not model_has_1):
        sd = dict(sd)
        for suf in ["weight", "bias"]:
            k1 = f"classifier.1.{suf}"
            k0 = f"classifier.0.{suf}"
            if k1 in sd and k0 not in sd:
                sd[k0] = sd.pop(k1)
        return sd

    if has_ckpt_0 and model_has_1 and (not model_has_0):
        sd = dict(sd)
        for suf in ["weight", "bias"]:
            k0 = f"classifier.0.{suf}"
            k1 = f"classifier.1.{suf}"
            if k0 in sd and k1 not in sd:
                sd[k1] = sd.pop(k0)
        return sd

    return sd


def _remap_classifier_fc_between_families(sd: dict, model_kind: str) -> dict:
    if not isinstance(sd, dict) or not sd:
        return sd

    sd = dict(sd)

    if model_kind.lower().startswith("res"):
        if ("fc.weight" not in sd) and any(
            isinstance(k, str) and k.startswith("classifier.") for k in sd.keys()
        ):
            for src in ["classifier.1", "classifier.0", "classifier"]:
                w = f"{src}.weight"
                b = f"{src}.bias"
                if w in sd and b in sd:
                    sd["fc.weight"] = sd.pop(w)
                    sd["fc.bias"] = sd.pop(b)
                    break
        return sd

    if model_kind.lower().startswith("eff"):
        if (
            ("classifier.1.weight" not in sd)
            and ("classifier.0.weight" not in sd)
            and ("fc.weight" in sd)
        ):
            sd["classifier.1.weight"] = sd.pop("fc.weight")
            if "fc.bias" in sd:
                sd["classifier.1.bias"] = sd.pop("fc.bias")
        return sd

    return sd


def _final_layer_changed(model: nn.Module, before_sd: dict) -> bool:
    msd = model.state_dict()
    candidates = []
    for k in ["fc.weight", "classifier.1.weight", "classifier.0.weight"]:
        if k in msd and k in before_sd:
            candidates.append(k)
    if not candidates:
        return True
    for k in candidates:
        if not torch.allclose(msd[k].detach().cpu(), before_sd[k].detach().cpu()):
            return True
    return False


def _count_tensor_key_matches(model: nn.Module, sd: dict) -> int:
    msd = model.state_dict()
    n = 0
    for k, v in sd.items():
        if (
            k in msd
            and torch.is_tensor(v)
            and torch.is_tensor(msd[k])
            and (v.shape == msd[k].shape)
        ):
            n += 1
    return n


def load_weights_safely(model, path, model_kind: str = ""):
    if not path or not os.path.exists(path):
        print(
            f"INFO: No custom weights found at {path}; keeping current initialization."
        )
        return False

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    ckpt = torch.load(path, map_location="cpu")
    candidates = _unwrap_checkpoint_to_state_dict(ckpt)
    if not candidates:
        print(
            f"WARNING: Unrecognized checkpoint format at {path}; keeping current initialization."
        )
        return False

    best_sd = None
    best_rank = None
    best_matchcount = None

    for sd in candidates:
        sd2 = sd
        sd2 = _strip_prefix_if_present_allkeys(sd2, "module.")
        sd2 = _strip_prefix_if_present_allkeys(sd2, "model.")
        sd2 = _strip_prefix_if_present_allkeys(sd2, "net.")
        sd2 = _strip_prefix_per_key(sd2, prefixes=("module.", "model.", "net."))

        sd2 = _remap_classifier_fc_between_families(sd2, model_kind=model_kind)
        if model_kind.lower().startswith("eff"):
            sd2 = _remap_effnet_classifier_keys(sd2, model)

        matchcount = _count_tensor_key_matches(model, sd2)

        incompatible = model.load_state_dict(sd2, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        head_changed = _final_layer_changed(model, before)

        rank = (-matchcount, -int(head_changed), len(missing), len(unexpected))

        if (best_rank is None) or (rank < best_rank):
            best_rank = rank
            best_sd = sd2
            best_matchcount = matchcount

        model.load_state_dict(before, strict=True)

    incompatible = model.load_state_dict(best_sd, strict=False)
    head_changed = _final_layer_changed(model, before)
    after = model.state_dict()
    delta_ratio = _state_dict_delta_ratio(before, after, sample_k=60)

    print(f"Loaded custom weights from: {path}")
    print(f"  Matched tensors (by key+shape): {best_matchcount}")
    print(
        f"  Missing keys: {len(list(incompatible.missing_keys))} | Unexpected keys: {len(list(incompatible.unexpected_keys))}"
    )
    print(f"  Approx. param delta ratio (sampled): {delta_ratio:.6f}")
    print(f"  Final layer changed check: {head_changed}")

    total_params = len(model.state_dict().keys())
    if total_params > 0 and len(list(incompatible.missing_keys)) > 0.9 * total_params:
        print(
            "WARNING: Checkpoint appears incompatible (too many missing keys). Keeping current initialization."
        )
        model.load_state_dict(before, strict=True)
        return False

    if (not head_changed) or (best_matchcount is None) or (best_matchcount < 2):
        print(
            "WARNING: Checkpoint load looks ineffective (unchanged head and/or too few matched tensors). Keeping current initialization."
        )
        model.load_state_dict(before, strict=True)
        return False

    return True


loaded_resnet = load_weights_safely(
    resnet_model, resnet_model_path_resolved, model_kind="resnet"
)
loaded_effnet = load_weights_safely(
    effnet_model, effnet_model_path_resolved, model_kind="effnet"
)

resnet_model.eval()
effnet_model.eval()

print("Loaded flags -> resnet:", loaded_resnet, "| effnet:", loaded_effnet)

if not (loaded_resnet or loaded_effnet):
    print(
        "WARNING: Neither custom checkpoint was loaded. Continuing with current initialization "
        "(ImageNet weights if available, otherwise random head). This will likely score low."
    )




## === cell 4
def albumentations_image_to_tensor(img: np.ndarray) -> torch.Tensor:
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)




## === cell 5
INFER_SIZE = 380

base_aug = A.Compose(
    [
        A.Resize(height=INFER_SIZE, width=INFER_SIZE),
        A.CenterCrop(height=INFER_SIZE, width=INFER_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



## === cell 6
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 7
sample_sub = pd.read_csv(sample_sub_path)

tta_count = 10
tta_ops = [
    ("id", lambda x: x),
    ("hflip", lambda x: np.ascontiguousarray(x[:, ::-1, :])),
    ("vflip", lambda x: np.ascontiguousarray(x[::-1, :, :])),
    ("hvflip", lambda x: np.ascontiguousarray(x[::-1, ::-1, :])),
]
tta_plan = [tta_ops[i % len(tta_ops)] for i in range(tta_count)]

predictions = []

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        tta_batch = []
        for _, op in tta_plan:
            aug_in = op(img_np)
            aug_img = base_aug(image=aug_in)["image"]
            tta_batch.append(albumentations_image_to_tensor(aug_img))
        tta_batch = (
            torch.stack(tta_batch, dim=0)
            .contiguous()
            .to(device=DEVICE, dtype=torch.float32)
        )

        res_logits = resnet_model(tta_batch).detach().cpu().numpy()
        eff_logits = effnet_model(tta_batch).detach().cpu().numpy()

        res_prob = softmax(res_logits, axis=1)
        eff_prob = softmax(eff_logits, axis=1)

        image_prob = (0.6 * res_prob + 0.4 * eff_prob).mean(axis=0, keepdims=True)
        predictions.extend(image_prob)

predictions = np.asarray(predictions).argmax(axis=1)
sub_df = pd.DataFrame(
    {"image_id": sample_sub.image_id, "label": predictions.astype(int)}
)
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Custom checkpoint loaded? resnet:", loaded_resnet, "| effnet:", loaded_effnet)
