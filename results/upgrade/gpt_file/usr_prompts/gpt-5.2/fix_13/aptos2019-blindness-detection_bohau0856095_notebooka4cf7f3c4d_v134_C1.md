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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.9

# 3. Installed packages

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
timm==1.0.19
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8976533744850824

# 6. Current score

-0.07353

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'You’re failing because the notebook expects external weight files that aren’t present (`../input/weights/...`) and it hard-forces CUDA on a CPU-only runtime, causing a crash before any predictions are generated (hence the empty submission). I (1) auto-detect CPU/GPU, (2) robustly locate the dataset under `/kaggle/input/...` while keeping the same relative paths style, and (3) load weights only if they exist; otherwise, run the exact same model forward pass with random-initialized weights so the pipeline completes and writes a valid `submission.csv`. I also fix a couple of CPU/GPU tensor device bugs in the helper functions and a Python string comparison bug (`is` vs `==`) that can break transforms. These changes are execution/stability fixes; without the provided pretrained weights, the score won’t reach the target, but you get a valid submission file end-to-end.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because this pipeline is effectively producing near-random predictions when the expected pretrained weights are missing, so the smallest meaningful step toward your target is to reliably load a real pretrained checkpoint if it exists anywhere in the dataset tree (instead of only checking two hardcoded paths). I keep your exact model and prediction logic, but broaden the weight search to recursively scan `/kaggle/input` and `/kaggle/data` for the expected `.pkl`, and I also harden image loading (handle `.jpeg/.jpg` if present) so the loop never silently fails mid-run. Finally, I ensure we always write a valid `submission.csv` and keep device handling safe (CPU/GPU) without changing evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from effectively random predictions because the notebook usually can’t find/load the intended pretrained weights, so the smallest change that meaningfully increases score toward your target is to (1) make weight loading robust to common checkpoint formats (raw state_dict vs wrapped dicts, DataParallel `module.` prefix) and (2) fall back to using `transform2` (ImageNet normalization) only if we detect the loaded checkpoint was trained with that preprocessing (simple heuristic), otherwise keep your existing transform. These changes preserve your exact model and prediction logic, but greatly increase the chance that the real pretrained weights (if present anywhere) actually get applied correctly. I also add a deterministic, safe per-image error fallback (predict 0) so submission generation never aborts mid-loop, which avoids invalid/partial submissions that can also lead to a 0.0 outcome.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is because the pipeline is still effectively producing random predictions when the intended pretrained weights aren’t present/aren’t being correctly loaded into the exact inference model you use (`ThreeStage_Model`, predicting from its regressor head). The smallest change that can genuinely move score toward your target is to robustly locate and correctly load the checkpoint even if it’s saved under common alternate names (e.g., `.pth/.pt`) and formats (wrapped dicts, `module.` prefix), and to verify that a meaningful fraction of parameters actually match before trusting the load. If no compatible weights are found, we still produce a valid submission (as before), but if weights do exist in the environment this change makes it far more likely they are actually applied—raising QWK substantially. I also clamp the final predicted class to `[0,4]` in `regress2class` to avoid rare out-of-range behavior from thresholding on malformed tensors, without changing the core model or metric semantics.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with predictions being effectively random due to missing/incompatible weights; the smallest legitimate step toward the 0.8976 target is to make checkpoint loading actually work when the weights exist in any common format/path. I keep your exact model and inference path (use `ThreeStage_Model` regressor head + `regress2class`), but broaden weight discovery to accept common filename variants and then load the checkpoint more robustly (handle wrapped dicts, `module.` prefix, and non-strict partial loads). To avoid accidentally “loading” the wrong file and hurting score, I verify compatibility by counting matched keys and only switching to the ImageNet preprocessing when compatibility is high. This preserves core logic and still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is almost certainly because no real pretrained weights are actually being loaded, so predictions are effectively random; the minimal legitimate way to move toward the 0.8976 target is to make weight discovery and loading succeed when the checkpoint exists in the Kaggle filesystem. I (1) broaden the checkpoint search to include common filename variants and prefer larger/more plausible weight files, and (2) make checkpoint loading robust to more save formats (nested dicts and `ema`/`model_ema`, `module.` prefixes) while keeping your exact model/inference logic unchanged. I also ensure we never “pretend” a bad load succeeded: we only accept a checkpoint when enough keys match and otherwise fall back exactly as before, still producing a valid `submission.csv`. These are stability/weight-loading fixes only; no architecture, loss, or inference semantics are changed.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with either (a) an invalid submission (misaligned ids/order/type) or (b) effectively random predictions because weights aren’t actually loaded into the inference model. To move toward the target with minimal risk and without changing your model/inference semantics, I (1) guarantee the submission rows exactly follow `sample_submission.csv` order (this avoids silent id/order mismatches that can yield very low/0 scores), and (2) make checkpoint selection stricter by preferring files whose names strongly match the expected weight name and verifying a higher key-match ratio before accepting them (reducing the chance of loading an incompatible checkpoint that behaves randomly). If no compatible checkpoint exists, the code still produces a valid `submission.csv` end-to-end, but this change maximizes the chance of using the intended weights when they are present.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with either (1) an “all-one-class” style submission due to random/unloaded weights or (2) an invalid/misaligned submission; your current code already writes a valid, correctly ordered CSV, so the smallest score-relevant improvement is to greatly increase the chance that a real pretrained checkpoint (if present) is actually found and correctly loaded. I keep your exact model and inference path unchanged, but (a) broaden checkpoint discovery to also consider `.ckpt` files and common directory tokens, (b) make state_dict extraction handle Lightning-style `state_dict` with prefixes like `net1.`/`model.` in addition to `module.`, and (c) select the best checkpoint by maximizing matched-parameter ratio rather than filename/size alone. If no compatible weights exist, behavior remains the same (valid submission with low score), but if they do exist this should move your score substantially toward the 0.8976 target without changing core logic.'
- What this solution (achieved 0.0) has done: 'I keep your exact model and inference path, but fix the one thing that can keep the score at ~0 even when the right checkpoint exists: you currently load only a *filtered* subset of matching keys, which can silently leave many weights randomly initialized and produce near-random predictions. I change checkpoint loading to prefer a full strict load when possible (after prefix-stripping), and only fall back to partial loading if strict fails—this preserves architecture/semantics while materially increasing the chance the intended pretrained weights are actually applied. I also pick the best checkpoint by testing strict-load feasibility (not just key-shape match ratio), and I remove the heuristic that rejects a checkpoint even after a successful load. These are minimal, score-relevant changes and still always write a valid `submission.csv` in sample order.'
- What this solution (achieved 0.00047) has done: 'Your 0.0 score is overwhelmingly likely coming from running inference with random/uninitialized weights (because no checkpoint is actually present/compatible), not from the submission formatting (which is already correct). The smallest score-moving change that preserves your model/inference semantics is to make the EfficientNet backbones load ImageNet weights **only when your task checkpoint is missing**, so the regressor head isn’t fed totally random features. I keep the same `ThreeStage_Model` forward path (`_, r_out, _` + `regress2class`) and do not add any training; this is just a safer initialization fallback to move QWK upward toward your target. I also keep your robust checkpoint search/load logic, but only switch to the ImageNet-normalized `transform2` when we’re using ImageNet-pretrained features (either via a found checkpoint or via the new fallback), to avoid hurting performance by mismatched preprocessing.'
- What this solution (achieved 0.05403) has done: 'Your current 0.00047 is far below the 0.8976 target, and the biggest remaining score limiter is that even when a task checkpoint is found, the inference preprocessing may be mismatched (you force ImageNet normalization whenever *any* checkpoint loads, but many DR checkpoints are trained with custom mean/std and resize). I keep your model and inference path identical, but make transform selection depend on the actual checkpoint identity: if we load the expected `B4_3stage_22epoch_320*` checkpoint (high confidence), use your original `transform1`; otherwise, only use `transform2` when we explicitly fall back to ImageNet-pretrained backbone. I also ensure we use the exact same resize as training for the task checkpoint case, which should move QWK upward toward the target without changing architecture/loss/training. Submission formatting and ordering stay anchored to `sample_submission.csv` to avoid any alignment-related score collapse.'
- What this solution (achieved -0.07353) has done: 'Your score gap to the 0.8976 target is very large, and the only minimal, score-relevant lever available without changing your architecture/training is making sure inference uses the *same* preprocessing the checkpoint was trained with (and avoiding a subtle but severe bug in your EfficientNet stem forward). I (1) fix the forward pass in `backboneNet_efficient` (it incorrectly feeds `x1` into `block0`, which breaks feature flow if that class is ever used by a loaded checkpoint), and (2) automatically choose the inference transform based on the checkpoint’s expected input resolution (derived from filename tokens like `320`) rather than assuming one fixed resize. This keeps your model/inference semantics the same (still using `ThreeStage_Model` regressor head + `regress2class`), but should move QWK upward if the correct checkpoint is present while remaining safe if it isn’t. Submission creation stays anchored to `sample_submission.csv` order to avoid any alignment-related score collapse.'

# 9. Code solution

## === cell 0
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm
import os
import glob
import warnings

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def _find_aptos_root():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for base in ["/kaggle/input", "/kaggle/data", "../input", "../data"]:
        if os.path.exists(base):
            hits = glob.glob(
                os.path.join(base, "**", "aptos2019-blindness-detection"),
                recursive=True,
            )
            for h in hits:
                if os.path.exists(h):
                    return h
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection directory in the filesystem."
    )


APTOS_ROOT = _find_aptos_root()
print("APTOS_ROOT:", APTOS_ROOT)


def _resolve_image_path(img_root: str, id_code: str):
    for ext in (".png", ".jpeg", ".jpg", ".PNG", ".JPEG", ".JPG"):
        p = os.path.join(img_root, f"{id_code}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(img_root, f"{id_code}.png")


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "ema",
            "model_ema",
            "model_ema_state_dict",
        ]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                ckpt_obj = ckpt_obj[k]
                break

    if isinstance(ckpt_obj, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt_obj.values()
    ):
        return ckpt_obj
    return ckpt_obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _strip_any_prefixes_to_match(state_dict, model_keys):
    """
    Score-relevant: handle common checkpoint key prefixes (Lightning 'model.', 'net.', etc.)
    so the intended pretrained weights can actually be loaded.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    if any(k in model_keys for k in state_dict.keys()):
        return state_dict

    prefixes = []
    for k in state_dict.keys():
        if "." in k:
            prefixes.append(k.split(".", 1)[0] + ".")
    prefixes = sorted(set(prefixes), key=len, reverse=True)

    for pref in prefixes:
        stripped = {
            k[len(pref) :]: v for k, v in state_dict.items() if k.startswith(pref)
        }
        if stripped and any(k in model_keys for k in stripped.keys()):
            return stripped

    stripped2 = {}
    for k, v in state_dict.items():
        parts = k.split(".")
        if len(parts) >= 3:
            stripped_k = ".".join(parts[2:])
            stripped2[stripped_k] = v
    if stripped2 and any(k in model_keys for k in stripped2.keys()):
        return stripped2

    return state_dict


def _find_weight_file_fuzzy(basenames, prefer_token=None):
    roots = ["/kaggle/input", "/kaggle/data", "../input", "../data"]
    exts = [".pkl", ".pth", ".pt", ".bin", ".ckpt"]
    patterns = []

    for b in basenames:
        b = str(b)
        root_b, ext_b = os.path.splitext(b)
        if ext_b.lower() in exts:
            patterns.append(b)
            patterns.append(root_b)
        else:
            patterns.append(b)
            patterns.append(b.replace("_", "-"))
            patterns.append(b.replace("-", "_"))
            for e in exts:
                patterns.append(b + e)
                patterns.append(b.replace("_", "-") + e)
                patterns.append(b.replace("-", "_") + e)

    candidates = []
    for r in roots:
        if not os.path.exists(r):
            continue
        for pat in patterns:
            hits = glob.glob(os.path.join(r, "**", pat), recursive=True)
            if not hits:
                hits = glob.glob(os.path.join(r, "**", f"*{pat}*"), recursive=True)
            for h in hits:
                if os.path.isfile(h):
                    try:
                        sz = os.path.getsize(h)
                    except OSError:
                        sz = -1
                    name = os.path.basename(h).lower()
                    prefer = 0
                    if prefer_token is not None and str(prefer_token).lower() in name:
                        prefer = 1
                    candidates.append((h, sz, prefer))

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda t: (-t[2], -t[1], len(os.path.basename(t[0])), len(t[0]), t[0]),
    )
    best_path, best_size, best_prefer = candidates[0]
    print(
        f"Weight candidate selected (by fuzzy name/size): {best_path} (prefer={best_prefer}, size={best_size/1024/1024:.2f} MB)"
    )
    return best_path


def _prepare_state_for_model(model: nn.Module, weight_path: str):
    ckpt = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _strip_module_prefix(state)
    model_state = model.state_dict()
    state = _strip_any_prefixes_to_match(state, set(model_state.keys()))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint did not resolve to a state_dict-like mapping.")
    return state


def _can_strict_load(model: nn.Module, weight_path: str):
    """
    Score-relevant: prefer checkpoints that can be loaded strictly into THIS model.
    """
    try:
        state = _prepare_state_for_model(model, weight_path)
        _ = model.load_state_dict(state, strict=True)
        return True
    except Exception:
        return False


def _checkpoint_match_ratio(model: nn.Module, weight_path: str):
    state = _prepare_state_for_model(model, weight_path)
    model_state = model.state_dict()

    matched = 0
    for k, v in state.items():
        if (
            k in model_state
            and isinstance(v, torch.Tensor)
            and model_state[k].shape == v.shape
        ):
            matched += 1
    return matched / max(1, len(model_state)), matched


def _find_best_checkpoint_for_model(model: nn.Module, basenames, prefer_token=None):
    roots = ["/kaggle/input", "/kaggle/data", "../input", "../data"]
    exts = [".pkl", ".pth", ".pt", ".bin", ".ckpt"]

    patterns = []
    for b in basenames:
        b = str(b)
        root_b, ext_b = os.path.splitext(b)
        if ext_b.lower() in exts:
            patterns.append(b)
        else:
            patterns.append(b)
            patterns.append(b.replace("_", "-"))
            patterns.append(b.replace("-", "_"))
            for e in exts:
                patterns.append(b + e)
                patterns.append(b.replace("_", "-") + e)
                patterns.append(b.replace("-", "_") + e)

    hits = set()
    for r in roots:
        if not os.path.exists(r):
            continue
        for pat in patterns:
            for h in glob.glob(os.path.join(r, "**", pat), recursive=True):
                if os.path.isfile(h):
                    hits.add(h)
            for h in glob.glob(os.path.join(r, "**", f"*{pat}*"), recursive=True):
                if os.path.isfile(h):
                    hits.add(h)

    if not hits:
        return None, 0.0

    scored = []
    for h in hits:
        name = os.path.basename(h).lower()
        prefer = 0
        if prefer_token is not None and str(prefer_token).lower() in name:
            prefer = 1
        try:
            sz = os.path.getsize(h)
        except OSError:
            sz = -1

        strict_ok = _can_strict_load(model, h)
        try:
            ratio, matched = _checkpoint_match_ratio(model, h)
        except Exception:
            ratio, matched = 0.0, 0

        scored.append((1 if strict_ok else 0, ratio, prefer, matched, sz, h))

    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]), reverse=True)
    strict_ok, best_ratio, best_prefer, best_matched, best_sz, best_path = scored[0]
    print(
        f"Best checkpoint selected: {best_path} | strict_ok={bool(strict_ok)} | match_ratio={best_ratio:.3f} | matched={best_matched} | prefer={best_prefer} | size={best_sz/1024/1024:.2f} MB"
    )
    return best_path, best_ratio


def _load_checkpoint_safely(model: nn.Module, weight_path: str):
    """
    Score-relevant: try strict load first (ensures no random leftovers). If strict fails,
    fall back to partial compatible load so inference still runs.
    """
    state = _prepare_state_for_model(model, weight_path)

    try:
        missing, unexpected = model.load_state_dict(state, strict=True)
        match_ratio = 1.0
        return match_ratio, len(missing), len(unexpected), True
    except Exception as strict_e:
        model_state = model.state_dict()
        filtered = {}
        matched = 0
        for k, v in state.items():
            if (
                k in model_state
                and isinstance(v, torch.Tensor)
                and model_state[k].shape == v.shape
            ):
                filtered[k] = v
                matched += 1

        if matched == 0:
            raise ValueError(
                f"No matching keys between checkpoint and model (0 matched). strict_error={repr(strict_e)}"
            )

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        match_ratio = matched / max(1, len(model_state))
        return match_ratio, len(missing), len(unexpected), False


def _infer_expected_img_size_from_path(weight_path: str):
    """
    Score-relevant minimal change: many DR checkpoints encode training resolution in the filename
    (e.g., '*_320*'). Using the correct resize at inference can materially increase QWK.
    """
    if not weight_path:
        return None
    base = os.path.basename(weight_path).lower()
    for sz in (512, 448, 384, 360, 352, 320, 300, 288, 280, 256):
        if str(sz) in base:
            return int(sz)
    return None




## === cell 1
import torch
import torch.nn as nn
import torchvision
import csv
import timm
import time
import glob
import copy
import os
import json
import cv2
import numpy as np
import pickle
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import lr_scheduler
from timm.models import *
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
from PIL import Image
from PIL import Image, ImageEnhance, ImageOps
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import misc as misc_nn_ops
from torch import Tensor, Size
from torch.jit.annotations import List, Optional, Tuple
from torch.nn.parameter import Parameter
import warnings


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        n: Optional[int] = None,
    ):
        if n is not None:
            warnings.warn(
                "`n` argument is deprecated and has been renamed `num_features`",
                DeprecationWarning,
            )
            num_features = n
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


class backboneNet_efficient(nn.Module):
    def __init__(self):
        super(backboneNet_efficient, self).__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)
        layers_to_train = ["blocks"]
        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        self.num_features = 1792
        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1
        self.act1 = net.act1
        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2
        self.act2 = net.act2
        self.global_pool = net.global_pool
        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        x1 = self.conv_stem(x)
        x2 = self.bn1(x1)
        x3 = self.act1(x2)
        x4 = self.block0(x3)
        x5 = self.block1(x4)
        x6 = self.block2(x5)
        x7 = self.block3(x6)
        x8 = self.block4(x7)
        x9 = self.block5(x8)
        x10 = self.block6(x9)
        x11 = self.conv_head(x10)
        x12 = self.bn2(x11)
        x13 = self.act2(x12)
        x14 = self.global_pool(x13)
        if self.drop_rate > 0.0:
            x14 = F.dropout(x14, p=self.drop_rate, training=self.training)

        x15 = self.rg_cls(x14)
        x16 = self.cls_cls(x14)
        x17 = self.ord_cls(x14)

        return x15, x16, x17




## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return int(np.clip(prediction, 0, 4))


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i])))
            l2 = int(math.ceil(float(out[i])))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def combine3output(r_out, c_out, o_out):
    R = regress2class(r_out.data)
    _, C = torch.max(c_out.data, 1)
    C = C.squeeze().item()
    _, O = torch.max(o_out.data, 1)
    O = O.squeeze().item()

    P = (R + C + O) / 3.0
    P = int(round(P))

    return P




## === cell 3
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5

        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 4
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]

        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)

        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size

        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w

        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h

        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img




## === cell 5
sample_sub_path = os.path.join(APTOS_ROOT, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["id_code"].astype(str).tolist()
print("Loaded sample_submission ids:", len(test_ids))

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.384, 0.258, 0.174],
            std=[0.124, 0.089, 0.094],
        ),
    ]
)

transform2 = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

net1 = ThreeStage_Model()

expected_weight_name = "B4_3stage_22epoch_320"

best_weight_path, best_ratio = _find_best_checkpoint_for_model(
    net1,
    basenames=[
        expected_weight_name,
        expected_weight_name + ".pkl",
        expected_weight_name + ".pth",
        expected_weight_name + ".pt",
        expected_weight_name + ".bin",
        expected_weight_name + ".ckpt",
        "B4_3stage_22epoch_320",
        "3stage",
        "3_stage",
        "three_stage",
        "three-stage",
        "aptos",
        "blindness",
        "efficientnet_b4",
        "tf_efficientnet_b4",
    ],
    prefer_token=expected_weight_name,
)

if best_weight_path is None:
    best_weight_path = _find_weight_file_fuzzy(
        [
            expected_weight_name,
            expected_weight_name + ".pkl",
            expected_weight_name + ".pth",
            expected_weight_name + ".pt",
            expected_weight_name + ".bin",
            expected_weight_name + ".ckpt",
            "B4_3stage_22epoch_320",
            "3stage",
            "3_stage",
            "three_stage",
        ],
        prefer_token=expected_weight_name,
    )

loaded_ok = False
match_ratio = 0.0
strict_loaded = False

loaded_expected_checkpoint = False

if best_weight_path is None:
    warnings.warn(
        f"Pretrained task weights like '{expected_weight_name}(.pkl/.pth/.pt/.ckpt)' not found under common Kaggle roots."
    )
else:
    print("Loading task weights from:", best_weight_path)
    try:
        match_ratio, n_missing, n_unexpected, strict_loaded = _load_checkpoint_safely(
            net1, best_weight_path
        )
        print(
            f"Task checkpoint load: strict_loaded={strict_loaded} | match_ratio={match_ratio:.3f}, missing={n_missing}, unexpected={n_unexpected}"
        )
        loaded_ok = True

        bn = os.path.basename(best_weight_path).lower()
        loaded_expected_checkpoint = expected_weight_name.lower() in bn
    except Exception as e:
        warnings.warn(
            f"Failed to load task checkpoint from {best_weight_path}: {repr(e)}; will use fallback init."
        )
        loaded_ok = False
        loaded_expected_checkpoint = False

fallback_imagenet_backbone = False
if not loaded_ok:
    try:
        net1.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
        net1.backbone.global_pool = GeM(flatten=True)
        fallback_imagenet_backbone = True
        warnings.warn(
            "No compatible task checkpoint loaded; using ImageNet-pretrained EfficientNet-B4 backbone as a fallback "
            "(heads remain randomly initialized)."
        )
    except Exception as e:
        warnings.warn(
            f"Failed to enable ImageNet backbone fallback: {repr(e)}; using fully random weights."
        )
        fallback_imagenet_backbone = False

net1 = net1.to(device)
net1.eval()

ckpt_expected_sz = (
    _infer_expected_img_size_from_path(best_weight_path) if loaded_ok else None
)
if ckpt_expected_sz is not None:
    dynamic_transform1 = transforms.Compose(
        [
            trim(),
            cropTo4_3(),
            transforms.Resize(
                (int(ckpt_expected_sz * 0.9), int(ckpt_expected_sz * 1.2))
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.384, 0.258, 0.174],
                std=[0.124, 0.089, 0.094],
            ),
        ]
    )
else:
    dynamic_transform1 = transform1

if loaded_ok and loaded_expected_checkpoint:
    infer_transform = dynamic_transform1
    infer_transform_name = f"transform1(custom; expected checkpoint; dynamic_resize_from_ckpt={ckpt_expected_sz})"
elif fallback_imagenet_backbone:
    infer_transform = transform2
    infer_transform_name = "transform2(ImageNet; backbone fallback)"
else:
    infer_transform = transform1
    infer_transform_name = "transform1(custom; default)"

print(
    "loaded_ok:",
    loaded_ok,
    f"| match_ratio={match_ratio:.3f}",
    f"| strict_loaded={strict_loaded}",
    "| loaded_expected_checkpoint:",
    loaded_expected_checkpoint,
    "| fallback_imagenet_backbone:",
    fallback_imagenet_backbone,
    "| infer_transform:",
    infer_transform_name,
)



## === cell 6
submission = []
test_img_root = os.path.join(APTOS_ROOT, "test_images")

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print("Predicting", i, "/", len(test_ids))

        try:
            image_name = _resolve_image_path(test_img_root, str(idx))
            img1 = Image.open(image_name).convert("RGB")
            img1 = infer_transform(img1).unsqueeze(0).to(device)

            _, r_out, _ = net1(img1)
            pred1 = int(regress2class(r_out.data.squeeze(1)))
            P = int(np.clip(pred1, 0, 4))
        except Exception as e:
            warnings.warn(f"Failed on id_code={idx}: {repr(e)}; using diagnosis=0")
            P = 0

        submission.append([str(idx), P])

submission = np.array(submission, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = sample_sub[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

assert len(df) == len(sample_sub), "Submission row count mismatch vs sample_submission."
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
