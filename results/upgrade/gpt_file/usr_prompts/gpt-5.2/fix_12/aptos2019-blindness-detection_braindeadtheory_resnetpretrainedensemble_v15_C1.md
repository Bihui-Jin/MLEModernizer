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

3.8

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

0.8911934928977304

# 6. Current score

0.34645

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.08916) has done: 'The runtime failure comes from trying to load pretrained weights from a non-existent `../input/resnet/` dataset; I make the script robust by falling back to ImageNet weights (same ResNet architectures) when those `.pt` files aren’t available. I also fix path handling to use Kaggle’s absolute `/kaggle/input/...` locations (so it runs consistently), and replace the notebook-only tqdm import with standard `tqdm` to avoid import/runtime issues. Finally, I keep the ensemble logic intact (average logits then argmax) but make it work even if only one model successfully loads, ensuring a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -0.05088) has done: 'Your current very low kappa is consistent with a label/target mismatch: you’re using ImageNet-pretrained backbones but random-initialized 5-class heads (because the custom `.pt` weights aren’t present), so predictions are essentially arbitrary. To move the score toward your target with minimal change and without altering the model/training core, I (1) make the script *actually use the provided trained weights* by searching common `/kaggle/input/**` locations for the expected `.pt` files, (2) only fall back to ImageNet when those files truly aren’t available, and (3) ensure inference is numerically stable and correct by running under `inference_mode()` and softmax-averaging probabilities (keeps the same ensemble semantics: average then argmax, but improves calibration for kappa). The output submission format and paths remain unchanged, and the rest of the pipeline (dataset, transforms, models, inference loop) stays intact.'
- What this solution (achieved 0.09033) has done: 'Your score is extremely low because the ensemble is effectively making near-random predictions when the custom `.pt` weights aren’t actually found/loaded; the current code still proceeds with ImageNet backbones + random 5-class heads. I make the weight-file discovery stricter and verifiable (only accept real matches), then only include models whose custom weights successfully load (so the ensemble isn’t poisoned by random heads). I also enforce a deterministic, correct `id_code` order by reading `test.csv` and aligning predictions to it before writing `submission.csv`. These are minimal changes that preserve your model architecture and inference approach (softmax-average then argmax) while moving performance upward toward the target.'
- What this solution (achieved -0.09594) has done: 'Your current 0.09033 score is consistent with still loading weight files incorrectly (or partially), so the ensemble is effectively using random 5-class heads. I keep your exact ResNet architectures and inference/ensembling logic, but make weight loading strict and verifiable: only accept checkpoints that actually contain the expected `fc.*` tensors with shape `[5, in_features]`, and load with `strict=True` once the keys are normalized. I also ensure the test-time preprocessing matches common ImageNet-pretrained expectations (center-crop after resize) without changing resolution, which is a minimal inference-only tweak that can improve stability when using pretrained backbones and trained heads. Finally, the script still always write a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved -0.04459) has done: 'Your current score is far below the target, which is consistent with one or more of your custom `.pt` checkpoints not actually being found/loaded (so you end up ensembling in effect-random 5-class heads, or falling back entirely). I make weight discovery and loading more robust while keeping the same ResNet architectures and the same inference/ensembling semantics (softmax -> average -> argmax): (1) search for the weight files by both exact name and common case/extension variants, (2) accept checkpoints saved either as full `state_dict`, under `state_dict`/`model_state_dict`, or wrapped in a dict, and (3) if the checkpoint uses `classifier.*` instead of `fc.*`, remap those keys to `fc.*` (still strict shape-checked). This is a minimal change aimed at ensuring you actually use the intended trained heads, which should move kappa substantially upward toward your target while still writing a valid `submission.csv`.'
- What this solution (achieved -0.10113) has done: 'Your score is far below target, and with your current constraints the most likely cause is that the intended trained `.pt` checkpoints still aren’t being found/loaded (so you’re effectively using ImageNet backbones with random 5-class heads). I make weight discovery more exhaustive across `/kaggle/input` (including `.ckpt` and common nested folder patterns) and add a second, safe loading path that can handle checkpoints saved as full `nn.Module` objects or with different head naming (while still strict shape-checking for a 5-class head). I also ensure that only successfully loaded custom-weight models participate in the ensemble (to avoid “poisoning” the average), while keeping your exact ResNet architectures and the same softmax-average-then-argmax inference semantics. Finally, I keep the `test.csv` alignment and always write a valid `submission.csv`.'
- What this solution (achieved -0.00326) has done: 'Your current kappa (-0.10113) is so far below the target that the most likely cause is still “not actually using the trained checkpoints”, even though the script runs and writes a CSV. I make one minimal, score-relevant fix: when searching for weights, prefer checkpoints that *actually match* the model (i.e., contain a valid 5-class `fc.*` head with the right `in_features`) instead of returning the first file with a matching name fragment somewhere under `/kaggle/input`. This keeps your exact model architectures and inference/ensembling semantics unchanged, but greatly increases the chance that the ensemble uses the intended trained heads rather than accidentally loading an unrelated file (or falling back). The submission writing stays identical (aligned to `test.csv`, `submission.csv` output).'
- What this solution (achieved 0.10276) has done: 'Your current score indicates the script is still effectively producing near-random labels (either because trained weights aren’t found/loaded, or because the loaded checkpoints don’t match what the model expects). To move the score upward toward the target with minimal change, I (1) broaden weight discovery to include common filename variants that your current search misses (notably underscore-vs-no-underscore and case variants), (2) avoid “poisoning” the ensemble by ensuring only truly compatible checkpoints are selected (still using your existing strict validation), and (3) make the submission alignment stricter by writing predictions in the exact `test.csv` order directly (no merge-needed), eliminating any chance of NaNs/default-0 hurting kappa. Core model architectures, preprocessing resolution, and ensemble semantics (softmax → mean → argmax) remain unchanged.'
- What this solution (achieved 0.12789) has done: 'Your very low kappa strongly suggests you’re still not actually using the intended trained checkpoints (so you’re effectively predicting with random 5-class heads). I make a minimal, score-relevant change: broaden and prioritize checkpoint discovery specifically inside the competition dataset folders (where weights are most likely to be), and rank candidates by “most likely real model checkpoint” (smallest file that still contains a valid 5-class `fc.*` head), while keeping your strict shape validation and exact ensemble semantics (softmax → mean → argmax). I also add a one-line guard to avoid accidentally selecting huge non-model files during fuzzy search, which can silently waste time and/or lead to wrong loads. Everything else (models, transforms, inference loop, submission alignment) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.26015) has done: 'Your current score (0.12789) is far below the target (0.89119), and with your constraints the most likely blocker is still that you’re not loading the intended trained checkpoints (or you’re picking the wrong matching file). I make weight discovery/loading more reliable without changing the model architectures, transforms, or inference/ensemble semantics by (1) extending strict validation beyond just `fc.*` to accept checkpoints whose head is stored under common ResNet naming (`fc.*` or `module.fc.*`) and also accept checkpoints where the whole model was saved but still contains a valid 5-class head, and (2) ranking candidate checkpoints by “best match” (head shape match + most overlapping keys with the current model) instead of smallest file size. This keeps the same ResNet152/ResNet101 ensemble and the same softmax→mean→argmax inference, but should materially increase the chance you actually ensemble the correct trained weights and move kappa upward toward the target band. The script still runs end-to-end and always writes a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.34645) has done: 'Your current score is far below the target, so the most likely remaining “minimal-change” gain is in metric-aligned post-processing rather than changing models: QWK typically improves when you optimize class cutpoints on a validation split and apply them to turn continuous predictions into ordinal labels. I keep your exact ResNet architectures and no-training inference, but add a small stratified validation split from `train.csv`, run the same ensemble inference on that split, and fit 4 thresholds (for 5 classes) that maximize quadratic weighted kappa. Then I apply those thresholds to the test ensemble’s expected class value (probability-weighted mean) and write `submission.csv` in `test.csv` order as before.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import glob

base_in = "/kaggle/input"
if os.path.exists(base_in):
    for dirname, _, filenames in os.walk(base_in):
        for filename in filenames[:5]:
            print(os.path.join(dirname, filename))
        break



## === cell 1
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score



## === cell 2
DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"

transform = transforms.Compose(
    [
        transforms.Resize(320),
        transforms.CenterCrop((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file).reset_index(drop=True)
        self.filetype = filetype
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_name = os.path.join(
                TRAIN_IMG_DIR, self.eye_frame.loc[idx, "id_code"] + ".png"
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(self.eye_frame.loc[idx, "diagnosis"])
        else:
            img_name = os.path.join(
                TEST_IMG_DIR, self.eye_frame.loc[idx, "id_code"] + ".png"
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, self.eye_frame.loc[idx, "id_code"]




## === cell 3
test_dataset = APTOSDataset(csv_file=TEST_CSV, filetype="test", transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 4
def _normalize_state_dict_keys(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        new_state[nk] = v
    return new_state


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        tensor_like = any(isinstance(v, torch.Tensor) for v in ckpt.values())
        if tensor_like:
            return ckpt
    return None


def _remap_classifier_to_fc(state):
    if "fc.weight" in state and "fc.bias" in state:
        return state
    if "classifier.weight" in state and "classifier.bias" in state:
        state = dict(state)
        state["fc.weight"] = state.pop("classifier.weight")
        state["fc.bias"] = state.pop("classifier.bias")
    return state


def _remap_head_variants_to_fc(state):
    if "fc.weight" in state and "fc.bias" in state:
        return state
    state = dict(state)

    candidates = [
        ("head.weight", "head.bias"),
        ("classifier.1.weight", "classifier.1.bias"),
        ("classifier.0.weight", "classifier.0.bias"),
        ("logits.weight", "logits.bias"),
        ("final.weight", "final.bias"),
        ("output.weight", "output.bias"),
    ]
    for w_key, b_key in candidates:
        if w_key in state and b_key in state:
            state["fc.weight"] = state.pop(w_key)
            state["fc.bias"] = state.pop(b_key)
            break

    return state


def _ensure_fc_keys_exist(state):
    if "fc.weight" in state and "fc.bias" in state:
        return state
    state = dict(state)
    if "fc.weight" not in state and "module.fc.weight" in state:
        state["fc.weight"] = state["module.fc.weight"]
    if "fc.bias" not in state and "module.fc.bias" in state:
        state["fc.bias"] = state["module.fc.bias"]
    return state


def _validate_fc_shapes(model, state):
    if "fc.weight" not in state or "fc.bias" not in state:
        return False, "no_fc_head_in_checkpoint"
    if not isinstance(state["fc.weight"], torch.Tensor) or not isinstance(
        state["fc.bias"], torch.Tensor
    ):
        return False, "fc_params_not_tensor"

    fc_w = state["fc.weight"]
    fc_b = state["fc.bias"]
    if fc_w.ndim != 2 or fc_w.shape[0] != 5:
        return False, f"fc_weight_bad_shape={tuple(fc_w.shape)}"
    if fc_b.ndim != 1 or fc_b.shape[0] != 5:
        return False, f"fc_bias_bad_shape={tuple(fc_b.shape)}"

    if getattr(model.fc, "in_features", None) is not None:
        if fc_w.shape[1] != model.fc.in_features:
            return (
                False,
                f"in_features_mismatch_ckpt={fc_w.shape[1]} model={model.fc.in_features}",
            )
    return True, "ok"


def _try_load_state_dict_strict(model, weight_path, device):
    if weight_path is None or (not os.path.exists(weight_path)):
        return False, "missing"

    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        state = ckpt.state_dict()
    else:
        state = _extract_state_dict(ckpt)

    if state is None or not isinstance(state, dict):
        return False, "unsupported_checkpoint_format"

    state = _normalize_state_dict_keys(state)
    state = _remap_classifier_to_fc(state)
    state = _remap_head_variants_to_fc(state)
    state = _ensure_fc_keys_exist(state)

    ok, msg = _validate_fc_shapes(model, state)
    if not ok:
        return False, msg

    model.load_state_dict(state, strict=True)
    return True, "loaded_strict"


def build_resnet(model_name, num_classes=5, weights_fallback=True):
    if model_name == "resnet152":
        if weights_fallback:
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            model = torchvision.models.resnet152(weights=weights)
        else:
            model = torchvision.models.resnet152(weights=None)
    elif model_name == "resnet101":
        if weights_fallback:
            weights = torchvision.models.ResNet101_Weights.DEFAULT
            model = torchvision.models.resnet101(weights=weights)
        else:
            model = torchvision.models.resnet101(weights=None)
    else:
        raise ValueError("Unsupported model_name")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def _filename_variants(filename):
    base, ext = os.path.splitext(filename)
    ext = ext if ext else ".pt"

    bases = {base}
    bases.add(base.replace("_", ""))
    bases.add(base.replace("_", "-"))
    bases.add(base.replace("-", "_"))
    bases.add(base.replace("-", ""))
    bases.add(base.lower())
    bases.add(base.upper())

    variants = set()
    for b in bases:
        for e in [ext, ext.lower(), ext.upper(), ".pt", ".pth", ".bin", ".ckpt"]:
            variants.add(b + e)
    return sorted(variants)


def _iter_weight_candidates(filename, preferred_dir="/kaggle/input/resnet"):
    candidates = []
    prioritized_roots = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input",
    ]

    for v in _filename_variants(filename):
        candidates.append(os.path.join(preferred_dir, v))

    nested_patterns = []
    for pr in prioritized_roots:
        nested_patterns.extend(
            [
                os.path.join(pr, "**", "{v}"),
                os.path.join(pr, "**", "weights", "{v}"),
                os.path.join(pr, "**", "model", "{v}"),
                os.path.join(pr, "**", "models", "{v}"),
                os.path.join(pr, "**", "checkpoints", "{v}"),
            ]
        )

    for v in _filename_variants(filename):
        for pat in nested_patterns:
            candidates.extend(glob.glob(pat.format(v=v), recursive=True))

    base, _ = os.path.splitext(filename)
    fuzzy_keys = {base, base.replace("_", ""), base.lower(), base.upper()}
    for fk in sorted(fuzzy_keys, key=len, reverse=True):
        for pr in prioritized_roots:
            candidates.extend(
                glob.glob(os.path.join(pr, "**", f"*{fk}*.pt"), recursive=True)
            )
            candidates.extend(
                glob.glob(os.path.join(pr, "**", f"*{fk}*.pth"), recursive=True)
            )
            candidates.extend(
                glob.glob(os.path.join(pr, "**", f"*{fk}*.ckpt"), recursive=True)
            )
            candidates.extend(
                glob.glob(os.path.join(pr, "**", f"*{fk}*.bin"), recursive=True)
            )

    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for p in ordered:
        if os.path.exists(p) and os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in [".pt", ".pth", ".bin", ".ckpt"]:
                try:
                    if os.path.getsize(p) > 1_200_000_000:  # > ~1.2GB
                        continue
                except OSError:
                    continue
                yield p


def find_weight_file_for_model(
    model, filename, device, preferred_dir="/kaggle/input/resnet"
):
    model_keys = set(model.state_dict().keys())

    scored = []
    for p in _iter_weight_candidates(filename, preferred_dir=preferred_dir):
        try:
            ckpt = torch.load(p, map_location="cpu")
            if isinstance(ckpt, nn.Module):
                state = ckpt.state_dict()
            else:
                state = _extract_state_dict(ckpt)
            if state is None or not isinstance(state, dict):
                continue

            state = _normalize_state_dict_keys(state)
            state = _remap_classifier_to_fc(state)
            state = _remap_head_variants_to_fc(state)
            state = _ensure_fc_keys_exist(state)

            ok, _ = _validate_fc_shapes(model, state)
            if not ok:
                continue

            state_keys = set(state.keys())
            overlap = len(model_keys.intersection(state_keys))
            try:
                size = os.path.getsize(p)
            except OSError:
                size = 10**18
            scored.append((-overlap, len(p), size, p))
        except Exception:
            continue

    if not scored:
        return None
    scored.sort()
    return scored[0][-1]


models_list = []

m0 = build_resnet("resnet152", num_classes=5, weights_fallback=True)
p0 = find_weight_file_for_model(m0, "FinalResnet152_0.pt", device=device)
loaded0, reason0 = _try_load_state_dict_strict(m0, p0, device)
print("model0 weights loaded:", loaded0, "| reason:", reason0, "| path:", p0)
if loaded0:
    models_list.append(m0.to(device))

for fname in [
    "FinalResnet02.pt",
    "FinalResnet01.pt",
    "FinalResnet00.pt",
    "FinalResnet0.pt",
]:
    m = build_resnet("resnet101", num_classes=5, weights_fallback=True)
    p = find_weight_file_for_model(m, fname, device=device)
    loaded, reason = _try_load_state_dict_strict(m, p, device)
    print(f"{fname} weights loaded:", loaded, "| reason:", reason, "| path:", p)
    if loaded:
        models_list.append(m.to(device))

if len(models_list) == 0:
    print(
        "WARNING: No valid custom .pt/.pth/.ckpt weight files found/loaded under /kaggle/input. "
        "Falling back to a single ImageNet backbone with random 5-class head (expected low kappa)."
    )
    fallback = build_resnet("resnet152", num_classes=5, weights_fallback=True).to(
        device
    )
    models_list = [fallback]




## === cell 5
def compute_predictions(model, model_type, data_loader, device):
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Predict(train)"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.detach().cpu())
            num_examples += labels.size(0)
            correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100.0
    else:
        predictions = []
        img_ids = []
        out = []
        for inputs, img_id in tqdm(data_loader, desc="Predict(test)"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().cpu())
            img_ids.extend(list(img_id))
            out.extend(outputs.detach().cpu())
        predictions = [int(pred.item()) for pred in predictions]
        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
        return final_predictions, out, img_ids


def _ensemble_predict_proba(models_list, loader, device):
    with torch.inference_mode():
        for m in models_list:
            m.eval()

        probs_list = []
        ids_ref = None
        for mi, m in enumerate(models_list):
            print(f"Computing Predictions for model {mi}")
            _, out_i, ids_i = compute_predictions(m, "test", loader, device)
            out_i = torch.stack(out_i, dim=0)  # [N, 5]
            prob_i = torch.softmax(out_i, dim=1)
            probs_list.append(prob_i)
            if ids_ref is None:
                ids_ref = ids_i
            else:
                if ids_i != ids_ref:
                    raise RuntimeError(
                        "Id order mismatch between models; cannot ensemble safely."
                    )

        prob_avg = torch.stack(probs_list, dim=0).mean(dim=0)  # [N, 5]
        return prob_avg, ids_ref


def _apply_thresholds_on_expectation(prob_avg, thresholds):
    exp = (
        prob_avg * torch.arange(5, device=prob_avg.device, dtype=prob_avg.dtype)
    ).sum(dim=1)
    t0, t1, t2, t3 = thresholds
    pred = torch.zeros_like(exp, dtype=torch.long)
    pred += (exp > t0).long()
    pred += (exp > t1).long()
    pred += (exp > t2).long()
    pred += (exp > t3).long()
    return pred.cpu().numpy().astype(int), exp.detach().cpu().numpy()


def _fit_qwk_thresholds(y_true, exp_pred, max_iter=4):
    y_true = np.asarray(y_true, dtype=int)
    exp_pred = np.asarray(exp_pred, dtype=float)

    thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=float)

    def score(thr_local):
        thr_local = np.sort(thr_local)
        pred = np.zeros_like(exp_pred, dtype=int)
        pred += (exp_pred > thr_local[0]).astype(int)
        pred += (exp_pred > thr_local[1]).astype(int)
        pred += (exp_pred > thr_local[2]).astype(int)
        pred += (exp_pred > thr_local[3]).astype(int)
        return cohen_kappa_score(y_true, pred, weights="quadratic")

    best = score(thr)

    grid = np.quantile(exp_pred, np.linspace(0.05, 0.95, 61))

    for _ in range(max_iter):
        improved = False
        for i in range(4):
            best_i_thr = thr[i]
            best_i_score = best
            for g in grid:
                thr_try = thr.copy()
                thr_try[i] = float(g)
                thr_try = np.sort(thr_try)
                if not (thr_try[0] < thr_try[1] < thr_try[2] < thr_try[3]):
                    continue
                s = score(thr_try)
                if s > best_i_score:
                    best_i_score = s
                    best_i_thr = float(g)
            if best_i_score > best:
                thr[i] = best_i_thr
                thr = np.sort(thr)
                best = best_i_score
                improved = True
        if not improved:
            break

    return np.sort(thr), float(best)




## === cell 6

train_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, va_idx = next(
    sss.split(train_df["id_code"].values, train_df["diagnosis"].values)
)

val_df = train_df.iloc[va_idx].reset_index(drop=True)
val_csv_path = "val_split.csv"
val_df.to_csv(val_csv_path, index=False)

val_dataset = APTOSDataset(csv_file=val_csv_path, filetype="train", transform=transform)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)


class _ValAsTestDataset(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds
        self.df = base_ds.eye_frame

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, y = self.base_ds[idx]
        return img, self.df.loc[idx, "id_code"]


val_as_test_loader = torch.utils.data.DataLoader(
    _ValAsTestDataset(val_dataset),
    batch_size=24,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

prob_val, ids_val = _ensemble_predict_proba(models_list, val_as_test_loader, device)
y_val_true = val_df["diagnosis"].to_numpy().astype(int)

_, val_exp = _apply_thresholds_on_expectation(prob_val, thresholds=[0.5, 1.5, 2.5, 3.5])
thr_opt, best_qwk = _fit_qwk_thresholds(y_val_true, val_exp, max_iter=4)
print("Optimized thresholds:", thr_opt, " | val QWK:", best_qwk)

prob_test, ids_ref = _ensemble_predict_proba(models_list, test_loader, device)
pred_test, _ = _apply_thresholds_on_expectation(prob_test, thresholds=thr_opt)

test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)
ids_test = test_df["id_code"].tolist()

if ids_ref != ids_test:
    pred_map = {k: int(v) for k, v in zip(ids_ref, pred_test)}
    final_predictions = pd.DataFrame(
        {"id_code": ids_test, "diagnosis": [pred_map.get(i, 0) for i in ids_test]}
    )
else:
    final_predictions = pd.DataFrame({"id_code": ids_test, "diagnosis": pred_test})

final_predictions["diagnosis"] = final_predictions["diagnosis"].astype(int)
final_predictions.to_csv("submission.csv", index=False)

print(final_predictions.head())
print("Wrote submission.csv with shape:", final_predictions.shape)
print("Loaded models count:", len(models_list))
