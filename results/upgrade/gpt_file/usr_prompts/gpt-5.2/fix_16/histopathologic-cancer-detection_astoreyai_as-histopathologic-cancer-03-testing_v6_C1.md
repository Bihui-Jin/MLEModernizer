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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.8230482640588473

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the `pip install` step (internet/offline installs are unreliable on Kaggle) and fix the Albumentations import crash by replacing Albumentations-based preprocessing with a minimal torchvision transform pipeline that preserves the same resize+normalize+tensor semantics. I also remove the dependency on `efficientnet_pytorch` (which wasn’t successfully imported) by switching to the built-in `torchvision.models.efficientnet_b0` while keeping the same “EfficientNet-B0 + linear head” core architecture and producing logits for 2 classes. Finally, I fix the submission logic to output probabilities (not thresholded 0/1 labels), which is required for AUC scoring and should move the score toward the target, and ensure model loading works even if the checkpoint keys differ (`module.` prefix, different classifier head naming).'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC is consistent with essentially random predictions, which in this pipeline most likely comes from the checkpoint not actually loading into the model due to key mismatches (especially around EfficientNet’s `features.*`/`classifier.*` naming) or producing degenerate probabilities. I keep the same EfficientNet-B0 + 2-class head logic and the same softmax→class-1 probability output, but make checkpoint loading robust by (1) detecting common key prefixes (`model.`, `net.`, `encoder.`, etc.), (2) remapping classifier head keys when the checkpoint used a different head name, and (3) explicitly warning and falling back to a deterministic baseline probability (class prior) only if loading truly fails (so you don’t silently submit random weights). I also add a quick sanity check on the probability distribution to catch “all 0.5 / constant” outputs early, without changing evaluation semantics. These minimal changes should move the score upward toward the target if the checkpoint contains real trained weights.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly suggests the model is not actually using trained weights (so predictions become near-random or constant), despite `strict=False` not throwing an error. I keep the exact same EfficientNet-B0 + 2-class linear head and the same softmax→class-1 probability submission, but make checkpoint loading deterministic and verifiable by (1) trying multiple common checkpoint layouts, (2) automatically remapping keys for both `torchvision` EfficientNet (`features.*`/`classifier.*`) and older wrappers (`model.*` nesting), and (3) validating that a substantial fraction of tensors loaded with matching shapes (otherwise we fall back to the prior). This is a minimal change focused on actually loading your trained weights, which should move AUC upward toward your target if the checkpoint is correct. I also align the resize interpolation with common ImageNet preprocessing (bicubic) without changing the overall transform semantics.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC is still consistent with either (a) falling back to a constant prior because `loaded_ok` never becomes True, or (b) loading only a tiny subset of weights (e.g., just the classifier) while leaving the backbone random—both lead to near-random ranking. I keep the same EfficientNet-B0 + 2-class head and the same softmax→class-1 probability submission, but make checkpoint loading both more compatible and safer by (1) explicitly handling checkpoints saved from your wrapper `CancerClassifier` (keys like `model.model.*`), and (2) allowing a controlled “classifier-only” load when the backbone key naming differs, rather than falling back to a constant 0.5-ish output. This should move predictions away from constant/random and improve AUC toward the target without changing training/inference core logic. I also keep submission id ordering aligned to `sample_submission.csv` to avoid any accidental mismatch risk.'
- What this solution (achieved 0.5) has done: 'I fix the most likely reason you’re stuck at AUC≈0.5: the checkpoint is not being loaded into the actual `model.model.*` keys because the current remapping adds an extra `model.` prefix and the classifier mapping points to a non-existent index for torchvision EfficientNet-B0. I minimally adjust the checkpoint key normalization to (a) map any bare `features.*`/`classifier.*` keys directly onto the `torchvision` EfficientNet keyspace, and (b) map classifier weights to `model.classifier.1.*` (not `.0.*`). I also keep your existing fallback behavior, but make the “classifier-only” extraction robust to both `.0.*` and `.1.*` so we don’t silently miss the head weights. These changes keep the same architecture/inference logic while making it far more likely your trained weights actually load, which should move AUC upward toward your target.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly indicates the model is still effectively untrained at inference (checkpoint not loading into the actual backbone), so the smallest score-improving change is to make the checkpoint key-mapping match your wrapper’s real keyspace and then verify we truly loaded most tensors. I minimally fix the remapping so it does not duplicate the `model.` prefix (which creates `model.model.*` and prevents loading), and I broaden the “unwrapping” of nested checkpoint dicts (common in Lightning/torch.save) so we actually find the real `state_dict`. Finally, I keep the same EfficientNet-B0+2-class head and the same softmax→class-1 probability output, but I require a high match fraction and print clear diagnostics so you don’t silently submit near-constant predictions again.'
- What this solution (achieved 0.5) has done: 'Your AUC=0.5 strongly suggests the backbone weights are still not being loaded (or you are accidentally loading only the classifier), yielding near-random ranking. I keep the exact same EfficientNet-B0 + 2-class linear head and the same softmax→class-1 probability submission, but I make checkpoint loading stricter and more compatible by (1) unwrapping nested checkpoints more aggressively (including common Lightning/EMA layouts), (2) remapping keys both directions for your wrapper (`model.*`) and plain torchvision (`features.*`/`classifier.*`) without creating `model.model.*`, and (3) requiring a high *backbone* match ratio (not just total) to consider the load successful. If we cannot confidently load the backbone, we not silently proceed with classifier-only weights; we fall back to the class prior (which is honest and avoids random noise), and you get clear diagnostics about what matched.'
- What this solution (achieved 0.5) has done: 'Your AUC=0.5 is most consistent with either constant predictions (fallback prior) or effectively-random weights because the checkpoint backbone keys still aren’t matching your `CancerClassifier.model.*` backbone. I keep your exact EfficientNet-B0 + 2-class head and the same softmax→class-1 probability submission, but make a minimal, more permissive checkpoint remapping that (a) correctly unwraps `model.model.*` to `model.*`, and (b) maps both `features.*`/`classifier.*` and `model.features.*`/`model.classifier.*` into the wrapper’s keyspace without creating duplicate prefixes. I also slightly relax the “accept load” criterion to allow a small number of missing buffers while still requiring a strong backbone match, so you’re much less likely to silently fall back to the prior. These changes should move predictions away from constant/random and increase AUC toward your 0.823 target, without changing core model/inference logic.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC is almost certainly because `loaded_ok` remains False (or the backbone isn’t actually loaded), so you submit a constant prior and get random ranking. I make the smallest changes to increase the chance that your checkpoint actually loads into the existing `CancerClassifier.model` EfficientNet-B0: (1) broaden prefix stripping to include `model.` (common when saving wrapper models) and (2) add a minimal remap for torchvision EfficientNet-B0’s head key (`classifier.1.*`) to your wrapper (`model.classifier.1.*`). I also slightly relax the acceptance criteria to allow typical missing buffers while still requiring a strong backbone match, so you don’t unnecessarily fall back to the prior. No changes to the model architecture, transforms, or probability computation—only checkpoint compatibility so predictions stop being constant and AUC can move toward the target.'
- What this solution (achieved 0.5) has done: 'I make the checkpoint loading actually usable by (1) fixing the remapping logic so it *replaces* mismatched prefixes (instead of only adding alternate keys), and (2) adding explicit support for the very common case where the checkpoint was saved from a plain `torchvision` EfficientNet (keys like `features.*`/`classifier.*`) while your current model expects `model.features.*`/`model.classifier.*`. This is the smallest change that directly targets why you’re stuck at AUC≈0.5: the backbone is likely not loading at all, so predictions are effectively random/constant. I also tighten the “best candidate” selection to prefer the candidate with the highest backbone match and then load that remapped dict (still `strict=False`) without changing the model, transforms, or probability computation. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I keep your exact EfficientNet-B0+2-class head and inference logic, but fix the most likely reason you’re stuck at AUC≈0.5: the checkpoint still isn’t actually loading backbone weights due to keyspace mismatches. The minimal change is to expand the checkpoint remapping to explicitly handle torchvision EfficientNet checkpoints saved with `features.*`/`classifier.*` and to also handle checkpoints saved from your wrapper with `model.*` or `model.model.*` without over/under-prefixing. I also loosen the “loaded_ok” acceptance just enough to avoid falling back to the class-prior when most backbone tensors match (buffers like `num_batches_tracked` can cause misses) while still requiring a strong backbone match so we don’t accidentally run on random weights. This should move predictions away from constant/prior and increase AUC toward your target without changing model architecture, transforms, or probability computation.'
- What this solution (achieved 0.5) has done: 'I make the smallest changes that directly target why you’re stuck at AUC≈0.5: your checkpoint is almost certainly not being applied to the backbone due to key mismatches, so inference runs on random weights or falls back to a constant prior. Concretely, I (1) broaden checkpoint “unwrapping” to find the real `state_dict` in more formats, (2) improve key remapping by generating a few explicit candidate keyspaces (plain torchvision EfficientNet, wrapped `model.*`, doubly-wrapped `model.model.*`) and choosing the one with the best *backbone* shape match, and (3) relax the load acceptance slightly to avoid rejecting good loads due to harmless buffer/key differences while still requiring strong backbone coverage. This keeps your exact model architecture, transforms, and probability computation unchanged, but makes it far more likely the trained weights actually load, which should move AUC upward toward your 0.823 target. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I keep your exact EfficientNet-B0 + 2-class head and the same softmax→class-1 probability submission, but fix the most likely cause of AUC=0.5: the checkpoint is not being loaded because `_auto_strip_known_prefixes()` strips `model.` even when the checkpoint already matches your wrapper (`model.features.*`, `model.classifier.*`), which destroys key alignment and forces the prior fallback. I make prefix stripping conditional so we only strip `model.` when it clearly improves match to the current model keys, and I broaden candidate variants to explicitly map between `features./classifier.` and `model.features./model.classifier.` in both directions. Finally, I keep your fallback behavior but make `loaded_ok` more likely to become True only when backbone match is genuinely high, so predictions stop being constant and AUC should move up toward your 0.823 target.'
- What this solution (achieved 0.5) has done: 'I make the checkpoint loading actually succeed by ensuring we don’t reject a good state_dict just because it’s a plain torchvision EfficientNet (keys `features.*` / `classifier.*`) while our wrapper expects `model.features.*` / `model.classifier.*`. Concretely, I (1) score backbone match on both possible prefixes (`model.features.` and `features.`) and accept whichever is relevant, and (2) pick the best candidate using that “effective backbone match” rather than only `model.features.` (which can incorrectly score as zero and force the constant-prior fallback that yields AUC≈0.5). This keeps the same model, transforms, and probability computation, but should move your score up toward the 0.823 target by stopping the silent fallback to constant predictions. The script still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I make checkpoint loading stop silently failing by changing the “best candidate” selection to prefer the variant that best matches your wrapper’s keyspace (especially `model.features.*`) and then explicitly try loading into the wrapper as well as into the inner torchvision model, picking whichever actually loads the backbone best. This keeps your exact model architecture and softmax-to-class-1 probability output, but greatly reduces the chance you end up in the constant-prior fallback that yields AUC≈0.5. I also add a tiny, score-relevant safety: if the loaded checkpoint is a 1-logit (binary) head, we still produce valid probabilities via sigmoid (instead of mis-softmaxing), without changing the model definition. Finally, inference and submission formatting remain identical, and the script still produce `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)



## === cell 1
DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"
TEST_DIR = f"{DATA_DIR}/test"

MODEL_PATH = "/kaggle/input/as-histopathologic-cancer-02-training/model_best.pth"

SUBMISSION_FILE = "submission.csv"

TARGET_SIZE = (96, 96)
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Model path exists:", os.path.exists(MODEL_PATH))



## === cell 2
test_transforms = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),  # [0,1], CHW
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)




## === cell 3
class HistologyTestDataset(Dataset):
    def __init__(self, img_ids, img_dir, transform):
        self.img_ids = list(img_ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_id




## === cell 4
class CancerClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


model = CancerClassifier().to(device)


def _is_state_dict_like(d):
    if not isinstance(d, dict) or len(d) == 0:
        return False
    str_keys = sum(isinstance(k, str) for k in d.keys())
    tensor_vals = sum(isinstance(v, torch.Tensor) for v in d.values())
    return str_keys >= max(1, int(0.9 * len(d))) and tensor_vals >= max(
        1, int(0.5 * len(d))
    )


def _iter_candidate_state_dicts(ckpt_obj):
    """
    Keep: broad nested checkpoint search to find the true state_dict.
    """
    cands = []
    if isinstance(ckpt_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
            "ema",
            "module",
            "checkpoint",
            "best_state_dict",
            "student",
            "teacher",
        ]:
            v = ckpt_obj.get(k, None)
            if _is_state_dict_like(v):
                cands.append((f"ckpt['{k}']", v))
            elif isinstance(v, dict):
                for kk in [
                    "state_dict",
                    "model_state_dict",
                    "weights",
                    "params",
                    "model",
                    "net",
                    "module",
                ]:
                    vv = v.get(kk, None)
                    if _is_state_dict_like(vv):
                        cands.append((f"ckpt['{k}']['{kk}']", vv))

        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            v = ckpt_obj["state_dict"]
            for kk in ["model", "net", "module"]:
                vv = v.get(kk, None)
                if _is_state_dict_like(vv):
                    cands.append((f"ckpt['state_dict']['{kk}']", vv))

        if _is_state_dict_like(ckpt_obj):
            cands.append(("ckpt", ckpt_obj))

    seen = set()
    out = []
    for name, sd in cands:
        if id(sd) not in seen:
            out.append((name, sd))
            seen.add(id(sd))
    return out


def _strip_prefix_if_present(sd, prefix):
    if not prefix:
        return sd
    out = {}
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _score_state_dict_fit(model_, sd):
    msd = model_.state_dict()
    matched = 0
    total = 0
    for k, v in msd.items():
        total += 1
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].shape == v.shape:
            matched += 1
    return matched, total, matched / max(1, total)


def _score_fit_by_prefix(model_, sd, prefix):
    msd = model_.state_dict()
    keys = [k for k in msd.keys() if k.startswith(prefix)]
    if not keys:
        return 0, 0, 0.0
    matched = 0
    total = 0
    for k in keys:
        total += 1
        v = msd[k]
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].shape == v.shape:
            matched += 1
    return matched, total, matched / max(1, total)


def _auto_strip_known_prefixes(model_, sd):
    """
    Why this change (score-relevant): Avoid destroying key alignment by stripping prefixes
    unless it improves actual shape-match against the current model.
    """
    always_try = ["module.", "net.", "encoder.", "backbone."]

    best_sd = dict(sd)
    best_score = _score_state_dict_fit(model_, best_sd)[2]

    for p in always_try:
        cnt = sum(1 for k in best_sd.keys() if isinstance(k, str) and k.startswith(p))
        if cnt >= max(1, int(0.7 * len(best_sd))):
            cand = _strip_prefix_if_present(best_sd, p)
            cand_score = _score_state_dict_fit(model_, cand)[2]
            if cand_score >= best_score:
                best_sd, best_score = cand, cand_score

    cnt_model = sum(
        1 for k in best_sd.keys() if isinstance(k, str) and k.startswith("model.")
    )
    if cnt_model >= max(1, int(0.7 * len(best_sd))):
        cand = _strip_prefix_if_present(best_sd, "model.")
        cand_score = _score_state_dict_fit(model_, cand)[2]
        if cand_score > best_score:
            best_sd, best_score = cand, cand_score

    return best_sd


def _generate_mapped_variants(sd):
    """
    Keep: generate both wrapper and plain torchvision keyspaces.
    """
    variants = []
    base = dict(sd)
    variants.append(("base", base))

    v_plain_to_wrapper = dict(base)
    for k, v in base.items():
        if isinstance(k, str) and (
            k.startswith("features.") or k.startswith("classifier.")
        ):
            v_plain_to_wrapper["model." + k] = v
    variants.append(("plain_to_wrapper", v_plain_to_wrapper))

    v_wrapper_to_plain = dict(base)
    for k, v in base.items():
        if isinstance(k, str) and (
            k.startswith("model.features.") or k.startswith("model.classifier.")
        ):
            v_wrapper_to_plain[k[len("model.") :]] = v
    variants.append(("wrapper_to_plain", v_wrapper_to_plain))

    v_modelmodel_to_model = dict(base)
    for k, v in base.items():
        if isinstance(k, str) and k.startswith("model.model."):
            v_modelmodel_to_model["model." + k[len("model.model.") :]] = v
    variants.append(("modelmodel_to_model", v_modelmodel_to_model))

    v_modelmodel_to_plain = dict(base)
    for k, v in base.items():
        if isinstance(k, str) and k.startswith("model.model."):
            v_modelmodel_to_plain[k[len("model.") :]] = v
    variants.append(("modelmodel_to_plainish", v_modelmodel_to_plain))

    return variants


def _effective_backbone_score(model_, sd):
    """
    Why this change (score-relevant): If the checkpoint is a plain torchvision EfficientNet
    (keys 'features.*'), backbone match against wrapper keys would look like 0 and we'd
    incorrectly fall back. Compute both and use the relevant one.
    """
    b1 = _score_fit_by_prefix(model_, sd, "model.features.")
    b2 = _score_fit_by_prefix(model_, sd, "features.")
    return b1 if b1[1] >= b2[1] else b2


def _effective_wrapper_backbone_frac(model_, sd):
    """
    Why this change (score-relevant): We ultimately need weights to land in the wrapper's
    'model.features.*' keyspace. Prefer candidates that already match that, to prevent
    ending up with a load that "looks good" only for the inner model.
    """
    bmw = _score_fit_by_prefix(model_, sd, "model.features.")
    return bmw[2]


loaded_ok = False
best_fit = None

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    candidates = _iter_candidate_state_dicts(ckpt)
    print(f"Found {len(candidates)} candidate state_dicts in checkpoint.")
    if not candidates:
        print("Warning: Could not interpret checkpoint format; will not load weights.")
    else:
        for name, sd0 in candidates:
            sd_raw = dict(sd0)
            sd_raw = _auto_strip_known_prefixes(model, sd_raw)

            for vname, sd in _generate_mapped_variants(sd_raw):
                matched, total, frac = _score_state_dict_fit(model, sd)

                b_m, b_t, b_frac = _effective_backbone_score(model, sd)
                bw_frac = _effective_wrapper_backbone_frac(model, sd)

                h1 = _score_fit_by_prefix(model, sd, "model.classifier.")
                h2 = _score_fit_by_prefix(model, sd, "classifier.")
                h_m, h_t, h_frac = h1 if h1[1] >= h2[1] else h2

                print(
                    f"Candidate {name} / {vname}: total match {matched}/{total} (frac={frac:.3f}), "
                    f"effective backbone {b_m}/{b_t} (frac={b_frac:.3f}), "
                    f"wrapper-backbone frac={bw_frac:.3f}, "
                    f"effective head {h_m}/{h_t} (frac={h_frac:.3f})"
                )

                score = (bw_frac, b_frac, frac)

                if best_fit is None or score > best_fit[0]:
                    best_fit = (
                        score,
                        f"{name} / {vname}",
                        sd,
                        (matched, total, frac),
                        (b_m, b_t, b_frac),
                        (h_m, h_t, h_frac),
                        bw_frac,
                    )

        if best_fit is not None:
            _, best_name, best_sd, tot_stat, bb_stat, hd_stat, bw_frac = best_fit
            matched, total, frac = tot_stat
            b_m, b_t, b_frac = bb_stat
            h_m, h_t, h_frac = hd_stat

            accept = (b_frac >= 0.70 and frac >= 0.60) or (
                bw_frac >= 0.60 and frac >= 0.55
            )
            if accept:
                missing_a, unexpected_a = model.load_state_dict(best_sd, strict=False)
                missing_b, unexpected_b = (None, None)
                loaded_into_inner = False
                try:
                    missing_b, unexpected_b = model.model.load_state_dict(
                        best_sd, strict=False
                    )
                    loaded_into_inner = True
                except Exception as e:
                    print("Inner-model load attempt failed (non-fatal):", repr(e))

                if bw_frac >= 0.60:
                    loaded_ok = True
                    print(
                        f"Loaded checkpoint into wrapper from {best_name}. "
                        f"total matched={matched}/{total} (frac={frac:.3f}), backbone (effective)={b_m}/{b_t} (frac={b_frac:.3f}), "
                        f"wrapper-backbone frac={bw_frac:.3f}. "
                        f"Missing keys: {len(missing_a)}, unexpected keys: {len(unexpected_a)}"
                    )
                    print("Sample missing keys:", list(missing_a)[:8])
                    print("Sample unexpected keys:", list(unexpected_a)[:8])
                else:
                    if loaded_into_inner and b_frac >= 0.70:
                        loaded_ok = True
                        print(
                            f"Loaded checkpoint into inner torchvision model from {best_name}. "
                            f"total matched(wrapper)={matched}/{total} (frac={frac:.3f}), effective backbone={b_m}/{b_t} (frac={b_frac:.3f}), "
                            f"wrapper-backbone frac={bw_frac:.3f}. "
                            f"Missing(inner): {len(missing_b)}, unexpected(inner): {len(unexpected_b)}"
                        )
                        print("Sample missing keys (inner):", list(missing_b)[:8])
                        print("Sample unexpected keys (inner):", list(unexpected_b)[:8])
                    else:
                        print(
                            f"Warning: Candidate accepted by coarse criteria but not loadable with strong backbone. "
                            f"Best was {best_name} with effective backbone frac={b_frac:.3f}, wrapper-backbone frac={bw_frac:.3f}, total frac={frac:.3f}. "
                            "Will fall back to prior."
                        )
            else:
                print(
                    f"Warning: No candidate achieved sufficient backbone match. Best was {best_name} "
                    f"with effective backbone frac={b_frac:.3f}, wrapper-backbone frac={bw_frac:.3f}, total frac={frac:.3f}. "
                    "Will fall back to prior (avoids random/untrained predictions causing AUC~0.5)."
                )
        else:
            print("Warning: No usable state_dict candidates found; will fall back.")
else:
    print("Warning: MODEL_PATH not found; will not load weights.")

model.eval()



## === cell 5
sample_path = f"{DATA_DIR}/sample_submission.csv"
if os.path.exists(sample_path):
    sample_df = pd.read_csv(sample_path)
    test_img_ids = sample_df["id"].tolist()
else:
    test_img_ids = sorted([p.stem for p in Path(TEST_DIR).glob("*.tif")])

print(f"Number of test images: {len(test_img_ids)}")

test_dataset = HistologyTestDataset(test_img_ids, TEST_DIR, test_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
predictions = []
ids = []

fallback_p = 0.5
train_labels_path = f"{DATA_DIR}/train_labels.csv"
if os.path.exists(train_labels_path):
    tl = pd.read_csv(train_labels_path)
    if "label" in tl.columns:
        fallback_p = float(tl["label"].mean())
print(f"Fallback probability (used only if loaded_ok is False): {fallback_p:.6f}")

model.eval()
with torch.no_grad():
    for images, img_ids in tqdm(test_loader, desc="Generating predictions"):
        ids.extend(list(img_ids))
        if loaded_ok:
            images = images.to(device, non_blocking=True)
            outputs = model(images)

            if outputs.ndim == 2 and outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs[:, 0]).detach().cpu().numpy()
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()

            predictions.extend(probs.tolist())
        else:
            predictions.extend([fallback_p] * len(img_ids))

submission_df = pd.DataFrame({"id": ids, "label": predictions})

assert submission_df.shape[0] == len(test_img_ids), "Submission row count mismatch."
assert submission_df["id"].is_unique, "Duplicate ids in submission."

pred_arr = np.asarray(predictions, dtype=np.float64)
print(
    "Pred stats:",
    "min",
    float(pred_arr.min()),
    "max",
    float(pred_arr.max()),
    "mean",
    float(pred_arr.mean()),
    "std",
    float(pred_arr.std()),
)
if float(pred_arr.std()) < 1e-6:
    print(
        "Warning: predictions are (near-)constant; AUC will be ~0.5. Check checkpoint compatibility."
    )

if os.path.exists(sample_path):
    submission_df = sample_df[["id"]].merge(submission_df, on="id", how="left")
    assert (
        submission_df["label"].notna().all()
    ), "Some ids missing predictions after merge."

submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission file '{SUBMISSION_FILE}' created with shape {submission_df.shape}.")
print(submission_df.head())
