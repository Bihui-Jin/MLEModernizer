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

3.12

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

0.8926602111463993

# 6. Current score

0.0154

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07285) has done: 'I fix the runtime failures by removing the hard dependency on missing external model weight files and making the ensemble robust when no weights are found. To preserve the core “multi-model softmax ensemble” logic, the code try to load each checkpoint if it exists, otherwise it skip that model and proceed with the remaining ones. If none are available (as in your environment), it fall back to using timm pretrained ImageNet weights for the same model architectures so inference can still run end-to-end and produce a valid `submission.csv`. I also fix image loading to ensure RGB inputs (some PNGs can be RGBA/grayscale) and make `state_dict` loading compatible with common checkpoint formats.'
- What this solution (achieved 0.00653) has done: 'Your current score is low because the fallback uses ImageNet-pretrained classifiers with randomly initialized 5-class heads, so predictions are essentially noise. To move the score toward your target while preserving the same “multi-model softmax ensemble” inference logic, the smallest legitimate fix is to actually use diabetic-retinopathy-trained weights; in this environment those are typically available via the competition’s dataset or your Kaggle “input” datasets. I make the code automatically search `/kaggle/input/**` for matching checkpoint filenames (e.g., `resnet18.pth`, `efficentNet_b3.pth`, etc.) and load them when found, otherwise fall back exactly as before. I also ensure that the ensemble weights are renormalized over only the successfully loaded models so the weighted sum stays correctly scaled.'
- What this solution (achieved -0.04809) has done: 'Your score is near-random because none of the DR-trained checkpoints are being found, so the code falls back to ImageNet-pretrained backbones with a random 5-class head. The smallest change to move the score toward your target is to (1) point `model_paths` at the correct dataset name (`aptos-ensamble-models` not `aptos_ensamble-models`) and (2) make checkpoint loading robust to common “head name” differences (e.g., `classifier.*` vs `fc.*`) while keeping the same architectures and weighted-softmax ensemble logic. This should enable actually loading the intended DR-trained weights when present, drastically improving QWK without changing your inference semantics. I also keep the weight renormalization over only the successfully loaded models (already correct) and leave transforms/model list intact.'
- What this solution (achieved 0.01011) has done: 'Your current score is far below the target because the intended DR-trained checkpoints still aren’t being loaded, so the ensemble falls back to ImageNet-pretrained backbones with effectively random 5-class heads. The smallest score-relevant change is to make checkpoint discovery robust to (a) slightly different dataset folder names and (b) the common “efficentNet_*.pth” misspelling by searching for multiple filename variants per model. I also keep your exact ensemble/softmax logic, but tighten checkpoint loading to prefer strict loading after a smarter key-remap (and only then fall back to non-strict), which increases the chance we load the trained classification head correctly. These changes should move QWK substantially upward toward your target without changing the model list, transforms, or prediction semantics.'
- What this solution (achieved 0.06608) has done: 'Your score is far below the target because the code is still not loading DR-trained checkpoints (so predictions are effectively random), so the smallest relevant change is to make checkpoint discovery actually find the common variants in `/kaggle/input` (including `.pt/.bin`, different casing, and common “_foldX”/“best” naming). Then, to increase the chance the trained 5-class head loads correctly without changing model logic, we add a minimal key-normalization step that also handles `model_state_dict`/Lightning-style keys and strips common prefixes (`model.`, `net.`). Finally, we keep your weighted-softmax ensemble exactly the same, but ensure we don’t silently accept a “loaded” checkpoint that has near-zero overlap (which would behave like random weights), falling back to ImageNet only in that case.'
- What this solution (achieved -0.4906) has done: 'Your score is far below the target because the pipeline is still mostly failing to load DR-trained checkpoints, so the ensemble behaves close to random. The smallest score-relevant change is to make checkpoint resolution prefer the intended dataset directory if it exists and to choose the “best match” among multiple hits (instead of the first arbitrary glob result), which increases the chance we load the correct trained weights and the correct 5-class head. I also add a minimal, safe fallback to handle checkpoints saved as a full `nn.Module` (common in Kaggle) without changing any model/inference logic. Everything else (models, transforms, weighted-softmax ensemble, argmax predictions, submission format) is kept identical.'
- What this solution (achieved 0.0154) has done: 'Your current score (-0.4906) indicates the ensemble is effectively random, which is most consistent with DR-trained checkpoints not being found/loaded in this environment (so you fall back to ImageNet models with a random 5-class head). The smallest score-relevant change is to avoid silently skipping missing checkpoints by falling back *per-model* to ImageNet-pretrained backbones **with a deterministic 5-class head initialization**, so at least the ensemble produces stable, non-degenerate outputs and avoids pathological randomness that can push QWK negative. This preserves your exact “multi-model softmax + weighted sum + argmax” inference semantics and keeps architectures/transforms unchanged. I also make checkpoint resolution consider both `/kaggle/input` and `/kaggle/data` (your files exist under both), which slightly increases the chance of finding the intended DR-trained weights without changing core logic.'
- What this solution (achieved 0.0154) has done: 'Your score is far below the target, and the most likely reason (given your logs/plans) is still that none of the DR-trained checkpoints are actually being found/loaded, so you’re effectively ensembling ImageNet backbones with a random 5-class head. To move the score upward toward the target without changing the ensemble/softmax/argmax core logic, I (1) add an explicit fast scan of `/kaggle/input` and `/kaggle/data` to auto-detect the real checkpoint root directory if it exists, then (2) resolve each model checkpoint by “best match” against the full filename patterns, and (3) strengthen checkpoint loading to handle common formats (full `nn.Module`, `state_dict`, and nested dicts) and common key prefixes/head names. These are minimal, score-relevant changes that only affect whether the intended trained weights are loaded; inference semantics remain the same. The script still fall back to the deterministic ImageNet-pretrained path per-model if a checkpoint truly isn’t present, and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.0154) has done: 'Your current score (0.0154) is far below the target (0.8927), and the most likely cause is still that none (or almost none) of the intended DR-trained checkpoints are being loaded—so the ensemble is effectively ImageNet backbones with a freshly initialized 5-class head. To move the score upward with minimal changes and identical ensemble/softmax/argmax semantics, I (1) add a fast existence check for the expected `/kaggle/input/aptos-ensamble-models/...` dataset and broaden checkpoint discovery to also consider `.pth` files under that dataset even when filenames don’t match, and (2) strengthen state_dict key normalization specifically for common timm EfficientNet naming (`classifier.*`/`head.*`) without changing architectures. Finally, I print a clear warning if we still fail to load any DR checkpoints so it’s obvious when the run is doomed to near-random performance, while still producing a valid `submission.csv`.'
- What this solution (achieved 0.0154) has done: 'I remove the hard failure when DR-specific checkpoints are missing and instead fall back per-model to a timm ImageNet-pretrained backbone with a deterministic 5-class head, so the pipeline always runs end-to-end and writes `submission.csv`. This fixes the immediate `FileNotFoundError` in model loading, which currently prevents any submission from being created. I also guard the inference loop so it cannot crash with `weighted_sum=None` (which happens when zero models are active). These changes preserve your core “multi-model softmax + weighted sum + argmax” ensemble semantics while making execution robust in this environment.'

# 9. Code solution

## === cell 0
import os
import glob
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def _ensure_cache(self):
        if not hasattr(self, "_id_codes"):
            self._id_codes = self.annotations.iloc[:, 0].astype(str).to_numpy()
            if not self.test and self.annotations.shape[1] > 1:
                self._labels = self.annotations.iloc[:, 1].astype(np.int64).to_numpy()
            else:
                self._labels = None

    def __getitem__(self, idx):
        self._ensure_cache()
        img_name = os.path.join(self.root_dir, self._id_codes[idx] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self._labels[idx])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)

_num_workers = min(4, (os.cpu_count() or 4))
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/resnet18.pth",
    "efficientnet_b0": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b0.pth",
    "efficientnet_b1": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "efficientnet_b4": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b4.pth",
    "efficientnet_b5": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 5
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}
total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_deterministic(seed: int = 1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_set_deterministic(1337)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if isinstance(ckpt, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        tensor_like = any(torch.is_tensor(v) for v in ckpt.values())
        if tensor_like:
            return ckpt

    return ckpt


def _normalize_state_dict_keys(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "student.", "encoder."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _candidate_checkpoint_filenames(model_key: str, original_path: str) -> list[str]:
    base = os.path.basename(original_path)
    stem, ext = os.path.splitext(base)
    exts = [ext] if ext else [".pth"]
    exts = list(dict.fromkeys(exts + [".pth", ".pt", ".bin"]))

    cands = set()
    for e in exts:
        cands.add(stem + e)

    suffixes = [
        "",
        "_best",
        "-best",
        "_final",
        "-final",
        "_last",
        "-last",
        "_best_kappa",
        "-best_kappa",
    ]
    fold_suffixes = [
        "",
        "_fold0",
        "_fold1",
        "_fold2",
        "_fold3",
        "_fold4",
        "-fold0",
        "-fold1",
        "-fold2",
        "-fold3",
        "-fold4",
        "_fold_0",
        "_fold_1",
        "_fold_2",
        "_fold_3",
        "_fold_4",
    ]
    for e in exts:
        for s in suffixes:
            for f in fold_suffixes:
                cands.add(f"{stem}{s}{f}{e}")

    if model_key.startswith("efficientnet_b"):
        b = model_key.split("_")[-1]  # b0..b5
        bases = [
            f"efficientnet_{b}",
            f"efficientnet-{b}",
            f"efficentNet_{b}",
            f"efficentnet_{b}",
            f"EfficientNet_{b}",
            f"EfficientNet-{b}",
            f"efficientNet_{b}",
        ]
        for bstem in bases:
            for e in exts:
                cands.add(bstem + e)
                for s in suffixes:
                    for f in fold_suffixes:
                        cands.add(f"{bstem}{s}{f}{e}")

    return list(cands)


_PREFERRED_CKPT_ROOTS = [
    "/kaggle/input/aptos-ensamble-models",
    "/kaggle/input/aptos_ensamble_models",
    "/kaggle/input/aptos-ensemble-models",
    "/kaggle/input/aptos_ensemble_models",
    "/kaggle/data/aptos-ensamble-models",
    "/kaggle/data/aptos_ensamble_models",
    "/kaggle/data/aptos-ensemble-models",
    "/kaggle/data/aptos_ensemble_models",
]


def _pick_best_checkpoint_hit(
    hits: list[str], original_path: str, model_key: str
) -> str | None:
    if not hits:
        return None
    target_base = os.path.basename(original_path).lower()

    def score(p: str) -> tuple:
        pl = p.lower()
        preferred_ds = any(
            root in pl
            for root in (
                "aptos-ensamble-models",
                "aptos_ensamble_models",
                "ensamble_v2",
            )
        )
        same_base = os.path.basename(p).lower() == target_base
        contains_model = model_key.replace("_", "") in pl.replace("_", "")
        return (preferred_ds, same_base, contains_model, -len(pl))

    return sorted(hits, key=score, reverse=True)[0]


def _resolve_checkpoint_path(original_path: str, model_key: str) -> str | None:
    if os.path.exists(original_path):
        return original_path

    orig_dir = os.path.dirname(original_path)
    for fname in _candidate_checkpoint_filenames(model_key, original_path):
        trial = os.path.join(orig_dir, fname)
        if os.path.exists(trial):
            return trial

    for root in _PREFERRED_CKPT_ROOTS:
        if not os.path.exists(root):
            continue
        patterns = [
            os.path.join(root, "**", os.path.basename(original_path)),
        ]
        for fname in _candidate_checkpoint_filenames(model_key, original_path):
            patterns.append(os.path.join(root, "**", fname))

        hits = []
        for pat in patterns:
            hits.extend(glob.glob(pat, recursive=True))
        best = _pick_best_checkpoint_hit(hits, original_path, model_key)
        if best is not None and os.path.exists(best):
            return best

    return None


def _remap_head_keys_if_needed(sd: dict, model: nn.Module) -> dict:
    if not isinstance(sd, dict):
        return sd

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    sd_keys = set(sd.keys())

    if len(sd_keys & model_keys) / max(1, len(model_keys)) > 0.90:
        return sd

    remap_candidates = [
        ("classifier.", "fc."),
        ("fc.", "classifier."),
        ("head.", "fc."),
        ("fc.", "head."),
        ("head.fc.", "fc."),
        ("model.classifier.", "classifier."),
        ("model.fc.", "fc."),
        ("classifier.fc.", "classifier."),
        ("head.fc.", "classifier."),
        ("classifier.", "head."),
    ]

    best_sd = sd
    best_overlap = len(sd_keys & model_keys)

    for src, dst in remap_candidates:
        trial = {}
        for k, v in sd.items():
            if k.startswith(src):
                trial[dst + k[len(src) :]] = v
            else:
                trial[k] = v
        overlap = len(set(trial.keys()) & model_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sd = trial

    return best_sd


def _init_5class_head_deterministically(model: nn.Module, seed: int = 1337):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for _, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight, mean=0.0, std=0.02, generator=g)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)




## === cell 7
models_list = []
active_model_keys = []
loaded_paths = {}
missing = []
fallback_used = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    resolved = _resolve_checkpoint_path(path, model_key)

    if resolved is None:
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        _init_5class_head_deterministically(model, seed=1337)
        model.to(device).eval()
        models_list.append(model)
        active_model_keys.append(model_key)
        fallback_used.append(model_key)
        missing.append((model_key, path))
        continue

    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    ckpt = torch.load(resolved, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    if isinstance(sd, dict):
        sd = _normalize_state_dict_keys(sd)
        sd = _remap_head_keys_if_needed(sd, model)

    model_keys_set = set(model.state_dict().keys())
    sd_keys_set = set(sd.keys()) if isinstance(sd, dict) else set()
    overlap_ratio = (
        (len(model_keys_set & sd_keys_set) / max(1, len(model_keys_set)))
        if isinstance(sd, dict)
        else 0.0
    )

    if overlap_ratio < 0.20:
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        _init_5class_head_deterministically(model, seed=1337)
        model.to(device).eval()
        models_list.append(model)
        active_model_keys.append(model_key)
        fallback_used.append(model_key)
        missing.append((model_key, path))
        continue

    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        model.load_state_dict(sd, strict=False)

    loaded_paths[model_key] = resolved
    model.to(device).eval()
    models_list.append(model)
    active_model_keys.append(model_key)

active_total = sum(weights[k] for k in active_model_keys)
active_weights = {k: (weights[k] / active_total) for k in active_model_keys}

print("Active models:", active_model_keys)
print("Loaded checkpoints:", {k: loaded_paths.get(k, None) for k in active_model_keys})
if missing:
    print("WARNING: Missing/unloadable checkpoints; used fallback for:", fallback_used)
    print("Missing list (first 5):", missing[:5])
print(
    "Num models:",
    len(models_list),
    "| weight sum:",
    float(sum(active_weights.values())),
)



## === cell 8
all_outputs = []
softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)

        weighted_sum = None
        for model_key, model in zip(active_model_keys, models_list):
            probs = softmax(model(images))
            if weighted_sum is None:
                weighted_sum = probs.mul(active_weights[model_key])
            else:
                weighted_sum.add_(probs, alpha=float(active_weights[model_key]))

        if weighted_sum is None:
            batch_size = images.shape[0]
            weighted_sum = torch.zeros((batch_size, 5), device=images.device)
            weighted_sum[:, 0] = 1.0

        all_outputs.append(weighted_sum.cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 9
test_ids = pd.read_csv(test_csv_file)["id_code"].astype(str).to_numpy()
assert len(test_ids) == len(final_predictions), (len(test_ids), len(final_predictions))

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("rows:", len(submission_df), "cols:", list(submission_df.columns))
