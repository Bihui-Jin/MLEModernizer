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

0.8883768355086908

# 6. Current score

0.04784

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02982) has done: 'I fix the model loading crash by preventing ImageNet weights from overwriting the final `fc` layer when the checkpoint is missing; this preserves your architecture and lets inference run. I also make the dataloader more robust in Kaggle by using a safe `num_workers` default and ensure `out` tensors are concatenated correctly (the current `stack(list_of_batches)` pattern can mis-shape). Finally, I keep the same argmax-over-ensemble-logits prediction logic but make sure ID ordering is validated and the submission is written with the required columns to `submission.csv`.'
- What this solution (achieved 0.02443) has done: 'Your current score is low because the “fallback” path is being triggered (the `/kaggle/input/resnet/*.pt` checkpoints aren’t present), so you are ensembling essentially untrained random classification heads on ImageNet backbones. The smallest change that moves you toward the target is to ensure the exact same architecture can actually load weights by (1) auto-discovering the checkpoint files if they exist anywhere under `/kaggle/input`, and (2) making the load tolerant to common checkpoint formats (`state_dict`, `model_state_dict`, `module.` prefixes). This preserves your model architecture and argmax-over-ensemble-logits semantics, but makes it much more likely you’re using real trained weights instead of random heads. If no checkpoints exist, the code still produces a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.01736) has done: 'Your score is extremely low because the code is almost certainly running the fallback path (ImageNet backbone + random 5-class head), so predictions are near-random for this task; the smallest legitimate improvement is to actually load the intended trained checkpoints. I keep the exact same ResNet architectures and the same “average logits then argmax” ensemble semantics, but make checkpoint discovery more robust by (1) searching for any `.pt/.pth` files whose names contain the expected substrings (not just exact basenames) and (2) supporting common checkpoint wrappers while also filtering to matching tensor shapes so a head-trained checkpoint can still load cleanly. I also ensure inference uses `torch.inference_mode()` (same semantics as `no_grad()` but safer/faster) and keep submission alignment against `sample_submission.csv` unchanged. If no compatible checkpoints exist anywhere under `/kaggle/input`, behavior stays the same as now (fallback), but when they do exist this should move your score substantially toward the target.'
- What this solution (achieved -0.22964) has done: 'Your score is near-random, which is consistent with the fallback path being used (ImageNet backbone + randomly initialized 5-class head) because no compatible checkpoints are found/loaded. The smallest change that should move you strongly toward the target (without changing architecture, loss, or training) is to: (1) search more broadly under both `/kaggle/input` and `/kaggle/data` for any plausible `.pt/.pth` checkpoints, and (2) also accept common formats where the entire `nn.Module` was saved (not just a state_dict). Finally, we keep your exact “average logits then argmax” ensembling semantics, but add `softmax` before averaging as a minimal calibration fix for mixing models whose logit scales may differ; this often improves quadratic kappa without changing the predicted label space or requiring retraining.'
- What this solution (achieved 0.09487) has done: 'Your current score indicates the models are still effectively “fallback” (ImageNet backbone + random 5-class head) because no compatible checkpoints are being found/loaded, so the smallest high-impact fix is to reliably load the competition weights if they exist anywhere in the dataset folders. I keep your exact ResNet architectures and the same ensemble semantics (softmax → average → argmax), but strengthen checkpoint discovery by prioritizing `.pt/.pth` files with matching `arch` substrings and preferring larger files (real checkpoints) over tiny placeholders. I also handle one more common wrapper format (`{"model": nn.Module}`) without changing any training/inference logic. If no checkpoints exist, behavior remains the same, but when they do, this should move the score substantially toward the target.'
- What this solution (achieved -0.0233) has done: 'Your score is still far from the target, and the most plausible cause (given your own logs/plans) is that you’re still falling back to ImageNet backbones with random 5-class heads because no real competition checkpoints are being loaded. The minimal, high-impact change is therefore to make checkpoint discovery actually find the common APTOS pretrained model files that are often shipped as `.pth` inside other datasets (and not necessarily containing your exact query tokens), by (1) widening the search to any reasonably-large `.pt/.pth` files and (2) selecting the best match by architecture keywords plus file size. This preserves your exact model architectures and the same softmax-average-argmax ensemble semantics; it only increases the likelihood that you load real trained weights instead of random heads. If no suitable checkpoints exist, it still run end-to-end and write a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.00832) has done: 'Your current kappa is far below target, which is consistent with the code still failing to load real trained checkpoints and falling back to ImageNet backbones + random 5-class heads. The smallest high-impact change is to make checkpoint discovery actually prioritize APTOS/DR-relevant `.pt/.pth` files (by adding competition-keyword scoring, excluding tiny/lightning/optimizer-only files, and preferring paths whose filename tokens match the requested `query`/`arch`). To avoid accidentally “loading something” that isn’t a classifier for 5 classes, we also add a very small validation that the loaded state_dict contains an `fc.*` compatible with 5 classes; otherwise we keep the current safe fallback behavior. This preserves your exact model architectures and the same softmax-average-argmax ensemble semantics, but increases the chance you’re ensembling real competition weights, which should move the score strongly toward the target.'
- What this solution (achieved 0.19983) has done: 'Your score is still near-random, which most plausibly means you’re not loading any real APTOS-trained weights and are instead ensembling ImageNet backbones with random 5-class heads. The minimal change that should move you strongly toward the target is to fix a bug in the checkpoint-loading path (`_has_5class_fc_in_state` is referenced but the defined function is `_has_5class_fc_in_state_dict`), which currently forces the fallback even when a good checkpoint is found. I also add one tiny, safe enhancement: if a checkpoint doesn’t contain `fc.*` but does contain a compatible `classifier.*` (common in some saved models), we map it to `fc.*` before loading—this keeps the same ResNet architecture and inference semantics, but increases the chance the 5-class head is actually loaded. Everything else (transforms, ResNet152/101 ensemble, softmax-average-argmax, submission alignment) is kept the same.'
- What this solution (achieved -0.00303) has done: 'Your current score (0.19983) is far below the target (0.88838), so we should only make a small, high-impact fix that preserves your exact ensemble/inference semantics. The most likely remaining issue is that checkpoints may load but still not actually populate the final 5-class head because the head keys can appear as `model.fc.*` (common when saving a wrapped module) and your `_has_5class_fc_in_state_dict` check then falsely rejects the checkpoint and falls back to ImageNet+random head. I minimally extend the state-dict normalization to also strip a leading `model.` prefix and map `model.fc.*` / `model.classifier.*` into `fc.*` before the 5-class-head check and shape-filtering. This keeps the same ResNet architectures, same softmax-average-argmax ensemble logic, and should substantially increase the chance you actually use the intended trained weights, moving QWK toward the target.'
- What this solution (achieved 0.0) has done: 'Your score is near-random, which most strongly suggests your ensemble is still running in “fallback” mode (ImageNet backbone + randomly initialized 5-class head) because no valid APTOS-trained checkpoints are actually being loaded. The smallest high-impact change that preserves your model/ensemble semantics is to allow loading checkpoints that *don’t* contain `fc.*` by still loading the backbone weights from the checkpoint (instead of discarding it and reverting to ImageNet), while keeping your `fc` randomly initialized in that case. Additionally, many real checkpoints store the head as `linear.*` (common in timm-style training), so we minimally map `linear.*` to `fc.*` the same way you already map `classifier.*`. These two changes increase the chance you’re using task-relevant pretrained features (moving QWK upward toward the target) without changing architecture, transforms, or prediction logic.'
- What this solution (achieved 0.00039) has done: 'Your current 0.0 score is consistent with the test prediction file being valid but the model outputs being essentially garbage (typically because the intended checkpoints are not actually being found/loaded, so you’re using ImageNet backbones with random 5-class heads). The smallest high-impact change toward your target is to make checkpoint discovery also consider common Kaggle dataset formats where weights are stored inside `.ckpt` (PyTorch Lightning) files, then extract and normalize their `state_dict` keys so they can load into your unchanged ResNet architectures. This keeps your ensemble logic (softmax-average-argmax) and transforms identical, but greatly increases the chance you load real DR-trained weights instead of random heads. If no compatible checkpoints exist, behavior remains the same and a valid `submission.csv` is still produced.'
- What this solution (achieved 0.02337) has done: 'Your current score is far below the target, which is consistent with the ensemble still not loading any real DR-trained checkpoints and effectively behaving like ImageNet backbones + random 5-class heads. The smallest high-impact change is to broaden checkpoint discovery to also consider common weight file extensions used on Kaggle (`.bin`/`.pkl`/`.ptl`) and, crucially, to also search inside common archive containers (`.zip`) under `/kaggle/input` and `/kaggle/data` for large DR-related checkpoints, extracting just the best-matching one to `/kaggle/working` for loading. This preserves your ResNet architectures, transforms, and the same softmax-average-argmax ensembling semantics; it only increases the chance that real pretrained competition weights are found and loaded. If no compatible checkpoints exist anywhere, the code still runs end-to-end and writes a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.04784) has done: 'Your current score is far below the target, so the highest-impact minimal fix is to ensure you actually load real APTOS-trained weights rather than silently falling back to ImageNet + a random 5-class head. I keep your exact ResNet152/101 ensemble, transforms, and softmax-average-argmax logic, but make checkpoint discovery also look for common “resnet*.pth” files anywhere under the provided APTOS dataset folders, and also load weights stored as `.npz` (a common Kaggle format) by converting them into a PyTorch state_dict. I also add a small safety fallback: if no usable checkpoints are found, we still write a valid `submission.csv` exactly as before. These changes are directly aimed at moving QWK upward toward your target without changing model semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm



## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        row = self.eye_frame.iloc[idx]
        img_id = row["id_code"]

        if self.filetype == "train":
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/train_images",
                img_id + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(row["diagnosis"])
        else:
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/test_images",
                img_id + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, img_id




## === cell 3
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
test_dataset = APTOSDataset(
    csv_file=test_csv_path, filetype="test", transform=transform
)

num_workers = 4
try:
    cpu_cnt = os.cpu_count() or 2
    num_workers = min(4, max(0, cpu_cnt - 1))
except Exception:
    num_workers = 0

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device, "| num_workers:", num_workers)



## === cell 4
import zipfile
import shutil

_SEARCH_ROOTS = ["/kaggle/input", "/kaggle/data"]
_EXTRACT_DIR = "/kaggle/working/_extracted_checkpoints"
os.makedirs(_EXTRACT_DIR, exist_ok=True)


def _tokenize(s: str):
    if not s:
        return []
    s = s.lower().replace("-", "_").replace(" ", "_")
    toks = [t for t in s.split("_") if t]
    return toks


def _looks_like_aptos_checkpoint(path: str) -> bool:
    """
    Avoid selecting large but irrelevant checkpoints which can partially load and behave
    like random heads. Keep search focused on DR/retina classification checkpoints.
    """
    pl = path.lower()
    bad_tokens = [
        "optimizer",
        "adam",
        "sched",
        "scheduler",
        "optstate",
        "scaler",
        "ema",
        "loss",
        "history",
        "metrics",
        "fold",
        "kfold",
        "tfrecord",
        "tokenizer",
        "bert",
        "gpt",
        "yolo",
        "unet",
        "seg",
        "segmentation",
        "mask",
        "deeplab",
        "detect",
        "fasterrcnn",
        "retinanet",
        "ssd",
        "rnn",
        "lstm",
        "transformer",
        "audio",
        "mmdet",
        "mmseg",
        "nvidia",
        "cuda",
        "trt",
        "tensorrt",
        "onnx",
    ]
    if any(tok in pl for tok in bad_tokens):
        return False

    good_tokens = [
        "aptos",
        "blind",
        "blindness",
        "retina",
        "retinopathy",
        "diabetic",
        "dr",
        "fundus",
        "eye",
        "resnet",
        "efficientnet",
        "densenet",
        "seresnet",
    ]
    return any(tok in pl for tok in good_tokens)


def _safe_extract_one_from_zip(zip_path: str, member_name: str, out_path: str) -> str:
    """
    Score-related change:
    Some Kaggle datasets store checkpoints inside zip archives. Extracting the best candidate
    increases the chance we load real DR-trained weights (vs fallback random head), without
    changing architecture/inference semantics.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name, "r") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    return out_path


def _npz_to_state_dict(npz_obj) -> dict:
    """
    Score-related change:
    Some Kaggle weights are distributed as .npz (numpy savez) containing key->array mappings.
    Converting those arrays into torch tensors allows us to load real pretrained weights
    without changing model architecture or inference logic.
    """
    sd = {}
    try:
        keys = list(npz_obj.files)
    except Exception:
        return sd
    for k in keys:
        arr = npz_obj[k]
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr)
            if t.dtype in (torch.float64, torch.float16, torch.bfloat16):
                t = t.float()
            sd[k] = t
    return sd


def _find_checkpoint_path(
    preferred_path: str, basename: str, query: str = None, arch: str = None
):
    """
    Minimal score-related change:
    - Also consider checkpoints inside '.zip' archives under /kaggle/input and /kaggle/data
      (common Kaggle dataset packaging), extracting only the best match to /kaggle/working.
    - Also allow common weight extensions besides .pt/.pth/.ckpt, without changing model logic.
    - Additionally, allow a slightly broader "resnet*.(pth/pt/ckpt/npz)" match even if it
      doesn't contain APTOS keywords; this is targeted to fix cases where weights are named
      generically (e.g., 'resnet101.pth') within the APTOS dataset folder.
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    if basename:
        for sr in _SEARCH_ROOTS:
            if os.path.exists(sr):
                for root, _, files in os.walk(sr):
                    if basename in files:
                        return os.path.join(root, basename)

    q_tokens = _tokenize(query) if query else []
    arch_tokens = _tokenize(arch) if arch else []

    candidates = []
    allowed_ext = (".pt", ".pth", ".ckpt", ".bin", ".pkl", ".ptl", ".npz")

    def _arch_name_ok(filename_lower: str) -> bool:
        if not arch_tokens:
            return True
        if "resnet152" in arch_tokens:
            return ("resnet152" in filename_lower) or ("resnet_152" in filename_lower)
        if "resnet101" in arch_tokens:
            return ("resnet101" in filename_lower) or ("resnet_101" in filename_lower)
        return any(tok in filename_lower for tok in arch_tokens)

    for sr in _SEARCH_ROOTS:
        if not os.path.exists(sr):
            continue
        for root, _, files in os.walk(sr):
            for f in files:
                fl = f.lower()
                if not fl.endswith(allowed_ext):
                    continue
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p)
                except Exception:
                    continue

                if sz < 2_000_000:
                    continue

                in_aptos_tree = "aptos2019-blindness-detection" in p.lower()
                generic_resnet_ok = (
                    in_aptos_tree and _arch_name_ok(fl) and ("resnet" in fl)
                )

                if not generic_resnet_ok and not _looks_like_aptos_checkpoint(p):
                    continue

                arch_match = sum(tok in fl for tok in arch_tokens) if arch_tokens else 0
                query_match = sum(tok in fl for tok in q_tokens) if q_tokens else 0
                comp_tokens = [
                    "aptos",
                    "blind",
                    "blindness",
                    "retina",
                    "retinopathy",
                    "diabetic",
                    "fundus",
                ]
                comp_match = sum(tok in fl for tok in comp_tokens)

                full_path_l = p.lower()
                path_query_bonus = (
                    sum(tok in full_path_l for tok in q_tokens) if q_tokens else 0
                )

                score = (
                    (12 * arch_match)
                    + (4 * query_match)
                    + (3 * comp_match)
                    + (2 * path_query_bonus)
                )
                if generic_resnet_ok:
                    score += 5  # small targeted boost to pick up generic resnet weights

                ext_bonus = (
                    2 if fl.endswith(".pth") else (1 if fl.endswith(".ckpt") else 0)
                )
                score += ext_bonus

                candidates.append(("file", score, sz, p, None))

    zip_candidates = []
    for sr in _SEARCH_ROOTS:
        if not os.path.exists(sr):
            continue
        for root, _, files in os.walk(sr):
            for f in files:
                if not f.lower().endswith(".zip"):
                    continue
                zip_path = os.path.join(root, f)
                try:
                    if os.path.getsize(zip_path) < 2_000_000:
                        continue
                except Exception:
                    continue

                zpl = zip_path.lower()
                if not _looks_like_aptos_checkpoint(zip_path) and not any(
                    tok in zpl
                    for tok in (
                        "resnet",
                        "aptos",
                        "retina",
                        "retinopathy",
                        "diabetic",
                        "fundus",
                        "dr",
                    )
                ):
                    continue

                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        for info in zf.infolist():
                            name = info.filename
                            nl = name.lower()
                            if not nl.endswith(allowed_ext):
                                continue
                            if info.file_size < 2_000_000:
                                continue

                            in_aptos_zip = (
                                "aptos2019-blindness-detection" in zip_path.lower()
                            )
                            generic_resnet_ok = (
                                in_aptos_zip and _arch_name_ok(nl) and ("resnet" in nl)
                            )
                            if (
                                not generic_resnet_ok
                                and not _looks_like_aptos_checkpoint(name)
                            ):
                                continue

                            arch_match = (
                                sum(tok in nl for tok in arch_tokens)
                                if arch_tokens
                                else 0
                            )
                            query_match = (
                                sum(tok in nl for tok in q_tokens) if q_tokens else 0
                            )
                            comp_tokens = [
                                "aptos",
                                "blind",
                                "blindness",
                                "retina",
                                "retinopathy",
                                "diabetic",
                                "fundus",
                            ]
                            comp_match = sum(tok in nl for tok in comp_tokens)
                            score = (
                                (12 * arch_match) + (4 * query_match) + (3 * comp_match)
                            )
                            if generic_resnet_ok:
                                score += 5

                            ext_bonus = (
                                2
                                if nl.endswith(".pth")
                                else (1 if nl.endswith(".ckpt") else 0)
                            )
                            score += ext_bonus

                            zip_candidates.append(
                                ("zip", score, info.file_size, zip_path, name)
                            )
                except Exception:
                    continue

    candidates.extend(zip_candidates)

    if candidates:
        candidates.sort(
            key=lambda x: (-x[1], -x[2], str(x[3]), str(x[4]) if x[4] else "")
        )
        best_kind, best_score, best_sz, best_path, best_member = candidates[0]

        if best_kind == "file":
            return best_path

        safe_name = os.path.basename(best_member)
        out_path = os.path.join(_EXTRACT_DIR, safe_name)
        if not os.path.exists(out_path) or os.path.getsize(out_path) != best_sz:
            try:
                _safe_extract_one_from_zip(best_path, best_member, out_path)
                print(
                    f"Extracted checkpoint from zip: {best_path}::{best_member} -> {out_path}"
                )
            except Exception as e:
                print(f"WARNING: Failed extracting {best_member} from {best_path}: {e}")
                return preferred_path
        return out_path

    return preferred_path


def _extract_state_dict(loaded_obj):
    """
    Minimal score-related change:
    Handle common PyTorch Lightning ckpt structure where weights live under 'state_dict'.
    Also support nested 'model' dicts.
    """
    if isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return loaded_obj["state_dict"]
        if "model_state_dict" in loaded_obj and isinstance(
            loaded_obj["model_state_dict"], dict
        ):
            return loaded_obj["model_state_dict"]
        if "model" in loaded_obj and isinstance(loaded_obj["model"], dict):
            return loaded_obj["model"]
    return loaded_obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _strip_model_prefix(state_dict):
    """
    Many training scripts save keys like 'model.fc.weight'/'model.layer1.*'.
    Stripping 'model.' improves checkpoint compatibility without changing semantics.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("model.") for k in state_dict.keys()):
        return {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _strip_pl_prefixes(state_dict):
    """
    Minimal score-related change:
    PyTorch Lightning often prefixes keys with 'net.'/'model.'/'backbone.' etc.
    We only strip a few common ones in a conservative way to increase compatibility.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ("net.", "network.", "backbone.", "encoder.")
    if any(k.startswith(prefixes) for k in state_dict.keys()):
        out = {}
        for k, v in state_dict.items():
            kk = k
            for pref in prefixes:
                if kk.startswith(pref):
                    kk = kk.replace(pref, "", 1)
            out[kk] = v
        return out
    return state_dict


def _filter_state_dict_by_shape(state_dict, model):
    if not isinstance(state_dict, dict):
        return state_dict, 0
    model_sd = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) == tuple(model_sd[k].shape):
                filtered[k] = v
            else:
                dropped += 1
        else:
            dropped += 1
    return filtered, dropped


def _has_5class_fc_in_state_dict(sd: dict) -> bool:
    w = sd.get("fc.weight", None) if isinstance(sd, dict) else None
    b = sd.get("fc.bias", None) if isinstance(sd, dict) else None
    if w is None or b is None:
        return False
    try:
        return (w.shape[0] == 5) and (b.shape[0] == 5)
    except Exception:
        return False


def _maybe_map_classifier_to_fc(sd: dict) -> dict:
    """
    Support additional common head key names (e.g. 'linear.*' from timm-style training),
    increasing the chance the 5-class head loads correctly.
    """
    if not isinstance(sd, dict):
        return sd

    if "fc.weight" in sd and "fc.bias" in sd:
        return sd

    sd = dict(sd)

    if "classifier.weight" in sd and "classifier.bias" in sd:
        sd["fc.weight"] = sd.pop("classifier.weight")
        sd["fc.bias"] = sd.pop("classifier.bias")

    if "linear.weight" in sd and "linear.bias" in sd and "fc.weight" not in sd:
        sd["fc.weight"] = sd.pop("linear.weight")
        sd["fc.bias"] = sd.pop("linear.bias")

    if "model.fc.weight" in sd and "model.fc.bias" in sd and "fc.weight" not in sd:
        sd["fc.weight"] = sd.pop("model.fc.weight")
        sd["fc.bias"] = sd.pop("model.fc.bias")

    if (
        "model.classifier.weight" in sd
        and "model.classifier.bias" in sd
        and "fc.weight" not in sd
    ):
        sd["fc.weight"] = sd.pop("model.classifier.weight")
        sd["fc.bias"] = sd.pop("model.classifier.bias")

    if (
        "model.linear.weight" in sd
        and "model.linear.bias" in sd
        and "fc.weight" not in sd
    ):
        sd["fc.weight"] = sd.pop("model.linear.weight")
        sd["fc.bias"] = sd.pop("model.linear.bias")

    return sd


def _try_load_entire_module(loaded_obj, model):
    if (
        isinstance(loaded_obj, dict)
        and "model" in loaded_obj
        and isinstance(loaded_obj["model"], nn.Module)
    ):
        loaded_obj = loaded_obj["model"]

    if isinstance(loaded_obj, nn.Module):
        sd = loaded_obj.state_dict()
        sd = _strip_module_prefix(sd)
        sd = _strip_model_prefix(sd)
        sd = _strip_pl_prefixes(sd)
        sd = _maybe_map_classifier_to_fc(sd)
        sd, dropped = _filter_state_dict_by_shape(sd, model)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return (
            True,
            f"Loaded from full nn.Module | kept: {len(sd)} dropped: {dropped} missing: {len(missing)} unexpected: {len(unexpected)}",
        )
    return False, ""


def _drop_fc_keys(sd: dict) -> dict:
    """
    If a checkpoint doesn't have a compatible 5-class head, still load its backbone
    weights while keeping our fc random (better than full fallback).
    """
    if not isinstance(sd, dict):
        return sd
    sd = dict(sd)
    sd.pop("fc.weight", None)
    sd.pop("fc.bias", None)
    return sd


def _load_checkpoint_object(path: str):
    """
    Score-related change:
    Add support for .npz weight files (common Kaggle format).
    This increases chance of loading real competition weights without changing model logic.
    """
    pl = path.lower()
    if pl.endswith(".npz"):
        npz = np.load(path, allow_pickle=True)
        return _npz_to_state_dict(npz)
    return torch.load(path, map_location="cpu")


def _load_resnet_with_optional_ckpt(
    arch: str, ckpt_path: str, num_classes: int = 5, query: str = None
):
    if arch == "resnet152":
        model = torchvision.models.resnet152(weights=None)
        base = torchvision.models.resnet152(
            weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        )
    elif arch == "resnet101":
        model = torchvision.models.resnet101(weights=None)
        base = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        )
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    ckpt_path_resolved = ckpt_path
    if ckpt_path is not None:
        ckpt_path_resolved = _find_checkpoint_path(
            ckpt_path, os.path.basename(ckpt_path), query=query, arch=arch
        )

    if ckpt_path_resolved and os.path.exists(ckpt_path_resolved):
        loaded = _load_checkpoint_object(ckpt_path_resolved)

        ok, msg = _try_load_entire_module(loaded, model)
        if ok:
            print(f"Loaded checkpoint: {ckpt_path_resolved}")
            print(msg)
        else:
            state = _extract_state_dict(loaded)
            state = _strip_module_prefix(state)
            state = _strip_model_prefix(state)
            state = _strip_pl_prefixes(state)
            state = _maybe_map_classifier_to_fc(state)

            if not _has_5class_fc_in_state_dict(state):
                print(
                    f"WARNING: Checkpoint lacks compatible 5-class head; loading backbone-only: {ckpt_path_resolved}"
                )
                backbone_state = _drop_fc_keys(state)
                backbone_state, dropped = _filter_state_dict_by_shape(
                    backbone_state, model
                )
                missing, unexpected = model.load_state_dict(
                    backbone_state, strict=False
                )
                print(
                    f"Backbone-only ckpt load (non-strict) | kept: {len(backbone_state)} dropped(shape/key): {dropped} "
                    f"missing: {len(missing)} unexpected: {len(unexpected)}"
                )
            else:
                state, dropped = _filter_state_dict_by_shape(state, model)
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint: {ckpt_path_resolved}")
                print(
                    f"Checkpoint load (non-strict) | kept: {len(state)} dropped(shape/key): {dropped} "
                    f"missing: {len(missing)} unexpected: {len(unexpected)}"
                )
    else:
        base_state = base.state_dict()
        base_state.pop("fc.weight", None)
        base_state.pop("fc.bias", None)
        missing, unexpected = model.load_state_dict(base_state, strict=False)
        print(
            f"WARNING: Checkpoint not found: {ckpt_path} (resolved: {ckpt_path_resolved})"
        )
        print(
            f"Falling back to ImageNet weights for {arch} (fc head randomly initialized). "
            f"Missing keys: {len(missing)}, unexpected: {len(unexpected)}"
        )

    return model.to(device)


model0 = _load_resnet_with_optional_ckpt(
    "resnet152", "../input/resnet/FinalResnet152_0.pt", query="finalresnet152"
)
model1 = _load_resnet_with_optional_ckpt(
    "resnet101", "../input/resnet/FinalResnet02.pt", query="finalresnet02"
)
model2 = _load_resnet_with_optional_ckpt(
    "resnet101", "../input/resnet/FinalResnet01.pt", query="finalresnet01"
)




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
        out_batches = []
        for inputs, img_id in tqdm(data_loader, desc="Predict(test)"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().cpu().tolist())
            img_ids.extend(list(img_id))
            out_batches.append(outputs.detach().cpu())
        final_predictions_df = pd.DataFrame(
            {"id_code": img_ids, "diagnosis": predictions}
        )
        return final_predictions_df, out_batches, img_ids




## === cell 6
with torch.inference_mode():
    model0.eval()
    model1.eval()
    model2.eval()

    print("Computing Test Predictions")
    test_predictions0, out0_batches, ids0 = compute_predictions(
        model0, "test", test_loader, device
    )
    test_predictions1, out1_batches, ids1 = compute_predictions(
        model1, "test", test_loader, device
    )
    test_predictions2, out2_batches, ids2 = compute_predictions(
        model2, "test", test_loader, device
    )



## === cell 7
if not (ids0 == ids1 == ids2):
    raise RuntimeError(
        "Model prediction ID orders do not match; cannot ensemble safely."
    )

out0 = torch.cat(out0_batches, dim=0)
out1 = torch.cat(out1_batches, dim=0)
out2 = torch.cat(out2_batches, dim=0)

p0 = torch.softmax(out0, dim=1)
p1 = torch.softmax(out1, dim=1)
p2 = torch.softmax(out2, dim=1)
p = (p0 + p1 + p2) / 3.0

img_ids = np.array(ids0)
predictions = torch.argmax(p, dim=1).numpy().astype(int)

final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})

sample_path = "../input/aptos2019-blindness-detection/sample_submission.csv"
sample = pd.read_csv(sample_path)

final_predictions = sample[["id_code"]].merge(
    final_predictions, on="id_code", how="left"
)
final_predictions["diagnosis"] = final_predictions["diagnosis"].fillna(0).astype(int)

final_predictions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_predictions.shape)
print(final_predictions.head())
print(
    "diagnosis value counts:\n",
    final_predictions["diagnosis"].value_counts().sort_index(),
)
