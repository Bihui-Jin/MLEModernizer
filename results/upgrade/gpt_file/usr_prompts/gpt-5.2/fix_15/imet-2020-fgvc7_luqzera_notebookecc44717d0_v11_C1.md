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
Label artwork images with significant attributes.

## Metric
Micro averaged F1 score.

## Submission Format
```
id,attribute_ids
00011f01965f141f5d1eea6592fa9862,0 1 2
00014abc91ed3e4bf1663fde8136fe80,0 1 2
0002e2054e303badc1a33463f6fb7973,0 1 2
```

## Dataset
Multiple modalities can be expected and the camera sources are unknown. The photographs are often centered for objects, and in the case where the museum artifact is an entire room, the images are scenic in nature.

Each object is annotated by a single annotator without a verification step. You should consider these annotations noisy.

The filename of each image is its `id`.

- **train.csv** gives the `attribute_ids` for the train images in **/train**
- **/test** contains the test images. You must predict the `attribute_ids` for these images.
- **sample_submission.csv** contains a submission in the correct format
- **labels.csv** provides descriptions of the attributes

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
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
        input/
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
        working/
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
```

-> data/imet-2020-fgvc7/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/imet-2020-fgvc7/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/imet-2020-fgvc7/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> data/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> (stopped after 10 files for performance)

# 5. Target score

0.5241436132292118

# 6. Current score

0.00982

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0033) has done: 'I fix the test dataset file handling so it only iterates real image files and constructs the correct `{id}.png` paths, which is why it currently tries to load a non-existent `test.png`. I also make the root directory handling consistent by using `Path` joins (without changing the model or transforms), and I ensure the submission is written as `submission.csv` with the required `id,attribute_ids` columns. These changes are execution-blocking bug fixes and should be score-neutral (they don’t change the model outputs, only which files are read and how they’re mapped back to `sample_submission`). Finally, I keep the inference order stable and map predictions by filename stem to avoid misalignment.'
- What this solution (achieved 0.0033) has done: 'Your score is extremely low because the inference-time transforms are stochastic (RandomCrop + RandomHorizontalFlip) and the model is set as `pretrained=True` even when a checkpoint is loaded, which can silently make inference inconsistent and hurt micro-F1 badly. I make test/eval transforms deterministic (Resize + CenterCrop, no flip) so predictions are stable and aligned with what the model expects, without changing the model architecture or training loop. I also prevent any accidental weight reinitialization at inference by constructing the model with `pretrained=False` when you intend to load a full checkpoint, which keeps core logic identical but avoids mixing ImageNet weights with your competition checkpoint. Finally, I keep the submission mapping logic but ensure it always produces a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0001) has done: 'Your current score is near-random because the prediction post-processing is not aligned with the competition’s micro-F1 objective: you’re emitting too many labels per image (all probs > 0.2 plus top-1), which creates many false positives and crushes micro-F1. I keep your model and inference pipeline intact, but change `save_predictions` to use a fixed top‑K strategy (a common, minimal post-process for iMet) so each image outputs a small, consistent number of attributes. This is a minimal, metric-aligned change that should move the score substantially upward toward your target without changing training, architecture, or transforms. I also keep the submission mapping exactly as you have it (by id) and still write `submission.csv` with the required columns.'
- What this solution (achieved 0.00036) has done: 'Your current 0.0001 score suggests the submission is structurally valid but the label selection post-process is far from the micro-F1 optimum (too many false positives/negatives relative to what the model outputs). To move toward your target with minimal risk and without changing the model or training, I replace the fixed `top_k=3` with a tiny “top‑K with probability gate” rule: select all classes above a conservative threshold, but always output at least K labels (and cap at a small max). This keeps your pipeline intact while better matching micro-F1 behavior on iMet-style multi-label data. I also ensure dtype/shape handling is consistent and keep the id-to-prediction mapping exactly as you already do.'
- What this solution (achieved 5e-05) has done: 'Your current score is far below the target, so we should make the smallest changes that legitimately increase micro‑F1 without altering the model or training loop. The biggest likely issue is that your post-processing still produces too many false positives (threshold 0.35 with min_k=2/max_k=7 is aggressive for iMet), and you’re also sorting selected classes by id which can accidentally drop higher-probability labels when capping. I keep your “threshold gate + min/max K” logic, but (1) cap by probability properly and (2) use a more conservative threshold and smaller max_k to reduce false positives (micro‑F1 is very sensitive to FP). I also ensure the submission mapping is aligned by id and that prediction ordering remains stable.'
- What this solution (achieved 0.00024) has done: 'Your current micro-F1 is far below the target, so the smallest legitimate improvement is to align the post-processing with iMet’s multi-label distribution: your current `threshold=0.55` is so strict that it likely predicts almost nothing (very low recall), collapsing micro-F1. I keep your model, transforms, and inference loop unchanged, and only adjust `save_predictions` to use a conservative but not extreme probability gate plus a fixed small top‑K fallback/cap, selecting labels by probability (not by id) to reduce both false negatives and false positives. This preserves your existing “threshold + min/max labels” core logic while making it behave sensibly for micro-F1. The output remains a valid `submission.csv` with `id,attribute_ids` aligned to `sample_submission.csv`.'
- What this solution (achieved 6e-05) has done: 'Your current score is far below the target, so the smallest meaningful change is to fix the prediction post-processing to better match micro-F1: your current `threshold=0.20` with `min_k=2` likely outputs far too many labels per image (very low precision), which collapses micro-F1. I keep your model, checkpoint loading, transforms, and inference loop identical, and only adjust `save_predictions` to a conservative “top-K with a higher probability gate” rule: select labels above a higher threshold, but always output a small fixed minimum (top-3), and cap at a small maximum (top-7), always choosing by probability. This is a minimal, metric-aligned change that should move the score substantially upward toward your target without altering the core logic. The submission writing remains `submission.csv` with exact `id,attribute_ids` columns mapped by image id.'
- What this solution (achieved 0.00032) has done: 'Your current score is far below the target, and the biggest likely cause (without changing your model/training) is overly-strict label selection at inference: `threshold=0.55` often output only the forced `min_k` labels, hurting recall and micro-F1 badly. I keep your existing “threshold + min/max K” post-processing logic, but make it less extreme by lowering the threshold and using a slightly higher `max_k`, while still selecting/capping by probability to avoid flooding false positives. I also make one small correctness fix in `make_mask` (used by validation) to avoid incorrect row indexing (even if you’re not training now, it keeps evaluation semantics sane). Everything else (model, checkpoint loading, transforms, inference loop, submission mapping) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.00025) has done: 'Your current score is far below the target, so we should make the smallest legitimate change that most directly improves micro-F1 without touching the model, checkpoint loading, or inference loop. The biggest remaining issue is the inference post-processing: `threshold=0.20` with `max_k=10` tends to output too many labels, causing massive false positives and collapsing micro-F1. I keep your existing “probability threshold + min_k fallback + max_k cap” logic, but make it more conservative (higher threshold, smaller cap) and ensure the cap is applied by probability before any sorting, which should materially raise precision and move the score toward your target. Everything else stays the same and it still write a valid `submission.csv` with `id,attribute_ids`.'
- What this solution (achieved 0.00187) has done: 'Your current score is far below the target, so we should make the smallest change that most directly improves micro‑F1 without touching the model, checkpoint loading, or inference loop. The biggest issue is the post-processing: using a high fixed threshold plus sorting labels by id (not probability) tends to output either too few labels (low recall) or the “wrong” capped labels, which collapses micro‑F1. I keep your existing “threshold + min_k + max_k cap” semantics, but (1) apply the cap strictly by probability and (2) tune the threshold/min/max to a safer iMet-style range (more recall, still capped to limit false positives). I also ensure the saved `attribute_ids` are in probability order (this doesn’t change scoring directly but avoids accidental degradation if you later debug/compare).'
- What this solution (achieved 0.00066) has done: 'Your current score is far below the target, so the smallest likely win (without touching the model, checkpoint, or training loop) is to align inference post-processing to micro-F1: your current `threshold=0.20` tends to emit far too many labels per image, creating massive false positives and collapsing micro-F1. I keep your exact “threshold gate + min_k fallback + max_k cap” logic, but make it more conservative by increasing the threshold and slightly tightening the cap, while still selecting/capping strictly by probability. This is a minimal change localized to `save_predictions` and should move the score upward toward your target band. Everything else stays the same and it still writes a valid `submission.csv` with `id,attribute_ids` mapped to `sample_submission.csv`.'
- What this solution (achieved 0.00042) has done: 'Your current score is far below the target, so the smallest likely win is to fix an evaluation-metric mismatch without changing your model, checkpoint, or inference loop. This competition is scored by micro-F1, but your pipeline was tuned around F2-style heuristics (min/max K + threshold), which can easily collapse micro-F1 if the label count per image is off. I keep your existing post-processing structure, but set `min_k/max_k/threshold` from the training-set label cardinality (median/percentiles) and also calibrate the threshold using a small validation fold by directly maximizing micro-F1, which should move the score substantially upward. All I/O paths and submission formatting stay the same (`submission.csv` with `id,attribute_ids`).'
- What this solution (achieved 0.00982) has done: 'Your current score (0.00042) is so far below the target (0.524) that the most likely cause is not “slightly suboptimal thresholding” but a fundamental mismatch between the checkpoint and the model definition you instantiate at inference (shape/key mismatches leading to mostly-random weights or partially-loaded weights). I keep your architecture and inference loop intact, but make checkpoint loading strict-and-verified: detect whether the checkpoint is a full state_dict vs a wrapped dict, handle common key prefixes, and fail loudly if too many keys are missing/unexpected. Then I keep your existing post-processing logic, but make calibration actually use the same `binarize_prediction()` parameters it returns (min_k/max_k/threshold) and add a safe default derived from train label cardinality if calibration can’t run. These changes are minimal, execution-safe, and directly target getting non-random predictions so micro-F1 can move toward your target.'

# 9. Code solution

## === cell 0
kaggle = True
should_train = False
input_dir = "../input/"
base_dir = "../input/imet-2020-fgvc7/"
model_path_for_test = f"../input/resnext-mod/best-model_resnext_mod.pt"
train_root = base_dir + "train"
number_of_classes = 3474
num_workers = 4
batch_size = 32



## === cell 1
from collections import defaultdict, Counter
from typing import Callable, List, Dict
from pathlib import Path
from itertools import islice
from functools import partial
from PIL import Image
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
from torch import nn, cuda
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    Resize,
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
)
import pandas as pd
import numpy as np
import torchvision.models as M
import os
import random
import cv2
import torch
import math
import json
import shutil
import warnings




## === cell 2
def make_folds(n_folds: int, labels_file_path: str) -> pd.DataFrame:
    df = pd.read_csv(labels_file_path)
    cls_counts = Counter(
        cls for classes in df["attribute_ids"].str.split() for cls in classes
    )
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in df.sample(frac=1, random_state=42).itertuples():
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df["fold"] = folds
    return df




## === cell 3
def save_folds(n_folds: int, labels_file_path: str, save_path: str):
    df = make_folds(n_folds=5, labels_file_path=base_dir + "train.csv")
    df.to_csv(save_path, index=None)




## === cell 4
class TrainDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        number_of_classes: int,
        df: pd.DataFrame,
        image_transform: Callable,
        tensor_transform: Callable,
    ):
        self.root_dir = Path(root_dir)
        self.df = df
        self.image_transform = image_transform
        self.tensor_transform = tensor_transform
        self.number_of_classes = number_of_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        image = load_transform_image(
            item.id, self.root_dir, self.image_transform, self.tensor_transform
        )
        target = torch.zeros(self.number_of_classes)

        for attribute in item.attribute_ids.split():
            target[int(attribute)] = 1

        return image, target


class PredictImageDataset(Dataset):
    def __init__(self, root_dir: Path, image_transform, tensor_transform):
        self.root_dir = Path(root_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = []
        for p in self.root_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p.name)
        self.files = sorted(files)
        self.tensor_transform = tensor_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = Path(self.files[idx]).stem
        image = load_transform_image(
            img_name, self.root_dir, self.image_transform, self.tensor_transform
        )
        return image

    def filenames(self):
        return [Path(x).stem for x in self.files]




## === cell 5
def load_transform_image(
    item_id, root_dir: Path, image_transform: Callable, tensor_transform: Callable
):
    image = load_image(item_id, root_dir)
    image = image_transform(image)
    return tensor_transform(image)


def load_image(item_id, root_dir: Path) -> Image.Image:
    root_dir = Path(root_dir)
    path = root_dir / f"{item_id}.png"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)




## === cell 6
def make_loader(
    df: pd.DataFrame, image_transform: Callable, tensor_transform: Callable
) -> DataLoader:
    dataset = TrainDataset(
        Path(train_root), number_of_classes, df, image_transform, tensor_transform
    )
    return DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )




## === cell 7
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = input_dir + f"{model_name}/{model_name}.pth"
        if os.path.exists(weights_path):
            net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            try:
                net = net_cls(weights="DEFAULT")
            except Exception:
                net = net_cls(pretrained=True)
    else:
        try:
            net = net_cls(weights=None)
        except Exception:
            net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.resnext50_32x4d, dropout=False
    ):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 8
def _strip_prefix_from_state_dict(state_dict: dict, prefixes: List[str]) -> dict:
    out = state_dict
    for pref in prefixes:
        if all((k.startswith(pref) for k in out.keys())):
            out = {k[len(pref) :]: v for k, v in out.items()}
    return out


def _extract_model_state(state):
    if isinstance(state, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def load_model(model: nn.Module, path: Path) -> Dict:
    if path is None or (
        isinstance(path, (str, Path)) and not os.path.exists(str(path))
    ):
        print(
            f"Warning: checkpoint not found at {path}. Using model as-initialized (no checkpoint loaded)."
        )
        return {"epoch": 0, "step": 0, "best_valid_loss": float("inf")}

    state = torch.load(str(path), map_location="cpu")
    model_state = _extract_model_state(state)
    if not isinstance(model_state, dict):
        raise ValueError(
            f"Unrecognized checkpoint format at {path}: type={type(state)}"
        )

    model_state = _strip_prefix_from_state_dict(
        model_state, prefixes=["module.", "model.", "net."]
    )

    missing, unexpected = model.load_state_dict(model_state, strict=False)

    total_keys = len(model.state_dict())
    loaded_keys = total_keys - len(missing)
    load_ratio = 0.0 if total_keys == 0 else loaded_keys / total_keys

    print(
        f"Checkpoint load: loaded {loaded_keys}/{total_keys} keys (ratio={load_ratio:.3f})."
    )
    if len(unexpected) > 0:
        print(f"Unexpected keys (showing up to 10): {unexpected[:10]}")
    if len(missing) > 0:
        print(f"Missing keys (showing up to 10): {missing[:10]}")

    if load_ratio < 0.95:
        raise RuntimeError(
            f"Checkpoint appears incompatible with model definition (load_ratio={load_ratio:.3f} < 0.95). "
            f"Refusing to run inference with mostly-uninitialized weights. Path: {path}"
        )

    if isinstance(state, dict):
        epoch = state.get("epoch", 0)
        step = state.get("step", 0)
        best_valid_loss = state.get("best_valid_loss", float("inf"))
        if "epoch" in state or "step" in state:
            print(f"Loaded model metadata: epoch={epoch}, step={step}")
        return {"epoch": epoch, "step": step, "best_valid_loss": best_valid_loss}

    return {"epoch": 0, "step": 0, "best_valid_loss": float("inf")}




## === cell 9
def reduce_loss(loss):
    return loss.sum() / loss.shape[0]




## === cell 10
def binarize_prediction(
    probabilities, threshold: float, argsorted=None, min_labels=1, max_labels=10
):
    """Return matrix of 0/1 predictions, same shape as probabilities."""
    assert probabilities.shape[1] == number_of_classes
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = make_mask(argsorted, max_labels)
    min_mask = make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def make_mask(argsorted, top_n: int):
    n_rows = argsorted.shape[0]
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = np.repeat(np.arange(n_rows), top_n)
    mask[row_indices, col_indices] = 1
    return mask




## === cell 11
def validation(
    model: nn.Module,
    criterion,
    valid_loader,
    use_cuda,
) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            all_losses.append(reduce_loss(loss).item())
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            return fbeta_score(all_targets, y_pred, beta=2, average="samples")

    metrics = {}
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.10, 0.20]:
        metrics[f"valid_f2_th_{threshold:.2f}"] = get_score(
            binarize_prediction(all_predictions, threshold, argsorted)
        )
    metrics["valid_loss"] = np.mean(all_losses)
    print(
        " | ".join(
            f"{k} {v:.3f}" for k, v in sorted(metrics.items(), key=lambda kv: -kv[1])
        )
    )

    return metrics




## === cell 12
def train(
    model: nn.Module,
    criterion,
    *,
    params,
    train_loader,
    valid_loader,
    init_optimizer,
    use_cuda,
    n_epochs=None,
    patience=2,
    max_lr_changes=2,
) -> bool:

    lr = 1e-4
    n_epochs = 100
    params = list(params)
    optimizer = init_optimizer(params, lr)

    model_path = "model.pt"
    best_model_path = "best-model.pt"
    uptrain = True
    if uptrain and os.path.exists(model_path):
        state = load_model(model, model_path)
        epoch = state["epoch"]
        step = state["step"]
        best_valid_loss = state["best_valid_loss"]
    else:
        epoch = 1
        step = 0
        best_valid_loss = float("inf")
    lr_changes = 0

    save = lambda ep: torch.save(
        {
            "model": model.state_dict(),
            "epoch": ep,
            "step": step,
            "best_valid_loss": best_valid_loss,
        },
        str(model_path),
    )

    report_each = 100
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        losses = []
        tl = train_loader
        mean_loss = 0
        for i, (inputs, targets) in enumerate(tl):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = reduce_loss(criterion(outputs, targets))
            batch_size_ = inputs.size(0)
            (batch_size_ * loss).backward()
            if (i + 1) % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
            losses.append(loss.item())
            mean_loss = np.mean(losses[-report_each:])
            print(f"mean_loss: {mean_loss:.3f}")
        save(epoch + 1)
        valid_metrics = validation(model, criterion, valid_loader, use_cuda)

        valid_loss = valid_metrics["valid_loss"]
        valid_losses.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            shutil.copy(str(model_path), str(best_model_path))
        elif (
            patience
            and epoch - lr_reset_epoch > patience
            and min(valid_losses[-patience:]) > best_valid_loss
        ):
            lr_changes += 1
            if lr_changes > max_lr_changes:
                break
            lr /= 5
            print(f"lr updated to {lr}")
            lr_reset_epoch = epoch
            optimizer = init_optimizer(params, lr)
    return True




## === cell 13
train_transform = Compose(
    [
        RandomCrop(288),
        RandomHorizontalFlip(),
    ]
)

test_transform = Compose(
    [
        RandomCrop(256),
        RandomHorizontalFlip(),
    ]
)

eval_transform = Compose(
    [
        Resize((288, 288)),
        CenterCrop(288),
    ]
)

tensor_transform = Compose(
    [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 14
def prepare_folds():
    folds = make_folds(n_folds=5, labels_file_path=base_dir + "train.csv")
    return folds




## === cell 15
def start_train():
    folds = prepare_folds()
    train_fold = folds[folds["fold"] != 0]
    valid_fold = folds[folds["fold"] == 0]

    train_loader = make_loader(train_fold, train_transform, tensor_transform)
    valid_loader = make_loader(valid_fold, test_transform, tensor_transform)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    model = ResNet(
        num_classes=number_of_classes, pretrained=True, net_cls=M.resnext50_32x4d
    )

    use_cuda = cuda.is_available()
    if use_cuda:
        model = model.cuda()

    train(
        params=model.parameters(),
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        patience=4,
        init_optimizer=lambda params, lr: Adam(params, lr),
        use_cuda=use_cuda,
    )




## === cell 16
def predict(model, test_dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cuda.is_available(),
    )

    model_result = []

    use_cuda = cuda.is_available()
    if use_cuda:
        model = model.cuda()
    model.eval()

    it = 0
    for batch in loader:
        print(it)
        if use_cuda:
            inputs = batch.cuda(non_blocking=True)
        else:
            inputs = batch
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            model_result.extend(probs.cpu().numpy())
        it += inputs.size(0)

    return model_result




## === cell 17
def _compute_label_count_priors(train_csv_path: str):
    df = pd.read_csv(train_csv_path, usecols=["attribute_ids"])
    counts = df["attribute_ids"].astype(str).str.split().map(len).to_numpy()
    if counts.size == 0:
        return {"min_k": 1, "max_k": 5}
    p50 = int(np.clip(np.median(counts), 1, 50))
    p90 = int(np.clip(np.percentile(counts, 90), 1, 50))
    min_k = int(np.clip(p50, 1, 10))
    max_k = int(np.clip(p90, min_k, 20))
    return {"min_k": min_k, "max_k": max_k}


def _micro_f1_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.uint8, copy=False)
    y_pred = y_pred.astype(np.uint8, copy=False)
    tp = np.logical_and(y_true == 1, y_pred == 1).sum(dtype=np.int64)
    fp = np.logical_and(y_true == 0, y_pred == 1).sum(dtype=np.int64)
    fn = np.logical_and(y_true == 1, y_pred == 0).sum(dtype=np.int64)
    denom = 2 * tp + fp + fn
    return float(0.0 if denom == 0 else (2 * tp) / denom)


def _calibrate_threshold_on_fold(model: nn.Module, fold_df: pd.DataFrame):
    use_cuda = cuda.is_available()
    valid_df = fold_df[fold_df["fold"] == 0].reset_index(drop=True)
    if len(valid_df) == 0:
        return None

    n = min(4000, len(valid_df))
    valid_df = valid_df.sample(n=n, random_state=42).reset_index(drop=True)

    valid_loader = DataLoader(
        TrainDataset(
            Path(train_root),
            number_of_classes,
            valid_df,
            eval_transform,
            tensor_transform,
        ),
        shuffle=False,
        batch_size=64,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            if use_cuda:
                inputs = inputs.cuda(non_blocking=True)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0).astype(np.uint8)

    argsorted = all_probs.argsort(axis=1)

    priors = _compute_label_count_priors(os.path.join(base_dir, "train.csv"))
    min_k = priors["min_k"]
    max_k = priors["max_k"]

    candidate_thresholds = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26]
    best = {"threshold": 0.20, "score": -1.0, "min_k": min_k, "max_k": max_k}
    for th in candidate_thresholds:
        y_pred = binarize_prediction(
            all_probs, th, argsorted=argsorted, min_labels=min_k, max_labels=max_k
        )
        score = _micro_f1_from_binary(all_targets, y_pred)
        if score > best["score"]:
            best = {
                "threshold": float(th),
                "score": float(score),
                "min_k": int(min_k),
                "max_k": int(max_k),
            }

    print(
        f"Calibration (fold0 sample n={n}): best micro-F1={best['score']:.4f} at threshold={best['threshold']:.2f}, min_k={min_k}, max_k={max_k}"
    )
    return best


def save_predictions(preds, test_dataset, *, postproc_params: dict):
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2 or preds.shape[1] != number_of_classes:
        raise ValueError(
            f"Unexpected preds shape {preds.shape}; expected (N, {number_of_classes})"
        )

    min_k = int(postproc_params.get("min_k", 2))
    max_k = int(postproc_params.get("max_k", 5))
    threshold = float(postproc_params.get("threshold", 0.20))

    argsorted = preds.argsort(axis=1)
    binary = binarize_prediction(
        preds,
        threshold=threshold,
        argsorted=argsorted,
        min_labels=min_k,
        max_labels=max_k,
    ).astype(bool)

    prediction = []
    for i in range(preds.shape[0]):
        sel = np.flatnonzero(binary[i]).astype(np.int32)
        sel = sel[np.argsort(preds[i, sel])[::-1]]
        prediction.append(" ".join(map(str, sel.tolist())))

    sample_path = os.path.join(base_dir, "sample_submission.csv")
    sample = pd.read_csv(sample_path)

    pred_map = dict(zip(test_dataset.filenames(), prediction))
    sample["attribute_ids"] = sample["id"].map(pred_map).fillna("")
    sample.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", sample.shape)
    print(f"Postproc used: threshold={threshold:.2f}, min_k={min_k}, max_k={max_k}")




## === cell 18
def test():
    batch_size = 200
    test_img_dir = os.path.join(base_dir, "test")
    test_dataset = PredictImageDataset(test_img_dir, eval_transform, tensor_transform)

    new_model = ResNet(
        num_classes=number_of_classes, pretrained=False, net_cls=M.resnext50_32x4d
    )
    load_model(new_model, model_path_for_test)
    new_model.eval()

    folds = prepare_folds()
    use_cuda = cuda.is_available()
    if use_cuda:
        new_model = new_model.cuda()

    postproc = _calibrate_threshold_on_fold(new_model, folds)
    if postproc is None:
        priors = _compute_label_count_priors(os.path.join(base_dir, "train.csv"))
        postproc = {"threshold": 0.20, **priors}

    preds = predict(new_model, test_dataset, batch_size)
    save_predictions(preds, test_dataset, postproc_params=postproc)




## === cell 19
test()
