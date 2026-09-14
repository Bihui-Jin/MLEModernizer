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

0.5399081545205416

# 6. Current score

0.31123

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0033) has done: 'I fix the missing model checkpoint by falling back to a torchvision ResNeXt-50 pretrained weights load when the provided `model_path_for_test` does not exist, so inference can run end-to-end. I also fix the prediction pipeline to use the correct number of classes (3474) and apply `sigmoid` before thresholding, which is required because the model outputs logits. Finally, I make test file ordering deterministic and ensure the submission uses the sample submission’s `id` order (to avoid id/prediction misalignment) while still producing a `submission.csv` with the required columns.'
- What this solution (achieved 0.00019) has done: 'Your current low score is mainly coming from a mismatch between how predictions are converted to labels versus what the competition’s micro-F1 rewards: using a fixed `> 0.01` threshold predicts far too many labels per image, crushing precision. I keep your model and inference logic intact, but change only the post-processing to use the same “top-K with a probability threshold and min/max label constraints” logic you already implemented (`binarize_prediction`), which is a standard fit for this dataset/metric. I also make the test-time transform deterministic by removing random horizontal flip (randomness at inference hurts consistency and typically lowers score). Finally, I ensure every row has at least one predicted label (per your min_labels=1), which avoids empty predictions hurting F1.'
- What this solution (achieved 0.31082) has done: 'The timeout is dominated by test-time inference over 21k images plus slow per-sample PIL/cv2 conversions and an O(N·C·logC) full argsort over 3474 classes. I keep the same model, transforms, thresholding logic, and output semantics, but remove redundant work by (1) decoding/resizing/normalizing directly into tensors without PIL, (2) enabling fast DataLoader settings (pinned memory, persistent workers, prefetch), (3) switching argsort to an equivalent top‑k selection (same result for top-10 masking) and a vectorized mask build, and (4) batching predictions into preallocated numpy arrays to avoid Python list overhead. These changes are performance-only and preserve the exact prediction rule (top-10 cap, min-1, prob>threshold) with only negligible floating-point differences from avoiding PIL roundtrips.'
- What this solution (achieved 0.30653) has done: 'Your current score gap to the target is large (0.31082 → 0.5399), so we need a small but meaningful improvement without changing the model/training core. The biggest low-risk lever here is post-processing: your fixed `threshold=0.20` and `max_labels=10` is likely too restrictive for micro-F1, which typically benefits from slightly more recall on this noisy multi-label dataset. I keep your exact model/inference pipeline, but tune the prediction rule minimally by (1) slightly lowering the threshold and (2) allowing a few more labels per image (still using the same `binarize_prediction` logic), which should move the score upward toward the target. I also make `topk_indices` return a sorted top-k to ensure the “min_labels” selection is truly the highest-prob labels (this is a correctness fix that can improve F1 without changing the model).'
- What this solution (achieved 0.31123) has done: 'The timeout is dominated by reading/decoding/resizing 21k PNGs (plus an extra fold-inference pass for calibration) with a small number of workers, plus expensive dense mask construction in `binarize_prediction` during calibration and submission formatting. I (1) cache the resize parameters once per transform instead of parsing it per image, (2) speed up dataset file discovery and dataloading (use `os.scandir`, `num_workers` tuned to CPU count, `prefetch_factor`, `inference_mode`), (3) replace the dense `(N, 3474)` mask construction with an equivalent top‑k sparse/row-wise approach that yields identical predictions, and (4) make target multi-hot construction vectorized to remove Python loops. These changes preserve the model, loss, training/inference semantics, and output formatting, while reducing CPU overhead and memory traffic substantially.'

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
from typing import Callable, List, Dict, Tuple
from pathlib import Path
from itertools import islice
from functools import partial
from PIL import Image
from sklearn.metrics import fbeta_score, f1_score
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
import time




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
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _parse_resize_size(image_transform: Callable):
    if isinstance(image_transform, Compose):
        for t in image_transform.transforms:
            if isinstance(t, Resize):
                sz = t.size
                if isinstance(sz, (list, tuple)) and len(sz) == 2:
                    return int(sz[0]), int(sz[1])
                if isinstance(sz, int):
                    return int(sz), int(sz)
    elif isinstance(image_transform, Resize):
        sz = image_transform.size
        if isinstance(sz, (list, tuple)) and len(sz) == 2:
            return int(sz[0]), int(sz[1])
        if isinstance(sz, int):
            return int(sz), int(sz)
    return None


_TRANSFORM_META_CACHE: Dict[int, Tuple[Tuple[int, int] or None, float or None]] = {}


def _get_transform_meta(image_transform: Callable):
    key = id(image_transform)
    meta = _TRANSFORM_META_CACHE.get(key)
    if meta is not None:
        return meta
    resize_hw = _parse_resize_size(image_transform)
    flip_p = None
    if isinstance(image_transform, Compose):
        for t in image_transform.transforms:
            if isinstance(t, RandomHorizontalFlip):
                flip_p = float(t.p)
                break
    elif isinstance(image_transform, RandomHorizontalFlip):
        flip_p = float(image_transform.p)
    meta = (resize_hw, flip_p)
    _TRANSFORM_META_CACHE[key] = meta
    return meta


def load_transform_image(
    item_id, root_dir: Path, image_transform: Callable, tensor_transform: Callable
):
    path = root_dir + f"/{item_id}.png"
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resize_hw, flip_p = _get_transform_meta(image_transform)
    if resize_hw is not None:
        h, w = resize_hw
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    if flip_p is not None:
        if random.random() < flip_p:
            image = cv2.flip(image, 1)

    x = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .contiguous()
        .to(torch.float32)
        .div_(255.0)
    )
    x.sub_(_IMAGENET_MEAN).div_(_IMAGENET_STD)
    return x


def load_image(item_id, root_dir: Path) -> Image.Image:
    path = root_dir + f"/{item_id}.png"
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)




## === cell 5
class TrainDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        number_of_classes: int,
        df: pd.DataFrame,
        image_transform: Callable,
        tensor_transform: Callable,
    ):
        self.root_dir = root_dir
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
        self.root_dir = root_dir
        self.files = sorted(
            [
                e.name
                for e in os.scandir(root_dir)
                if e.is_file() and e.name.lower().endswith(".png")
            ]
        )
        self.tensor_transform = tensor_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.splitext(self.files[idx])[0]
        image = load_transform_image(
            img_name, self.root_dir, self.image_transform, self.tensor_transform
        )
        return image

    def filenames(self):
        return [f[:-4] for f in self.files]




## === cell 6
def _effective_num_workers(requested: int) -> int:
    try:
        cpu = os.cpu_count() or 4
    except Exception:
        cpu = 4
    return max(0, min(int(requested), max(4, cpu // 2)))


def make_loader(
    df: pd.DataFrame, image_transform: Callable, tensor_transform: Callable
) -> DataLoader:
    dataset = TrainDataset(
        train_root, number_of_classes, df, image_transform, tensor_transform
    )
    use_cuda = cuda.is_available()
    nw = _effective_num_workers(num_workers)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )




## === cell 7
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if pretrained:
        model_name = net_cls.__name__
        weights_path = os.path.join(input_dir, f"{model_name}/{model_name}.pth")
        if os.path.exists(weights_path):
            net = net_cls(weights=None)
            net.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
        else:
            try:
                weights_enum = getattr(M, f"{model_name}_Weights").DEFAULT
                net = net_cls(weights=weights_enum)
            except Exception:
                net = net_cls(weights=None)
    else:
        net = net_cls(weights=None)
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
def load_model(model: nn.Module, path: Path) -> Dict:
    """
    Bugfix: PyTorch 2.6 defaults torch.load(weights_only=True), which can fail for
    our checkpoint dict (contains numpy scalars etc.). Our checkpoints are produced
    by this script, so we explicitly load with weights_only=False for compatibility.
    """
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        epoch = state.get("epoch", -1)
        step = state.get("step", -1)
        print(f"Loaded model from epoch {epoch}, step {step}")
        return state
    else:
        model.load_state_dict(state)
        print("Loaded model from a plain state_dict")
        return {"model": state, "epoch": -1, "step": -1, "best_valid_loss": None}




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
        argsorted = topk_indices(probabilities, k=max_labels)
    max_mask = make_mask(argsorted, max_labels)
    min_mask = (
        make_mask(argsorted[:, -min_labels:], min_labels)
        if min_labels < max_labels
        else make_mask(argsorted, min_labels)
    )
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def topk_indices(probabilities: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(probabilities, -k, axis=1)[:, -k:]  # (N, k) unsorted
    rows = np.arange(probabilities.shape[0])[:, None]
    vals = probabilities[rows, idx]
    order = np.argsort(vals, axis=1)  # ascending within top-k
    return idx[rows, order]


def make_mask(topk_idx, top_n: int):
    idx = topk_idx[:, -top_n:]
    n = idx.shape[0]
    mask = np.zeros((n, number_of_classes), dtype=np.uint8)
    rows = np.arange(n)[:, None]
    mask[rows, idx] = 1
    return mask


def binarize_prediction_sparse(
    probabilities: np.ndarray,
    *,
    threshold: float,
    argsorted: np.ndarray,
    min_labels: int = 1,
    max_labels: int = 10,
) -> List[np.ndarray]:
    n = probabilities.shape[0]
    rows = np.arange(n)[:, None]
    top_max = argsorted[:, -max_labels:]  # (N, max_labels)
    top_min = argsorted[:, -min_labels:]  # (N, min_labels)

    keep = probabilities[rows, top_max] > threshold  # (N, max_labels) bool
    out = []
    for i in range(n):
        sel = top_max[i][keep[i]]
        if min_labels > 0:
            sel = np.union1d(sel, top_min[i])
        out.append(sel.astype(np.int32, copy=False))
    return out




## === cell 11
def validation(model: nn.Module, criterion, valid_loader, use_cuda) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.inference_mode():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(
                    non_blocking=True
                )
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
    top10 = topk_indices(all_predictions, k=10)
    for threshold in [0.10, 0.20]:
        metrics[f"valid_f2_th_{threshold:.2f}"] = get_score(
            binarize_prediction(
                all_predictions, threshold, argsorted=top10, min_labels=1, max_labels=10
            )
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
        if best_valid_loss is None:
            best_valid_loss = float("inf")
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
        Resize((288, 288)),
        RandomHorizontalFlip(),
    ]
)

test_transform = Compose(
    [
        Resize((288, 288)),
    ]
)

eval_transform = Compose([Resize((288, 288))])

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
    use_cuda = cuda.is_available()
    nw = _effective_num_workers(num_workers)
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )

    if use_cuda:
        model = model.cuda()
    model.eval()

    n = len(test_dataset)
    out = np.empty((n, number_of_classes), dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for it, inputs in enumerate(loader):
            if it % 20 == 0:
                print(offset)
            if use_cuda:
                inputs = inputs.cuda(non_blocking=True)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
            bs = probs.shape[0]
            out[offset : offset + bs] = probs
            offset += bs
    return out




## === cell 17
def _encode_targets_multi_hot(df: pd.DataFrame, n_classes: int) -> np.ndarray:
    y = np.zeros((len(df), n_classes), dtype=np.uint8)
    attr_lists = df["attribute_ids"].astype(str).str.split()
    rows = []
    cols = []
    for i, lst in enumerate(attr_lists):
        if not lst:
            continue
        cols.extend(map(int, lst))
        rows.extend([i] * len(lst))
    if rows:
        y[np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)] = 1
    return y


def _calibrate_postprocess_params(
    model: nn.Module,
    folds_df: pd.DataFrame,
    *,
    fold_idx: int = 0,
    max_valid_samples: int = 2500,
    thresholds: List[float] = None,
    max_labels_grid: List[int] = None,
    min_labels: int = 1,
    infer_batch_size: int = 128,
) -> Tuple[float, int]:
    """
    Change (score-related): choose threshold/max_labels that best improves MICRO-F1
    on a held-out fold, using the same binarize_prediction logic as submission.
    This is a minimal, metric-alignment change and does not alter the model.
    """
    if thresholds is None:
        thresholds = [0.06, 0.08, 0.10, 0.12, 0.14]
    if max_labels_grid is None:
        max_labels_grid = [10, 15, 20, 25]

    valid_df = folds_df[folds_df["fold"] == fold_idx].reset_index(drop=True)
    if len(valid_df) > max_valid_samples:
        valid_df = valid_df.iloc[:max_valid_samples].reset_index(drop=True)

    valid_ds = TrainDataset(
        train_root, number_of_classes, valid_df, eval_transform, tensor_transform
    )
    use_cuda = cuda.is_available()
    nw = _effective_num_workers(num_workers)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )

    model.eval()
    probs_list = []
    with torch.inference_mode():
        for xb, yb in valid_loader:
            if use_cuda:
                xb = xb.cuda(non_blocking=True)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
            probs_list.append(probs)
    probs = np.concatenate(probs_list, axis=0)

    y_true = _encode_targets_multi_hot(valid_df, number_of_classes)

    best = (-1.0, None, None)
    for ml in max_labels_grid:
        topk = topk_indices(probs, k=ml)
        for th in thresholds:
            pred_idx = binarize_prediction_sparse(
                probs,
                threshold=th,
                argsorted=topk,
                min_labels=min_labels,
                max_labels=ml,
            )

            total_true = int(y_true.sum())
            total_pred = int(sum(len(p) for p in pred_idx))
            tp = 0
            for i, cols in enumerate(pred_idx):
                if cols.size:
                    tp += int(y_true[i, cols].sum())
            fp = total_pred - tp
            fn = total_true - tp
            denom = 2 * tp + fp + fn
            score = (2 * tp / denom) if denom > 0 else 0.0

            if score > best[0]:
                best = (score, th, ml)

    print(
        f"Calibration (fold={fold_idx}, n={len(valid_df)}): best_micro_f1={best[0]:.5f} at th={best[1]}, max_labels={best[2]}"
    )
    return float(best[1]), int(best[2])




## === cell 18
def save_predictions(
    preds,
    test_dataset,
    *,
    pred_threshold: float,
    pred_max_labels: int,
    pred_min_labels: int = 1,
):
    probs = np.asarray(preds, dtype=np.float32)  # (N, C)

    topk = topk_indices(probs, k=pred_max_labels)

    pred_idx = binarize_prediction_sparse(
        probs,
        threshold=pred_threshold,
        argsorted=topk,
        min_labels=pred_min_labels,
        max_labels=pred_max_labels,
    )
    prediction = [
        " ".join(map(str, idx.tolist())) if idx.size else "" for idx in pred_idx
    ]

    sample_path = os.path.join(base_dir, "sample_submission.csv")
    sample = pd.read_csv(sample_path)

    fnames = test_dataset.filenames()
    pred_map = dict(zip(fnames, prediction))
    sample["attribute_ids"] = sample["id"].map(pred_map).fillna("")

    sample.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")




## === cell 19
def _quick_train_if_no_checkpoint(max_seconds: int = 540) -> str:
    """
    If the external checkpoint is missing, train the same model briefly to avoid random-head inference.
    """
    local_best = "best-model.pt"
    local_last = "model.pt"
    if os.path.exists(local_best) or os.path.exists(local_last):
        return local_best if os.path.exists(local_best) else local_last

    folds = prepare_folds()
    train_fold = folds[folds["fold"] != 0]
    valid_fold = folds[folds["fold"] == 0]

    n_train = min(6000, len(train_fold))
    n_valid = min(1500, len(valid_fold))
    train_fold = train_fold.iloc[:n_train].reset_index(drop=True)
    valid_fold = valid_fold.iloc[:n_valid].reset_index(drop=True)

    train_loader = make_loader(train_fold, train_transform, tensor_transform)
    valid_loader = make_loader(valid_fold, test_transform, tensor_transform)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model = ResNet(
        num_classes=number_of_classes, pretrained=True, net_cls=M.resnext50_32x4d
    )

    use_cuda = cuda.is_available()
    if use_cuda:
        model = model.cuda()

    lr = 1e-4
    optimizer = Adam(model.parameters(), lr)

    best_valid_loss = float("inf")
    start = time.time()
    step = 0

    def _save(ep: int):
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": ep,
                "step": step,
                "best_valid_loss": best_valid_loss,
            },
            local_last,
        )

    for epoch in range(1, 1000):
        model.train()
        for inputs, targets in train_loader:
            if time.time() - start > max_seconds:
                print("Quick-train time budget reached; stopping.")
                _save(epoch)
                return local_best if os.path.exists(local_best) else local_last

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = reduce_loss(criterion(outputs, targets))
            (inputs.size(0) * loss).backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

        _save(epoch)
        valid_metrics = validation(model, criterion, valid_loader, use_cuda)
        valid_loss = valid_metrics["valid_loss"]
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            shutil.copy(local_last, local_best)

        if time.time() - start > max_seconds:
            break

    return local_best if os.path.exists(local_best) else local_last




## === cell 20
def test():
    batch_size = 200
    test_img_dir = os.path.join(base_dir, "test")
    test_dataset = PredictImageDataset(test_img_dir, eval_transform, tensor_transform)

    new_model = ResNet(num_classes=number_of_classes)
    if model_path_for_test and os.path.exists(model_path_for_test):
        load_model(new_model, model_path_for_test)
    else:
        fallback_ckpt = _quick_train_if_no_checkpoint(max_seconds=540)
        load_model(new_model, fallback_ckpt)
        print(
            f"Warning: checkpoint not found at {model_path_for_test}. Using locally trained fallback checkpoint: {fallback_ckpt}"
        )

    folds = prepare_folds()
    use_cuda = cuda.is_available()
    if use_cuda:
        new_model = new_model.cuda()

    cal_th, cal_max_labels = _calibrate_postprocess_params(
        new_model,
        folds,
        fold_idx=0,
        max_valid_samples=2500,
        thresholds=[0.06, 0.08, 0.10, 0.12, 0.14],
        max_labels_grid=[10, 15, 20, 25],
        min_labels=1,
        infer_batch_size=128,
    )

    new_model.eval()
    preds = predict(new_model, test_dataset, batch_size)
    save_predictions(
        preds,
        test_dataset,
        pred_threshold=cal_th,
        pred_max_labels=cal_max_labels,
        pred_min_labels=1,
    )




## === cell 21
test()
