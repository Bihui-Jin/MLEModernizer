# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from typing import Callable, List, Dict, Tuple, Optional
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
    RandomHorizontalFlip,
)
import pandas as pd
import numpy as np
import torchvision.models as M
import os
import random
import cv2
import torch
import shutil
import warnings
import time


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)

cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass




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


_TRANSFORM_META_CACHE: Dict[int, Tuple[Optional[Tuple[int, int]], Optional[float]]] = {}


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

    if flip_p is not None and flip_p > 0.0:
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
def _build_multi_hot_targets(df: pd.DataFrame, n_classes: int) -> torch.Tensor:
    y = torch.zeros((len(df), n_classes), dtype=torch.uint8)
    attr_lists = df["attribute_ids"].astype(str).str.split()

    counts = attr_lists.map(len).to_numpy(dtype=np.int32, copy=False)
    total = int(counts.sum())
    if total == 0:
        return y

    rows = np.repeat(np.arange(len(df), dtype=np.int32), counts)
    cols = np.fromiter(
        (int(c) for lst in attr_lists for c in lst),
        dtype=np.int32,
        count=total,
    )
    y[torch.from_numpy(rows).long(), torch.from_numpy(cols).long()] = 1
    return y


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
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform
        self.tensor_transform = tensor_transform
        self.number_of_classes = number_of_classes
        self._targets = _build_multi_hot_targets(self.df, self.number_of_classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        image = load_transform_image(
            item.id, self.root_dir, self.image_transform, self.tensor_transform
        )
        target = self._targets[idx].to(dtype=torch.float32)
        return image, target


class PredictImageDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        image_transform,
        tensor_transform,
        ids: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Test directory not found: {root_dir}")

        if ids is None:
            files = sorted(root.glob("*.png"))
            self._stems = [p.stem for p in files]
        else:
            self._stems = list(ids)

        self.tensor_transform = tensor_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self._stems)

    def __getitem__(self, idx):
        img_name = self._stems[idx]
        image = load_transform_image(
            img_name, self.root_dir, self.image_transform, self.tensor_transform
        )
        return image

    def filenames(self):
        return self._stems




## === cell 6
def _effective_num_workers(requested: int) -> int:
    try:
        cpu = os.cpu_count() or 4
    except Exception:
        cpu = 4
    return max(0, min(int(requested), max(4, cpu // 2)))


def _worker_init_fn(worker_id: int):
    cv2.setNumThreads(0)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def make_loader(
    df: pd.DataFrame, image_transform: Callable, tensor_transform: Callable
) -> DataLoader:
    dataset = TrainDataset(
        train_root, number_of_classes, df, image_transform, tensor_transform
    )
    use_cuda = cuda.is_available()
    nw = _effective_num_workers(num_workers)
    kwargs = dict(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        worker_init_fn=_worker_init_fn if nw > 0 else None,
    )
    if nw > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)




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
    assert probabilities.shape[1] == number_of_classes
    if argsorted is None:
        argsorted = topk_indices(probabilities, k=max_labels)
    idx = argsorted[:, :max_labels]
    n = probabilities.shape[0]
    rows = np.arange(n)[:, None]
    top_probs = probabilities[rows, idx]
    keep = top_probs > threshold
    if min_labels > 0:
        keep[:, :min_labels] = True
    mask = np.zeros((n, number_of_classes), dtype=np.uint8)
    sel_rows, sel_pos = np.nonzero(keep)
    sel_cols = idx[sel_rows, sel_pos]
    mask[sel_rows, sel_cols] = 1
    return mask


def topk_indices(probabilities: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(probabilities, -k, axis=1)[:, -k:]  # (N, k) unsorted
    rows = np.arange(probabilities.shape[0])[:, None]
    vals = probabilities[rows, idx]
    order = np.argsort(-vals, axis=1)  # DESC within top-k
    return idx[rows, order]


def make_mask(topk_idx, top_n: int):
    idx = topk_idx[:, :top_n]
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
    top_max = argsorted[:, :max_labels]
    top_probs = probabilities[rows, top_max]

    keep = top_probs > threshold
    if min_labels > 0:
        keep[:, :min_labels] = True

    out = []
    for i in range(n):
        sel = top_max[i][keep[i]]
        out.append(sel.astype(np.int32, copy=False))
    return out




## === cell 11
def validation(model: nn.Module, criterion, valid_loader, use_cuda) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.inference_mode():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy())
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
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(
                    non_blocking=True
                )
            outputs = model(inputs)
            loss = reduce_loss(criterion(outputs, targets))
            batch_size_ = inputs.size(0)
            (batch_size_ * loss).backward()
            if (i + 1) % 1 == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
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
_FOLDS_CACHE_PATH = os.path.join("/kaggle/working", "train_folds_5.csv")


def prepare_folds():
    if os.path.exists(_FOLDS_CACHE_PATH):
        return pd.read_csv(_FOLDS_CACHE_PATH)
    folds = make_folds(n_folds=5, labels_file_path=base_dir + "train.csv")
    folds.to_csv(_FOLDS_CACHE_PATH, index=False)
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
    kwargs = dict(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        worker_init_fn=_worker_init_fn if nw > 0 else None,
    )
    if nw > 0:
        kwargs["prefetch_factor"] = 4
    loader = torch.utils.data.DataLoader(**kwargs)

    if use_cuda:
        model = model.cuda()
    model.eval()

    n = len(test_dataset)
    out = np.empty((n, number_of_classes), dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for it, inputs in enumerate(loader):
            if it % 50 == 0:
                print(offset)
            if use_cuda:
                inputs = inputs.cuda(non_blocking=True)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            probs = probs.detach().cpu().numpy().astype(np.float32, copy=False)
            bs = probs.shape[0]
            out[offset : offset + bs] = probs
            offset += bs
    return out




## === cell 17
def _encode_targets_multi_hot(df: pd.DataFrame, n_classes: int) -> np.ndarray:
    y = np.zeros((len(df), n_classes), dtype=np.uint8)
    attr_lists = df["attribute_ids"].astype(str).str.split()

    counts = attr_lists.map(len).to_numpy(dtype=np.int32, copy=False)
    total = int(counts.sum())
    if total == 0:
        return y

    rows = np.repeat(np.arange(len(df), dtype=np.int32), counts)
    cols = np.fromiter(
        (int(c) for lst in attr_lists for c in lst),
        dtype=np.int32,
        count=total,
    )
    y[rows, cols] = 1
    return y


def _micro_f1_from_sparse_predictions(
    y_true: np.ndarray, sel_rows: np.ndarray, sel_cols: np.ndarray
) -> float:
    total_true = int(y_true.sum())
    total_pred = int(sel_cols.shape[0])
    if total_pred == 0 and total_true == 0:
        return 1.0
    if total_pred == 0 or total_true == 0:
        return 0.0
    tp = int(y_true[sel_rows, sel_cols].sum())
    fp = total_pred - tp
    fn = total_true - tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _calibrate_postprocess_params(
    model: nn.Module,
    folds_df: pd.DataFrame,
    *,
    fold_idx: int = 0,
    max_valid_samples: int = 6000,
    thresholds: List[float] = None,
    max_labels_grid: List[int] = None,
    min_labels: int = 1,
    infer_batch_size: int = 128,
) -> Tuple[float, int]:
    if thresholds is None:
        thresholds = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    if max_labels_grid is None:
        max_labels_grid = [5, 8, 10, 12, 15, 20]

    valid_df = folds_df[folds_df["fold"] == fold_idx].reset_index(drop=True)
    if len(valid_df) > max_valid_samples:
        valid_df = valid_df.iloc[:max_valid_samples].reset_index(drop=True)

    valid_ds = TrainDataset(
        train_root, number_of_classes, valid_df, eval_transform, tensor_transform
    )
    use_cuda = cuda.is_available()
    nw = _effective_num_workers(num_workers)
    kwargs = dict(
        dataset=valid_ds,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        worker_init_fn=_worker_init_fn if nw > 0 else None,
    )
    if nw > 0:
        kwargs["prefetch_factor"] = 4
    valid_loader = DataLoader(**kwargs)

    model.eval()
    probs_list = []
    with torch.inference_mode():
        for xb, yb in valid_loader:
            if use_cuda:
                xb = xb.cuda(non_blocking=True)
            logits = model(xb)
            probs = (
                torch.sigmoid(logits)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            probs_list.append(probs)
    probs = np.concatenate(probs_list, axis=0)

    y_true = _encode_targets_multi_hot(valid_df, number_of_classes)

    max_k = int(max(max_labels_grid))
    topk_all = topk_indices(probs, k=max_k)  # (N, max_k), DESC
    rows = np.arange(probs.shape[0])[:, None]
    topk_all_probs = probs[rows, topk_all]  # (N, max_k)

    best_score, best_th, best_ml = -1.0, None, None

    for ml in max_labels_grid:
        topk = topk_all[:, :ml]
        topk_probs = topk_all_probs[:, :ml]

        for th in thresholds:
            keep = topk_probs > th
            if min_labels > 0 and min_labels < ml:
                keep[:, :min_labels] = True
            elif min_labels >= ml:
                keep[:, :] = True

            sel_rows, sel_pos = np.nonzero(keep)
            sel_cols = topk[sel_rows, sel_pos]
            score = _micro_f1_from_sparse_predictions(y_true, sel_rows, sel_cols)

            if score > best_score:
                best_score, best_th, best_ml = score, float(th), int(ml)

    print(
        f"Calibration (fold={fold_idx}, n={len(valid_df)}): best_micro_f1={best_score:.5f} at th={best_th}, max_labels={best_ml}"
    )
    return float(best_th), int(best_ml)




## === cell 18
def save_predictions(
    preds,
    test_dataset,
    *,
    pred_threshold: float,
    pred_max_labels: int,
    pred_min_labels: int = 1,
):
    probs = np.asarray(preds, dtype=np.float32)

    topk = topk_indices(probs, k=pred_max_labels)

    pred_idx = binarize_prediction_sparse(
        probs,
        threshold=pred_threshold,
        argsorted=topk,
        min_labels=pred_min_labels,
        max_labels=pred_max_labels,
    )

    prediction = [
        " ".join(map(str, np.sort(idx).tolist())) if idx.size else ""
        for idx in pred_idx
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
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(
                    non_blocking=True
                )
            outputs = model(inputs)
            loss = reduce_loss(criterion(outputs, targets))
            (inputs.size(0) * loss).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
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
    batch_size = 128

    test_img_dir = os.path.join(base_dir, "test")

    sample_path = os.path.join(base_dir, "sample_submission.csv")
    sample = pd.read_csv(sample_path)
    test_ids = sample["id"].astype(str).tolist()
    test_dataset = PredictImageDataset(
        test_img_dir, eval_transform, tensor_transform, ids=test_ids
    )

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
        max_valid_samples=6000,
        thresholds=[0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
        max_labels_grid=[5, 8, 10, 12, 15, 20],
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

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
