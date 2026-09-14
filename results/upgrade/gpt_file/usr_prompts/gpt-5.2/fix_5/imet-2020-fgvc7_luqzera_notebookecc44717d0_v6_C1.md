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
model_path_for_test = f"../input/resnet50/best-model_wc.pt"
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
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        False
    )  # allow fastest kernels; evaluation semantics unchanged
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)


def _resolve_base_dir(preferred: str) -> str:
    candidates = [
        preferred,
        "../input/imet-2020-fgvc7/",
        "../input/imet-2020-fgvc7/imet-2020-fgvc7/",
        "/kaggle/input/imet-2020-fgvc7/",
        "/kaggle/input/imet-2020-fgvc7/imet-2020-fgvc7/",
    ]
    for c in candidates:
        if (
            c
            and os.path.exists(os.path.join(c, "train.csv"))
            and os.path.exists(os.path.join(c, "train"))
        ):
            return c if c.endswith("/") else c + "/"
    return preferred if preferred.endswith("/") else preferred + "/"


def _resolve_input_dir(preferred: str) -> str:
    candidates = [preferred, "../input/", "/kaggle/input/"]
    for c in candidates:
        if c and os.path.exists(c):
            return c if c.endswith("/") else c + "/"
    return preferred if preferred.endswith("/") else preferred + "/"


input_dir = _resolve_input_dir(input_dir)
base_dir = _resolve_base_dir(base_dir)
train_root = os.path.join(base_dir, "train")
print("Resolved input_dir:", input_dir)
print("Resolved base_dir:", base_dir)
print("Resolved train_root:", train_root)




## === cell 3
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




## === cell 4
def save_folds(n_folds: int, labels_file_path: str, save_path: str):
    df = make_folds(n_folds=5, labels_file_path=base_dir + "train.csv")
    df.to_csv(save_path, index=None)




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
        self.files = sorted(os.listdir(root_dir))
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
        return [os.path.splitext(x)[0] for x in self.files]




## === cell 6
from functools import lru_cache


@lru_cache(maxsize=2048)
def _read_png_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def load_transform_image(
    item_id, root_dir: Path, image_transform: Callable, tensor_transform: Callable
):
    image = load_image(item_id, root_dir)
    image = image_transform(image)
    return tensor_transform(image)


def load_image(item_id, root_dir: Path) -> Image.Image:
    path = os.path.join(root_dir, f"{item_id}.png")
    b = _read_png_bytes(path)
    arr = np.frombuffer(b, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)




## === cell 7
def make_loader(
    df: pd.DataFrame, image_transform: Callable, tensor_transform: Callable
) -> DataLoader:
    dataset = TrainDataset(
        train_root, number_of_classes, df, image_transform, tensor_transform
    )
    use_cuda = torch.cuda.is_available()
    w = min(num_workers, os.cpu_count() or 1)
    loader_kwargs = dict(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=w,
        pin_memory=use_cuda,
    )
    if w > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    return DataLoader(**loader_kwargs)




## === cell 8
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = os.path.join(input_dir, f"{model_name}", f"{model_name}.pth")
        if os.path.exists(weights_path):
            net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            try:
                net = net_cls(weights=M.ResNet50_Weights.IMAGENET1K_V2)
            except Exception:
                net = net_cls(weights=None)
    else:
        try:
            net = net_cls(weights=None)
        except TypeError:
            net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.resnet50, dropout=False
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




## === cell 9
def load_model(model: nn.Module, path: Path) -> Dict:
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        epoch = state.get("epoch", -1)
        step = state.get("step", -1)
        print(f"Loaded model from epoch {epoch}, step {step}")
        return state
    else:
        model.load_state_dict(state)
        print("Loaded model state_dict checkpoint")
        return {"model": state, "epoch": -1, "step": -1, "best_valid_loss": None}




## === cell 10
def reduce_loss(loss):
    return loss.sum() / loss.shape[0]




## === cell 11
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
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = np.repeat(np.arange(argsorted.shape[0]), top_n)
    mask[row_indices, col_indices] = 1
    return mask




## === cell 12
def validation(model: nn.Module, criterion, valid_loader, use_cuda) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
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
            all_predictions.append(predictions.detach().cpu().numpy())
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

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
    metrics["valid_loss"] = float(np.mean(all_losses))
    print(
        " | ".join(
            f"{k} {v:.3f}" for k, v in sorted(metrics.items(), key=lambda kv: -kv[1])
        )
    )
    return metrics




## === cell 13
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
        for i, (inputs, targets) in enumerate(train_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(
                    non_blocking=True
                )
            outputs = model(inputs)
            loss = reduce_loss(criterion(outputs, targets))
            batch_size_local = inputs.size(0)
            (batch_size_local * loss).backward()
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




## === cell 14
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

eval_transform = Compose([Resize((288, 288))])

tensor_transform = Compose(
    [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 15
def prepare_folds():
    folds = make_folds(n_folds=5, labels_file_path=os.path.join(base_dir, "train.csv"))
    return folds




## === cell 16
def start_train():
    folds = prepare_folds()
    train_fold = folds[folds["fold"] != 0]
    valid_fold = folds[folds["fold"] == 0]

    train_loader = make_loader(train_fold, train_transform, tensor_transform)
    valid_loader = make_loader(valid_fold, test_transform, tensor_transform)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    model = ResNet(num_classes=number_of_classes, pretrained=True, net_cls=M.resnet50)

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




## === cell 17
def predict(model, test_dataset, batch_size):
    use_cuda = torch.cuda.is_available()
    infer_workers = min(num_workers, os.cpu_count() or 1)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=infer_workers,
        pin_memory=use_cuda,
    )
    if infer_workers > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    loader = torch.utils.data.DataLoader(test_dataset, **loader_kwargs)

    if use_cuda:
        model = model.cuda()
    model.eval()

    n = len(test_dataset)
    out = np.empty((n, number_of_classes), dtype=np.float32)

    it = 0
    with torch.no_grad():
        for batch in loader:
            if it % (batch_size * 10) == 0:
                print(it)
            if use_cuda:
                inputs = batch.cuda(non_blocking=True)
            else:
                inputs = batch
            logits = model(inputs)
            probs = torch.sigmoid(logits)  # align with validation semantics
            bsz = probs.size(0)
            out[it : it + bsz] = probs.detach().cpu().numpy()
            it += bsz
    return out




## === cell 18
def save_predictions(preds, test_dataset):
    probs = np.array(preds, dtype=np.float32)
    argsorted = probs.argsort(axis=1)
    preds_fin = binarize_prediction(
        probs, threshold=0.20, argsorted=argsorted, min_labels=1, max_labels=10
    ).astype(int)

    prediction = []
    for i in range(preds_fin.shape[0]):
        pred1 = np.argwhere(preds_fin[i] == 1).reshape(-1).tolist()
        pred_str = " ".join(list(map(str, pred1)))
        prediction.append(pred_str)

    sample_path = os.path.join(base_dir, "sample_submission.csv")
    sample = pd.read_csv(sample_path)

    fname_to_pred = dict(zip(test_dataset.filenames(), prediction))
    sample["attribute_ids"] = sample["id"].map(fname_to_pred).fillna("")
    sample.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", sample.shape)




## === cell 19
def _find_model_checkpoint(preferred_path: str) -> str:
    candidates = [
        preferred_path,
        model_path_for_test,
        os.path.join(input_dir, "resnet50", "best-model_wc.pt"),
        os.path.join(input_dir, "resnet50", "best-model.pt"),
        os.path.join(input_dir, "resnet50", "best-model_wc.pth"),
        os.path.join(input_dir, "resnet50", "best-model.pth"),
        os.path.join(input_dir, "resnet50", "model.pt"),
        os.path.join(input_dir, "resnet50", "model.pth"),
        os.path.join(base_dir, "best-model.pt"),
        os.path.join(base_dir, "model.pt"),
        "best-model.pt",
        "model.pt",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            print("Using checkpoint:", p)
            return p
    return ""


def test():
    batch_size_local = 200
    test_img_dir = os.path.join(base_dir, "test")
    test_dataset = PredictImageDataset(test_img_dir, eval_transform, tensor_transform)

    ckpt_path = _find_model_checkpoint(model_path_for_test)

    if not ckpt_path:
        print("No checkpoint found; starting training to generate best-model.pt ...")
        start_train()
        ckpt_path = "best-model.pt"

    new_model = ResNet(num_classes=number_of_classes)
    load_model(new_model, ckpt_path)

    preds = predict(new_model, test_dataset, batch_size_local)
    save_predictions(preds, test_dataset)




## === cell 20
test()
