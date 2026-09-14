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
            [f for f in os.listdir(root_dir) if f.lower().endswith(".png")]
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
        return [os.path.splitext(x)[0] for x in self.files]




## === cell 5
def load_transform_image(
    item_id, root_dir: Path, image_transform: Callable, tensor_transform: Callable
):
    image = load_image(item_id, root_dir)
    image = image_transform(image)
    return tensor_transform(image)


def load_image(item_id, root_dir: Path) -> Image.Image:
    path = root_dir + f"/{item_id}.png"
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)




## === cell 6
def make_loader(
    df: pd.DataFrame, image_transform: Callable, tensor_transform: Callable
) -> DataLoader:
    dataset = TrainDataset(
        train_root, number_of_classes, df, image_transform, tensor_transform
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
        argsorted = probabilities.argsort(axis=1)
    max_mask = make_mask(argsorted, max_labels)
    min_mask = make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask




## === cell 11
def validation(model: nn.Module, criterion, valid_loader, use_cuda) -> Dict[str, float]:
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

    for it, inputs in enumerate(loader):
        print(it * batch_size)
        if use_cuda:
            inputs = inputs.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            model_result.extend(probs.detach().cpu().numpy())
    return model_result




## === cell 17
def save_predictions(preds, test_dataset):
    probs = np.asarray(preds, dtype=np.float32)  # (N, C)
    argsorted = probs.argsort(axis=1)
    bin_pred = binarize_prediction(
        probs, threshold=0.20, argsorted=argsorted, min_labels=1, max_labels=10
    ).astype(np.uint8)

    prediction = []
    for i in range(bin_pred.shape[0]):
        pred1 = np.flatnonzero(bin_pred[i]).tolist()
        prediction.append(" ".join(map(str, pred1)))

    sample_path = os.path.join(base_dir, "sample_submission.csv")
    sample = pd.read_csv(sample_path)

    fnames = test_dataset.filenames()
    pred_map = dict(zip(fnames, prediction))
    sample["attribute_ids"] = sample["id"].map(pred_map).fillna("")

    sample.to_csv("submission.csv", index=False)




## === cell 18
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




## === cell 19
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

    new_model.eval()
    preds = predict(new_model, test_dataset, batch_size)
    save_predictions(preds, test_dataset)




## === cell 20
test()
