# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00373) has done: 'I add a robust path lookup for the dataset directory so the script can find the CSV files and image folders regardless of the execution environment, and I fix a typo and indexing bug in the mask‑creation logic used during binarisation. These changes resolve the FileNotFound errors and the undefined‑variable error, allowing the pipeline to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.00112) has done: 'The fix adds missing imports, defines required global paths and hyper‑parameters, and corrects undefined names so the pipeline runs end‑to‑end and writes a proper `submission.csv`. No core model logic is altered, only the surrounding setup and tiny bug fixes.'

# 9. Code solution

## === cell 0
import os
import random
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import fbeta_score
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torchvision.models as M
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
    Normalize,
)

base_dir = Path(os.getcwd()) / "kaggle" / "data" / "imet-2020-fgvc7"
train_root = base_dir / "train"
test_root = base_dir / "test"

model_path = base_dir / "model.pt"
model_path_for_test = model_path
input_dir = base_dir  # used by create_net for custom weights

batch_size = 64
num_workers = 0
number_of_classes = pd.read_csv(base_dir / "labels.csv")["attribute_id"].max() + 1




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_54/223879711.py in <cell line: 0>()
     44 batch_size = 64
     45 num_workers = 0
---> 46 number_of_classes = pd.read_csv(base_dir / "labels.csv")["attribute_id"].max() + 1
     47 
     48 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/kaggle/data/imet-2020-fgvc7/labels.csv'

## === cell 1
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




## === cell 2
def save_folds(n_folds: int, labels_file_path: str, save_path: str):
    df = make_folds(n_folds=n_folds, labels_file_path=labels_file_path)
    df.to_csv(save_path, index=False)




## === cell 3
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        image = load_transform_image(
            item.id, self.root_dir, self.image_transform, self.tensor_transform
        )
        target = torch.zeros(self.number_of_classes, dtype=torch.float32)
        for attribute in item.attribute_ids.split():
            target[int(attribute)] = 1.0
        return image, target


class PredictImageDataset(Dataset):
    def __init__(self, root_dir: Path, image_transform, tensor_transform):
        self.root_dir = root_dir
        self.files = [
            f
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.tensor_transform = tensor_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.splitext(self.files[idx])[0]  # without extension
        image = load_transform_image(
            img_name, self.root_dir, self.image_transform, self.tensor_transform
        )
        return image

    def filenames(self):
        return [os.path.splitext(x)[0] for x in self.files]




## === cell 4
def load_transform_image(
    item_id, root_dir: Path, image_transform: Callable, tensor_transform: Callable
):
    image = load_image(item_id, root_dir)
    image = image_transform(image)
    return tensor_transform(image)


def load_image(item_id, root_dir: Path) -> Image.Image:
    path = os.path.join(str(root_dir), f"{item_id}.png")
    image = cv2.imread(path)
    if image is None:
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        image = dummy
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)




## === cell 5
def make_loader(
    df: pd.DataFrame, image_transform: Callable, tensor_transform: Callable
) -> DataLoader:
    dataset = TrainDataset(
        Path(train_root), number_of_classes, df, image_transform, tensor_transform
    )
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )




## === cell 6
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    """
    Create a network instance.
    If pretrained is True we first try to load custom weights from
    ``input_dir/<model_name>/<model_name>.pth``.
    If that file does not exist we fall back to the standard ImageNet weights
    provided by torchvision (i.e. ``net_cls(pretrained=True)``).
    """
    model_name = net_cls.__name__
    weights_path = os.path.join(input_dir, model_name, f"{model_name}.pth")
    if pretrained and os.path.exists(weights_path):
        net = net_cls()
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
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




## === cell 7
def load_model(model: nn.Module, path: Path) -> Dict:
    if not os.path.exists(path):
        print(
            f"Warning: checkpoint {path} not found – proceeding with randomly initialized model."
        )
        return {
            "model": model.state_dict(),
            "epoch": 0,
            "step": 0,
            "best_valid_loss": float("inf"),
        }
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state["model"])
    epoch = state.get("epoch", 0)
    step = state.get("step", 0)
    print(f"Loaded model from epoch {epoch}, step {step}")
    return state




## === cell 8
def reduce_loss(loss):
    return loss.sum() / loss.shape[0]




## === cell 9
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
    """Create a binary mask that selects the top_n indices per row."""
    if top_n <= 0:
        return np.zeros_like(argsorted, dtype=np.uint8)
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = np.repeat(np.arange(argsorted.shape[0]), top_n)
    mask[row_indices, col_indices] = 1
    return mask




## === cell 10
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
            all_targets.append(targets.cpu().numpy().copy())
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




## === cell 11
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
    if n_epochs is None:
        n_epochs = 100
    params = list(params)
    optimizer = init_optimizer(params, lr)

    model_path_local = model_path
    best_model_path = os.path.join(base_dir, "best-model-copy.pt")
    uptrain = True
    if uptrain and os.path.exists(model_path_local):
        state = load_model(model, model_path_local)
        epoch = state["epoch"]
        step = state["step"]
        best_valid_loss = state["best_valid_loss"]
    else:
        epoch = 1
        step = 0
        best_valid_loss = float("inf")
    lr_changes = 0

    def save(ep):
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": ep,
                "step": step,
                "best_valid_loss": best_valid_loss,
            },
            model_path_local,
        )

    report_each = 100
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        losses = []
        for i, (inputs, targets) in enumerate(train_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = reduce_loss(criterion(outputs, targets))
            (inputs.size(0) * loss).backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            losses.append(loss.item())
            if (i + 1) % report_each == 0:
                mean_loss = np.mean(losses[-report_each:])
                print(f"mean_loss: {mean_loss:.3f}")
        save(epoch + 1)
        valid_metrics = validation(model, criterion, valid_loader, use_cuda)

        valid_loss = valid_metrics["valid_loss"]
        valid_losses.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if os.path.abspath(model_path_local) != os.path.abspath(best_model_path):
                shutil.copy(model_path_local, best_model_path)
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




## === cell 12
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




## === cell 13
def prepare_folds():
    folds = make_folds(n_folds=5, labels_file_path=os.path.join(base_dir, "train.csv"))
    return folds




## === cell 14
def start_train():
    folds = prepare_folds()
    train_fold = folds[folds["fold"] != 0]
    valid_fold = folds[folds["fold"] == 0]

    train_loader = make_loader(train_fold, train_transform, tensor_transform)
    valid_loader = make_loader(valid_fold, eval_transform, tensor_transform)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    model = ResNet(
        num_classes=number_of_classes, pretrained=True, net_cls=M.resnext50_32x4d
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    train(
        model=model,
        criterion=criterion,
        params=model.parameters(),
        train_loader=train_loader,
        valid_loader=valid_loader,
        patience=4,
        init_optimizer=lambda params, lr: Adam(params, lr),
        use_cuda=use_cuda,
        n_epochs=10,
    )




## === cell 15
def predict(model, test_dataset, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    model_result = []
    with torch.no_grad():
        for i_batch in loader:
            inputs = i_batch.to(device)
            batch_out = model(inputs)
            probs = torch.sigmoid(batch_out)
            model_result.extend(probs.cpu().numpy())
    return model_result




## === cell 16
def save_predictions(preds, test_dataset):
    probs = np.array(preds)
    argsorted = probs.argsort(axis=1)
    bin_pred = binarize_prediction(probs, threshold=0.20, argsorted=argsorted)

    prediction = []
    for row in bin_pred:
        labels = np.where(row == 1)[0].tolist()
        prediction.append(" ".join(map(str, labels)))

    sample_path = os.path.join(base_dir, "sample_submission.csv")
    sample = pd.read_csv(sample_path)
    sample["id"] = test_dataset.filenames()
    sample["attribute_ids"] = prediction
    sample.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")
    print(sample.head())




## === cell 17
def test():
    batch_size = 200
    test_dataset = PredictImageDataset(
        Path(test_root), eval_transform, tensor_transform
    )
    new_model = ResNet(
        num_classes=number_of_classes, pretrained=True, net_cls=M.resnext50_32x4d
    )
    if os.path.exists(model_path_for_test):
        load_model(new_model, model_path_for_test)
    else:
        print("Checkpoint not found – using pretrained ImageNet weights.")
    new_model.eval()
    preds = predict(new_model, test_dataset, batch_size)
    save_predictions(preds, test_dataset)




## === cell 18
start_train()
test()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_54/1527150311.py in <cell line: 0>()
      1 # Run the whole pipeline
----> 2 start_train()
      3 test()

/tmp/ipykernel_54/2254623822.py in start_train()
      1 def start_train():
----> 2     folds = prepare_folds()
      3     train_fold = folds[folds["fold"] != 0]
      4     valid_fold = folds[folds["fold"] == 0]
      5 

/tmp/ipykernel_54/3472170511.py in prepare_folds()
      1 def prepare_folds():
----> 2     folds = make_folds(n_folds=5, labels_file_path=os.path.join(base_dir, "train.csv"))
      3     return folds
      4 
      5 

/tmp/ipykernel_54/3717037617.py in make_folds(n_folds, labels_file_path)
      1 def make_folds(n_folds: int, labels_file_path: str) -> pd.DataFrame:
----> 2     df = pd.read_csv(labels_file_path)
      3     # count how many times each class appears
      4     cls_counts = Counter(
      5         cls for classes in df["attribute_ids"].str.split() for cls in classes

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/kaggle/data/imet-2020-fgvc7/train.csv'
