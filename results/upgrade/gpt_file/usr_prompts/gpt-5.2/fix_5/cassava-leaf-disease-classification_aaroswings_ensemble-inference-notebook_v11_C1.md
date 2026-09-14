# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import io
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
batch_size = 32
valid_input_size = (512, 512)
num_classes = 5

DATA_ROOT = Path("/kaggle/input/cassava-leaf-disease-classification")
train_csv_path = DATA_ROOT / "train.csv"
train_img_path = DATA_ROOT / "train_images"
test_img_path = DATA_ROOT / "test_images"
sample_sub_path = DATA_ROOT / "sample_submission.csv"

assert train_csv_path.exists(), f"Missing train.csv: {train_csv_path}"
assert train_img_path.exists(), f"Missing train images path: {train_img_path}"
assert test_img_path.exists(), f"Missing test images path: {test_img_path}"
assert sample_sub_path.exists(), f"Missing sample submission: {sample_sub_path}"



## === cell 2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tfms = A.Compose(
    [
        A.LongestMaxSize(max_size=max(valid_input_size), always_apply=True),
        A.PadIfNeeded(
            min_height=valid_input_size[0],
            min_width=valid_input_size[1],
            border_mode=0,
            value=(0, 0, 0),
            always_apply=True,
        ),
        A.RandomCrop(
            height=valid_input_size[0], width=valid_input_size[1], always_apply=True
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)

test_tfms = A.Compose(
    [
        A.LongestMaxSize(max_size=max(valid_input_size), always_apply=True),
        A.PadIfNeeded(
            min_height=valid_input_size[0],
            min_width=valid_input_size[1],
            border_mode=0,
            value=(0, 0, 0),
            always_apply=True,
        ),
        A.CenterCrop(
            height=valid_input_size[0], width=valid_input_size[1], always_apply=True
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, tfms):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = row["image_id"]
        label = int(row["label"])
        with Image.open(self.img_dir / fname) as im:
            img = np.array(im.convert("RGB"))
        img = self.tfms(image=img)["image"]
        return img, label


class TestDataset(Dataset):
    def __init__(self, path, tfms):
        super().__init__()
        path = Path(path)
        self.files = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        self.tfms = tfms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with Image.open(file) as im:
            img = np.array(im.convert("RGB"))
        img = self.tfms(image=img)["image"]
        return file.name, img




## === cell 3
import torchvision


def _make_model(num_classes: int = 5) -> nn.Module:
    m = torchvision.models.resnext101_32x8d(
        weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
    )
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _try_load_state_dict_into_resnext101(
    checkpoint_path: Path, num_classes: int = 5
) -> nn.Module:
    obj = torch.load(checkpoint_path, map_location="cpu")
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        sd = obj
    else:
        if hasattr(obj, "eval") and hasattr(obj, "to"):
            return obj
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v

    m = torchvision.models.resnext101_32x8d(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)

    for bad_key in ["fc.weight", "fc.bias"]:
        if bad_key in cleaned:
            cleaned.pop(bad_key, None)

    m.load_state_dict(cleaned, strict=False)
    return m




## === cell 4
train_df = pd.read_csv(train_csv_path)


def stratified_split(
    df: pd.DataFrame, label_col: str = "label", val_frac: float = 0.1, seed: int = 42
):
    rng = np.random.default_rng(seed)
    val_idx = []
    for lbl, sub in df.groupby(label_col):
        idx = sub.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx.extend(idx[:n_val].tolist())
    val_idx = sorted(val_idx)
    val_df = df.loc[val_idx].copy()
    tr_df = df.drop(index=val_idx).copy()
    return tr_df, val_df


tr_df, val_df = stratified_split(train_df, "label", val_frac=0.1, seed=42)

train_ds = TrainDataset(tr_df, train_img_path, train_tfms)
val_ds = TrainDataset(val_df, train_img_path, test_tfms)

_nw = min(8, os.cpu_count() or 1)
_dl_common = dict(
    num_workers=_nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_nw > 0),
    prefetch_factor=4 if _nw > 0 else None,
)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    **{k: v for k, v in _dl_common.items() if v is not None},
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in _dl_common.items() if v is not None},
)

model = None
explicit_ckpt = os.environ.get("CASSAVA_CKPT", "").strip()
if explicit_ckpt:
    cp = Path(explicit_ckpt)
    if cp.exists():
        model = _try_load_state_dict_into_resnext101(cp, num_classes=num_classes)
        print("Warm-started from explicit checkpoint:", str(cp))

if model is None:
    model = _make_model(num_classes=num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)


@torch.inference_mode()
def evaluate_accuracy(m: nn.Module, loader: DataLoader) -> float:
    m.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = m(x)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


epochs = 2
best_val = -1.0
best_state = None

for ep in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())

    val_acc = evaluate_accuracy(model, val_loader)
    avg_loss = running_loss / max(1, len(train_loader))
    print(f"epoch {ep}/{epochs} - train_loss={avg_loss:.4f} - val_acc={val_acc:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state, strict=True)

model = model.eval()



## === cell 5
test_ds = TestDataset(test_img_path, test_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in _dl_common.items() if v is not None},
)


class EnsemblePredictor:
    def __init__(self, models):
        super().__init__()
        self.models = models

    @torch.inference_mode()
    def predict_on_loader(self, loader):
        predictions = []
        filenames = []

        for file, img in loader:
            img = img.to(device, non_blocking=True)
            pred = None
            for model in self.models:
                out = model(img)
                out = out.float().detach().cpu().numpy()
                pred = out if pred is None else (pred + out)

            predictions.extend(np.asarray(pred).argmax(axis=1).astype(int).tolist())
            filenames.extend(list(file))

        return predictions, filenames


predictor = EnsemblePredictor([model])
predictions, filenames = predictor.predict_on_loader(test_loader)



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(filenames, predictions))

missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if missing:
    for img_id in missing:
        pred_map[img_id] = 0

submission_df = sample_sub.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map).astype(int)

out_path = Path("submission.csv")
submission_df.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print("Submission shape:", submission_df.shape)
print(submission_df.head())
