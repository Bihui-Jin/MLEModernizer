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

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
import os
import math
import random
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)



## === cell 1
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 8

FALLBACK_TRAIN_EPOCHS = 2
FALLBACK_LR = 2e-4
FALLBACK_WD = 1e-4
FALLBACK_SAVE_PATH = "fallback_effb4_finetuned.pth"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICES = (
    [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)


def _resolve_path(path: str) -> str:
    """Resolve common Kaggle mount differences without changing intended relative paths."""
    if os.path.exists(path):
        return path
    if path.startswith("../input/"):
        alt = os.path.join("/kaggle/input", path[len("../input/") :])
        if os.path.exists(alt):
            return alt
    if path.startswith("../input/"):
        alt2 = os.path.join("/kaggle/data/input", path[len("../input/") :])
        if os.path.exists(alt2):
            return alt2
    return path


INPUT_PATH = _resolve_path(INPUT_PATH)
TRAIN_CSV_PATH = _resolve_path(TRAIN_CSV_PATH)
TRAIN_IMAGE_PATH = _resolve_path(TRAIN_IMAGE_PATH)
TEST_IMAGE_PATH = _resolve_path(TEST_IMAGE_PATH)




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor)
        y = y + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor).to(y_hat.device)

    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1 - y_true) * (1 - p)

    ce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * ce, dim=-1)




## === cell 3
def lr_tune(epoch, num_epochs=NUM_EPOCHS):
    lr_start = LR_START
    lr_max = LR_MAX
    lr_final = LR_FINAL
    lr_warmup_epoch = 4
    lr_sustain_epoch = 0
    lr_decay_epoch = num_epochs - lr_warmup_epoch - lr_sustain_epoch - 1

    if epoch <= lr_warmup_epoch:
        lr = lr_start + (lr_max - lr_start) * (epoch / lr_warmup_epoch) ** 2.5
    elif epoch < lr_warmup_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        epoch_diff = epoch - lr_warmup_epoch - lr_sustain_epoch
        decay_factor = (epoch_diff / lr_decay_epoch) * math.pi
        decay_factor = (torch.cos(torch.tensor(decay_factor)).numpy() + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr


x = [i for i in range(NUM_EPOCHS)]
y = [lr_tune(i) for i in x]
plt.plot(x, y)



## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(p=0.5),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)



## === cell 5
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), p=1.0),
            ],
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 6
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)



## === cell 7
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=False)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)
my_model_1



## === cell 8
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)
my_model_2



## === cell 9
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 10
import multiprocessing
import torchvision


def _find_weights_file(input_dir: str, fname: str) -> str | None:
    candidates = [
        os.path.join(input_dir, fname),
        os.path.join(
            "/kaggle/input", os.path.basename(os.path.normpath(input_dir)), fname
        ),
        os.path.join("/kaggle/input", fname),
        os.path.join(
            "/kaggle/data/input", os.path.basename(os.path.normpath(input_dir)), fname
        ),
        os.path.join("/kaggle/data/input", fname),
        os.path.join("../input", fname),
        os.path.join("/kaggle/working", fname),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _load_state_dict_flexible(model: nn.Module, state: dict) -> None:
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format.")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)


def _default_num_workers() -> int:
    if os.name == "nt":
        return 0
    cpu = multiprocessing.cpu_count()
    return max(2, min(8, cpu // 2))


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        augs: A.Compose,
        with_labels: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.augs = augs
        self.with_labels = with_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_id"])
        img = torchvision.io.read_image(
            image_path, mode=torchvision.io.ImageReadMode.RGB
        )  # uint8 CHW
        image = img.permute(1, 2, 0).contiguous().numpy()
        x = self.augs(image=image)["image"]
        if self.with_labels:
            y = int(row["label"])
            return x, y
        return x, row["image_id"]


class CassavaBaseTensorDataset(torch.utils.data.Dataset):
    def __init__(self, image_names: list[str], image_dir: str, base_augs: A.Compose):
        self.image_names = image_names
        self.image_dir = image_dir
        self.base_augs = base_augs

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        path = os.path.join(self.image_dir, name)
        img = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)
        image = img.permute(1, 2, 0).contiguous().numpy()
        x = self.base_augs(image=image)["image"]  # float tensor CHW normalized
        return x, idx


def _collate_tensor_and_index(batch):
    xs, idxs = zip(*batch)
    xb = torch.stack(xs, dim=0)
    idx = torch.as_tensor(idxs, dtype=torch.int64)
    return xb, idx


def _make_loader(ds, batch_size: int, shuffle: bool, *, num_workers: int | None = None):
    if num_workers is None:
        num_workers = _default_num_workers()
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if (num_workers > 0) else None,
        collate_fn=_collate_tensor_and_index,
    )


def _tta_transforms_list(tta: int):
    def _id(x):
        return x

    def _t(x):
        return x.transpose(2, 3)

    def _h(x):
        return torch.flip(x, dims=(3,))

    def _v(x):
        return torch.flip(x, dims=(2,))

    def _th(x):
        return torch.flip(x.transpose(2, 3), dims=(3,))

    def _tv(x):
        return torch.flip(x.transpose(2, 3), dims=(2,))

    def _hv(x):
        return torch.flip(x, dims=(2, 3))

    def _thv(x):
        return torch.flip(x.transpose(2, 3), dims=(2, 3))

    base = [_id, _t, _h, _v, _th, _tv, _hv, _thv]
    return [base[i % len(base)] for i in range(int(tta))]


@torch.inference_mode()
def _predict_logits_loader(
    model: nn.Module,
    image_names: list[str],
    image_dir: str,
    augs: A.Compose,
    tta: int = 1,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    model.eval()
    n = len(image_names)
    out_sum = torch.zeros((n, OUT_FEATURES), dtype=torch.float32)

    ds = CassavaBaseTensorDataset(image_names, image_dir, base_augs=valid_augs)
    num_workers = _default_num_workers()
    loader = _make_loader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    tta_fns = _tta_transforms_list(tta)

    for fn in tta_fns:
        for xb, idx in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            xb = fn(xb)
            logits = model(xb)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            out_sum.index_add_(0, idx, logits.detach().cpu())

    out_sum /= float(tta)
    return out_sum


def _train_fallback_effb4(train_csv: str, train_img_dir: str, save_path: str) -> str:
    df = pd.read_csv(train_csv)
    assert {"image_id", "label"}.issubset(df.columns)

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_valid = max(1, int(0.05 * len(df)))
    df_train = df.iloc[:-n_valid].reset_index(drop=True)
    df_valid = df.iloc[-n_valid:].reset_index(drop=True)

    train_ds = CassavaDataset(df_train, train_img_dir, train_augs, with_labels=True)
    valid_ds = CassavaDataset(df_valid, train_img_dir, valid_augs, with_labels=True)

    num_workers = max(2, min(4, _default_num_workers()))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if (num_workers > 0) else None,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if (num_workers > 0) else None,
    )

    model = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=OUT_FEATURES
    ).to(DEVICE)
    if torch.cuda.is_available() and len(DEVICES) > 1:
        model = nn.DataParallel(model).to(DEVICE)

    optimizer = OPTIMIZER(model.parameters(), lr=FALLBACK_LR, weight_decay=FALLBACK_WD)

    for epoch in range(FALLBACK_TRAIN_EPOCHS):
        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"Finetune fallback epoch {epoch+1}/{FALLBACK_TRAIN_EPOCHS}",
        )
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            logits = model(xb)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            y_onehot = F.one_hot(yb, num_classes=OUT_FEATURES).float()
            loss = sigmoid_focal_cross_entropy(logits, y_onehot).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                logits = model(xb)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        print(f"Fallback valid acc (monitor only): {correct/total:.4f}")

    state = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state, save_path)
    return save_path




## === cell 11
assert os.path.isdir(TEST_IMAGE_PATH), f"TEST_IMAGE_PATH not found: {TEST_IMAGE_PATH}"
test_image_list = sorted(
    [fn for fn in os.listdir(TEST_IMAGE_PATH) if fn.lower().endswith(".jpg")]
)

resnext_ckpt = _find_weights_file(INPUT_PATH, RESNEXT_PATH)
b4_ckpt = _find_weights_file(INPUT_PATH, B4_PATH)

use_ensemble = (resnext_ckpt is not None) and (b4_ckpt is not None)

_prev_benchmark = torch.backends.cudnn.benchmark
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

try:
    if use_ensemble:
        my_model_1 = my_model_1.to(DEVICE)
        state1 = torch.load(resnext_ckpt, map_location="cpu")
        _load_state_dict_flexible(my_model_1, state1)
        if torch.cuda.is_available() and len(DEVICES) > 1:
            my_model_1 = nn.DataParallel(my_model_1).to(DEVICE)

        predictions_1 = _predict_logits_loader(
            my_model_1,
            test_image_list,
            TEST_IMAGE_PATH,
            augs=valid_augs,
            tta=1,
            batch_size=BATCH_SIZE,
        )
        normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        my_model_2 = my_model_2.to(DEVICE)
        state2 = torch.load(b4_ckpt, map_location="cpu")
        _load_state_dict_flexible(my_model_2, state2)
        if torch.cuda.is_available() and len(DEVICES) > 1:
            my_model_2 = nn.DataParallel(my_model_2).to(DEVICE)

        predictions_2 = _predict_logits_loader(
            my_model_2,
            test_image_list,
            TEST_IMAGE_PATH,
            augs=valid_augs,
            tta=TTA,
            batch_size=BATCH_SIZE,
        )
        normalize_pred_2 = F.normalize(predictions_2.T, p=2, dim=0).T

        final_pred = (normalize_pred_1 * 0.4) + (normalize_pred_2 * 0.6)
    else:
        finetuned_ckpt = _find_weights_file("/kaggle/working", FALLBACK_SAVE_PATH)
        if finetuned_ckpt is None or not os.path.exists(finetuned_ckpt):
            assert os.path.isfile(
                TRAIN_CSV_PATH
            ), f"TRAIN_CSV_PATH not found: {TRAIN_CSV_PATH}"
            assert os.path.isdir(
                TRAIN_IMAGE_PATH
            ), f"TRAIN_IMAGE_PATH not found: {TRAIN_IMAGE_PATH}"
            finetuned_ckpt = _train_fallback_effb4(
                TRAIN_CSV_PATH, TRAIN_IMAGE_PATH, FALLBACK_SAVE_PATH
            )

        fallback_name = "tf_efficientnet_b4_ns"
        fallback_model = timm.create_model(
            fallback_name, pretrained=True, num_classes=OUT_FEATURES
        )
        _load_state_dict_flexible(
            fallback_model, torch.load(finetuned_ckpt, map_location="cpu")
        )
        fallback_model = fallback_model.to(DEVICE)
        if torch.cuda.is_available() and len(DEVICES) > 1:
            fallback_model = nn.DataParallel(fallback_model).to(DEVICE)

        final_pred = _predict_logits_loader(
            fallback_model,
            test_image_list,
            TEST_IMAGE_PATH,
            augs=valid_augs,
            tta=TTA,
            batch_size=BATCH_SIZE,
        )

    label = final_pred.argmax(dim=-1).numpy().astype(int).tolist()

    df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
    df_submission.to_csv(SUBMISSION_PATH, index=False)

    print(
        f"Wrote submission: {SUBMISSION_PATH} | rows={len(df_submission)} | use_ensemble={use_ensemble}"
    )
    print(df_submission.head())
finally:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = _prev_benchmark
