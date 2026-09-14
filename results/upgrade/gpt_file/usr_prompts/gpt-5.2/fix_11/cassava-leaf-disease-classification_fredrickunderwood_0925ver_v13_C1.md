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
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
from tqdm import tqdm




## === cell 1
def _resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if p.startswith("../input/"):
        alt = "/kaggle/input/" + p[len("../input/") :]
        if os.path.exists(alt):
            return alt
    return p




## === cell 2
INPUT_PATH = "../input/modelparam1003"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

OUT_FEATURES = 5
NUM_CLASSES = 5

NUM_EPOCHS = 20
BATCH_SIZE = 16
IMAGE_SIZE = 224
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 3

INPUT_PATH = _resolve_path(INPUT_PATH)
TRAIN_CSV_PATH = _resolve_path(TRAIN_CSV_PATH)
TRAIN_IMAGE_PATH = _resolve_path(TRAIN_IMAGE_PATH)
TEST_IMAGE_PATH = _resolve_path(TEST_IMAGE_PATH)




## === cell 3
def softmax_focal_cross_entropy(
    logits, targets, alpha=1.0, gamma=2.0, label_smoothing=0.1
):
    """
    logits: (N, C)
    targets: (N,) int64 class indices OR (N, C) one-hot/soft targets.
    returns: (N,) per-sample loss
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N,C), got {tuple(logits.shape)}")

    n, c = logits.shape

    if isinstance(targets, (list, tuple, np.ndarray)):
        targets = torch.tensor(targets)

    if isinstance(targets, torch.Tensor) and targets.ndim == 2:
        targets = targets.argmax(dim=1)

    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    targets = targets.to(device=logits.device)
    if targets.dtype != torch.long:
        targets = targets.long()

    log_probs = F.log_softmax(logits, dim=1)  # (N,C)
    probs = log_probs.exp()  # (N,C)

    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(label_smoothing / (c - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)

    ce = -(true_dist * log_probs).sum(dim=1)  # (N,)

    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_(1e-8, 1.0)  # (N,)
    focal_factor = (1.0 - p_t).pow(gamma)

    return alpha * focal_factor * ce




## === cell 4
def to_onehot(labels, num_classes):
    labels = torch.as_tensor(labels, dtype=torch.long)
    return F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)




## === cell 5
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




## === cell 6
def _rrc(h, w, p=1.0):
    return A.RandomResizedCrop(
        size=(h, w), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333), p=p
    )




## === cell 7
train_augs = A.Compose(
    [
        _rrc(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
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



## === cell 8
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                _rrc(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
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



## === cell 9
if len(DEVICES) == 0:
    DEVICES = [torch.device("cpu")]




## === cell 10
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


seed_everything(SEED)



## === cell 11
from torchvision.io import read_image, ImageReadMode


def _load_image_rgb(path):
    img = read_image(path, mode=ImageReadMode.RGB)  # uint8, (C,H,W)
    return img.permute(1, 2, 0).contiguous().numpy()  # (H,W,C) uint8




## === cell 12
class MyCassavaLeafDataset(Dataset):
    @staticmethod
    def generate_index(num_total, ratio):
        k = int(ratio * 10)
        if k <= 0:
            k = 1
        valid_index = np.arange(0, num_total, k)
        mask = np.ones(num_total, dtype=bool)
        mask[valid_index] = False
        train_index = np.nonzero(mask)[0]
        return train_index, valid_index

    def __init__(
        self,
        csv_path=None,
        images_path=None,
        transform=None,
        mode="train",
        train_ratio=0.5,
        df=None,
    ):
        super().__init__()
        self.transform = transform
        self.mode = mode
        self.images_path = images_path

        self.data_info = df if df is not None else pd.read_csv(csv_path)
        self.data_len = self.data_info.shape[0]

        if self.mode == "train":
            train_index, _ = MyCassavaLeafDataset.generate_index(
                self.data_len, train_ratio
            )
            self.image_arr = np.asarray(self.data_info.iloc[train_index, 0])
            self.label_arr = np.asarray(self.data_info.iloc[train_index, 1])
            self.real_len = len(self.image_arr)
        elif self.mode == "valid":
            _, valid_index = MyCassavaLeafDataset.generate_index(
                self.data_len, train_ratio
            )
            self.image_arr = np.asarray(self.data_info.iloc[valid_index, 0])
            self.label_arr = np.asarray(self.data_info.iloc[valid_index, 1])
            self.real_len = len(self.image_arr)
        else:
            self.image_arr = np.asarray(self.data_info.iloc[:, 0])
            self.label_arr = None
            self.real_len = len(self.image_arr)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        image = _load_image_rgb(os.path.join(self.images_path, single_image_name))
        if self.mode != "test":
            label = int(self.label_arr[index])
            return self.transform(image=image)["image"], label
        return self.transform(image=image)["image"], single_image_name

    def __len__(self):
        return self.real_len




## === cell 13
train_df_full = pd.read_csv(TRAIN_CSV_PATH)
labels = train_df_full["label"].values.astype(int)
idx = np.arange(len(train_df_full))

rng = np.random.RandomState(SEED)
train_idx = []
valid_idx = []

valid_frac = 0.1
for cls in np.unique(labels):
    cls_idx = idx[labels == cls]
    rng.shuffle(cls_idx)
    n_valid = max(1, int(round(valid_frac * len(cls_idx))))
    valid_idx.append(cls_idx[:n_valid])
    train_idx.append(cls_idx[n_valid:])

train_idx = np.concatenate(train_idx)
valid_idx = np.concatenate(valid_idx)
rng.shuffle(train_idx)
rng.shuffle(valid_idx)

train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
valid_df = train_df_full.iloc[valid_idx].reset_index(drop=True)

_num_workers = min(8, (os.cpu_count() or 2))
pin = torch.cuda.is_available()


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

train_set = MyCassavaLeafDataset(
    df=train_df,
    images_path=TRAIN_IMAGE_PATH,
    transform=train_augs,
    mode="train",
)
my_train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=8 if _num_workers > 0 else None,
    worker_init_fn=_seed_worker if _num_workers > 0 else None,
    generator=g,
    drop_last=False,
)
valid_set = MyCassavaLeafDataset(
    df=valid_df,
    images_path=TRAIN_IMAGE_PATH,
    transform=valid_augs,
    mode="valid",
)
my_valid_dataloader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=8 if _num_workers > 0 else None,
    worker_init_fn=_seed_worker if _num_workers > 0 else None,
    generator=g,
    drop_last=False,
)



## === cell 14
my_model = torchvision.models.efficientnet_b4(weights=None)
my_model.classifier[-1] = nn.Linear(my_model.classifier[-1].in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model.classifier[-1].weight)



## === cell 15
if torch.cuda.is_available():
    my_model = my_model.to(memory_format=torch.channels_last)
my_model = my_model.to(DEVICES[0])



## === cell 16
criterion = softmax_focal_cross_entropy




## === cell 17
def _maybe_compile(model, mode="max-autotune"):
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode=mode, fullgraph=False, dynamic=False)
    except Exception:
        return model


class MyTrainer:
    @staticmethod
    def accurate_count(y_hat, y_true):
        y_pred = y_hat.argmax(dim=1)
        correct = (y_pred == y_true).sum().item()
        return float(correct)

    @staticmethod
    def calc_valid_acc(model, valid_dataloader):
        model.eval()
        device = next(iter(model.parameters())).device
        test_num = 0
        test_acc_num = 0
        with torch.no_grad():
            for x, y_true in valid_dataloader:
                if isinstance(x, list):
                    x = [x_1.to(device, non_blocking=True) for x_1 in x]
                else:
                    x = x.to(device, non_blocking=True)
                    if device.type == "cuda":
                        x = x.to(memory_format=torch.channels_last)
                y_true = y_true.to(device, non_blocking=True).long()
                test_num += y_true.shape[0]
                test_acc_num += MyTrainer.accurate_count(model(x), y_true)
        return test_acc_num / test_num

    def __init__(
        self,
        optimizer,
        model,
        criterion,
        train_dataloader,
        valid_dataloader,
        param_group=True,
        learning_rate=lr_tune,
        num_epochs=NUM_EPOCHS,
        devices=DEVICES,
    ):
        self.optimizer_class = optimizer
        self.model = model
        self.criterion = criterion
        self.devices = devices
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.param_group = param_group
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.optimizer = None
        self._last_lr = None

        excluded = {
            "module.classifier.1.weight",
            "module.classifier.1.bias",
            "classifier.1.weight",
            "classifier.1.bias",
        }
        self._param_1x = [
            param
            for name, param in self.model.named_parameters()
            if name not in excluded
        ]
        self._classifier_params = (
            self.model.module.classifier.parameters()
            if hasattr(self.model, "module")
            else self.model.classifier.parameters()
        )

    def _ensure_optimizer(self, epoch):
        lr = float(self.learning_rate(epoch))
        if self.optimizer is None:
            self.optimizer = self.optimizer_class(
                [
                    {"params": self._param_1x, "lr": lr},
                    {"params": self._classifier_params, "lr": lr * 10},
                ],
                lr=lr,
                weight_decay=0.001,
            )
            self._last_lr = lr
        elif lr != self._last_lr:
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = lr * 10
            self._last_lr = lr

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        train_num = 0
        train_acc_num = 0
        batch_num = len(self.train_dataloader)

        self._ensure_optimizer(epoch)
        optimizer = self.optimizer
        device0 = self.devices[0]

        print(f"epoch{epoch + 1} begins:")

        tk0 = tqdm(
            enumerate(self.train_dataloader),
            total=batch_num,
            mininterval=2.0,
            leave=False,
        )
        for _, (x, y_true) in tk0:
            x = x.to(device0, non_blocking=True)
            if device0.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            y_true = y_true.to(device0, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y_true)  # (N,)
            loss_sum = loss.sum()
            loss_sum.backward()
            optimizer.step()

            total_loss += float(loss_sum.detach().item())
            train_num += y_true.shape[0]
            train_acc_num += MyTrainer.accurate_count(y_hat.detach(), y_true)

        return total_loss / train_num, train_acc_num / train_num

    def train(self):
        best_valid_acc = 0
        if len(self.devices) > 1 and self.devices[0].type == "cuda":
            self.model = nn.DataParallel(
                self.model, device_ids=list(range(len(self.devices)))
            ).to(self.devices[0])
        else:
            self.model = self.model.to(self.devices[0])

        self.model = _maybe_compile(self.model)

        for epoch in range(self.num_epochs):
            train_loss, train_acc = MyTrainer.train_epoch(self, epoch)
            valid_acc = MyTrainer.calc_valid_acc(self.model, self.valid_dataloader)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), os.path.join("best_model.pth"))
            print(
                f"epoch{epoch + 1}:train_loss:{train_loss}, train_acc:{train_acc}, valid_acc:{valid_acc}"
            )




## === cell 18
torch.cuda.empty_cache()




## === cell 19
def _load_checkpoint_into_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICES[0])

    has_module = any(k.startswith("module.") for k in ckpt.keys())
    model_is_module = any(k.startswith("module.") for k in model.state_dict().keys())

    state = ckpt
    if has_module and not model_is_module:
        state = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    elif (not has_module) and model_is_module:
        state = {"module." + k: v for k, v in ckpt.items()}

    for w_key, b_key in [
        ("classifier.1.weight", "classifier.1.bias"),
        ("module.classifier.1.weight", "module.classifier.1.bias"),
    ]:
        if w_key in state and b_key in state:
            w = state[w_key]
            b = state[b_key]
            if w.ndim == 2 and w.shape[0] != OUT_FEATURES:
                if w.shape[0] > OUT_FEATURES:
                    state[w_key] = w[:OUT_FEATURES].contiguous()
                    state[b_key] = b[:OUT_FEATURES].contiguous()

    try:
        model.load_state_dict(state, strict=True)
        return True
    except RuntimeError:
        try:
            model.load_state_dict(state, strict=False)
            return True
        except Exception:
            return False




## === cell 20
candidate_ckpts = [
    os.path.join(_resolve_path(INPUT_PATH), "best_model.pth"),
    os.path.join(INPUT_PATH, "best_model.pth"),
    os.path.join(".", "best_model.pth"),
]
ckpt_to_use = None
for p in candidate_ckpts:
    if os.path.exists(p):
        ckpt_to_use = p
        break



## === cell 21
if ckpt_to_use is None:
    trainer = MyTrainer(
        optimizer=OPTIMIZER,
        model=my_model,
        criterion=criterion,
        train_dataloader=my_train_dataloader,
        valid_dataloader=my_valid_dataloader,
        num_epochs=NUM_EPOCHS,
        devices=DEVICES,
    )
    trainer.train()
    if os.path.exists("best_model.pth"):
        ckpt_to_use = os.path.join(".", "best_model.pth")



## === cell 22
my_model = my_model.to(DEVICES[0])
my_model.eval()
if ckpt_to_use is not None:
    ok = _load_checkpoint_into_model(my_model, ckpt_to_use)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_to_use}")
else:
    print(
        "WARNING: No best_model.pth found. Proceeding with randomly initialized weights."
    )

my_model = _maybe_compile(my_model)

test_device = DEVICES[0]
sample_sub_path = _resolve_path(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.exists(sample_sub_path):
    sample_sub_path = _resolve_path(
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
sample_df = pd.read_csv(sample_sub_path)

image_ids = sample_df["image_id"].values
preds = np.empty(len(sample_df), dtype=np.int64)

infer_bs = 64 if test_device.type == "cuda" else 8


class _TestBaseDataset(Dataset):
    def __init__(self, image_ids, images_path):
        self.image_ids = image_ids
        self.images_path = images_path

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_path, image_id)
        image = _load_image_rgb(img_path)  # decode once
        return image, idx  # return index to avoid string->pos dict in the hot loop


def _tta_collate(batch, transform=test_augs, tta=TTA):
    images, indices = zip(*batch)  # list of HWC uint8 arrays, and ints
    bsz = len(images)
    tta_views = []
    for img in images:
        views = [transform(image=img)["image"] for _ in range(tta)]
        tta_views.append(torch.stack(views, dim=0))
    x_tta = torch.stack(tta_views, dim=0)  # (B, TTA, C, H, W)
    return x_tta, torch.tensor(indices, dtype=torch.int64)


_test_workers = min(8, (os.cpu_count() or 2))
test_ds = _TestBaseDataset(image_ids=image_ids, images_path=TEST_IMAGE_PATH)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=max(1, infer_bs // max(1, TTA)),
    shuffle=False,
    num_workers=_test_workers,
    pin_memory=pin,
    persistent_workers=(_test_workers > 0),
    prefetch_factor=8 if _test_workers > 0 else None,
    worker_init_fn=_seed_worker if _test_workers > 0 else None,
    generator=g,
    collate_fn=_tta_collate,
)

with torch.no_grad():
    for x_tta, batch_pos in tqdm(test_dl, total=len(test_dl), mininterval=2.0):
        bsz = x_tta.shape[0]
        x = x_tta.view(bsz * TTA, *x_tta.shape[2:]).to(test_device, non_blocking=True)
        if test_device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)

        logits = my_model(x)[:, :NUM_CLASSES]  # (B*TTA, C)
        logits = logits.view(bsz, TTA, NUM_CLASSES).sum(dim=1)  # (B,C)
        batch_pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

        preds[batch_pos.numpy()] = batch_pred

submission = sample_df.copy()
submission["label"] = preds
submission.to_csv(SUBMISSION_PATH, index=False)
print(
    f"Wrote submission to: {SUBMISSION_PATH} with shape {submission.shape} and columns {list(submission.columns)}"
)
