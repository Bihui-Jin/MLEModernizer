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

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm import tqdm
import timm



## === cell 1
INPUT_PATH = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_CSV_PATH = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
SAMPLE_SUB_PATH = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32

IMAGE_SIZE_RESNEXT = 224
IMAGE_SIZE_EFFB4 = 384

OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5

TTA = 5

device = DEVICES[0] if len(DEVICES) else torch.device("cpu")




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p_t = y_true * y_hat + (1 - y_true) * (1 - y_hat)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




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




## === cell 4
IMAGE_SIZE = 512

train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.3333), p=1.0
        ),
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
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 64),
            hole_width_range=(8, 64),
            p=0.5,
        ),
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

valid_augs = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)



## === cell 5
test_augs_base = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
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
my_model_1 = timm.create_model(model_name1, pretrained=True, num_classes=OUT_FEATURES)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True, num_classes=OUT_FEATURES)



## === cell 8
torch.cuda.empty_cache()


def _find_weight_file(fname: str):
    candidates = [
        os.path.join(INPUT_PATH, fname),
        os.path.join(INPUT_PATH, "weights", fname),
        os.path.join(INPUT_PATH, "models", fname),
        os.path.join("/kaggle/working", fname),
        os.path.join("/kaggle/input", fname),
        os.path.join("/kaggle/input/cassava-leaf-disease-classification", fname),
        os.path.join(
            "/kaggle/input/cassava-leaf-disease-classification", "weights", fname
        ),
        os.path.join(
            "/kaggle/input/cassava-leaf-disease-classification", "models", fname
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    for root, _, files in os.walk("/kaggle/input"):
        if fname in files:
            return os.path.join(root, fname)

    return None


def _try_load_weights(model, weights_path):
    """
    If weights exist, load them; otherwise keep current model weights.
    With pretrained=True above, "current weights" are strong ImageNet weights instead of random.
    """
    if not weights_path or (not os.path.exists(weights_path)):
        print(
            f"[WARN] Weights not found for: {weights_path}. Using current (pretrained) weights."
        )
        return model

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(
                f"[WARN] Missing keys while loading {weights_path}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            print(
                f"[WARN] Unexpected keys while loading {weights_path}: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
    else:
        print(
            f"[WARN] Unsupported checkpoint format in {weights_path}. Using current weights."
        )
    return model


class CassavaTestDataset(Dataset):
    def __init__(self, image_dir, image_files, aug):
        self.image_dir = image_dir
        self.image_files = image_files
        self.aug = aug

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        x = self.aug(image=np.array(img))["image"]
        return fname, x


def _make_test_aug(image_size: int, flip_mode: str):
    def _flip_aug(mode_):
        if mode_ == "none":
            return A.NoOp(p=1.0)
        if mode_ == "h":
            return A.HorizontalFlip(p=1.0)
        if mode_ == "v":
            return A.VerticalFlip(p=1.0)
        if mode_ == "hv":
            return A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])
        return A.NoOp(p=1.0)

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.CenterCrop(height=image_size, width=image_size),
            _flip_aug(flip_mode),
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


@torch.no_grad()
def predict_model_batched(
    model, test_files, image_size: int, tta=1, batch_size=32, num_workers=2
):
    model.eval()

    flip_modes = ["none", "h", "v", "hv"]
    n_modes = max(1, min(int(tta), len(flip_modes)))
    use_modes = flip_modes[:n_modes]

    preds_sum = None

    for mode in use_modes:
        aug = _make_test_aug(image_size=image_size, flip_mode=mode)
        ds = CassavaTestDataset(TEST_IMAGE_PATH, test_files, aug)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        mode_logits = []
        for _, x in tqdm(
            dl, desc=f"Predict (tta={mode}, sz={image_size})", leave=False
        ):
            x = x.to(device, non_blocking=True)
            out = model(x)
            mode_logits.append(out.detach().cpu())
        mode_logits = torch.cat(mode_logits, dim=0)

        if preds_sum is None:
            preds_sum = mode_logits
        else:
            preds_sum += mode_logits

    preds = preds_sum / float(len(use_modes))
    return preds




## === cell 9
class CassavaTrainDataset(Dataset):
    def __init__(self, df, image_dir, aug, with_label=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.aug = aug
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["image_id"]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        x = self.aug(image=np.array(img))["image"]
        if self.with_label:
            y = int(row["label"])
            return x, y
        return fname, x


def _make_train_aug(image_size: int):
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333),
                p=1.0,
            ),
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
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 64),
                hole_width_range=(8, 64),
                p=0.5,
            ),
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


def _make_valid_aug(image_size: int):
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.CenterCrop(height=image_size, width=image_size),
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


def _train_one_model(model, df_train, df_val, image_size: int, save_path: str):
    train_aug = _make_train_aug(image_size)
    val_aug = _make_valid_aug(image_size)

    ds_tr = CassavaTrainDataset(df_train, TRAIN_IMAGE_PATH, train_aug, with_label=True)
    ds_va = CassavaTrainDataset(df_val, TRAIN_IMAGE_PATH, val_aug, with_label=True)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = model.to(device)
    opt = OPTIMIZER(model.parameters(), lr=LR_START)
    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(NUM_EPOCHS):
        lr = float(lr_tune(epoch, NUM_EPOCHS))
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        tr_loss = 0.0
        tr_n = 0
        for x, y in tqdm(
            dl_tr, desc=f"Train ep{epoch+1}/{NUM_EPOCHS} sz={image_size}", leave=False
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach().cpu()) * x.size(0)
            tr_n += x.size(0)

        model.eval()
        va_correct = 0
        va_total = 0
        with torch.no_grad():
            for x, y in tqdm(
                dl_va,
                desc=f"Valid ep{epoch+1}/{NUM_EPOCHS} sz={image_size}",
                leave=False,
            ):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x)
                pred = out.argmax(dim=1)
                va_correct += int((pred == y).sum().item())
                va_total += int(y.numel())

        va_acc = va_correct / max(1, va_total)
        tr_loss = tr_loss / max(1, tr_n)
        print(
            f"[INFO] ep {epoch+1:02d}/{NUM_EPOCHS} sz={image_size} lr={lr:.2e} train_loss={tr_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best weights to {save_path} (val_acc={best_acc:.4f})")

    model.load_state_dict(torch.load(save_path, map_location="cpu"), strict=True)
    return model


def _need_train(resnext_weights_found, b4_weights_found):
    return (resnext_weights_found is None) or (b4_weights_found is None)




## === cell 10
test_image_list = sorted(
    [fn for fn in os.listdir(TEST_IMAGE_PATH) if fn.lower().endswith(".jpg")]
)

resnext_weights = _find_weight_file(RESNEXT_PATH)
b4_weights = _find_weight_file(B4_PATH)
print(f"[INFO] resnext weights: {resnext_weights}")
print(f"[INFO] b4 weights: {b4_weights}")

if _need_train(resnext_weights, b4_weights):
    print(
        "[WARN] One or more checkpoints missing. Starting in-notebook finetune to obtain cassava-specific weights."
    )
    df = pd.read_csv(TRAIN_CSV_PATH)

    rng = np.random.RandomState(SEED)
    idxs = np.arange(len(df))
    labels = df["label"].values
    val_frac = 0.1
    val_indices = []
    for c in range(OUT_FEATURES):
        c_idxs = idxs[labels == c]
        rng.shuffle(c_idxs)
        n_val = max(1, int(len(c_idxs) * val_frac))
        val_indices.append(c_idxs[:n_val])
    val_indices = np.concatenate(val_indices)
    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[val_indices] = True
    df_train = df.loc[~val_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)
    print(f"[INFO] Split train={len(df_train)} val={len(df_val)}")

    resnext_local = os.path.join("/kaggle/working", "finetuned_resnext50_32x4d.pth")
    b4_local = os.path.join("/kaggle/working", "finetuned_tf_efficientnet_b4_ns.pth")

    my_model_1 = _train_one_model(
        my_model_1, df_train, df_val, IMAGE_SIZE_RESNEXT, resnext_local
    )
    my_model_2 = _train_one_model(
        my_model_2, df_train, df_val, IMAGE_SIZE_EFFB4, b4_local
    )

    resnext_weights = resnext_local
    b4_weights = b4_local
else:
    my_model_1 = _try_load_weights(my_model_1, resnext_weights)
    my_model_2 = _try_load_weights(my_model_2, b4_weights)

my_model_1 = my_model_1.to(device)
my_model_2 = my_model_2.to(device)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    my_model_1 = nn.DataParallel(my_model_1).to(device)
    my_model_2 = nn.DataParallel(my_model_2).to(device)

predictions_1 = predict_model_batched(
    my_model_1,
    test_image_list,
    image_size=IMAGE_SIZE_RESNEXT,
    tta=min(4, TTA),
    batch_size=BATCH_SIZE,
    num_workers=2,
)

torch.cuda.empty_cache()

predictions_2 = predict_model_batched(
    my_model_2,
    test_image_list,
    image_size=IMAGE_SIZE_EFFB4,
    tta=min(4, TTA),
    batch_size=BATCH_SIZE,
    num_workers=2,
)

final_pred = (predictions_1 * 0.45) + (predictions_2 * 0.55)
label = final_pred.argmax(dim=-1).cpu().numpy().astype(int)



## === cell 11
pred_map = pd.DataFrame({"image_id": test_image_list, "label": label})

if os.path.exists(SAMPLE_SUB_PATH):
    df_submission = pd.read_csv(SAMPLE_SUB_PATH)
    df_submission = df_submission.merge(
        pred_map, on="image_id", how="left", suffixes=("", "_pred")
    )
    df_submission["label"] = df_submission["label_pred"].fillna(0).astype(int)
    df_submission = df_submission[["image_id", "label"]]
else:
    df_submission = pred_map[["image_id", "label"]].copy()

df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Wrote submission to: {SUBMISSION_PATH} with shape {df_submission.shape}")
print(df_submission.head())
print(df_submission["label"].value_counts().sort_index())
