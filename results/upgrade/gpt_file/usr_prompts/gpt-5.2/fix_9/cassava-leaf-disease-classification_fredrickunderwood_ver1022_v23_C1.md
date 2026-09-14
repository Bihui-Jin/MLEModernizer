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

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from torchvision.io import read_image




## === cell 1
def _first_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if len(state_dict) > 0 and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj




## === cell 2
INPUT_PATH = _first_existing_path(
    [
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
)
if INPUT_PATH is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset folder."
    )

TRAIN_CSV_PATH = os.path.join(INPUT_PATH, "train.csv")
TRAIN_IMAGE_PATH = os.path.join(INPUT_PATH, "train_images")
TEST_IMAGE_PATH = os.path.join(INPUT_PATH, "test_images")
SAMPLE_SUB_PATH = os.path.join(INPUT_PATH, "sample_submission.csv")

SUBMISSION_PATH = "submission.csv"

RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
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



## === cell 3
if torch.cuda.is_available() and len(DEVICES) > 0:
    DEVICE0 = DEVICES[0]
else:
    DEVICE0 = torch.device("cpu")




## === cell 4
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor) + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor).to(dtype=y_hat.dtype, device=y_hat.device)

    bce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1 - y_true) * (1 - p)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * bce, dim=-1)




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
        decay_factor = (math.cos(decay_factor) + 1.0) / 2.0
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return float(lr)




## === cell 6
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




## === cell 7
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

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


seed_everything(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True



## === cell 8
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=False)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)



## === cell 9
torch.cuda.empty_cache()


def _find_checkpoint(filename):
    direct = os.path.join(INPUT_PATH, filename)
    if os.path.exists(direct):
        return direct

    common = [
        os.path.join("/kaggle/input", filename),
        os.path.join("../input", filename),
    ]
    for p in common:
        if os.path.exists(p):
            return p

    return None


RESNEXT_CKPT = _find_checkpoint(RESNEXT_PATH)
B4_CKPT = _find_checkpoint(B4_PATH)

print(f"External checkpoints: resnext={RESNEXT_CKPT}, b4={B4_CKPT}")




## === cell 10
class CassavaDataset(Dataset):
    def __init__(self, df, image_root, augs, return_label=True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.augs = augs
        self.return_label = return_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        img_path = os.path.join(self.image_root, image_id)

        img = read_image(img_path)  # uint8, CHW, RGB
        image = img.permute(1, 2, 0).numpy()

        x = self.augs(image=image)["image"]
        if self.return_label:
            y = int(row["label"])
            return x, y
        return x, image_id


def _one_hot(labels, num_classes=OUT_FEATURES, device=None, dtype=torch.float32):
    y = torch.zeros((labels.shape[0], num_classes), device=device, dtype=dtype)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y




## === cell 11
def _train_one_model(model, train_loader, num_epochs=NUM_EPOCHS):
    model = model.to(DEVICE0)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = OPTIMIZER(model.parameters(), lr=LR_START)
    for epoch in range(num_epochs):
        model.train()
        lr = lr_tune(epoch, num_epochs=num_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        running = 0.0
        n = 0
        for xb, yb in tqdm(
            train_loader, desc=f"Train epoch {epoch+1}/{num_epochs}", leave=False
        ):
            xb = xb.to(DEVICE0, non_blocking=True).float()
            yb = yb.to(DEVICE0, non_blocking=True).long()
            y_oh = _one_hot(
                yb, num_classes=OUT_FEATURES, device=DEVICE0, dtype=xb.dtype
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = sigmoid_focal_cross_entropy(logits, y_oh).mean()
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)

        print(
            f"epoch {epoch+1}/{num_epochs} - lr={lr:.6g} - loss={running/max(n,1):.4f}"
        )

    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()




## === cell 12
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_ds = CassavaDataset(train_df, TRAIN_IMAGE_PATH, train_augs, return_label=True)

_num_workers = min(8, os.cpu_count() or 1)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

if RESNEXT_CKPT is None:
    print(
        "RESNEXT checkpoint missing; training resnext50_32x4d and saving to working directory."
    )
    sd1 = _train_one_model(my_model_1, train_loader, num_epochs=NUM_EPOCHS)
    RESNEXT_CKPT = os.path.join("/kaggle/working", "trained_resnext50_32x4d.pth")
    torch.save({"state_dict": sd1}, RESNEXT_CKPT)
    torch.cuda.empty_cache()

if B4_CKPT is None:
    print(
        "B4 checkpoint missing; training tf_efficientnet_b4_ns and saving to working directory."
    )
    sd2 = _train_one_model(my_model_2, train_loader, num_epochs=NUM_EPOCHS)
    B4_CKPT = os.path.join("/kaggle/working", "trained_tf_efficientnet_b4_ns.pth")
    torch.save({"state_dict": sd2}, B4_CKPT)
    torch.cuda.empty_cache()

print(f"Using checkpoints: resnext={RESNEXT_CKPT}, b4={B4_CKPT}")




## === cell 13
class TestImageDataset(Dataset):
    """
    CHANGE (runtime): perform test_augs inside Dataset so DataLoader workers handle CPU aug/normalize/tensor conversion.
    Preserves identical Albumentations pipeline/semantics, but removes per-sample work from the main process.
    """

    def __init__(self, image_ids, image_root, augs):
        self.image_ids = list(image_ids)
        self.image_root = image_root
        self.augs = augs

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_root, image_id)
        img = read_image(img_path)  # uint8 CHW RGB
        img_np = img.permute(1, 2, 0).numpy()  # HWC uint8
        x = self.augs(image=img_np)["image"]  # CHW float tensor (from ToTensorV2)
        return x, image_id


def _apply_tta_single(x, do_t, do_h, do_v):
    if do_t:
        x = x.transpose(-1, -2)
    if do_h:
        x = torch.flip(x, dims=[-1])
    if do_v:
        x = torch.flip(x, dims=[-2])
    return x


def _infer_model_no_tta(model, test_loader):
    preds = []
    model.eval()
    with torch.inference_mode():
        for xb, _image_ids in tqdm(test_loader, desc="Infer resnext50", leave=False):
            xb = xb.to(DEVICE0, non_blocking=True).float()
            if xb.is_cuda:
                xb = xb.contiguous(memory_format=torch.channels_last)
            logits = model(xb)
            preds.append(logits.detach().cpu())
    return torch.cat(preds, dim=0)


def _infer_model_tta(model, test_loader, tta=TTA):
    preds = []
    model.eval()
    with torch.inference_mode():
        for xb0, _image_ids in tqdm(
            test_loader, desc="Infer efficientnet_b4_ns", leave=False
        ):
            xb0 = xb0.to(DEVICE0, non_blocking=True).float()
            if xb0.is_cuda:
                xb0 = xb0.contiguous(memory_format=torch.channels_last)

            bsz = xb0.size(0)

            r = torch.rand((tta, bsz, 3), device=DEVICE0)
            t_mask = r[:, :, 0] < 0.5
            h_mask = r[:, :, 1] < 0.5
            v_mask = r[:, :, 2] < 0.5

            acc = None
            for k in range(tta):
                out_k = torch.empty(
                    (bsz, OUT_FEATURES), device=DEVICE0, dtype=xb0.dtype
                )
                tm = t_mask[k]
                hm = h_mask[k]
                vm = v_mask[k]

                for dt in (False, True):
                    for dh in (False, True):
                        for dv in (False, True):
                            m = (tm == dt) & (hm == dh) & (vm == dv)
                            if not torch.any(m):
                                continue
                            xb = _apply_tta_single(xb0[m], dt, dh, dv)
                            logits = model(xb)
                            out_k[m] = logits

                acc = out_k if acc is None else (acc + out_k)

            acc = acc / tta
            preds.append(acc.detach().cpu())
    return torch.cat(preds, dim=0)


test_image_list = np.array(
    sorted([fn for fn in os.listdir(TEST_IMAGE_PATH) if fn.lower().endswith(".jpg")])
)

test_ds = TestImageDataset(test_image_list, TEST_IMAGE_PATH, test_augs)

_num_workers_test = min(8, os.cpu_count() or 1)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=_num_workers_test,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=(_num_workers_test > 0),
    prefetch_factor=4 if _num_workers_test > 0 else None,
)


def _maybe_compile_for_infer(m):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(m, mode="reduce-overhead", fullgraph=False)
        except Exception:
            return m
    return m


model_param = torch.load(RESNEXT_CKPT, map_location="cpu")
model_param = _unwrap_state_dict(model_param)
my_model_1.load_state_dict(_strip_module_prefix(model_param), strict=True)
my_model_1 = my_model_1.to(DEVICE0)
if torch.cuda.device_count() > 1:
    my_model_1 = nn.DataParallel(my_model_1)
my_model_1 = _maybe_compile_for_infer(my_model_1)

predictions_1 = _infer_model_no_tta(my_model_1, test_loader)
normalize_pred_1 = F.normalize(predictions_1, p=2, dim=1)
torch.cuda.empty_cache()

model_param = torch.load(B4_CKPT, map_location="cpu")
model_param = _unwrap_state_dict(model_param)
my_model_2.load_state_dict(_strip_module_prefix(model_param), strict=True)
my_model_2 = my_model_2.to(DEVICE0)
if torch.cuda.device_count() > 1:
    my_model_2 = nn.DataParallel(my_model_2)
my_model_2 = _maybe_compile_for_infer(my_model_2)

predictions_2 = _infer_model_tta(my_model_2, test_loader, tta=TTA)
normalize_pred_2 = F.normalize(predictions_2, p=2, dim=1)

final_pred = (normalize_pred_1 * 0.44) + (normalize_pred_2 * 0.56)
label = final_pred.argmax(dim=-1).numpy().astype(int).tolist()

sub = pd.read_csv(SAMPLE_SUB_PATH)
sub = sub.sort_values("image_id").reset_index(drop=True)

pred_df = (
    pd.DataFrame({"image_id": test_image_list, "label": label})
    .sort_values("image_id")
    .reset_index(drop=True)
)
sub = sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if sub["label"].isna().any():
    missing = sub[sub["label"].isna()]["image_id"].tolist()[:5]
    raise RuntimeError(f"Some test images missing predictions (example ids: {missing})")

sub["label"] = sub["label"].astype(int)
sub.to_csv(SUBMISSION_PATH, index=False)

print(f"Wrote {SUBMISSION_PATH} with shape {sub.shape} and columns {list(sub.columns)}")
print(sub.head())
