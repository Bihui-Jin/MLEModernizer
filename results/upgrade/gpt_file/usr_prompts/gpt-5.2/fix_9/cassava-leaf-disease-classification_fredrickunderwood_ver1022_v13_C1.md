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
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import timm




## === cell 1
def _find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


BASE_INPUT = _find_existing_path(
    [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
    ]
)



## === cell 2
INPUT_PATH = BASE_INPUT  # dataset root (images/csv)
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMAGE_PATH = os.path.join(BASE_INPUT, "train_images")
TEST_IMAGE_PATH = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

CKPT_SEARCH_ROOTS = [
    ".",  # current working dir
    "/kaggle/working",
    os.path.join("/kaggle/working", "cassava-leaf-disease-classification"),
    BASE_INPUT,  # dataset dir (if user attached as dataset with checkpoints)
    "/kaggle/input",
    "/kaggle/data",
]

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
TTA = 6

FALLBACK_TRAIN_EPOCHS = (
    2  # small but impactful; avoids heavy changes while improving score
)
FALLBACK_NUM_WORKERS = 2
FALLBACK_VAL_FRAC = 0.1

PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4



## === cell 3
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")




## === cell 4
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

    y_true = smooth(y_true, smooth_factor).to(y_hat.device).type_as(y_hat)

    p = torch.sigmoid(y_hat)
    ce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p_t = y_true * p + (1 - y_true) * (1 - p)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * ce, dim=-1)




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


x = [i for i in range(NUM_EPOCHS)]
y = [lr_tune(i) for i in x]
plt.plot(x, y)



## === cell 6
train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333333333333),
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



## === cell 7
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop(
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    scale=(0.9, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),
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




## === cell 8
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



## === cell 9
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True, num_classes=OUT_FEATURES)
my_model_1



## === cell 10
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True, num_classes=OUT_FEATURES)
my_model_2



## === cell 11
torch.cuda.empty_cache()




## === cell 12
def _load_checkpoint_if_exists(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        return False

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        return False

    def _strip_prefix(d, prefix):
        if any(k.startswith(prefix) for k in d.keys()):
            return {k.replace(prefix, "", 1): v for k, v in d.items()}
        return d

    state_a = _strip_prefix(state, "module.")
    state_a = _strip_prefix(state_a, "model.")
    state_b = state.copy()
    if not any(k.startswith("module.") for k in state_b.keys()):
        state_b = {("module." + k): v for k, v in state_b.items()}

    target = model.module if isinstance(model, nn.DataParallel) else model

    try:
        target.load_state_dict(state_a, strict=True)
        return True
    except Exception:
        pass

    try:
        target.load_state_dict(state_a, strict=False)
        return True
    except Exception:
        pass

    try:
        model.load_state_dict(state_b, strict=False)
        return True
    except Exception:
        return False


def _find_ckpt_file(filename, roots):
    checked = []
    for r in roots:
        cand = os.path.join(r, filename)
        checked.append(cand)
        if os.path.exists(cand):
            return cand

    max_depth = 4
    for r in roots:
        if not os.path.isdir(r):
            continue
        r = os.path.abspath(r)
        for dirpath, dirnames, filenames in os.walk(r):
            rel = os.path.relpath(dirpath, r)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > max_depth:
                dirnames[:] = []
                continue
            if filename in filenames:
                return os.path.join(dirpath, filename)

    return checked[0] if checked else os.path.join(roots[0], filename)




## === cell 13
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_list = sample_sub["image_id"].astype(str).tolist()

if len(test_image_list) == 0:
    test_image_list = sorted(
        [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")]
    )




## === cell 14
class CassavaDataset(Dataset):
    def __init__(self, df, image_root, augs, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.augs = augs
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        img_path = os.path.join(self.image_root, image_id)

        with Image.open(img_path) as im:
            img = im.convert("RGB")
            img_np = np.asarray(img)

        x = self.augs(image=img_np)["image"]  # torch.Tensor CHW float32
        if self.has_labels:
            y = int(row["label"])
            return x, y
        return x, image_id


def _make_loader(df, image_root, augs, has_labels, shuffle, batch_size):
    num_workers = int(FALLBACK_NUM_WORKERS)
    use_workers = num_workers > 0
    return DataLoader(
        CassavaDataset(df, image_root=image_root, augs=augs, has_labels=has_labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(PERSISTENT_WORKERS and use_workers),
        prefetch_factor=(PREFETCH_FACTOR if use_workers else None),
    )




## === cell 15
def _train_if_needed(model, model_name, train_df, device, epochs):
    model.train()
    optimizer = OPTIMIZER(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        lr = lr_tune(min(ep, NUM_EPOCHS - 1), num_epochs=NUM_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = float(lr)

        loader = _make_loader(
            train_df,
            TRAIN_IMAGE_PATH,
            train_augs,
            has_labels=True,
            shuffle=True,
            batch_size=BATCH_SIZE,
        )

        running = 0.0
        n = 0
        for xb, yb in tqdm(
            loader,
            desc=f"Fallback training {model_name} ep {ep+1}/{epochs}",
            leave=False,
        ):
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu()) * xb.size(0)
            n += xb.size(0)

        print(f"[{model_name}] ep {ep+1}/{epochs} train_loss={running/max(1,n):.4f}")

    model.eval()
    return model




## === cell 16
@torch.no_grad()
def _predict_logits_loader(model, df_images, image_root, augs, device, tta, batch_size):
    model.eval()

    base_ds = CassavaDataset(
        df_images, image_root=image_root, augs=None, has_labels=False
    )

    def _collate_raw(batch):
        raise RuntimeError("unreachable")

    class _RawCassavaDataset(Dataset):
        def __init__(self, df, image_root):
            self.df = df.reset_index(drop=True)
            self.image_root = image_root

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_id = str(row["image_id"])
            img_path = os.path.join(self.image_root, image_id)
            with Image.open(img_path) as im:
                img = im.convert("RGB")
                img_np = np.asarray(img)
            return img_np, image_id

    raw_loader = DataLoader(
        _RawCassavaDataset(df_images, image_root=image_root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(FALLBACK_NUM_WORKERS),
        pin_memory=False,  # raw numpy arrays; pinning is done after tensor creation
        drop_last=False,
        persistent_workers=(PERSISTENT_WORKERS and int(FALLBACK_NUM_WORKERS) > 0),
        prefetch_factor=(PREFETCH_FACTOR if int(FALLBACK_NUM_WORKERS) > 0 else None),
    )

    all_logits = []
    all_ids = []

    for imgs_np, ids in raw_loader:
        if isinstance(imgs_np, torch.Tensor):
            imgs_list = [imgs_np[i].numpy() for i in range(imgs_np.shape[0])]
        else:
            imgs_list = list(imgs_np)

        bs = len(imgs_list)
        all_ids.extend(list(ids))

        logits_sum = None
        for _ in range(tta):
            xb_list = [
                augs(image=im)["image"] for im in imgs_list
            ]  # list of CHW tensors
            xb = torch.stack(xb_list, dim=0)
            if torch.cuda.is_available():
                xb = xb.pin_memory()
            xb = xb.to(device, non_blocking=True)

            logits = model(xb).detach().cpu()
            logits_sum = logits if logits_sum is None else (logits_sum + logits)

        all_logits.append(logits_sum / float(tta))

    logits_all = torch.cat(all_logits, dim=0)  # (N, C)
    return logits_all, all_ids




## === cell 17
ckpt1 = _find_ckpt_file(RESNEXT_PATH, CKPT_SEARCH_ROOTS)
ckpt2 = _find_ckpt_file(B4_PATH, CKPT_SEARCH_ROOTS)

print("Checkpoint candidates:")
print(" - resnext:", ckpt1, "exists:", os.path.exists(ckpt1))
print(" - b4ns  :", ckpt2, "exists:", os.path.exists(ckpt2))

my_model_1 = (
    nn.DataParallel(my_model_1).to(DEVICE)
    if torch.cuda.device_count() > 1
    else my_model_1.to(DEVICE)
)
my_model_2 = (
    nn.DataParallel(my_model_2).to(DEVICE)
    if torch.cuda.device_count() > 1
    else my_model_2.to(DEVICE)
)

has_ckpt1 = _load_checkpoint_if_exists(my_model_1, ckpt1, DEVICE)
has_ckpt2 = _load_checkpoint_if_exists(my_model_2, ckpt2, DEVICE)
print("Loaded checkpoints:", {"resnext": has_ckpt1, "b4ns": has_ckpt2})

train_df_full = pd.read_csv(TRAIN_CSV_PATH)
train_df_full["image_id"] = train_df_full["image_id"].astype(str)
train_df_full["label"] = train_df_full["label"].astype(int)

if (not has_ckpt1) or (not has_ckpt2):
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(train_df_full))
    rng.shuffle(idx)
    val_n = int(len(idx) * FALLBACK_VAL_FRAC)
    train_idx = idx[val_n:]
    train_df = train_df_full.iloc[train_idx].reset_index(drop=True)

    if not has_ckpt1:
        my_model_1 = _train_if_needed(
            my_model_1, "resnext50_32x4d", train_df, DEVICE, FALLBACK_TRAIN_EPOCHS
        )
    if not has_ckpt2:
        my_model_2 = _train_if_needed(
            my_model_2, "tf_efficientnet_b4_ns", train_df, DEVICE, FALLBACK_TRAIN_EPOCHS
        )



## === cell 18
test_df = pd.DataFrame({"image_id": test_image_list})

logits1, ids1 = _predict_logits_loader(
    my_model_1,
    test_df,
    TEST_IMAGE_PATH,
    test_augs,
    DEVICE,
    tta=1,
    batch_size=BATCH_SIZE,
)
logits2, ids2 = _predict_logits_loader(
    my_model_2,
    test_df,
    TEST_IMAGE_PATH,
    test_augs,
    DEVICE,
    tta=TTA,
    batch_size=BATCH_SIZE,
)

normalize_pred_1 = F.normalize(logits1.T, p=2, dim=0).T
normalize_pred_2 = F.normalize(logits2.T, p=2, dim=0).T
final_pred = (normalize_pred_1 * 0.43) + (normalize_pred_2 * 0.57)

labels = final_pred.argmax(dim=-1).numpy().astype(np.int64)



## === cell 19
df_submission = sample_sub.copy()
df_submission["label"] = labels
df_submission.to_csv(SUBMISSION_PATH, index=False)

print("Wrote:", SUBMISSION_PATH)
print(df_submission.head())
print("Row count:", len(df_submission), "Expected:", len(sample_sub))
print(
    "Unique predicted labels:", pd.Series(labels).value_counts().sort_index().to_dict()
)
