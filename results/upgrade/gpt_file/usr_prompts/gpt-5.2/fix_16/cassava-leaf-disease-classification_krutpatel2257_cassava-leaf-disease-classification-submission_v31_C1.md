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
import os
import glob
import random
import math

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.io import read_image, ImageReadMode


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 1
config = {
    "DATA": {
        "IMAGES": "train_images",
        "LABELS": "train.csv",
        "SUB_IMAGES": "test_images",
        "SUB_LABELS": "sample_submission.csv",
        "SUB_OUTPUT": "submission.csv",
    },
    "DEVICE": "cuda",
    "NUM_GPU": torch.cuda.device_count(),
    "TRAIN_BATCH_SIZE": 32,
    "VAL_BATCH_SIZE": 16,
    "CLASSES": 5,
    "CV_FOLDS": 5,
    "NUM_EPOCHS": 15,
    "MODEL_PATH": "model.pth",
    "SGD": {"LR": 0.0005, "MOMENTUM": 0.9, "WEIGHT_DECAY": 0.001},
    "COS_ANN_LR": {"ETA_MIN": 0.00001},
    "MODEL_TYPE": "RESNET_50",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "GPUs:", config["NUM_GPU"])



## === cell 2
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_images_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"

model_path = "../input/rn-tta-calr-clahe-v2/model(19).pth"


def find_first_checkpoint():
    if isinstance(model_path, str) and os.path.exists(model_path):
        return model_path

    base = "/kaggle/input"
    patterns = (
        os.path.join(base, "*", "*.pth"),
        os.path.join(base, "*", "*.pt"),
        os.path.join(base, "*", "*", "*.pth"),
        os.path.join(base, "*", "*", "*.pt"),
        os.path.join(base, "*", "*", "*", "*.pth"),
        os.path.join(base, "*", "*", "*", "*.pt"),
    )

    best_path = None
    best_size = -1
    for pat in patterns:
        for p in glob.iglob(pat):
            try:
                st = os.stat(p)
            except OSError:
                continue
            if not os.path.isfile(p):
                continue
            if st.st_size > best_size:
                best_size = st.st_size
                best_path = p
    return best_path


resolved_ckpt_path = find_first_checkpoint()
print("Resolved checkpoint:", resolved_ckpt_path)



## === cell 3
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.2, 1.0)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

to_tensor = transforms.ToTensor()



## === cell 4
if config["MODEL_TYPE"] == "EFFICIENT_NET_B4":
    raise RuntimeError(
        "EFFICIENT_NET_B4 selected but efficientnet_pytorch is not available in this environment."
    )
elif config["MODEL_TYPE"] == "RESNET_50":
    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Linear(2048, config["CLASSES"])
else:
    raise ValueError(f"Unknown MODEL_TYPE: {config['MODEL_TYPE']}")

model.to(device)


def load_checkpoint_flexible(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = None

    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None:
            state_dict = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    cleaned = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return model


if resolved_ckpt_path is not None:
    model = load_checkpoint_flexible(model, resolved_ckpt_path, device)
    model.eval()
else:
    print(
        "No checkpoint found under /kaggle/input; training a model to ensure a valid submission is produced."
    )
    model.train()



## === cell 5
if resolved_ckpt_path is None:
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    train_images_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    df = pd.read_csv(train_csv_path)

    idx = np.arange(len(df))
    np.random.shuffle(idx)
    split = int(0.9 * len(df))
    train_idx, val_idx = idx[:split], idx[split:]

    train_df = df.iloc[train_idx].reset_index(drop=True)

    train_aug = A.Compose(
        [
            A.RandomResizedCrop(size=(512, 512), scale=(0.2, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    class CassavaDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, aug=None):
            self.image_ids = df["image_id"].values
            self.labels = df["label"].values.astype(np.int64, copy=False)
            self.img_dir = img_dir
            self.aug = aug

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, i):
            img = np.array(
                Image.open(os.path.join(self.img_dir, self.image_ids[i])).convert("RGB")
            )
            if self.aug is not None:
                img = self.aug(image=img)["image"]
            img = to_tensor(img)
            y = int(self.labels[i])
            return img, y

    g = torch.Generator()
    g.manual_seed(42)

    _cpu = os.cpu_count() or 2
    _train_workers = min(8, max(2, _cpu // 2))

    train_loader = torch.utils.data.DataLoader(
        CassavaDataset(train_df, train_images_path, aug=train_aug),
        batch_size=config["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=_train_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(_train_workers > 0),
        prefetch_factor=(4 if _train_workers > 0 else None),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["SGD"]["LR"],
        momentum=config["SGD"]["MOMENTUM"],
        weight_decay=config["SGD"]["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["NUM_EPOCHS"], eta_min=config["COS_ANN_LR"]["ETA_MIN"]
    )

    for epoch in range(config["NUM_EPOCHS"]):
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
            running_loss += loss.item() * x.size(0)
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{config['NUM_EPOCHS']} - loss: {running_loss/len(train_loader.dataset):.4f}"
        )

    model.eval()



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 5


class TTATestDatasetOnce(torch.utils.data.Dataset):
    def __init__(self, image_ids, img_dir):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, img_idx):
        image_id = self.image_ids[img_idx]
        img_path = os.path.join(self.img_dir, image_id)
        img_chw = read_image(img_path, mode=ImageReadMode.RGB)
        img_hwc = img_chw.permute(1, 2, 0).contiguous().numpy()
        return img_hwc, img_idx


def collate_keep_numpy(batch):
    imgs, idxs = zip(*batch)
    return list(imgs), torch.as_tensor(idxs, dtype=torch.long)


_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_STD_CPU = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
_MEAN_DEV = _MEAN_CPU.to(device=device, non_blocking=True)
_STD_DEV = _STD_CPU.to(device=device, non_blocking=True)


def _sample_rrc_params_vectorized(
    H: int,
    W: int,
    N: int,
    scale=(0.2, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
    device="cpu",
):
    area = float(H * W)
    log_ratio0 = math.log(ratio[0])
    log_ratio1 = math.log(ratio[1])

    num_attempts = 10
    target_area = area * torch.empty((num_attempts, N), device=device).uniform_(
        scale[0], scale[1]
    )
    aspect = torch.exp(
        torch.empty((num_attempts, N), device=device).uniform_(log_ratio0, log_ratio1)
    )
    w = torch.round(torch.sqrt(target_area * aspect)).to(torch.int64)
    h = torch.round(torch.sqrt(target_area / aspect)).to(torch.int64)

    valid = (w > 0) & (h > 0) & (w <= W) & (h <= H)
    has_any = valid.any(dim=0)
    first_idx = torch.argmax(valid.to(torch.int64), dim=0)

    idx = torch.arange(N, device=device)
    h_ch = h[first_idx, idx]
    w_ch = w[first_idx, idx]

    max_i = (H - h_ch + 1).clamp_min(1)
    max_j = (W - w_ch + 1).clamp_min(1)
    i_ch = torch.randint(0, int(H), (N,), device=device) % max_i
    j_ch = torch.randint(0, int(W), (N,), device=device) % max_j

    if (~has_any).any():
        in_ratio = float(W) / float(H)
        if in_ratio < ratio[0]:
            w_fb = W
            h_fb = int(round(w_fb / ratio[0]))
        elif in_ratio > ratio[1]:
            h_fb = H
            w_fb = int(round(h_fb * ratio[1]))
        else:
            w_fb = W
            h_fb = H
        i_fb = (H - h_fb) // 2
        j_fb = (W - w_fb) // 2

        m = ~has_any
        h_ch = torch.where(m, torch.full_like(h_ch, h_fb), h_ch)
        w_ch = torch.where(m, torch.full_like(w_ch, w_fb), w_ch)
        i_ch = torch.where(m, torch.full_like(i_ch, i_fb), i_ch)
        j_ch = torch.where(m, torch.full_like(j_ch, j_fb), j_ch)

    return (
        i_ch.to(torch.int64),
        j_ch.to(torch.int64),
        h_ch.to(torch.int64),
        w_ch.to(torch.int64),
    )


_BASE_GRID_CACHE = {}


def _get_base_grid(out_h: int, out_w: int, device, dtype):
    key = (out_h, out_w, str(device), dtype)
    base = _BASE_GRID_CACHE.get(key, None)
    if base is None or base.device != device or base.dtype != dtype:
        ys = torch.linspace(-1.0, 1.0, out_h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, out_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([xx, yy], dim=-1).contiguous()
        _BASE_GRID_CACHE[key] = base
    return base


def _pack_uint8_nchw_pinned_inplace(imgs_np, out_tensor_u8_cpu: torch.Tensor):
    for bi, im in enumerate(imgs_np):
        t = torch.from_numpy(im)  # H,W,3 uint8 (CPU)
        out_tensor_u8_cpu[bi].copy_(t.permute(2, 0, 1))
    return out_tensor_u8_cpu


def tta_batch_torch(imgs_np, tta_count: int, device, buffers: dict):
    B = len(imgs_np)
    H0, W0 = imgs_np[0].shape[0], imgs_np[0].shape[1]
    N = B * tta_count

    x_u8_cpu = buffers.get("x_u8_cpu", None)
    if (
        x_u8_cpu is None
        or x_u8_cpu.shape != (B, 3, H0, W0)
        or x_u8_cpu.dtype != torch.uint8
        or (device.type == "cuda" and not x_u8_cpu.is_pinned())
    ):
        x_u8_cpu = torch.empty(
            (B, 3, H0, W0),
            dtype=torch.uint8,
            pin_memory=(device.type == "cuda"),
        )
        buffers["x_u8_cpu"] = x_u8_cpu

    _pack_uint8_nchw_pinned_inplace(imgs_np, x_u8_cpu)
    x_u8 = x_u8_cpu.to(device=device, non_blocking=True)
    x_f = x_u8.to(dtype=torch.float32).mul_(1.0 / 255.0)

    out = buffers.get("out", None)
    if out is None or out.shape != (N, 3, 512, 512) or out.device != device:
        out = torch.empty((N, 3, 512, 512), dtype=torch.float32, device=device)
        buffers["out"] = out

    grid = buffers.get("grid", None)
    if grid is None or grid.shape != (N, 512, 512, 2) or grid.device != device:
        grid = torch.empty((N, 512, 512, 2), dtype=torch.float32, device=device)
        buffers["grid"] = grid

    base = _get_base_grid(512, 512, device=device, dtype=torch.float32)
    base_exp = base.unsqueeze(0).expand(N, -1, -1, -1)

    top, left, height, width = _sample_rrc_params_vectorized(
        H0, W0, N, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), device=device
    )

    top_f = top.to(dtype=torch.float32)
    left_f = left.to(dtype=torch.float32)
    height_f = height.to(dtype=torch.float32).clamp_min(1)
    width_f = width.to(dtype=torch.float32).clamp_min(1)

    a = (width_f - 1.0) / W0
    b = (height_f - 1.0) / H0
    tx = (2.0 * left_f + width_f - W0) / W0
    ty = (2.0 * top_f + height_f - H0) / H0

    grid[..., 0].copy_(base_exp[..., 0])
    grid[..., 1].copy_(base_exp[..., 1])
    grid[..., 0].mul_(a.view(N, 1, 1)).add_(tx.view(N, 1, 1))
    grid[..., 1].mul_(b.view(N, 1, 1)).add_(ty.view(N, 1, 1))

    rep_cache = buffers.get("rep_cache", None)
    if rep_cache is None:
        rep_cache = {}
        buffers["rep_cache"] = rep_cache
    rep_key = (B, tta_count, device.type, device.index if device.type == "cuda" else -1)
    rep_idx = rep_cache.get(rep_key, None)
    if rep_idx is None or rep_idx.device != device:
        rep_idx = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(
            tta_count
        )
        rep_cache[rep_key] = rep_idx
    x_rep = x_f.index_select(0, rep_idx)

    x1 = F.grid_sample(
        x_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    m_t = buffers.get("m_t", None)
    m_h = buffers.get("m_h", None)
    m_v = buffers.get("m_v", None)
    r_t = buffers.get("r_t", None)
    r_h = buffers.get("r_h", None)
    r_v = buffers.get("r_v", None)
    if m_t is None or m_t.shape != (N,) or m_t.device != device:
        m_t = torch.empty((N,), dtype=torch.bool, device=device)
        m_h = torch.empty((N,), dtype=torch.bool, device=device)
        m_v = torch.empty((N,), dtype=torch.bool, device=device)
        r_t = torch.empty((N,), dtype=torch.float32, device=device)
        r_h = torch.empty((N,), dtype=torch.float32, device=device)
        r_v = torch.empty((N,), dtype=torch.float32, device=device)
        buffers["m_t"], buffers["m_h"], buffers["m_v"] = m_t, m_h, m_v
        buffers["r_t"], buffers["r_h"], buffers["r_v"] = r_t, r_h, r_v

    r_t.uniform_()
    m_t.copy_(r_t < 0.5)
    if m_t.any():
        xt = x1.transpose(-1, -2)
        x1 = torch.where(m_t.view(N, 1, 1, 1), xt, x1)

    r_h.uniform_()
    m_h.copy_(r_h < 0.5)
    if m_h.any():
        xf = x1.flip(-1)
        x1 = torch.where(m_h.view(N, 1, 1, 1), xf, x1)

    r_v.uniform_()
    m_v.copy_(r_v < 0.5)
    if m_v.any():
        xf = x1.flip(-2)
        x1 = torch.where(m_v.view(N, 1, 1, 1), xf, x1)

    x1.sub_(_MEAN_DEV).div_(_STD_DEV)
    out.copy_(x1)
    return out


_cpu = os.cpu_count() or 2
num_workers = min(4, max(2, _cpu // 2))
test_ds = TTATestDatasetOnce(sample_sub["image_id"].values, test_images_path)

g = torch.Generator()
g.manual_seed(42)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=collate_keep_numpy,
    worker_init_fn=seed_worker,
    generator=g,
)

model.eval()
num_images = len(sample_sub)

logits_sum = torch.zeros(
    (num_images, config["CLASSES"]),
    dtype=torch.float32,
    device=(device if torch.cuda.is_available() else "cpu"),
)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = model.to(memory_format=torch.channels_last)

buffers = {}
with torch.inference_mode():
    for imgs_np, img_idx in test_loader:
        img_idx_dev = (
            img_idx.to(device=device, non_blocking=True)
            if logits_sum.device.type == "cuda"
            else img_idx
        )

        x_bt = tta_batch_torch(
            imgs_np, tta_count=tta_count, device=device, buffers=buffers
        )

        if x_bt.device.type == "cuda":
            x_bt = x_bt.to(memory_format=torch.channels_last)

        out_bt = model(x_bt)
        B = len(imgs_np)
        out = out_bt.view(tta_count, B, config["CLASSES"]).sum(dim=0)

        logits_sum.index_add_(0, img_idx_dev, out)

logits_avg = logits_sum / float(tta_count)
pred_labels = (
    torch.argmax(logits_avg, dim=1).detach().to("cpu").numpy().astype(np.int64)
)

sub_df = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": pred_labels})
out_path = config["DATA"]["SUB_OUTPUT"]
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.head())
print("Rows:", len(sub_df))
