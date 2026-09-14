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

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



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

    candidates = []
    for ext in ("*.pth", "*.pt"):
        candidates.extend(glob.glob(f"/kaggle/input/**/{ext}", recursive=True))

    candidates = sorted(candidates, key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0] if candidates else None


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
            self.df = df
            self.img_dir = img_dir
            self.aug = aug

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            img = np.array(
                Image.open(os.path.join(self.img_dir, row.image_id)).convert("RGB")
            )
            if self.aug is not None:
                img = self.aug(image=img)["image"]
            img = to_tensor(img)
            y = int(row.label)
            return img, y

    train_loader = torch.utils.data.DataLoader(
        CassavaDataset(train_df, train_images_path, aug=train_aug),
        batch_size=config["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
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
        with Image.open(img_path) as im:
            img0 = np.array(im.convert("RGB"))
        return img0, img_idx


def collate_keep_numpy(batch):
    imgs, idxs = zip(*batch)
    return list(imgs), torch.as_tensor(idxs, dtype=torch.long)


num_workers = min(4, os.cpu_count() or 2)
test_ds = TTATestDatasetOnce(sample_sub["image_id"].values, test_images_path)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=32,  # each item is a full decoded image; keep batch moderate to limit CPU RAM pressure
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    collate_fn=collate_keep_numpy,
)

model.eval()
num_images = len(sample_sub)
logits_sum = torch.zeros((num_images, config["CLASSES"]), dtype=torch.float32)

with torch.no_grad():
    for imgs_np, img_idx in test_loader:
        bsz = len(imgs_np)

        for _ in range(int(tta_count)):
            xs = [to_tensor(sub_aug(image=im)["image"]) for im in imgs_np]
            x = torch.stack(xs, dim=0)
            x = x.to(device, non_blocking=True)
            out = model(x).detach().cpu()  # (B, C)
            logits_sum.index_add_(0, img_idx, out)

logits_avg = logits_sum / float(tta_count)
pred_labels = torch.argmax(logits_avg, dim=1).numpy().astype(np.int64)

sub_df = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": pred_labels})
out_path = config["DATA"]["SUB_OUTPUT"]
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.head())
print("Rows:", len(sub_df))
