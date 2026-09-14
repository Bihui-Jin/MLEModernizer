# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.8943789664551224

# 6. Current score

0.59567

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58371) has done: 'I fixed the Albumentations augmentation definitions which were causing validation errors and prevented `test_augs` from being created. The `RandomResizedCrop` now uses the correct `size` tuple argument, allowing the script to run end‑to‑end and produce a valid `submission.csv` file.'
- What this solution (achieved 0.57661) has done: 'The update skips the expensive training loop because pretrained checkpoints are already loaded, while keeping all model definitions, data loading, and inference unchanged. By surrounding the training code with a `if not SKIP_TRAIN:` guard, the script runs only the inference stage, staying within the 600‑second limit without altering model architecture or final prediction logic.'
- What this solution (achieved 0.59567) has done: 'The changes skip the costly fine‑tuning (the pretrained checkpoints are already loaded) and replace the heavyweight test‑time augmentation with a fast resize‑normalize pipeline. In inference we apply the augmentation once per image and reuse the tensor for both models, eliminating duplicate work while keeping the exact model architecture and prediction logic unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split




## === cell 1
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
DEVICE = DEVICES[0] if DEVICES else torch.device("cpu")

OUT_FEATURES = 5
NUM_EPOCHS = 17  # total schedule length (kept for LR scheduler)
TRAIN_EPOCHS = 5  # unchanged fine‑tuning schedule
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 1  # reduced to stay within time budget
SKIP_TRAIN = True  # skip training; use pretrained checkpoints directly




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    """Optional focal loss (not used in the main loop but kept for completeness)."""

    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat, dtype=torch.float32)

    y_true = smooth(y_true, smooth_factor)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p_t = y_true * y_hat + (1 - y_true) * (1 - y_hat)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




## === cell 3
def lr_tune(epoch, num_epochs=NUM_EPOCHS):
    """Cosine‑like learning‑rate scheduler with warm‑up."""
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
        decay_factor = (math.cos(decay_factor) + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr


x = list(range(NUM_EPOCHS))
y = [lr_tune(i) for i in x]
plt.plot(x, y)
plt.title("Learning‑rate schedule")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.show()




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
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

test_augs = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)




## === cell 5
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


seed_everything(SEED)




## === cell 6
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)




## === cell 7
my_model_1.eval()
try:
    ckpt = torch.load(os.path.join(INPUT_PATH, RESNEXT_PATH), map_location=DEVICE)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    my_model_1.load_state_dict(ckpt, strict=False)
except FileNotFoundError:
    pass
my_model_1 = nn.DataParallel(my_model_1).to(DEVICE)

my_model_2.eval()
try:
    ckpt = torch.load(os.path.join(INPUT_PATH, B4_PATH), map_location=DEVICE)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    my_model_2.load_state_dict(ckpt, strict=False)
except FileNotFoundError:
    pass
my_model_2 = nn.DataParallel(my_model_2).to(DEVICE)

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMAGE_PATH, x)
)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=SEED,
)


class CassavaDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = np.array(img)
        aug = self.transforms(image=img)
        tensor = aug["image"]
        label = int(row["label"])
        return tensor, label


train_dataset = CassavaDataset(train_split, train_augs)
val_dataset = CassavaDataset(val_split, valid_augs)

num_workers = min(8, os.cpu_count() or 1)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = OPTIMIZER(my_model_2.parameters(), lr=LR_START)

use_amp = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler() if use_amp else None

if not SKIP_TRAIN:
    for epoch in range(TRAIN_EPOCHS):
        lr = lr_tune(epoch, num_epochs=NUM_EPOCHS)
        for g in optimizer.param_groups:
            g["lr"] = lr

        my_model_2.train()
        running_loss = 0.0
        for imgs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS} [train]"
        ):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = my_model_2(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = my_model_2(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        my_model_2.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS} [val]"
            ):
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = my_model_2(imgs)
                else:
                    outputs = my_model_2(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(
            f"Epoch {epoch+1}/{TRAIN_EPOCHS} - LR {lr:.6f} - Train loss {avg_loss:.4f} - Val Acc {val_acc:.4f}"
        )




## === cell 8
torch.cuda.empty_cache()

test_image_list = sorted(
    [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith((".jpg", ".png"))]
)


class TestImageDataset(Dataset):
    def __init__(self, image_names, img_dir):
        self.image_names = image_names
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.img_dir, name)
        img_np = np.array(Image.open(path).convert("RGB"))
        return img_np, name


test_dataset = TestImageDataset(test_image_list, TEST_IMAGE_PATH)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda batch: batch,
)

my_model_1.eval()
my_model_2.eval()

preds_1, preds_2, ordered_names = [], [], []

use_amp = DEVICE.type == "cuda"

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        img_batch, name_batch = zip(*batch)  # unzip
        aug_tensors = [test_augs(image=img)["image"] for img in img_batch]
        batch_tensor = torch.stack(aug_tensors).to(DEVICE, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                out1 = my_model_1(batch_tensor)
                out2 = my_model_2(batch_tensor)
        else:
            out1 = my_model_1(batch_tensor)
            out2 = my_model_2(batch_tensor)

        preds_1.append(out1.cpu())
        preds_2.append(out2.cpu())
        ordered_names.extend(name_batch)

predictions_1 = torch.cat(preds_1, dim=0)
predictions_2 = torch.cat(preds_2, dim=0)

norm_pred_1 = F.normalize(predictions_1, p=2, dim=1)
norm_pred_2 = F.normalize(predictions_2, p=2, dim=1)

final_pred = norm_pred_1 * 0.4 + norm_pred_2 * 0.6
label_ids = final_pred.argmax(dim=1).numpy()

df_submission = pd.DataFrame({"image_id": ordered_names, "label": label_ids})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH} with {len(df_submission)} rows.")
