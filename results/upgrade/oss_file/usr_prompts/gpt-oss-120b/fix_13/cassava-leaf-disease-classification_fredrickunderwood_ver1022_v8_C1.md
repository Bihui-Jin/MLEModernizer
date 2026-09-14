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

0.894983378664249

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.58782) has done: 'I fixed the Albumentations API usage (providing size as a tuple), added safe device handling, made the model loading robust by using pretrained weights when the checkpoint files are missing, and streamlined the inference pipeline to always produce a valid `submission.csv` file. These changes resolve the runtime errors and ensure the script runs end‑to‑end while keeping the original modeling logic intact.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split



## === cell 1
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

if torch.cuda.is_available():
    DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
else:
    DEVICES = [torch.device("cpu")]
DEVICE = DEVICES[0]

OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32  # reduced to fit GPU memory
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 1  # reduced to 1 to eliminate heavy test‑time augmentation loops




## === cell 2
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
        decay_factor = (math.cos(decay_factor) + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr


x = list(range(NUM_EPOCHS))
y = [lr_tune(i) for i in x]
plt.plot(x, y)
plt.title("Learning rate schedule")
plt.close()  # ensure no blocking GUI



## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), p=1.0),
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
    ],
    p=1.0,
)



## === cell 5
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), p=1.0),
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # enable benchmark for speed


seed_everything(SEED)



## === cell 7
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




## === cell 8
def load_checkpoint(model, ckpt_path):
    """Attempt to load a checkpoint; if missing, keep the current (pretrained) weights."""
    full_path = os.path.join(INPUT_PATH, ckpt_path)
    if os.path.isfile(full_path):
        try:
            state = torch.load(full_path, map_location=DEVICE)
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state, strict=False)
            print(f"Loaded checkpoint from {full_path}")
        except Exception as e:
            print(f"Failed to load {full_path}: {e}")
    else:
        print(f"Checkpoint {full_path} not found – using pretrained weights.")
    return model


my_model_1 = load_checkpoint(my_model_1, RESNEXT_PATH)
my_model_2 = load_checkpoint(my_model_2, B4_PATH)

ckpt1_exists = os.path.isfile(os.path.join(INPUT_PATH, RESNEXT_PATH))
ckpt2_exists = os.path.isfile(os.path.join(INPUT_PATH, B4_PATH))
skip_training = True

if torch.cuda.device_count() > 1:
    my_model_1 = nn.DataParallel(my_model_1)
    my_model_2 = nn.DataParallel(my_model_2)

my_model_1 = my_model_1.to(DEVICE)
my_model_2 = my_model_2.to(DEVICE)

torch.cuda.empty_cache()



## === cell 9
NUM_WORKERS = min(16, os.cpu_count())


class CassavaDataset(Dataset):
    """Lazy‑loading dataset: store paths and read each image on demand."""

    def __init__(self, df, img_dir, augmentations):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.aug = augmentations
        self.paths = [
            os.path.join(self.img_dir, img_name) for img_name in self.df["image_id"]
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img_tensor = torchvision.io.read_image(img_path)  # C,H,W uint8
        img_np = img_tensor.permute(1, 2, 0).numpy()
        aug_image = self.aug(image=img_np)["image"]
        label = int(self.df.iloc[idx]["label"])
        return aug_image, label


train_df = pd.read_csv(TRAIN_CSV_PATH)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=SEED,
)

train_dataset = CassavaDataset(train_split, TRAIN_IMAGE_PATH, train_augs)
val_dataset = CassavaDataset(val_split, TRAIN_IMAGE_PATH, valid_augs)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

criterion = nn.CrossEntropyLoss()
optimizer1 = OPTIMIZER(my_model_1.parameters(), lr=LR_START)
optimizer2 = OPTIMIZER(my_model_2.parameters(), lr=LR_START)

scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

MAX_TRAIN_EPOCHS = 5  # unchanged – will be skipped due to `skip_training`

if not skip_training:
    for epoch in range(min(NUM_EPOCHS, MAX_TRAIN_EPOCHS)):
        my_model_1.train()
        my_model_2.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE)

            optimizer1.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                out1 = my_model_1(imgs)
                loss1 = criterion(out1, labels)
            if scaler:
                scaler.scale(loss1).backward()
                scaler.step(optimizer1)
                scaler.update()
            else:
                loss1.backward()
                optimizer1.step()

            optimizer2.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                out2 = my_model_2(imgs)
                loss2 = criterion(out2, labels)
            if scaler:
                scaler.scale(loss2).backward()
                scaler.step(optimizer2)
                scaler.update()
            else:
                loss2.backward()
                optimizer2.step()

        for g in optimizer1.param_groups:
            g["lr"] = lr_tune(epoch)
        for g in optimizer2.param_groups:
            g["lr"] = lr_tune(epoch)

        my_model_1.eval()
        my_model_2.eval()
        correct1 = correct2 = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    out1 = my_model_1(imgs)
                    out2 = my_model_2(imgs)
                pred1 = out1.argmax(dim=1)
                pred2 = out2.argmax(dim=1)
                correct1 += (pred1 == labels).sum().item()
                correct2 += (pred2 == labels).sum().item()
                total += labels.size(0)
        print(
            f"Epoch {epoch+1}/{MAX_TRAIN_EPOCHS} – Val Acc Model1: {correct1/total:.4f}, Model2: {correct2/total:.4f}"
        )
else:
    print("Both checkpoints found – skipping training and using loaded weights.")

my_model_1.eval()
my_model_2.eval()




## === cell 10
class TestDataset(Dataset):
    """Lazy‑loading test dataset; stores only file names and paths."""

    def __init__(self, img_dir, img_names):
        self.img_dir = img_dir
        self.img_names = img_names
        self.paths = [os.path.join(self.img_dir, name) for name in img_names]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img_tensor = torchvision.io.read_image(img_path)  # C,H,W uint8
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_name = self.img_names[idx]
        return img_np, img_name


test_image_list = sorted(
    [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")]
)

test_dataset = TestDataset(TEST_IMAGE_PATH, test_image_list)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,  # reduced batch size for memory safety
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

preds_1 = []
preds_2 = []
ordered_names = []

amp_enabled = torch.cuda.is_available()  # use AMP during inference when possible

with torch.no_grad():
    for batch_np, batch_names in tqdm(test_loader, desc="Inference"):
        ordered_names.extend(batch_names)

        batch_aug = torch.stack(
            [test_augs(image=img_np)["image"] for img_np in batch_np]
        ).to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out1 = my_model_1(batch_aug)
            out2 = my_model_2(batch_aug)

        preds_1.append(out1.cpu())
        preds_2.append(out2.cpu())

predictions_1 = torch.cat(preds_1, dim=0)
predictions_2 = torch.cat(preds_2, dim=0)

norm_pred_1 = F.normalize(predictions_1, p=2, dim=1)
norm_pred_2 = F.normalize(predictions_2, p=2, dim=1)

final_pred = norm_pred_1 * 0.43 + norm_pred_2 * 0.57
label_ids = final_pred.argmax(dim=1).numpy()

submission_df = pd.DataFrame({"image_id": ordered_names, "label": label_ids})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1151281415.py in <cell line: 0>()
     45         # Apply the same augmentation once (TTA=1) and reuse for both models
     46         batch_aug = torch.stack(
---> 47             [test_augs(image=img_np)["image"] for img_np in batch_np]
     48         ).to(DEVICE, non_blocking=True)
     49 

/tmp/ipykernel_55/1151281415.py in <listcomp>(.0)
     45         # Apply the same augmentation once (TTA=1) and reuse for both models
     46         batch_aug = torch.stack(
---> 47             [test_augs(image=img_np)["image"] for img_np in batch_np]
     48         ).to(DEVICE, non_blocking=True)
     49 

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in __call__(self, force_apply, *args, **data)
    605             return data
    606 
--> 607         self.preprocess(data)
    608 
    609         for t in self.transforms:

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in preprocess(self, data)
    628                     continue
    629 
--> 630                 shape = self._get_data_shape(data_name, internal_name, data_value)
    631                 if shape is not None:
    632                     if internal_name in CHECKED_VOLUME | CHECKED_MASK3D:

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in _get_data_shape(self, data_name, internal_name, data)
    894         # Handle single images and masks
    895         if internal_name in CHECKED_SINGLE:
--> 896             return self._get_single_data_shape(data_name, data)
    897 
    898         # Handle volumes

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in _get_single_data_shape(self, data_name, data)
    913         """Get shape of single image or mask."""
    914         if not isinstance(data, np.ndarray):
--> 915             raise TypeError(f"{data_name} must be numpy array type")
    916         return data.shape
    917 

TypeError: image must be numpy array type
