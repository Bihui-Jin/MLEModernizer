# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.14

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, math, warnings, random, collections

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from torch.cuda.amp import autocast, GradScaler  # <-- mixed precision

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # ensure reproducibility on GPU
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = True  # allow cuDNN to find fastest algorithm
torch.set_float32_matmul_precision("high")  # enable highest‑precision matmul speed




## === cell 1
class args:
    batch_size = 64  # larger batch fits in GPU memory with smaller images
    image_size = 256  # down‑sample from 384 to cut FLOPs
    epochs = 5  # keep original training length
    lr = 2.5e-5
    weight_decay = 0.01




## === cell 2
class MelanomaDataset(torch.utils.data.Dataset):
    """Dataset with optional in‑memory caching (cache_size=-1 loads everything)."""

    def __init__(self, image_paths, targets, augmentations, cache_size=0):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.cache = collections.OrderedDict()
        self.cache_size = cache_size

        if self.cache_size == -1:
            for p in self.image_paths:
                self.cache[p] = self._load_image(p)

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        """Load image, apply augmentations, and return a float32 numpy array."""
        img = cv2.imread(path)  # BGR uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augmentations:
            img = self.augmentations(image=img)["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        return img

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if path in self.cache:
            img = self.cache[path]
        else:
            img = self._load_image(path)
            if self.cache_size > 0:
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                self.cache[path] = img
        img_tensor = torch.from_numpy(img)  # shares memory, no extra copy
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        return img_tensor, target




## === cell 3
class MelanomaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, in_chans=3)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(self.backbone.get_classifier().in_features, 1)
        self.backbone.reset_classifier(0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.head(x)
        return x




## === cell 4
train_aug = A.Compose(
    [
        A.Resize(args.image_size, args.image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_aug = A.Compose(
    [
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 5
train_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
train_img_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train"
train_df["img_path"] = train_df["image_name"].apply(
    lambda x: f"{train_img_dir}/{x}.jpg"
)
train_df = train_df.dropna(subset=["img_path"])  # safety

train_idx, val_idx = train_test_split(
    train_df.index, test_size=0.2, stratify=train_df["target"], random_state=42
)
train_paths = train_df.loc[train_idx, "img_path"].tolist()
train_targets = train_df.loc[train_idx, "target"].values.astype(np.float32)
val_paths = train_df.loc[val_idx, "img_path"].tolist()
val_targets = train_df.loc[val_idx, "target"].values.astype(np.float32)

train_dataset = MelanomaDataset(train_paths, train_targets, train_aug, cache_size=0)
val_dataset = MelanomaDataset(val_paths, val_targets, test_aug, cache_size=-1)

worker_count = min(8, os.cpu_count() or 2)  # increase parallelism for I/O

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=worker_count,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=worker_count,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MelanomaModel().to(device)

model = torch.compile(model, mode="reduce-overhead")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=1, eta_min=1e-6
)

use_amp = device.type == "cuda"
scaler = GradScaler() if use_amp else None  # mixed‑precision scaler

for epoch in range(args.epochs):
    model.train()
    epoch_losses = []
    for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        if use_amp:
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, tgts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, tgts)
            loss.backward()
            optimizer.step()
        epoch_losses.append(loss.item())
    scheduler.step()
    avg_loss = np.mean(epoch_losses)

    model.eval()
    val_probs = []
    val_true = []
    with torch.no_grad():
        for imgs, tgts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            val_probs.extend(probs)
            val_true.extend(tgts.numpy())
    auc = metrics.roc_auc_score(val_true, val_probs)
    print(f"Epoch {epoch+1}: Train loss {avg_loss:.4f} | Val AUC {auc:.4f}")



## === cell 7
test_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
test_img_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test"
test_df["img_path"] = test_df["image_name"].apply(lambda x: f"{test_img_dir}/{x}.jpg")
test_paths = test_df["img_path"].tolist()
test_dataset = MelanomaDataset(
    test_paths, np.zeros(len(test_paths), dtype=np.float32), test_aug, cache_size=-1
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=worker_count,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

model.eval()
test_preds = []
with torch.no_grad():
    for imgs, _ in tqdm(test_loader, desc="Predicting"):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        test_preds.extend(probs)

test_df["target"] = test_preds
submission = test_df[["image_name", "target"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
