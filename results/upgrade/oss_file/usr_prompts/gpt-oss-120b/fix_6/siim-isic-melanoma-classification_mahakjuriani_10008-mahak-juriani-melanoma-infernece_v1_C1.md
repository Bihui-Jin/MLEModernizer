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
import os
import pandas as pd
import numpy as np
import cv2
import albumentations
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
import math
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(4)




## === cell 1
class args:
    batch_size = 64
    image_size = 384
    epochs = 2  # modest fine‑tuning
    lr = 2.5e-05
    weight_decay = 0.01




## === cell 2
def sigmoid(x):
    return 1 / (1 + math.exp(-x))




## === cell 3
class MelanomaDataset(Dataset):
    def __init__(self, image_paths, dense_features, targets, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augmentations is not None:
            img = self.augmentations(image=img)["image"]

        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))).astype(np.float32)
        img_tensor = torch.from_numpy(img).float()

        features = torch.from_numpy(self.dense_features[idx]).float()
        target = torch.tensor(self.targets[idx], dtype=torch.float)

        return {
            "image": img_tensor,
            "features": features,
            "target": target,
        }




## === cell 4
class MelanomaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True, in_chans=3)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(1000, 1)

    def monitor_metrics(self, outputs, targets, loss):
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        accuracy = (preds == targets.view(-1, 1)).float().mean()
        return {"accuracy": accuracy, "loss": loss}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt, sch

    def forward(self, image, features, target=None):
        x = self.model(image)
        x = self.dropout(x)
        logits = self.out(x)
        if target is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, target.view(-1, 1))
            metrics = self.monitor_metrics(logits, target, loss)
            return logits, loss, metrics
        return logits, 0, {}




## === cell 5
df_train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")

train_img_paths = [
    f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{name}.jpg"
    for name in df_train["image_name"].values
]

dense_vals_train = np.empty((len(df_train), 0), dtype=np.float32)

train_dataset = MelanomaDataset(
    image_paths=train_img_paths,
    dense_features=dense_vals_train,
    targets=df_train["target"].values.astype(np.float32),
    augmentations=albumentations.Compose(
        [
            albumentations.Resize(args.image_size, args.image_size, p=1.0),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MelanomaModel().to(device)
model.train()

num_workers = min(8, os.cpu_count() or 4)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

optimizer, scheduler = model.optimizer_scheduler()

use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler() if use_amp else None

for epoch in range(args.epochs):
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{args.epochs}"):
        optimizer.zero_grad()
        imgs = batch["image"].to(device, non_blocking=True)
        feats = batch["features"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits, loss, metrics = model(imgs, feats, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss, metrics = model(imgs, feats, targets)
            loss.backward()
            optimizer.step()

        epoch_losses.append(loss.item())
        epoch_accs.append(metrics["accuracy"].item())
    scheduler.step()
    print(
        f"Epoch {epoch+1} – loss: {np.mean(epoch_losses):.4f}, acc: {np.mean(epoch_accs):.4f}"
    )

model.eval()



## === cell 6
test_aug = albumentations.Compose(
    [
        albumentations.Resize(args.image_size, args.image_size, p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



## === cell 7
df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")

test_img_paths = [
    f"/kaggle/input/siim-isic-melanoma-classification/jpeg/test/{name}.jpg"
    for name in df_test["image_name"].values
]

dense_vals_test = np.empty((len(df_test), 0), dtype=np.float32)

test_dataset = MelanomaDataset(
    image_paths=test_img_paths,
    dense_features=dense_vals_test,
    targets=np.ones(len(test_img_paths)),  # dummy targets for compatibility
    augmentations=test_aug,
)

loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

all_preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        imgs = batch["image"].to(device, non_blocking=True)
        feats = batch["features"].to(device, non_blocking=True)
        logits, _, _ = model(imgs, feats)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs.ravel().tolist())

df_test["target"] = all_preds
df_test = df_test[["image_name", "target"]]
df_test.to_csv("submission.csv", index=False)



## === cell 8
df_test.head()
