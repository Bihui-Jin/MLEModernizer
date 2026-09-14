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

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models as tv_models
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # auto‑tuner for convs
torch.set_float32_matmul_precision("medium")  # faster matmul with negligible diff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 1
class MelanomaDataset(Dataset):
    """
    Faster dataset: load images lazily on demand instead of pre‑loading all
    images into RAM. This keeps I/O out of the training loop while preserving
    the original transforms (including random augmentations) applied per sample.
    """

    def __init__(self, df: pd.DataFrame, img_folder: str, transform=None):
        self.transform = transform
        self.img_folder = img_folder

        self.image_names = df["image_name"].values.astype(str)
        if "target" in df.columns:
            self.targets = df["target"].values.astype(np.float32)
        else:
            self.targets = np.full(len(df), -1.0, dtype=np.float32)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = os.path.join(self.img_folder, name + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        target = float(self.targets[idx])
        return img, target




## === cell 2
class DeeperNetwork(nn.Module):
    """
    Wraps a torchvision EfficientNet backbone and adds extra fully‑connected layers.
    The backbone's classifier is replaced by an identity layer so that we can
    feed its feature vector into our custom head.
    """

    def __init__(self, backbone):
        super().__init__()
        if hasattr(backbone, "_fc"):  # legacy EfficientNet (not used here)
            in_features = backbone._fc.in_features
            backbone._fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            if isinstance(backbone.classifier, nn.Sequential):
                in_features = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Identity()
            else:
                in_features = backbone.classifier.in_features
                backbone.classifier = nn.Identity()
        else:
            raise AttributeError(
                "Backbone does not have a recognizable classifier attribute"
            )

        self.backbone = backbone
        self.extra = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.extra(x)
        return x




## === cell 3
BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"

train_csv = os.path.join(BASE_PATH, "train.csv")
test_csv = os.path.join(BASE_PATH, "test.csv")
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

assert "image_name" in train_df.columns
assert "target" in train_df.columns




## === cell 4
train_transform = T.Compose(
    [
        T.ToPILImage(),
        T.RandomResizedCrop(256, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 5
train_img_dir = os.path.join(BASE_PATH, "jpeg/train")
test_img_dir = os.path.join(BASE_PATH, "jpeg/test")

num_workers = min(8, os.cpu_count() or 1)

train_dataset = MelanomaDataset(train_df, train_img_dir, transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=256,  # reduced batch size to stay within GPU memory limits
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

test_dataset = MelanomaDataset(test_df, test_img_dir, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)




## === cell 6
backbone = tv_models.efficientnet_b1(weights=tv_models.EfficientNet_B1_Weights.DEFAULT)
model = DeeperNetwork(backbone).to(device)

model = torch.compile(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # for mixed‑precision training




## === cell 7
model.train()
num_epochs = 2
for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    total = 0
    correct = 0
    for images, targets in train_loader:
        images = images.to(device, dtype=torch.float, non_blocking=True)
        targets = targets.to(device, dtype=torch.float, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.round(torch.sigmoid(outputs))
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    print(
        f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%"
    )




## === cell 8
model.eval()
all_preds = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device, dtype=torch.float, non_blocking=True)
        with autocast():
            outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
        if probs.ndim == 0:
            probs = np.array([probs])
        all_preds.extend(probs.tolist())

submission = test_df[["image_name"]].copy()
submission["target"] = all_preds
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission file saved to:", submission_path)




## === cell 9
import os

print("Files in /kaggle/working:", os.listdir("/kaggle/working"))
