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

No external packages required in the script and installed.

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import math
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]

LOCAL_DS_PATH = None
for root in CANDIDATE_ROOTS:
    if (
        os.path.exists(os.path.join(root, "train.csv"))
        and os.path.exists(os.path.join(root, "test.csv"))
        and (
            os.path.exists(os.path.join(root, "jpeg"))
            or os.path.exists(os.path.join(root, "jpeg", "test"))
        )
    ):
        LOCAL_DS_PATH = root
        break

if LOCAL_DS_PATH is None:
    raise FileNotFoundError(f"Could not locate dataset root. Tried: {CANDIDATE_ROOTS}")

print("Using dataset root:", LOCAL_DS_PATH)

train_csv_path = os.path.join(LOCAL_DS_PATH, "train.csv")
test_csv_path = os.path.join(LOCAL_DS_PATH, "test.csv")
sample_path = os.path.join(LOCAL_DS_PATH, "sample_submission.csv")

if not os.path.exists(sample_path):
    alt = "/kaggle/input/sample_submission.csv"
    if os.path.exists(alt):
        sample_path = alt

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample = pd.read_csv(sample_path)

jpeg_test_dir_candidates = [
    os.path.join(LOCAL_DS_PATH, "jpeg", "test"),
    os.path.join(LOCAL_DS_PATH, "siim-isic-melanoma-classification", "jpeg", "test"),
]
JPEG_TEST_DIR = None
for d in jpeg_test_dir_candidates:
    if os.path.exists(d):
        JPEG_TEST_DIR = d
        break
if JPEG_TEST_DIR is None:
    raise FileNotFoundError(
        f"Could not find jpeg test directory. Tried: {jpeg_test_dir_candidates}"
    )

jpeg_train_dir_candidates = [
    os.path.join(LOCAL_DS_PATH, "jpeg", "train"),
    os.path.join(LOCAL_DS_PATH, "siim-isic-melanoma-classification", "jpeg", "train"),
]
JPEG_TRAIN_DIR = None
for d in jpeg_train_dir_candidates:
    if os.path.exists(d):
        JPEG_TRAIN_DIR = d
        break
if JPEG_TRAIN_DIR is None:
    raise FileNotFoundError(
        f"Could not find jpeg train directory. Tried: {jpeg_train_dir_candidates}"
    )

print("Using JPEG train dir:", JPEG_TRAIN_DIR)
print("Using JPEG test dir:", JPEG_TEST_DIR)
print(
    "Train rows:",
    len(train_df),
    "Test rows:",
    len(test_df),
    "Sample rows:",
    len(sample),
)



## === cell 2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

BATCH_SIZE = 64

weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)
model = model.to(device)

for p in model.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

preprocess = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean, std=weights.transforms().std
        ),
    ]
)


class MelanomaTrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image_name"]
        y = float(row["target"])
        path = os.path.join(self.img_dir, f"{image_name}.jpg")
        try:
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
        except Exception:
            x = torch.zeros(3, 128, 128, dtype=torch.float32)
        return x, torch.tensor([y], dtype=torch.float32)


class MelanomaTestDataset(Dataset):
    def __init__(self, image_names, img_dir, transform):
        self.image_names = list(image_names)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        path = os.path.join(self.img_dir, f"{image_name}.jpg")
        try:
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
        except Exception:
            x = torch.zeros(3, 128, 128, dtype=torch.float32)
        return x, image_name


train_dataset = MelanomaTrainDataset(
    train_df[["image_name", "target"]], JPEG_TRAIN_DIR, preprocess
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_dataset = MelanomaTestDataset(
    test_df["image_name"].values, JPEG_TEST_DIR, preprocess
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 3
pos = float(train_df["target"].sum())
neg = float(len(train_df) - pos)
pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-4
)

model.train()
EPOCHS = 2  # keep within 600s; training only the head is fast and already a big improvement over random

for epoch in range(EPOCHS):
    running = 0.0
    n = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        running += float(loss.detach().cpu()) * bs
        n += bs

    print(f"epoch {epoch+1}/{EPOCHS} - train_loss: {running/max(n,1):.5f}")

model.eval()



## === cell 4
all_ids = []
all_preds = []

with torch.no_grad():
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_ids.extend(list(ids))
        all_preds.extend(list(probs))

sub_raw = pd.DataFrame(
    {"image_name": np.array(all_ids), "target": np.array(all_preds, dtype=np.float32)}
)

sub = sample[["image_name"]].merge(sub_raw, on="image_name", how="left")
sub["target"] = (
    sub["target"]
    .fillna(sub["target"].mean() if sub["target"].notna().any() else 0.5)
    .clip(0.0, 1.0)
)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Target stats:",
    float(sub["target"].min()),
    float(sub["target"].mean()),
    float(sub["target"].max()),
)
