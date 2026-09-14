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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import random
import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam




## === cell 1
def _find_base_input_dir():
    candidates = [
        "/kaggle/input/siim-isic-melanoma-classification",
        "../input/siim-isic-melanoma-classification",
        "/kaggle/data/siim-isic-melanoma-classification",
        "../data/siim-isic-melanoma-classification",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "../input/siim-isic-melanoma-classification"


BASE = _find_base_input_dir()

train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
test_path = os.path.join(BASE, "jpeg", "test")
train_path = os.path.join(BASE, "jpeg", "train")
sample_submission = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))

print("BASE:", BASE)
print(train.shape, test.shape, sample_submission.shape)



## === cell 2
IMG_SIZE = 32

cpu = os.cpu_count() or 4
cv2.setNumThreads(max(1, min(8, cpu)))
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def _load_jpeg_rgb(image_fp: str) -> np.ndarray:
    img = cv2.imread(image_fp, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_fp}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_images_from_jpegs_cached(
    image_names, split_dir: str, img_size=32, cache_name="cache.npy"
):
    cache_path = os.path.join(".", cache_name)
    if os.path.exists(cache_path):
        X = np.load(cache_path, mmap_mode=None)
        if (
            X.shape[0] == len(image_names)
            and X.shape[1] == img_size
            and X.shape[2] == img_size
            and X.shape[3] == 3
        ):
            return X
        try:
            os.remove(cache_path)
        except Exception:
            pass

    X = np.empty((len(image_names), img_size, img_size, 3), dtype=np.float32)

    for i, nm in enumerate(
        tqdm(
            image_names,
            desc=f"Loading JPEGs ({os.path.basename(split_dir)})",
            mininterval=1.0,
        )
    ):
        fp = os.path.join(split_dir, f"{nm}.jpg")
        img = _load_jpeg_rgb(fp)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        X[i] = img.astype(np.float32) * (1.0 / 255.0)

    np.save(cache_path, X)
    return X


train_y = train["target"].values.astype(np.int64)
train_names = train["image_name"].values
test_names = test["image_name"].values

train_x = load_images_from_jpegs_cached(
    train_names,
    split_dir=train_path,
    img_size=IMG_SIZE,
    cache_name=f"train_{IMG_SIZE}.npy",
)
test_x = load_images_from_jpegs_cached(
    test_names,
    split_dir=test_path,
    img_size=IMG_SIZE,
    cache_name=f"test_{IMG_SIZE}.npy",
)

print("train_x:", train_x.shape, "train_y:", train_y.shape, "test_x:", test_x.shape)



## === cell 3
i = 0
plt.figure(figsize=(8, 8))
plt.subplot(221), plt.imshow(train_x[i])
plt.subplot(222), plt.imshow(train_x[min(i + 25, len(train_x) - 1)])
plt.subplot(223), plt.imshow(train_x[min(i + 50, len(train_x) - 1)])
plt.subplot(224), plt.imshow(train_x[min(i + 75, len(train_x) - 1)])
plt.tight_layout()
plt.show()



## === cell 4
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
)
print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape))



## === cell 5
use_cuda = torch.cuda.is_available()
pin = True if use_cuda else False

train_x = torch.from_numpy(np.transpose(train_x, (0, 3, 1, 2))).contiguous()
val_x = torch.from_numpy(np.transpose(val_x, (0, 3, 1, 2))).contiguous()
test_x = torch.from_numpy(np.transpose(test_x, (0, 3, 1, 2))).contiguous()

if pin:
    train_x = train_x.pin_memory()
    val_x = val_x.pin_memory()
    test_x = test_x.pin_memory()

train_y = torch.from_numpy(train_y.astype(np.int64))
val_y = torch.from_numpy(val_y.astype(np.int64))
if pin:
    train_y = train_y.pin_memory()
    val_y = val_y.pin_memory()

print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape)



## === cell 6
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 7
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.adapt = nn.AdaptiveMaxPool2d((5, 7))
        self.fc1 = nn.Linear(16 * 5 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Sequential(nn.Linear(84, 2))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x.float())), (2, 2))
        x = self.adapt(F.relu(self.conv2(x.float())))
        x = x.view(-1, 16 * 5 * 7)
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x.float()))
        x = self.fc3(x.float())
        return x




## === cell 8
model = Net()
optimizer = Adam(model.parameters(), lr=0.07)
criterion = CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

train_x_dev = train_x.to(device, non_blocking=pin)
train_y_dev = train_y.to(device, non_blocking=pin)
val_x_dev = val_x.to(device, non_blocking=pin)
val_y_dev = val_y.to(device, non_blocking=pin)
test_x_dev = test_x.to(device, non_blocking=pin)

print(model)
print("device:", device)




## === cell 9
def train_epoch(epoch):
    model.train()

    optimizer.zero_grad()
    output_train = model(train_x_dev)
    loss_train = criterion(output_train, train_y_dev)

    train_losses.append(loss_train.detach().cpu().item())

    loss_train.backward()
    optimizer.step()

    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            output_val = model(val_x_dev)
            loss_val = criterion(output_val, val_y_dev)
        val_losses.append(loss_val.detach().cpu().item())
        print("Epoch :", epoch + 1, "\t", "loss :", loss_val.detach().cpu().item())
        model.train()




## === cell 10
n_epochs = 11
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    train_epoch(epoch)



## === cell 11
model.eval()
with torch.no_grad():
    out = model(train_x_dev)
    probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)

print("train acc:", accuracy_score(train_y.numpy(), preds))
print("train auc:", roc_auc_score(train_y.numpy(), probs))



## === cell 12
model.eval()
with torch.no_grad():
    out = model(val_x_dev)
    probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)

print("val acc:", accuracy_score(val_y.numpy(), preds))
print("val auc:", roc_auc_score(val_y.numpy(), probs))



## === cell 13
model.eval()
with torch.no_grad():
    out = model(test_x_dev)
    probs_test = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()

sub = pd.DataFrame(
    {"image_name": test["image_name"].values, "target": probs_test.astype(np.float32)}
)
sub.to_csv("sub_05.csv", index=False)

print(sub.head())
print("Wrote submission:", "sub_05.csv", "rows:", len(sub))
print("target min/max:", float(sub["target"].min()), float(sub["target"].max()))
