# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
try:
    import sys

    sys.path.append("../input/torchcontrib/contrib-master/")
    import torchcontrib  # noqa: F401
    from torchcontrib.optim import SWA  # noqa: F401
    from torch.optim.swa_utils import AveragedModel, SWALR  # noqa: F401
except Exception:
    pass




## === cell 1
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.000
LR = 0.0001
IM_SIZE = 640

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
if not os.path.isdir(BASE_INPUT):
    BASE_INPUT = "../input/plant-pathology-2021-fgvc8"

TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")




## === cell 2
train_df = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))

try:
    labeled_train_df = pd.read_csv("../input/train-labeled/train.csv")
except Exception:
    labeled_train_df = pd.DataFrame(columns=train_df.columns)  # empty placeholder




## === cell 3
_ = train_df["labels"].value_counts()




## === cell 4
NUM_CL = len(train_df["labels"].value_counts())
NUM_CL




## === cell 5
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])




## === cell 6
my_dict = {}
if not labeled_train_df.empty:
    for i in range(NUM_CL):
        if not (labeled_train_df["label_id"] == i).any():
            continue
        equality = (
            labeled_train_df.loc[labeled_train_df["label_id"] == i].values[0][1]
            != train_df.loc[train_df["label_id"] == i].values[0][1]
        )
        if equality:
            for l in range(NUM_CL):
                if not (labeled_train_df["label_id"] == i).any():
                    continue
                equality = (
                    labeled_train_df.loc[labeled_train_df["label_id"] == i].values[0][1]
                    == train_df.loc[train_df["label_id"] == l].values[0][1]
                )
                if equality:
                    my_dict[l] = i
    if my_dict:
        train_df = train_df.replace({"label_id": my_dict})




## === cell 7
class_map = dict(sorted(train_df[["label_id", "labels"]].values.tolist()))




## === cell 8
tr_df = train_df[:trainnum]
print("Training set size:", len(tr_df))
X_Train, Y_Train = tr_df["image"].values, tr_df["label_id"].values




## === cell 9
Transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 10
Transformval = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 11
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index])).convert("RGB")
        if "train" in self.dir:
            return self.transform(x), self.labels[index]
        else:  # test directory
            return self.transform(x), self.fnames[index]




## === cell 12
NUM_WORKERS = min(8, os.cpu_count() or 1)
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 13
val_df = train_df[-valnum:]
print("Validation set size:", len(val_df))
X_val, Y_val = val_df["image"].values, val_df["label_id"].values
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 14
batch_shape = next(iter(trainloader))[0].shape
print("Batch tensor shape:", batch_shape)




## === cell 15
def PREDS(l):
    word = train_df.loc[train_df["label_id"] == l].values[0][1]  # finds label's name
    words = word.split(" ")
    return words




## === cell 16
def metrics(preds, labels, tp, fn, fp):
    a = preds.tolist()
    b = labels.tolist()
    for i in range(len(preds)):
        tp += len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        fn += len(PREDS(b[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        fp += len(PREDS(a[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
    return tp, fn, fp




## === cell 17
model = torchvision.models.resnext101_32x8d(pretrained=True)
model.fc = nn.Linear(2048, NUM_CL, bias=True)

checkpoint_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.exists(checkpoint_path):
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Custom checkpoint loaded.")
    except Exception as e:
        print(
            f"Failed to load custom checkpoint: {e}. Proceeding with pretrained weights."
        )
else:
    print("Custom checkpoint not found; using default pretrained weights.")

model = model.to(DEVICE)

if torch.cuda.is_available():
    model = torch.compile(model, mode="reduce-overhead")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)




## === cell 18
scaler = GradScaler()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, targets in trainloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(trainloader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training loss: {epoch_loss:.4f}")

model.eval()
val_loss = 0.0
with torch.no_grad():
    for imgs, targets in valloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        val_loss += loss.item() * imgs.size(0)
val_loss /= len(valloader.dataset)
print(f"Validation loss after training: {val_loss:.4f}")




## === cell 19
if os.path.isdir(os.path.join(TEST_DIR, "test_images")):
    TEST_DIR = os.path.join(TEST_DIR, "test_images")

X_Test = [
    f
    for f in os.listdir(TEST_DIR)
    if os.path.isfile(os.path.join(TEST_DIR, f))
    and f.lower().endswith((".jpg", ".jpeg", ".png"))
]




## === cell 20
testset = GetData(TEST_DIR, X_Test, None, Transformval)
testloader = DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 21
TOP_K = 3
s_ls = []
model.eval()
with torch.no_grad():
    for image, fname in testloader:
        image = image.to(DEVICE, non_blocking=True)
        with autocast():
            logits = model(image)
        topk = logits.topk(TOP_K, dim=1)
        ids = topk.indices.squeeze().cpu().numpy()
        if np.isscalar(ids):
            ids = [int(ids)]
        else:
            ids = [int(i) for i in ids]
        label_str = " ".join([class_map[i] for i in ids])
        if isinstance(fname, (list, tuple)):
            fname = fname[0]
        if isinstance(fname, torch.Tensor):
            fname = fname.item()
        s_ls.append([fname, label_str])




## === cell 22
pred_df = pd.DataFrame.from_records(s_ls, columns=["image", "labels"])




## === cell 23
sub = pred_df




## === cell 24
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
