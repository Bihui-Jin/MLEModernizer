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
import os, copy, time, sys, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

try:
    sys.path.append("../input/torchcontrib/contrib-master/")
    import torchcontrib  # noqa: F401
except Exception:
    pass




## === cell 1
BATCH = 64
EPOCHS = 3
LR = 1e-5
WEIGHT_DECAY = 0.0
IM_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"




## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train_df.head()




## === cell 3
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())

all_classes = sorted({lbl for sublist in train_df["label_list"] for lbl in sublist})
NUM_CL = len(all_classes)
print("Number of distinct classes:", NUM_CL)

class_to_id = {c: i for i, c in enumerate(all_classes)}
id_to_class = {i: c for c, i in class_to_id.items()}

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=all_classes)
label_matrix = mlb.fit_transform(train_df["label_list"]).astype(np.float32)




## === cell 4
indices = np.arange(len(train_df))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

X_train = train_df["image"].values[train_idx]
Y_train = label_matrix[train_idx]

X_val = train_df["image"].values[val_idx]
Y_val = label_matrix[val_idx]

print("train:", len(X_train), "val:", len(X_val))




## === cell 5
Transform = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
TransformVal = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 6
class GetData(Dataset):
    def __init__(self, root_dir, fnames, labels, transform, preload=False):
        self.root = root_dir
        self.fnames = fnames
        self.transform = transform
        self.preload = preload
        if labels is None:
            self.label_tensors = None
        else:
            self.label_tensors = [torch.from_numpy(lbl).float() for lbl in labels]
        if self.preload:
            self.cached_data = []
            for idx, fname in enumerate(self.fnames):
                img_path = os.path.join(self.root, fname)
                img = Image.open(img_path).convert("RGB")
                img_t = self.transform(img)  # (C, H, W) tensor
                if self.label_tensors is None:
                    self.cached_data.append((img_t, fname))
                else:
                    self.cached_data.append((img_t, self.label_tensors[idx]))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if self.preload:
            return self.cached_data[idx]
        img_path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)
        if self.label_tensors is None:
            return img_t, self.fnames[idx]
        else:
            return img_t, self.label_tensors[idx]




## === cell 7
torch.backends.cudnn.benchmark = True

model = torchvision.models.resnext101_32x8d(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, NUM_CL)
model = model.to(DEVICE)
model = model.to(memory_format=torch.channels_last)

model = torch.compile(model, mode="reduce-overhead")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scaler = torch.cuda.amp.GradScaler()




## === cell 8
trainset = GetData(TRAIN_DIR, X_train, Y_train, Transform, preload=True)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=0,  # no extra workers needed when data is cached
    pin_memory=True,
    persistent_workers=False,
)

valset = GetData(TRAIN_DIR, X_val, Y_val, TransformVal, preload=True)
valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
)




## === cell 9
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for imgs, targets in trainloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * imgs.size(0)

    epoch_loss /= len(trainloader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for imgs, targets in valloader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            val_loss += loss.item() * imgs.size(0)
        val_loss /= len(valloader.dataset)
    print(f"            Validation loss: {val_loss:.4f}")
    model.train()




## === cell 10
all_test_files = sorted(os.listdir(TEST_DIR))
test_fnames = [
    f for f in all_test_files if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
print("Number of test images:", len(test_fnames))

testset = GetData(TEST_DIR, test_fnames, None, TransformVal, preload=False)
testloader = DataLoader(
    testset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
)




## === cell 11
top_k = 3
pred_records = []

model.eval()
with torch.no_grad():
    for imgs, fnames in testloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        probs = torch.sigmoid(logits)  # [batch, NUM_CL]

        for i in range(probs.size(0)):
            prob = probs[i]
            mask = prob > 0.5
            if mask.any():
                selected_ids = mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
            else:
                selected_ids = prob.topk(top_k).indices.cpu().numpy()
            pred_labels = " ".join([id_to_class[int(j)] for j in selected_ids])
            pred_records.append([fnames[i], pred_labels])




## === cell 12
pred_df = pd.DataFrame(pred_records, columns=["image", "labels"])




## === cell 13
submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape:", pred_df.shape)
