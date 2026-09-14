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
import os
import numpy as np
import pandas as pd
from PIL import Image  # kept for compatibility but not used in data loading
import torchvision.io as io  # new import for fast image reading

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import torchmetrics
from torch.cuda.amp import autocast, GradScaler

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True




## === cell 1
BATCH = 24  # was 12
EPOCHS = 10
WEIGHT_DECAY = 0.0
LR = 1e-4
IM_SIZE = 728
TOP_K = 3  # fallback if no prob > 0.5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True




## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
print(f"Train records: {len(train_df)}")




## === cell 3
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(train_df["label_list"])
train_df["label_vec"] = list(mlb.transform(train_df["label_list"]))
NUM_CL = len(mlb.classes_)
class_names = mlb.classes_
print(f"Number of disease classes: {NUM_CL}")




## === cell 4
total_len = len(train_df)
trainnum = int(0.90 * total_len)
valnum = total_len - trainnum

tr_df = train_df.iloc[:trainnum]
X_Train, Y_Train = tr_df["image"].values, np.array(
    tr_df["label_vec"].tolist(), dtype=np.float32
)

val_df = train_df.iloc[trainnum:]
X_val, Y_val = val_df["image"].values, np.array(
    val_df["label_vec"].tolist(), dtype=np.float32
)




## === cell 5
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

Transform = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.Lambda(lambda x: x.float() / 255.0),  # scale to [0,1]
        transforms.Normalize(mean, std),
    ]
)

Transformval = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Normalize(mean, std),
    ]
)




## === cell 6
class GetData(Dataset):
    def __init__(self, dir_path, fnames, labels, transform):
        self.dir = dir_path
        self.fnames = fnames
        self.transform = transform
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.fnames[idx])
        img = io.read_image(img_path)  # shape: C x H x W, uint8
        img = self.transform(img)
        if self.labels is None:
            return img, self.fnames[idx]  # test mode
        else:
            return img, self.labels[idx]




## === cell 7
NUM_WORKERS = min(8, os.cpu_count() or 1)

trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,  # uses the increased batch size
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(
    valset,
    batch_size=BATCH,  # uses the increased batch size
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 8
model = torchvision.models.resnext101_32x8d()
model.fc = nn.Linear(2048, NUM_CL, bias=True)

checkpoint_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    print("Loaded custom ResNext checkpoint.")
else:
    print("Custom checkpoint not found – using ImageNet‑pretrained weights.")

model = model.to(DEVICE)

model = torch.compile(model, mode="reduce-overhead")




## === cell 9
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

f1_metric = torchmetrics.F1Score(
    task="multilabel", num_labels=NUM_CL, average="macro", threshold=0.5
)

scaler = GradScaler()  # AMP scaler

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for imgs, targets in trainloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(trainloader.dataset)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in valloader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            all_preds.append(probs)  # keep on GPU
            all_targets.append(targets)  # keep on GPU
    preds = torch.cat(all_preds)  # GPU tensor
    targets = torch.cat(all_targets)  # GPU tensor
    val_f1 = f1_metric(preds, targets.int())
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Macro F1: {val_f1:.4f}"
    )

model.eval()




## === cell 10
test_fnames = [
    f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
print(f"Number of test images: {len(test_fnames)}")




## === cell 11
testset = GetData(TEST_DIR, test_fnames, None, Transformval)
testloader = DataLoader(
    testset,
    batch_size=BATCH,  # larger batch for faster inference (matches training batch size)
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 12
pred_records = []

with torch.no_grad():
    for imgs, fnames in testloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast():
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
        for prob_vec, fname in zip(probs, fnames):
            idxs = np.where(prob_vec > 0.5)[0]
            if len(idxs) == 0:
                idxs = np.argsort(prob_vec)[-TOP_K:]  # fallback to top‑K
            pred_labels = " ".join(class_names[idx] for idx in idxs)
            pred_records.append([fname, pred_labels])




## === cell 13
submission = pd.DataFrame(pred_records, columns=["image", "labels"])
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
