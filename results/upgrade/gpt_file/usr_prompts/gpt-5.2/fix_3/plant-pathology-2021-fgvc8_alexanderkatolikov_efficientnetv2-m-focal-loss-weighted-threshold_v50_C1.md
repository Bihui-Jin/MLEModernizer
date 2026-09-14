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
import numpy as np
import pandas as pd
from PIL import Image
import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 2
BATCH = 6
EPOCHS = 10
WEIGHT_DECAY = 0.000
LR = 0.000001
IM_SIZE = 728

trainnum = 14800
valnum = 3700

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images/"

TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB_CSV = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
train_df.head()



## === cell 4
train_df["labels"].value_counts().head(10)



## === cell 5
all_tokens = sorted(
    {t for s in train_df["labels"].astype(str).tolist() for t in s.split(" ") if t}
)
NUM_CL = len(all_tokens)
NUM_CL, all_tokens[:10]



## === cell 6
token2id = {t: i for i, t in enumerate(all_tokens)}
id2token = {i: t for t, i in token2id.items()}


def encode_multilabel(label_str):
    y = torch.zeros(NUM_CL, dtype=torch.float32)
    for t in str(label_str).split(" "):
        if t in token2id:
            y[token2id[t]] = 1.0
    return y




## === cell 7
n = len(train_df)
all_idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(all_idx)

tr_idx = all_idx[: min(trainnum, n - valnum)]
va_idx = all_idx[min(trainnum, n - valnum) : min(trainnum + valnum, n)]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[va_idx].reset_index(drop=True)

print("Train size:", len(tr_df), "Val size:", len(val_df))

X_Train = tr_df["image"].values
Y_Train = tr_df["labels"].values

X_val = val_df["image"].values
Y_val = val_df["labels"].values



## === cell 8
Transform = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)



## === cell 9
Transformval = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 10
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = list(FNames)
        self.transform = Transform
        self.labels = None if Labels is None else list(Labels)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index])).convert("RGB")

        if self.labels is not None:
            y = encode_multilabel(self.labels[index])
            return self.transform(x), y
        else:
            return self.transform(x), self.fnames[index]




## === cell 11
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 12
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 13
next(iter(trainloader))[0].shape




## === cell 14
def metrics(pred_tokens, true_tokens, tp, fn, fp):
    a = pred_tokens
    b = true_tokens
    for i in range(len(a)):
        tp += len(list(set(a[i]) & set(b[i])))
        fn += len(b[i]) - len(list(set(a[i]) & set(b[i])))
        fp += len(a[i]) - len(list(set(a[i]) & set(b[i])))
    return tp, fn, fp


def mean_f1_from_strings(y_true_str, y_pred_str):
    true_tokens = [str(s).split() if str(s).strip() else [] for s in y_true_str]
    pred_tokens = [str(s).split() if str(s).strip() else [] for s in y_pred_str]
    tp = fn = fp = 0
    tp, fn, fp = metrics(pred_tokens, true_tokens, tp, fn, fp)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0




## === cell 15
model = torchvision.models.resnext101_32x8d(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, NUM_CL, bias=True)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)




## === cell 16
def train_one_epoch(model, loader):
    model.train()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        running += loss.item() * bs
        n += bs
    return running / max(1, n)


@torch.no_grad()
def predict_probs_on_loader(model, loader):
    model.eval()
    probs_all = []
    y_all = []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        y_all.append(yb.numpy())
    return np.vstack(probs_all), np.vstack(y_all)


def probs_to_label_strings(probs, thresh):
    out = []
    for p in probs:
        idx = np.where(p >= thresh)[0].tolist()
        if len(idx) == 0:
            idx = [int(np.argmax(p))]
        out.append(" ".join([id2token[j] for j in idx]))
    return out


best_thresh = 0.35  # starting point, will be tuned on val to improve mean F1
for epoch in range(EPOCHS):
    tr_loss = train_one_epoch(model, trainloader)

    val_probs, val_y = predict_probs_on_loader(model, valloader)
    y_true_str = val_df["labels"].astype(str).tolist()

    candidate_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_f1 = -1.0
    best_t = best_thresh
    for t in candidate_thresholds:
        y_pred_str = probs_to_label_strings(val_probs, t)
        f1 = mean_f1_from_strings(y_true_str, y_pred_str)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    best_thresh = best_t

    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss: {tr_loss:.5f} - val_f1@t: {best_f1:.5f} (t={best_thresh:.2f})"
    )



## === cell 17
X_Test = [name for name in os.listdir(TEST_DIR) if name.lower().endswith(".jpg")]
len(X_Test), X_Test[:3]



## === cell 18
testset = GetData(TEST_DIR, X_Test, None, Transformval)
testloader = DataLoader(
    testset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 19
model.eval()
s_ls = []

with torch.no_grad():
    for images, fnames in testloader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        for i in range(probs.shape[0]):
            p = probs[i]
            idx = np.where(p >= best_thresh)[0].tolist()
            if len(idx) == 0:
                idx = [int(np.argmax(p))]
            labels_str = " ".join([id2token[j] for j in idx])
            s_ls.append([fnames[i], labels_str])



## === cell 20
pred_df = pd.DataFrame.from_records(s_ls, columns=["image", "labels"])
pred_df.head()



## === cell 21
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
pred_df = sample_sub[["image"]].merge(pred_df, on="image", how="left")
pred_df["labels"] = pred_df["labels"].fillna("healthy")
pred_df.head()



## === cell 22
sub = pred_df[["image", "labels"]]
sub.head()



## === cell 23
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head(3).to_string(index=False))
