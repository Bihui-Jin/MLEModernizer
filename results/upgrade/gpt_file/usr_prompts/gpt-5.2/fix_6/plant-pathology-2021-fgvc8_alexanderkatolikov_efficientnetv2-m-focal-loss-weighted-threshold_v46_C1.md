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
import time
import copy

import sys
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



## === cell 1
pass



## === cell 2
BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.000
LR = 0.000001
IM_SIZE = 640

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images/"



## === cell 3
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train_df



## === cell 4
train_df["labels"].value_counts()



## === cell 5
NUM_CL = len(train_df["labels"].value_counts())
NUM_CL



## === cell 6
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])
print(train_df)



## === cell 7
class_map = {i: c for i, c in enumerate(le.classes_)}



## === cell 8
tr_df = train_df.iloc[:trainnum].copy()
print(len(tr_df))
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
        elif "test" in self.dir:
            return self.transform(x), self.fnames[index]




## === cell 12
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)



## === cell 13
val_df = train_df.iloc[-valnum:].copy()
print(len(val_df))
X_val, Y_val = val_df["image"].values, val_df["label_id"].values
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(valset, batch_size=BATCH, shuffle=True)



## === cell 14
next(iter(trainloader))[0].shape




## === cell 15
def PREDS(l):
    word = train_df.loc[train_df["label_id"] == l].values[0][1]
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
try:
    model = torchvision.models.resnext101_32x8d(weights=None)
except TypeError:
    model = torchvision.models.resnext101_32x8d()

model.fc = nn.Linear(2048, NUM_CL, bias=True)

ckpt_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
else:
    try:
        model = torchvision.models.resnext101_32x8d(
            weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
        )
    except Exception:
        model = torchvision.models.resnext101_32x8d(pretrained=True)
    model.fc = nn.Linear(2048, NUM_CL, bias=True)

model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)



## === cell 18
testnum = 3700
test_df = train_df.iloc[-testnum:].copy()
X_test, Y_test = test_df["image"].values, test_df["label_id"].values
testset = GetData(TRAIN_DIR, X_test, Y_test, Transformval)
testloader = DataLoader(testset, batch_size=BATCH, shuffle=True)



## === cell 19
_valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
X_Test = []
for name in os.listdir(TEST_DIR):
    full = os.path.join(TEST_DIR, name)
    if os.path.isfile(full) and os.path.splitext(name.lower())[1] in _valid_ext:
        X_Test.append(name)

X_Test = sorted(X_Test)

print("Found test images:", len(X_Test))
print("Example:", X_Test[:5])



## === cell 20
testset = GetData(TEST_DIR, X_Test, None, Transformval)
testloader = DataLoader(testset, batch_size=1, shuffle=False)




## === cell 21
def _labels_str_to_multi_hot(labels_str, class_tokens):
    s = set(labels_str.split(" "))
    return np.array([1 if t in s else 0 for t in class_tokens], dtype=np.int32)


def mean_f1_from_multilabel(pred_bin, true_bin, eps=1e-9):
    tp = (pred_bin & true_bin).sum(axis=0).astype(np.float32)
    fp = (pred_bin & (~true_bin.astype(bool))).sum(axis=0).astype(np.float32)
    fn = ((~pred_bin.astype(bool)) & true_bin).sum(axis=0).astype(np.float32)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


all_tokens = sorted(
    {t for s in train_df["labels"].astype(str).values for t in s.split(" ")}
)
token_to_idx = {t: i for i, t in enumerate(all_tokens)}
C_tok = len(all_tokens)

id_to_multihot = {}
for lab_id, lab_str in zip(
    train_df["label_id"].values, train_df["labels"].astype(str).values
):
    if int(lab_id) not in id_to_multihot:
        id_to_multihot[int(lab_id)] = _labels_str_to_multi_hot(lab_str, all_tokens)

healthy_token_idx = token_to_idx.get("healthy", None)

val_probs_cls = []
val_true_tokens = []
model.eval()
with torch.no_grad():
    for xb, yb in valloader:
        xb = xb.to(DEVICE)
        logits = model(xb)  # [B, NUM_CL]
        ps = torch.sigmoid(logits).detach().cpu().numpy()  # [B, NUM_CL]
        val_probs_cls.append(ps)
        for yi in yb.detach().cpu().numpy().tolist():
            val_true_tokens.append(id_to_multihot[int(yi)])

val_probs_cls = np.concatenate(val_probs_cls, axis=0)  # [Nv, NUM_CL]
val_true_tokens = np.stack(val_true_tokens, axis=0).astype(bool)  # [Nv, C_tok]

token_in_cls = np.zeros((C_tok, NUM_CL), dtype=bool)
for cls_id, cls_str in class_map.items():
    toks = cls_str.split(" ")
    for t in toks:
        if t in token_to_idx:
            token_in_cls[token_to_idx[t], int(cls_id)] = True

val_probs_tok = np.zeros((val_probs_cls.shape[0], C_tok), dtype=np.float32)
for ti in range(C_tok):
    mask = token_in_cls[ti]  # [NUM_CL]
    if mask.any():
        val_probs_tok[:, ti] = val_probs_cls[:, mask].max(axis=1)
    else:
        val_probs_tok[:, ti] = 0.0

thr_grid = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50], dtype=np.float32)
best_thr = np.full((C_tok,), 0.35, dtype=np.float32)

for ti in range(C_tok):
    if val_true_tokens[:, ti].sum() == 0:
        best_thr[ti] = 0.50
        continue
    best_score = -1.0
    best_t = 0.35
    for t in thr_grid:
        pred_bin = val_probs_tok[:, ti] >= t
        true_bin = val_true_tokens[:, ti]
        tp = float((pred_bin & true_bin).sum())
        fp = float((pred_bin & (~true_bin)).sum())
        fn = float(((~pred_bin) & true_bin).sum())
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        if f1 > best_score:
            best_score = f1
            best_t = float(t)
    best_thr[ti] = best_t

if healthy_token_idx is not None:
    best_thr[healthy_token_idx] = max(best_thr[healthy_token_idx], 0.55)

val_pred_tokens = val_probs_tok >= best_thr[None, :]
if healthy_token_idx is not None:
    has_other = val_pred_tokens.copy()
    has_other[:, healthy_token_idx] = False
    has_other = has_other.any(axis=1)
    val_pred_tokens[has_other, healthy_token_idx] = False

max_tok = val_probs_tok.argmax(axis=1)
empty = ~val_pred_tokens.any(axis=1)
val_pred_tokens[empty, max_tok[empty]] = True

print(
    "Val mean-F1 (token-level, postprocess):",
    mean_f1_from_multilabel(val_pred_tokens, val_true_tokens),
)



## === cell 22
s_ls = []

with torch.no_grad():
    model.eval()
    for image, fname in testloader:
        image = image.to(DEVICE)
        logits = model(image)  # [1, NUM_CL]
        ps_cls = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()  # [NUM_CL]

        ps_tok = np.zeros((C_tok,), dtype=np.float32)
        for ti in range(C_tok):
            mask = token_in_cls[ti]
            if mask.any():
                ps_tok[ti] = float(ps_cls[mask].max())
            else:
                ps_tok[ti] = 0.0

        pred_tok = ps_tok >= best_thr

        if healthy_token_idx is not None and pred_tok[healthy_token_idx]:
            if pred_tok.sum() > 1:
                pred_tok[healthy_token_idx] = False

        if not pred_tok.any():
            pred_tok[int(np.argmax(ps_tok))] = True

        pred_labels = [all_tokens[i] for i in np.where(pred_tok)[0].tolist()]
        pred_label_str = " ".join(pred_labels)

        s_ls.append([fname[0], pred_label_str])



## === cell 23
pred_df = pd.DataFrame.from_records(s_ls, columns=["image", "labels"])
pred_df



## === cell 24
sub = pred_df[["image", "labels"]]
sub.head()



## === cell 25
sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
sub = sample_sub[["image"]].merge(sub, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")



## === cell 26
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
