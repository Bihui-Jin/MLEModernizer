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
import time
import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2 as T_v2



## === cell 1
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_CPU = os.cpu_count() or 2
NUM_WORKERS = min(4, max(2, _CPU // 2))



## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train_df



## === cell 3
train_df["labels"].value_counts()



## === cell 4
NUM_CL = len(train_df["labels"].value_counts())
NUM_CL



## === cell 5
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])
print(train_df)



## === cell 6
class_map = {i: c for i, c in enumerate(le.classes_)}



## === cell 7
tr_df = train_df.iloc[:trainnum].copy()
print(len(tr_df))
X_Train, Y_Train = tr_df["image"].values, tr_df["label_id"].values



## === cell 8
Transform = T_v2.Compose(
    [
        T_v2.ToDtype(torch.float32, scale=True),  # uint8 -> float32 in [0,1]
        T_v2.Resize((IM_SIZE, IM_SIZE)),
        T_v2.CenterCrop(int(IM_SIZE * 0.8)),
        T_v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)



## === cell 9
Transformval = T_v2.Compose(
    [
        T_v2.ToDtype(torch.float32, scale=True),
        T_v2.Resize((IM_SIZE, IM_SIZE)),
        T_v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 10
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels
        self._is_train = "train" in self.dir
        self._is_test = "test" in self.dir

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.fnames[index])
        x = read_image(path, mode=ImageReadMode.RGB)  # uint8, [C,H,W]
        x = self.transform(x)

        if self._is_train:
            return x, self.labels[index]
        elif self._is_test:
            return x, self.fnames[index]
        else:
            return x, self.labels[index]




## === cell 11
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)



## === cell 12
val_df = train_df.iloc[trainnum : trainnum + valnum].copy()
print(len(val_df))
X_val, Y_val = val_df["image"].values, val_df["label_id"].values
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)

valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)



## === cell 13
next(iter(trainloader))[0].shape



## === cell 14
_labelid_to_tokens = {}
for lab_id, lab_str in zip(
    train_df["label_id"].values, train_df["labels"].astype(str).values
):
    lab_id = int(lab_id)
    if lab_id not in _labelid_to_tokens:
        _labelid_to_tokens[lab_id] = lab_str.split(" ")


def PREDS(l):
    return _labelid_to_tokens[int(l)]




## === cell 15
def metrics(preds, labels, tp, fn, fp):
    a = preds.tolist()
    b = labels.tolist()
    for i in range(len(preds)):
        tp += len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        fn += len(PREDS(b[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        fp += len(PREDS(a[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
    return tp, fn, fp




## === cell 16
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



## === cell 17
scaler = torch.cuda.amp.GradScaler(
    enabled=False
)  # keep semantics identical; no mixed precision

model.train()
for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0
    n_seen = 0

    for xb, yb in trainloader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        running_loss += float(loss.detach().cpu().item()) * bs
        n_seen += bs

    dt = time.time() - t0
    print(
        f"Epoch {epoch+1}/{EPOCHS} - loss: {running_loss/max(1,n_seen):.5f} - time: {dt:.1f}s"
    )



## === cell 18
testnum = 3700
test_df = train_df.iloc[-testnum:].copy()
X_test, Y_test = test_df["image"].values, test_df["label_id"].values
testset = GetData(TRAIN_DIR, X_test, Y_test, Transformval)
testloader = DataLoader(
    testset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)



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
testloader = DataLoader(
    testset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)




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
    lab_id = int(lab_id)
    if lab_id not in id_to_multihot:
        id_to_multihot[lab_id] = _labels_str_to_multi_hot(lab_str, all_tokens)

healthy_token_idx = token_to_idx.get("healthy", None)

_token_order = []
_token_seen = set()
for s in train_df["labels"].astype(str).values:
    for t in s.split(" "):
        if t not in _token_seen:
            _token_seen.add(t)
            _token_order.append(t)
_token_order_rank = {t: i for i, t in enumerate(_token_order)}


def _order_tokens_canonical(tokens_list):
    return sorted(tokens_list, key=lambda t: _token_order_rank.get(t, 10**9))


def predict_tokens_from_class_probs_topk(
    p_cls, K=2, p_min=0.20, max_tokens=None, token_keep_priority=None
):
    """
    p_cls: [N, NUM_CL] softmax class probs
    max_tokens: if not None, cap number of predicted tokens per image (after healthy suppression).
    token_keep_priority: np.array shape [C_tok], higher means keep earlier when capping.
    Returns: pred_tok bool [N, C_tok]
    """
    N = p_cls.shape[0]
    pred_tok = np.zeros((N, C_tok), dtype=bool)

    topk_idx = np.argpartition(-p_cls, kth=min(K, p_cls.shape[1] - 1), axis=1)[:, :K]
    topk_prob = np.take_along_axis(p_cls, topk_idx, axis=1)

    for i in range(N):
        chosen = topk_idx[i][topk_prob[i] >= p_min]
        if chosen.size == 0:
            chosen = np.array([int(p_cls[i].argmax())], dtype=np.int64)

        toks = set()
        for cid in chosen.tolist():
            toks.update(class_map[int(cid)].split(" "))

        if "healthy" in toks and len(toks) > 1:
            toks.discard("healthy")

        if len(toks) == 0:
            if healthy_token_idx is not None:
                toks.add("healthy")
            else:
                toks.add(class_map[int(p_cls[i].argmax())].split(" ")[0])

        if (max_tokens is not None) and (len(toks) > int(max_tokens)):
            toks_list = list(toks)
            if token_keep_priority is not None:
                toks_list = sorted(
                    toks_list,
                    key=lambda t: (
                        (
                            -float(token_keep_priority[token_to_idx[t]])
                            if t in token_to_idx
                            else 0.0
                        ),
                        _token_order_rank.get(t, 10**9),
                    ),
                )
            else:
                toks_list = _order_tokens_canonical(toks_list)
            toks = set(toks_list[: int(max_tokens)])

            if len(toks) == 0:
                toks.add(
                    "healthy"
                    if healthy_token_idx is not None
                    else class_map[int(p_cls[i].argmax())].split(" ")[0]
                )

        for t in toks:
            ti = token_to_idx.get(t, None)
            if ti is not None:
                pred_tok[i, ti] = True

    return pred_tok


token_counts = np.zeros(C_tok, dtype=np.int64)
for s in train_df["labels"].astype(str).values:
    for t in set(s.split(" ")):
        ti = token_to_idx.get(t, None)
        if ti is not None:
            token_counts[ti] += 1
token_prior = token_counts / max(1, len(train_df))
token_keep_priority = token_prior.astype(np.float32)

class_tokens_list = [class_map[i].split(" ") for i in range(NUM_CL)]
class_to_token_idx = [
    np.array([token_to_idx[t] for t in toks if t in token_to_idx], dtype=np.int64)
    for toks in class_tokens_list
]


def token_probs_from_class_probs_max(p_cls):
    """
    p_cls: [N, NUM_CL] softmax probs
    returns p_tok: [N, C_tok] where p_tok[:,t] = max_c p_cls[:,c] for classes c containing token t
    """
    N = p_cls.shape[0]
    p_tok = np.zeros((N, C_tok), dtype=np.float32)
    for c in range(NUM_CL):
        idxs = class_to_token_idx[c]
        if idxs.size == 0:
            continue
        pc = p_cls[:, c].astype(np.float32)  # [N]
        for ti in idxs.tolist():
            p_tok[:, ti] = np.maximum(p_tok[:, ti], pc)
    return p_tok


def tokens_from_token_probs_threshold(
    p_tok, thr=0.20, max_tokens=None, token_keep_priority=None
):
    """
    p_tok: [N, C_tok] token probabilities
    returns pred_tok: [N, C_tok] bool
    """
    pred_tok = p_tok >= float(thr)

    if healthy_token_idx is not None:
        for i in range(pred_tok.shape[0]):
            if pred_tok[i, healthy_token_idx] and pred_tok[i].sum() > 1:
                pred_tok[i, healthy_token_idx] = False

    for i in range(pred_tok.shape[0]):
        if pred_tok[i].sum() == 0:
            if healthy_token_idx is not None:
                pred_tok[i, healthy_token_idx] = True
            else:
                pred_tok[i, int(p_tok[i].argmax())] = True

        if (max_tokens is not None) and (pred_tok[i].sum() > int(max_tokens)):
            on_idx = np.where(pred_tok[i])[0].tolist()
            if token_keep_priority is not None:
                on_idx = sorted(
                    on_idx,
                    key=lambda ti: (
                        -float(token_keep_priority[ti]),
                        _token_order_rank.get(all_tokens[ti], 10**9),
                    ),
                )
            else:
                on_idx = sorted(
                    on_idx, key=lambda ti: _token_order_rank.get(all_tokens[ti], 10**9)
                )
            keep = set(on_idx[: int(max_tokens)])
            pred_tok[i, :] = False
            pred_tok[i, list(keep)] = True

            if pred_tok[i].sum() == 0:
                if healthy_token_idx is not None:
                    pred_tok[i, healthy_token_idx] = True
                else:
                    pred_tok[i, int(p_tok[i].argmax())] = True

    return pred_tok.astype(bool)


val_probs_cls = []
val_true_tokens = []

model.eval()
with torch.no_grad():
    for xb, yb in valloader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)  # [B, NUM_CL]
        ps = torch.softmax(logits, dim=1).detach().cpu().numpy()  # [B, NUM_CL]
        val_probs_cls.append(ps)
        yb_np = yb.detach().cpu().numpy()
        val_true_tokens.extend([id_to_multihot[int(yi)] for yi in yb_np.tolist()])

val_probs_cls = np.concatenate(val_probs_cls, axis=0)  # [Nv, NUM_CL]
val_true_tokens = np.stack(val_true_tokens, axis=0).astype(bool)  # [Nv, C_tok]

K_grid = [1, 2, 3, 4]
pmin_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
max_tokens_grid = [None, 1, 2, 3]

tokthr_grid = [0.10, 0.15, 0.20, 0.25, 0.30]

val_p_tok = token_probs_from_class_probs_max(val_probs_cls)

best = (-1.0, None, None, None, None)
for K in K_grid:
    for pmin in pmin_grid:
        for mt in max_tokens_grid:
            pred_tok_topk = predict_tokens_from_class_probs_topk(
                val_probs_cls,
                K=K,
                p_min=pmin,
                max_tokens=mt,
                token_keep_priority=token_keep_priority,
            )
            for tokthr in tokthr_grid:
                pred_tok_thr = tokens_from_token_probs_threshold(
                    val_p_tok,
                    thr=tokthr,
                    max_tokens=mt,
                    token_keep_priority=token_keep_priority,
                )
                pred_tok_union = pred_tok_topk | pred_tok_thr
                score = mean_f1_from_multilabel(pred_tok_union, val_true_tokens)
                if score > best[0]:
                    best = (score, K, pmin, mt, tokthr)

best_val_score, best_K, best_pmin, best_max_tokens, best_tokthr = best
print(
    "Best val mean-F1 (topK-union + token-thr union):",
    best_val_score,
    "K:",
    best_K,
    "p_min:",
    best_pmin,
    "max_tokens:",
    best_max_tokens,
    "tok_thr:",
    best_tokthr,
)



## === cell 22
s_ls = []

with torch.no_grad():
    model.eval()
    for images, fnames in testloader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)  # [B, NUM_CL]
        ps_cls_batch = (
            torch.softmax(logits, dim=1).detach().cpu().numpy()
        )  # [B, NUM_CL]

        pred_tok_topk = predict_tokens_from_class_probs_topk(
            ps_cls_batch,
            K=int(best_K),
            p_min=float(best_pmin),
            max_tokens=best_max_tokens,
            token_keep_priority=token_keep_priority,
        )
        p_tok_batch = token_probs_from_class_probs_max(ps_cls_batch)
        pred_tok_thr = tokens_from_token_probs_threshold(
            p_tok_batch,
            thr=float(best_tokthr),
            max_tokens=best_max_tokens,
            token_keep_priority=token_keep_priority,
        )
        pred_tok_batch = pred_tok_topk | pred_tok_thr

        for i in range(ps_cls_batch.shape[0]):
            tok_idx = np.where(pred_tok_batch[i])[0].tolist()
            pred_labels = [all_tokens[j] for j in tok_idx]
            pred_labels = _order_tokens_canonical(pred_labels)
            pred_label_str = " ".join(pred_labels)
            s_ls.append([fnames[i], pred_label_str])



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
