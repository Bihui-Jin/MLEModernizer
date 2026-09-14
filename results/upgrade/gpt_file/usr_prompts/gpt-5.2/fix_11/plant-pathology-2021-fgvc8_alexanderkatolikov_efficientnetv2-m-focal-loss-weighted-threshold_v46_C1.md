# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.7485687903970452

# 6. Current score

0.26647

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12452) has done: 'Your notebook currently can’t yield a Kaggle score because it references two external inputs that don’t exist in your environment (`../input/train-labeled/train.csv` and `../input/resnet-model/ResNext16.pth`). I make the smallest fixes to (1) remove the dependency on `train-labeled` by building `class_map` directly from your `LabelEncoder`, and (2) safely fall back to ImageNet weights if the `.pth` file isn’t available, so the pipeline runs end-to-end and writes a valid `submission.csv`. I also fix two inference-time issues that can hurt F1: applying `torch.exp` to raw logits (should use `softmax`) and ensuring images are RGB before normalization. These changes preserve your core approach (single-label classifier → label string submission) while producing a valid submission and moving score upward from “not yielded” toward your target.'
- What this solution (achieved 0.12459) has done: 'I fix the inference crash by filtering `X_Test` to include only actual image files and skipping any directories inside `test_images` (the current error comes from a nested `test_images/` directory being returned by `os.listdir`). I also make the train/val/test slicing robust by using `.iloc` (avoids potential indexing surprises) while keeping your exact split logic and model setup unchanged. These changes are execution-stability focused and score-neutral except that they allow the notebook to complete and generate a valid `submission.csv`. The submission writing and label mapping logic remain the same.'
- What this solution (achieved 0.20149) has done: 'Your current score is far below the target, and the main reason is that the code treats this multi-label F1 competition as single-label classification (argmax), which severely underpredicts multiple diseases per image. To move the score upward toward the target while keeping your ResNeXt + CrossEntropy training core intact, I only change the inference/post-processing to emit *space-delimited multi-label predictions* using per-class probabilities and tuned thresholds. Specifically: (1) compute softmax probabilities, (2) select all classes above a threshold (with a safe top-1 fallback), and (3) special-case `healthy` so it’s predicted only when nothing else is confident. This is a minimal change to evaluation semantics (output formatting) and should substantially increase mean F1 without changing your architecture, loss, or training loop.'
- What this solution (achieved 0.272) has done: 'Your score is far below the target, so we should push it upward with the smallest change that meaningfully improves mean F1 while keeping your single-label training intact. The biggest remaining issue is that using `softmax` forces probabilities to compete (bad for multi-label F1); switching only the *inference* probabilities to `sigmoid` (independent per class) typically boosts multi-label recall/precision balance without changing your model or loss. I also make the threshold selection slightly more conservative and keep your “don’t output healthy with other labels” rule plus a top-1 fallback to guarantee a valid label string. Finally, I keep submission alignment with `sample_submission.csv` unchanged to ensure a valid `.csv`.'
- What this solution (achieved 0.28088) has done: 'I remove the biggest Python-side bottlenecks without altering the model/training logic: (1) speed up image input by using `torchvision.io.read_image` + tensor-based transforms (avoids slow PIL decoding), (2) make DataLoaders fast and GPU-friendly with `num_workers`, `pin_memory`, `persistent_workers`, and prefetching, and (3) vectorize the token-probability aggregation and per-token threshold search to eliminate large nested Python loops. I also avoid repeated pandas lookups in `PREDS`/`metrics` by caching label-id→tokens, and use batched inference for test instead of batch_size=1 (same outputs, far fewer forward passes). These changes are provably equivalent in semantics (same transforms, same logits→sigmoid mapping, same threshold grid search) and should bring runtime under the 600s limit.'
- What this solution (achieved 0.18013) has done: 'Your current gap to target is large (0.28088 → 0.74857), so the smallest score-relevant improvement is to align the *inference-time* multi-label probabilities with your *training* objective: CrossEntropy trains mutually-exclusive classes, so using `sigmoid(logits)` (independent probabilities) is mismatched and tends to over-predict tokens. I keep your model, loss, training loop, transforms, and token-aggregation logic intact, but change only the probability mapping at validation threshold-search and test-time prediction to `softmax(logits)` (class probabilities that sum to 1). This single semantic fix typically yields a substantial F1 increase in this specific “single-class trained, multi-label evaluated via token expansion” setup, while preserving your overall approach and still writing a valid `submission.csv`. I also keep all your post-processing rules (healthy suppression, non-empty fallback) unchanged.'
- What this solution (achieved 0.30137) has done: 'Your current score is far below the target, so we should increase it with the smallest score-relevant change while keeping your ResNeXt + CrossEntropy training intact. The main issue is that the per-token aggregation `max(class_prob)` plus a fixed threshold tends to overpredict rare tokens and hurts mean F1; a minimal improvement is to (1) calibrate token probabilities using a simple “noisy-OR” union over all classes that contain a token, and (2) tune **one global threshold** on the validation set (instead of per-token thresholds) to better match the competition’s mean F1. This preserves your core logic (single-label trained model → token expansion → thresholding + healthy suppression + non-empty fallback) but generally improves precision/recall balance and stability. I keep I/O paths and submission formatting identical, and the notebook still writes a valid `submission.csv`.'
- What this solution (achieved 0.19044) has done: 'We keep your single-label ResNeXt + CrossEntropy training setup and your token-expansion + global-threshold inference, but fix one split bug and one inference aggregation choice that are holding F1 down. First, your current train/val split overlaps heavily (both include the last 3700 rows), which makes threshold selection less reliable; we make val a disjoint slice without changing dataset size. Second, the “noisy-OR” token aggregation tends to inflate token probabilities when many classes share a token; we switch aggregation to a calibrated `max` over member classes (minimal semantic change, often improves precision/mean-F1 here). Finally, we make the threshold grid slightly denser around the typical operating region to better match mean-F1 without changing the model/training loop.'
- What this solution (achieved 0.26647) has done: 'Your current score (0.19044) is far below the target (0.74857), so we should increase it with the smallest change that corrects the main metric-mismatch while keeping your ResNeXt + CrossEntropy training untouched. The key issue is that you’re expanding a *single-label* classifier into multi-label tokens and then thresholding tokens independently; a much more reliable minimal post-process for mean F1 here is to predict the **top-K most likely label-strings** (classes) and union their tokens, with K and a class-probability cutoff tuned on the validation set. This preserves your architecture, loss, training loop, transforms, and token expansion logic, but changes only inference aggregation/thresholding to better match the competition’s multi-label mean-F1. We tune (K, p_min) on the existing validation loader, keep your “healthy suppression” and “non-empty fallback”, and still write a valid `submission.csv`.'

# 9. Code solution

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
    shuffle=True,
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



## === cell 18
_valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
X_Test = []
for name in os.listdir(TEST_DIR):
    full = os.path.join(TEST_DIR, name)
    if os.path.isfile(full) and os.path.splitext(name.lower())[1] in _valid_ext:
        X_Test.append(name)

X_Test = sorted(X_Test)

print("Found test images:", len(X_Test))
print("Example:", X_Test[:5])



## === cell 19
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




## === cell 20
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


def predict_tokens_from_class_probs_topk(p_cls, K=2, p_min=0.20):
    """
    p_cls: [N, NUM_CL] softmax class probs
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
            toks.add(
                all_tokens[int(np.argmax([t == "healthy" for t in all_tokens]))]
                if healthy_token_idx is not None
                else class_map[int(p_cls[i].argmax())].split(" ")[0]
            )

        for t in toks:
            ti = token_to_idx.get(t, None)
            if ti is not None:
                pred_tok[i, ti] = True

    return pred_tok


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

best = (-1.0, None, None)
for K in K_grid:
    for pmin in pmin_grid:
        pred_tok = predict_tokens_from_class_probs_topk(val_probs_cls, K=K, p_min=pmin)
        score = mean_f1_from_multilabel(pred_tok, val_true_tokens)
        if score > best[0]:
            best = (score, K, pmin)

best_val_score, best_K, best_pmin = best
print(
    "Best val mean-F1 (topK-union):", best_val_score, "K:", best_K, "p_min:", best_pmin
)



## === cell 21
s_ls = []

with torch.no_grad():
    model.eval()
    for images, fnames in testloader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)  # [B, NUM_CL]
        ps_cls_batch = (
            torch.softmax(logits, dim=1).detach().cpu().numpy()
        )  # [B, NUM_CL]

        pred_tok_batch = predict_tokens_from_class_probs_topk(
            ps_cls_batch, K=int(best_K), p_min=float(best_pmin)
        )

        for i in range(ps_cls_batch.shape[0]):
            pred_labels = [
                all_tokens[j] for j in np.where(pred_tok_batch[i])[0].tolist()
            ]
            pred_label_str = " ".join(pred_labels)
            s_ls.append([fnames[i], pred_label_str])



## === cell 22
pred_df = pd.DataFrame.from_records(s_ls, columns=["image", "labels"])
pred_df



## === cell 23
sub = pred_df[["image", "labels"]]
sub.head()



## === cell 24
sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
sub = sample_sub[["image"]].merge(sub, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")



## === cell 25
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
