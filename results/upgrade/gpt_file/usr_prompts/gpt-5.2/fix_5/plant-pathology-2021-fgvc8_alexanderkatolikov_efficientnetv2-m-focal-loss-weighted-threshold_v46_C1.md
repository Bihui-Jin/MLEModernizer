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

0.272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12452) has done: 'Your notebook currently can’t yield a Kaggle score because it references two external inputs that don’t exist in your environment (`../input/train-labeled/train.csv` and `../input/resnet-model/ResNext16.pth`). I make the smallest fixes to (1) remove the dependency on `train-labeled` by building `class_map` directly from your `LabelEncoder`, and (2) safely fall back to ImageNet weights if the `.pth` file isn’t available, so the pipeline runs end-to-end and writes a valid `submission.csv`. I also fix two inference-time issues that can hurt F1: applying `torch.exp` to raw logits (should use `softmax`) and ensuring images are RGB before normalization. These changes preserve your core approach (single-label classifier → label string submission) while producing a valid submission and moving score upward from “not yielded” toward your target.'
- What this solution (achieved 0.12459) has done: 'I fix the inference crash by filtering `X_Test` to include only actual image files and skipping any directories inside `test_images` (the current error comes from a nested `test_images/` directory being returned by `os.listdir`). I also make the train/val/test slicing robust by using `.iloc` (avoids potential indexing surprises) while keeping your exact split logic and model setup unchanged. These changes are execution-stability focused and score-neutral except that they allow the notebook to complete and generate a valid `submission.csv`. The submission writing and label mapping logic remain the same.'
- What this solution (achieved 0.20149) has done: 'Your current score is far below the target, and the main reason is that the code treats this multi-label F1 competition as single-label classification (argmax), which severely underpredicts multiple diseases per image. To move the score upward toward the target while keeping your ResNeXt + CrossEntropy training core intact, I only change the inference/post-processing to emit *space-delimited multi-label predictions* using per-class probabilities and tuned thresholds. Specifically: (1) compute softmax probabilities, (2) select all classes above a threshold (with a safe top-1 fallback), and (3) special-case `healthy` so it’s predicted only when nothing else is confident. This is a minimal change to evaluation semantics (output formatting) and should substantially increase mean F1 without changing your architecture, loss, or training loop.'
- What this solution (achieved 0.272) has done: 'Your score is far below the target, so we should push it upward with the smallest change that meaningfully improves mean F1 while keeping your single-label training intact. The biggest remaining issue is that using `softmax` forces probabilities to compete (bad for multi-label F1); switching only the *inference* probabilities to `sigmoid` (independent per class) typically boosts multi-label recall/precision balance without changing your model or loss. I also make the threshold selection slightly more conservative and keep your “don’t output healthy with other labels” rule plus a top-1 fallback to guarantee a valid label string. Finally, I keep submission alignment with `sample_submission.csv` unchanged to ensure a valid `.csv`.'

# 9. Code solution

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
healthy_id = None
for k, v in class_map.items():
    if v.strip() == "healthy":
        healthy_id = k
        break

TH = 0.42  # slightly conservative multi-label threshold for sigmoid probs to reduce overprediction
TH_HEALTHY = 0.55  # keep healthy relatively harder to predict unless it's clearly the best option

s_ls = []

with torch.no_grad():
    model.eval()
    for image, fname in testloader:
        image = image.to(DEVICE)
        logits = model(image)  # [1, NUM_CL]

        ps = torch.sigmoid(logits).squeeze(0)  # [NUM_CL], multi-label probabilities

        cand = (ps >= TH).nonzero(as_tuple=False).view(-1).tolist()

        if healthy_id is not None and healthy_id in cand:
            cand = [c for c in cand if c != healthy_id]

        if len(cand) == 0:
            top_p, top_idx = torch.max(ps, dim=0)
            top_idx = int(top_idx.item())
            top_p = float(top_p.item())

            if healthy_id is not None and top_idx == healthy_id and top_p < TH_HEALTHY:
                cand = [healthy_id]
            else:
                cand = [top_idx]

        pred_label_str = " ".join([class_map[int(i)] for i in cand])
        s_ls.append([fname[0], pred_label_str])



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
