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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.12

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
wandb==0.21.0

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

1.38008

# 6. Current score

0.38336

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.38336) has done: 'Diagnosis: Cell 5 raises intentionally because `test_list` is picking up 15,000 `.jpg` files (from multiple nested `test/**` locations under `/kaggle`, including duplicates), while the competition `sample_submission.csv` expects exactly 2,500 unique test images. The root cause is that the fallback globbing in cell 5 is too broad and aggregates multiple copies of the test set across mirrored folders, so the length check fails.  
Patch summary: In cell 5 only, keep the same discovery logic but, when a `sample_submission.csv` is available, restrict `test_list` to the exact set of IDs in that file by mapping `{id -> path}` and selecting only those IDs (and deduplicating). This preserves downstream semantics while making `test_list` match the expected 2,500 files deterministically.  
Updated cells: Only cell 5 is modified.  
Compatibility notes for cell k+1: `test_list` remains a sorted list of image file paths; `train_list` and `labels` are unchanged. Cell 6 continues to split `train_list` with `labels` exactly as before.  
Assumptions: The Kaggle test image filenames are numeric (e.g., `900.jpg`) and `sample_submission.csv` contains those numeric IDs; at least one copy of each required test image exists somewhere under the searched paths.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
import zipfile
import glob
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ.setdefault("PYTHONHASHSEED", "0")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"
device



## === cell 2
train_dir = "train"
test_dir = "test"


def _extract_if_needed(zip_path, out_dir, expected_glob):
    if glob.glob(expected_glob):
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


_extract_if_needed(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip",
    "",
    os.path.join(train_dir, "*.jpg"),
)
_extract_if_needed(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip",
    "",
    os.path.join(test_dir, "*.jpg"),
)

train_list = glob.glob(os.path.join(train_dir, "*.jpg"))
test_list = glob.glob(os.path.join(test_dir, "*.jpg"))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")



## === cell 3
import os
import glob


def _pick_first_existing(patterns):
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches
    return []


train_list = _pick_first_existing(
    [
        os.path.join("train", "*.jpg"),
        os.path.join("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train", "*.jpg"),
        os.path.join("/kaggle/data", "train", "*.jpg"),
    ]
)

test_list = _pick_first_existing(
    [
        os.path.join("test", "test", "*.jpg"),
        os.path.join(
            "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test", "test", "*.jpg"
        ),
        os.path.join("/kaggle/data", "test", "test", "*.jpg"),
        os.path.join("test", "*.jpg"),
        os.path.join("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test", "*.jpg"),
        os.path.join("/kaggle/data", "test", "*.jpg"),
    ]
)

if not train_list:
    train_list = glob.glob(
        os.path.join("/kaggle", "**", "train", "*.jpg"), recursive=True
    )

if not test_list:
    candidates = glob.glob(
        os.path.join("/kaggle", "**", "test", "*.jpg"), recursive=True
    )
    candidates = [
        p for p in candidates if os.path.splitext(os.path.basename(p))[0].isdigit()
    ]
    test_list = sorted(set(candidates))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

train_list[0] if len(train_list) > 0 else None




## === cell 4
def _label_from_path(p):
    base = os.path.basename(p)
    parts = base.split(".")
    if len(parts) >= 2 and parts[0] in ("dog", "cat"):
        return parts[0]
    parent = os.path.basename(os.path.dirname(p))
    if parent in ("dog", "cat"):
        return parent
    return parts[0]


labels = [_label_from_path(path) for path in train_list]
len(labels)



## === cell 5
if len(train_list) == 0:
    candidates = []
    for p in [
        os.path.join("train", "*", "*.jpg"),
        os.path.join(
            "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train", "*", "*.jpg"
        ),
        os.path.join("/kaggle/data", "train", "*", "*.jpg"),
        os.path.join("/kaggle", "**", "train", "*", "*.jpg"),
    ]:
        candidates.extend(glob.glob(p, recursive=True))
    candidates = sorted(set(candidates))
    if candidates:
        train_list = candidates
        labels = [_label_from_path(path) for path in train_list]

if len(test_list) == 0:
    test_candidates = []
    for p in [
        os.path.join("test", "test", "*.jpg"),
        os.path.join("test", "unknown", "*.jpg"),
        os.path.join("test", "*.jpg"),
        os.path.join("/kaggle/data", "test", "test", "*.jpg"),
        os.path.join("/kaggle/data", "test", "unknown", "*.jpg"),
        os.path.join("/kaggle/data", "test", "*.jpg"),
        os.path.join(
            "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test", "test", "*.jpg"
        ),
        os.path.join(
            "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test", "unknown", "*.jpg"
        ),
        os.path.join("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test", "*.jpg"),
        os.path.join("/kaggle", "**", "test", "test", "*.jpg"),
        os.path.join("/kaggle", "**", "test", "unknown", "*.jpg"),
        os.path.join("/kaggle", "**", "test", "*.jpg"),
    ]:
        test_candidates.extend(glob.glob(p, recursive=True))

    if not test_candidates:
        test_candidates.extend(
            glob.glob(
                os.path.join("/kaggle", "**", "test", "**", "*.jpg"), recursive=True
            )
        )

    test_candidates = [
        p
        for p in set(test_candidates)
        if os.path.splitext(os.path.basename(p))[0].isdigit()
        and os.path.splitext(p)[1].lower() == ".jpg"
    ]
    test_list = sorted(
        test_candidates, key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )

if len(train_list) == 0:
    raise ValueError(
        "train_list is empty; no training images were found. Check extraction/path globs in earlier cells."
    )

sample_path = _pick_first_existing(
    [
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "sample_submission.csv",
    ]
)
if sample_path:
    sample_sub = pd.read_csv(sample_path[0])
    expected_n_test = len(sample_sub)

    if len(test_list) != expected_n_test:
        id_to_path = {}
        for p in test_list:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.isdigit():
                k = int(stem)
                if k not in id_to_path:
                    id_to_path[k] = p

        desired_ids = sample_sub["id"].astype(int).tolist()
        filtered = [id_to_path[i] for i in desired_ids if i in id_to_path]

        if len(filtered) == expected_n_test:
            test_list = filtered
        else:
            raise ValueError(
                f"Found {len(test_list)} test images, but sample_submission expects {expected_n_test}. "
                f"Could only match {len(filtered)}/{expected_n_test} IDs from sample_submission; "
                f"your test_list glob likely points to the wrong folder."
            )

if os.environ.get("SHOW_SAMPLES", "0") == "1":
    n_show = min(9, len(train_list))
    random_idx = np.random.randint(0, len(train_list), size=n_show)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for idx, ax in zip(random_idx, axes.ravel()):
        img = Image.open(train_list[idx])
        ax.set_title(labels[idx])
        ax.imshow(img)

    for ax in axes.ravel()[n_show:]:
        ax.axis("off")


## === cell 6
train_list, valid_list = train_test_split(
    train_list,
    test_size=0.2,
    stratify=labels,
    random_state=0,
)



## === cell 7
print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")



## === cell 8
SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transforms = transforms.Compose(
    [
        transforms.Resize((SIZE, SIZE)),
        transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)




## === cell 9
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None, return_id=False):
        self.file_list = list(file_list)
        self.transform = transform
        self.filelength = len(self.file_list)
        self.return_id = return_id

        if self.return_id:
            self._ids = np.fromiter(
                (int(os.path.splitext(os.path.basename(p))[0]) for p in self.file_list),
                dtype=np.int32,
                count=self.filelength,
            )
            self._labels = None
        else:
            self._labels = np.fromiter(
                (1 if _label_from_path(p) == "dog" else 0 for p in self.file_list),
                dtype=np.int64,
                count=self.filelength,
            )
            self._ids = None

    def __len__(self):
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_id:
            return img, int(self._ids[idx])
        return img, int(self._labels[idx])




## === cell 10
train_data = CatsDogsDataset(train_list, transform=train_transforms, return_id=False)
valid_data = CatsDogsDataset(valid_list, transform=test_transforms, return_id=False)
test_data = CatsDogsDataset(test_list, transform=test_transforms, return_id=True)



## === cell 11
pass



## === cell 12
CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = min(8, CPU_COUNT)
NUM_WORKERS



## === cell 13
batch_size = 64
pin_memory = device == "cuda"


def _seed_worker(worker_id):
    seed = 0 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


g = torch.Generator()
g.manual_seed(0)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
    in_order=False if NUM_WORKERS > 0 else True,
)
valid_loader = DataLoader(
    dataset=valid_data,
    batch_size=batch_size,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
    in_order=False if NUM_WORKERS > 0 else True,
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
    in_order=False if NUM_WORKERS > 0 else True,
)



## === cell 14
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.batch1 = nn.BatchNorm2d(96)
        self.maxPool = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.batch2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.batch3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.batch4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.batch5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(5 * 5 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.fc4 = nn.Linear(1000, 256)
        self.fc5 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.maxPool(self.relu(self.batch1(self.conv1(x))))
        x = self.maxPool(self.relu(self.batch2(self.conv2(x))))
        x = self.dropout(self.relu(self.batch3(self.conv3(x))))
        x = self.dropout(self.relu(self.batch4(self.conv4(x))))
        x = self.dropout(self.relu(self.batch5(self.conv5(x))))
        x = self.maxPool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        return self.fc5(x)


net = AlexNet().to(device)



## === cell 15
import torch.optim as optim

learning_rate = 0.005
weight_decay = 0.00001
momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum
)




## === cell 16
def _run_epoch_train(net, loader, optimizer, criterion, device):
    net.train()
    running_loss = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = net(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        running_loss += loss.detach().item() * bs
        n += bs
    return running_loss / max(n, 1)


@torch.no_grad()
def _run_epoch_valid_logloss(net, loader, device):
    net.eval()
    losses = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = net(X)
        loss = F.cross_entropy(logits, y, reduction="sum")
        losses += loss.item()
        n += X.size(0)
    return losses / max(n, 1)


EPOCHS = (
    1  # minimal change to get a meaningful (non-random) model within time constraints
)

for epoch in range(EPOCHS):
    train_loss = _run_epoch_train(net, train_loader, optimizer, criterion, device)
    val_logloss = _run_epoch_valid_logloss(net, valid_loader, device)
    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_logloss: {val_logloss:.4f}"
    )



## === cell 17
net.eval()
test_probs = np.empty(len(test_data), dtype=np.float32)
test_ids = np.empty(len(test_data), dtype=np.int32)

offset = 0
with torch.no_grad():
    for X, ids in test_loader:
        b = X.size(0)
        X = X.to(device, non_blocking=True)
        output = F.softmax(net(X), dim=1)[..., 1].detach().cpu().numpy()
        test_probs[offset : offset + b] = output
        test_ids[offset : offset + b] = np.asarray(ids, dtype=np.int32)
        offset += b

out_df = pd.DataFrame({"id": test_ids, "label": test_probs})
out_df = out_df.sort_values("id").reset_index(drop=True)

if sample_path:
    if len(out_df) != len(sample_sub):
        raise ValueError(
            f"submission has {len(out_df)} rows but sample_submission has {len(sample_sub)} rows."
        )

out_df.to_csv("submission.csv", index=False)

print(out_df.head())
print("Wrote submission.csv with", len(out_df), "rows")



## === cell 18
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

valid_labels = valid_data._labels.astype(np.int64, copy=False)

net.eval()
val_pred = np.empty(len(valid_labels), dtype=np.int64)

offset = 0
with torch.no_grad():
    for X, y in valid_loader:
        b = X.size(0)
        X = X.to(device, non_blocking=True)
        pred = F.softmax(net(X), dim=1).argmax(1).detach().cpu().numpy()
        val_pred[offset : offset + b] = pred
        offset += b

confusion_matrix = metrics.confusion_matrix(valid_labels, val_pred)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=["cat", "dog"]
)
cm_display.plot()
plt.show()



## === cell 19
metrics.accuracy_score(valid_labels, val_pred)
