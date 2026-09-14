# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.7627146814404437

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, json, time
import cv2, pandas as pd, numpy as np
import torch, torch.nn as nn
import torchvision  # added import for torchvision models
from torch.utils import data
from torchvision import transforms
from PIL import Image  # use Pillow for faster image loading

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # enable cuDNN auto‑tuner for speed
torch.set_float32_matmul_precision("high")  # faster matmul on GPU
USE_AMP = torch.cuda.is_available()  # only use AMP on GPU
try:
    from efficientnet_pytorch import model as enet
except Exception:
    enet = None  # fallback not needed for baseline



## === cell 1
TEST = True
VER = "v103"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
else:
    DATA_PATH = "./data"
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
if not os.path.isdir(IMGS_PATH):
    raise FileNotFoundError(f"Image directory not found: {IMGS_PATH}")



## === cell 2
from sklearn.metrics import f1_score
from pathlib import Path

train_csv_path = os.path.join(DATA_PATH, "train.csv")
df_train = pd.read_csv(train_csv_path)
all_labels = sorted({lbl for row in df_train["labels"] for lbl in str(row).split()})
label2idx = {lbl: i for i, lbl in enumerate(all_labels)}
num_classes = len(all_labels)
print(f"Detected {num_classes} unique labels: {all_labels}")


class PlantDataset(data.Dataset):
    """
    Fast dataset with optional caching:
    - Loads images with Pillow once and reuses them.
    - Creates one‑hot targets only when a 'labels' column exists.
    """

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self._cache = {}  # idx → PIL.Image

        if "labels" in self.df.columns:
            target_list = []
            for labels in self.df["labels"]:
                vec = np.zeros(num_classes, dtype=np.float32)
                for lbl in str(labels).split():
                    vec[label2idx[lbl]] = 1.0
                target_list.append(torch.from_numpy(vec))
            self.targets = torch.stack(target_list)  # (N, num_classes)
        else:
            self.targets = torch.zeros(len(self.df), num_classes, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if idx in self._cache:
            img = self._cache[idx]
        else:
            img_path = os.path.join(self.img_dir, row["image"])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            self._cache[idx] = img  # store for future epochs
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_frac = 0.1
val_size = int(len(df_train) * val_frac)
train_df = df_train.iloc[val_size:].reset_index(drop=True)
val_df = df_train.iloc[:val_size].reset_index(drop=True)

train_dataset = PlantDataset(
    train_df, os.path.join(DATA_PATH, "train_images"), transform=train_transform
)
val_dataset = PlantDataset(
    val_df, os.path.join(DATA_PATH, "train_images"), transform=val_transform
)

worker_cnt = min(8, os.cpu_count())  # more workers for faster loading
BATCH_SIZE = 256  # larger batch → fewer optimizer steps

torch.set_num_threads(os.cpu_count())  # use all available CPU threads

train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=worker_cnt,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=worker_cnt > 0,
    prefetch_factor=8,  # larger prefetch to reduce waiting
)
val_loader = data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=worker_cnt,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=worker_cnt > 0,
    prefetch_factor=8,
)

if torch.cuda.is_available() and hasattr(torchvision.models, "efficientnet_b0"):
    from torchvision import models

    backbone = models.efficientnet_b0(pretrained=True)
    backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, num_classes)
else:

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    backbone = SimpleCNN(num_classes)

model = backbone.to(DEVICE)
if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"torch.compile failed: {e}")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if USE_AMP:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None  # placeholder for CPU path


def compute_best_thresholds(y_true, y_prob):
    best_thr = np.full(num_classes, 0.5)
    best_f1 = 0.0
    for thr in np.arange(0.3, 0.71, 0.05):
        preds = (y_prob > thr).astype(int)
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr.fill(thr)
    return best_thr


def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        if USE_AMP:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch} - Train loss: {epoch_loss:.4f}")


def evaluate():
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True)
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
            else:
                outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu()
            all_targets.append(targets)
            all_probs.append(probs)
    y_true = torch.cat(all_targets, dim=0).numpy()
    y_prob = torch.cat(all_probs, dim=0).numpy()
    preds = (y_prob > 0.5).astype(int)
    f1 = f1_score(y_true, preds, average="macro")
    print(f"Validation macro F1 (0.5 thr): {f1:.4f}")
    return f1, y_true, y_prob


best_f1 = 0.0
best_state = None
best_thresholds = np.full(num_classes, 0.5)

for epoch in range(1, 3):  # two epochs
    train_one_epoch(epoch)
    val_f1, val_true, val_prob = evaluate()
    thr = compute_best_thresholds(val_true, val_prob)
    calibrated_preds = (val_prob > thr).astype(int)
    calib_f1 = f1_score(val_true, calibrated_preds, average="macro")
    print(f"Validation macro F1 with calibrated thresholds: {calib_f1:.4f}")

    if calib_f1 > best_f1:
        best_f1 = calib_f1
        best_state = model.state_dict()
        best_thresholds = thr.copy()

if best_state is not None:
    model.load_state_dict(best_state)
else:
    print("Warning: no best model captured; using last epoch weights.")

test_imgs = os.listdir(IMGS_PATH)
test_df = pd.DataFrame(test_imgs, columns=["image"])
test_dataset = PlantDataset(
    test_df, IMGS_PATH, transform=val_transform
)  # targets are dummy zeros
test_loader = data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=worker_cnt,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=worker_cnt > 0,
    prefetch_factor=8,
)

model.eval()
pred_labels = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True)
        if USE_AMP:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
        else:
            outputs = model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        for prob_vec in probs:
            idxs = np.where(prob_vec > best_thresholds)[0]
            if len(idxs) == 0:
                pred_labels.append("healthy")
            else:
                pred_labels.append(" ".join([all_labels[i] for i in idxs]))

df_sub = pd.DataFrame(os.listdir(IMGS_PATH), columns=["image"])
df_sub["labels"] = pred_labels
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 261) is killed by signal: Killed. 

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3804345360.py in <cell line: 0>()
    218 
    219 for epoch in range(1, 3):  # two epochs
--> 220     train_one_epoch(epoch)
    221     val_f1, val_true, val_prob = evaluate()
    222     thr = compute_best_thresholds(val_true, val_prob)

/tmp/ipykernel_55/3804345360.py in train_one_epoch(epoch)
    169     model.train()
    170     running_loss = 0.0
--> 171     for imgs, targets in train_loader:
    172         imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True)
    173         targets = targets.to(DEVICE, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 261) exited unexpectedly
