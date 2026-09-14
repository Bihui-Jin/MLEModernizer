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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.738670360110804

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.24378) has done: 'I fix the two execution blockers: (1) the missing pretrained weights path by falling back to loading ImageNet weights when the custom `.pth` file isn’t available, so inference can run end-to-end; and (2) the deprecated `DataFrame.append` usage by collecting rows and building the submission DataFrame once (pandas 2.x compatible). I also make label extraction in the dataset robust to non-default indices by using `.iloc` instead of assuming the index equals `idx`, which prevents subtle misalignment bugs. These changes preserve the core approach (ResNet50 + sigmoid + thresholding) and ensure a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import glob
import matplotlib.pyplot as plt
import gc
import albumentations as A
import torchmetrics
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torchvision.models as models
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    cv2.setNumThreads(0)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.cuda.empty_cache()
gc.collect()
DEBUG = False
DIMENTION = (256, 171)  # image size (W, H)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images_names = glob.glob(test_images_path + "*.jpg")

train_data["labels"] = train_data["labels"].apply(lambda string: string.split(" "))

s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)
labels_size = len(train_labels.columns)

train_df = pd.concat([train_data[["image"]], train_labels], axis=1)



## === cell 1
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class PlantDataSet(Dataset):
    def __init__(self, dataset, images_path, transform=None):
        super(PlantDataSet, self).__init__()
        self.images_path = images_path
        self.transform = transform  # kept for API compatibility; not used

        if self.images_path is not None:
            self.image_names = dataset["image"].to_numpy()
            self.labels = dataset.drop(columns=["image"]).to_numpy(
                dtype=np.float32, copy=False
            )
            self.is_test = False
            self._base = self.images_path
            self._dummy_target = None
        else:
            self.image_paths = list(dataset)
            self.is_test = True
            self._dummy_target = torch.zeros((0,), dtype=torch.float32)

    def __getitem__(self, idx):
        if not self.is_test:
            img_name = self.image_names[idx]
            path = self._base + img_name
            labels = torch.from_numpy(self.labels[idx])
        else:
            path = self.image_paths[idx]
            labels = self._dummy_target

        img = read_image(path, mode=ImageReadMode.RGB)  # uint8, CHW, variable H/W

        img = img.unsqueeze(0).to(dtype=torch.float32).div_(255.0)
        img = F.interpolate(
            img, size=(DIMENTION[1], DIMENTION[0]), mode="bilinear", align_corners=False
        )
        img = img.squeeze(0).mul_(255.0).clamp_(0, 255).to(dtype=torch.uint8)

        return img, labels

    def __len__(self):
        return len(self.image_paths) if self.is_test else len(self.image_names)


transform = None  # normalization is done later (on device)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.1, random_state=42, shuffle=True
)
train_subset = train_df.iloc[train_idx].reset_index(drop=True)
val_subset = train_df.iloc[val_idx].reset_index(drop=True)

train_dataset = PlantDataSet(
    train_subset, "/kaggle/input/plant-pathology-2021-fgvc8/train_images/", transform
)
val_dataset = PlantDataSet(
    val_subset, "/kaggle/input/plant-pathology-2021-fgvc8/train_images/", transform
)

BS = 30


def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

NUM_WORKERS = min(8, (os.cpu_count() or 2))
PIN = torch.cuda.is_available()
PREFETCH = 4 if NUM_WORKERS > 0 else None

train_loader = DataLoader(
    train_dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH,
    worker_init_fn=_seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH,
    worker_init_fn=_seed_worker,
    generator=g,
)

test_dataset = PlantDataSet(test_images_names, None, transform)
plants_test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH,
    worker_init_fn=_seed_worker,
    generator=g,
)




## === cell 2
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)

        def _preload():
            nonlocal next_images, next_targets
            try:
                next_images, next_targets = next(it)
            except StopIteration:
                next_images = None
                next_targets = None
                return
            with torch.cuda.stream(self.stream):
                next_images = next_images.to(self.device, non_blocking=True)
                if torch.is_tensor(next_targets):
                    next_targets = next_targets.to(self.device, non_blocking=True)

        next_images = None
        next_targets = None
        _preload()
        while next_images is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            images = next_images
            targets = next_targets
            _preload()
            yield images, targets


def preprocess_on_device(images_u8: torch.Tensor) -> torch.Tensor:
    images = images_u8.to(dtype=torch.float32).div_(255.0)
    images = images.sub_(_MEAN.to(images.device)).div_(_STD.to(images.device))
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def test(test_dataloader, model):
    model.eval()
    model = model.to(device)
    preds = []
    dl = CUDAPrefetcher(test_dataloader, device)
    with torch.inference_mode():
        for images, _ in dl:
            if device.type == "cuda":
                torch.cuda.synchronize()
            images = preprocess_on_device(images)
            output = model(images)
            probabilities = torch.sigmoid(output)
            preds.append(probabilities.cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 3
def create_submission(test_images_path, predictions, threshold=0.6):
    names = np.fromiter((p.rsplit("/", 1)[-1] for p in test_images_path), dtype=object)
    cls = np.array(train_labels.columns.tolist(), dtype=object)

    pred_mask = predictions > threshold  # strict '>' preserved
    any_pos = pred_mask.any(axis=1)

    labels_out = np.empty(pred_mask.shape[0], dtype=object)
    labels_out[~any_pos] = "healthy"

    pos_rows = np.flatnonzero(any_pos)
    for i in pos_rows:
        idx = np.flatnonzero(pred_mask[i])
        labels_out[i] = " ".join(cls[idx].tolist())

    submission_df = pd.DataFrame({"image": names, "labels": labels_out})
    submission_df.to_csv("submission.csv", index=False)
    return submission_df




## === cell 4
weights_path = "../input/resnet50-final/resnet50_final.pth"

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_features = resnet50.fc.in_features
resnet50.fc = torch.nn.Linear(in_features, labels_size)

if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    try:
        resnet50.load_state_dict(state, strict=True)
    except Exception:
        pass




## === cell 5
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    dl = CUDAPrefetcher(loader, device)
    for images, targets in dl:
        if device.type == "cuda":
            torch.cuda.synchronize()
        images = preprocess_on_device(images)
        targets = targets  # already on device via prefetcher
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    all_targets = []
    dl = CUDAPrefetcher(loader, device)
    for images, targets in dl:
        if device.type == "cuda":
            torch.cuda.synchronize()
        images = preprocess_on_device(images)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def macro_f1_from_probs(y_true, y_prob, thr):
    y_pred = y_prob >= thr
    yt = y_true.astype(np.int32) == 1
    yp = y_pred

    tp = np.sum(yt & yp, axis=0).astype(np.float64)
    fp = np.sum((~yt) & yp, axis=0).astype(np.float64)
    fn = np.sum(yt & (~yp), axis=0).astype(np.float64)

    eps = 1e-9
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return float(np.mean(f1))


resnet50 = resnet50.to(device)
if device.type == "cuda":
    resnet50 = resnet50.to(memory_format=torch.channels_last)

if device.type == "cuda":
    try:
        resnet50 = torch.compile(resnet50, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=2e-4)

EPOCHS = 2
for epoch in range(EPOCHS):
    loss = train_one_epoch(resnet50, train_loader, optimizer, criterion)
    print(f"epoch {epoch+1}/{EPOCHS} - train_loss: {loss:.5f}")

val_probs, val_true = predict_probs(resnet50, val_loader)
threshold_grid = np.round(np.arange(0.30, 0.71, 0.05), 2)
best_thr = 0.6
best_f1 = -1.0
for thr in threshold_grid:
    f1 = macro_f1_from_probs(val_true, val_probs, thr)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = float(thr)
print(f"selected_threshold: {best_thr} (val_macro_f1={best_f1:.5f})")

test_predictions = test(plants_test_data_loader, resnet50)
print(test_predictions.shape)
submission_df = create_submission(
    test_images_names, test_predictions, threshold=best_thr
)
print(submission_df.head())
print("Wrote submission.csv")
