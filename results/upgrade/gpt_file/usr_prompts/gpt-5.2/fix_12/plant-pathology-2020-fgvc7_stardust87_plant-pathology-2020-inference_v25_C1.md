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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

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
plotly==5.24.1
plotly-express==0.4.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, time, random

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold

import timm

import warnings

warnings.filterwarnings("ignore")

DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_INPUT = DIR_INPUT

SEED = 42
N_FOLDS = 5
N_EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SIZE = (409, 273)  # (width, height) for cv2.resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_CPU = os.cpu_count() or 2
N_WORKERS = min(4, _CPU)


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


_dl_generator = torch.Generator()
_dl_generator.manual_seed(SEED)

torch.set_num_threads(min(4, _CPU))

try:
    cv2.setNumThreads(0)
except Exception:
    pass

_USE_AMP = bool(torch.cuda.is_available())
_scaler = torch.cuda.amp.GradScaler(enabled=_USE_AMP)

device



## === cell 1
_IMAGES_DIR = os.path.join(IMAGE_INPUT, "images")


def find_image_path(image_id: str) -> str:
    p1 = os.path.join(_IMAGES_DIR, f"{image_id}.jpg")
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(_IMAGES_DIR, image_id)
    if os.path.isfile(p2):
        return p2
    raise FileNotFoundError(
        f"Could not find image for image_id={image_id} at {p1} or {p2}"
    )




## === cell 2
transforms_resize_only = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], p=1.0),
    ]
)

transforms_to_tensor_norm = A.Compose(
    [
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_train = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_valid = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)



## === cell 3
_GLOBAL_RESIZED_RGB_CACHE = {}


class PlantDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
        test_set=False,
        cache_images=True,
        cache_stage="resize_rgb",
    ):
        self.df = df.reset_index(drop=True).copy()
        if "image_path" not in self.df.columns:
            self.df["image_path"] = self.df["image_id"].map(find_image_path)

        self.transforms = transforms
        self.test_set = test_set
        if not self.transforms:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

        self.image_paths = self.df["image_path"].to_numpy()

        if not self.test_set:
            self._targets = self.df[
                ["healthy", "multiple_diseases", "rust", "scab"]
            ].to_numpy(dtype=np.float32, copy=True)
        else:
            self._targets = None

        self.cache_images = bool(cache_images)
        self.cache_stage = cache_stage

        self._cache = None

        self._tensor_tf = transforms_to_tensor_norm

    def __len__(self):
        return self.image_paths.shape[0]

    def _imread_rgb(self, image_src: str):
        img = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cv2.imread failed for {image_src}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_resized_rgb(self, image_src: str):
        cached = _GLOBAL_RESIZED_RGB_CACHE.get(image_src)
        if cached is None:
            img = self._imread_rgb(image_src)
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
            _GLOBAL_RESIZED_RGB_CACHE[image_src] = img
            return img
        return cached

    def _final_transform(self, img):
        out = self._tensor_tf(image=img)["image"]
        if isinstance(out, torch.Tensor):
            out = out.contiguous()
        return out

    def __getitem__(self, idx):
        image_src = self.image_paths[idx]
        img = self._load_resized_rgb(image_src)
        image = self._final_transform(img)

        if not self.test_set:
            labels = torch.from_numpy(self._targets[idx])
            return image, labels
        else:
            return image


def _build_cache_in_main_process_from_paths(image_paths):
    for p in image_paths:
        if p in _GLOBAL_RESIZED_RGB_CACHE:
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cv2.imread failed for {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        _GLOBAL_RESIZED_RGB_CACHE[p] = img




## === cell 4
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model("resnest269e", pretrained=True)
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  # remove classifier head
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        else:
            feats = feats.flatten(1)
        x = self.logit(feats)
        return x




## === cell 5
criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running = 0.0
    for images, targets in tqdm(loader, leave=False, disable=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if targets.ndim == 2:
            targets_ce = targets.argmax(dim=1)
        else:
            targets_ce = targets

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=_USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets_ce)

        _scaler.scale(loss).backward()
        _scaler.step(optimizer)
        _scaler.update()

        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.inference_mode()
def predict_proba(model, loader):
    model.eval()
    probs_all = []
    for batch in tqdm(loader, leave=False, disable=True):
        images = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=_USE_AMP):
            logits = model(images)
            probs = F.softmax(logits, dim=1)
        probs_all.append(probs.detach().cpu().numpy())
    return np.concatenate(probs_all, axis=0)




## === cell 6
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

train_df["image_path"] = train_df["image_id"].map(find_image_path)
test_df["image_path"] = test_df["image_id"].map(find_image_path)

y_single = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values.argmax(
    axis=1
)

train_df.shape, test_df.shape



## === cell 7
_all_paths = pd.concat([train_df["image_path"], test_df["image_path"]], axis=0).unique()
_build_cache_in_main_process_from_paths(_all_paths)

dataset_test = PlantDataset(
    df=test_df, test_set=True, transforms=transforms_valid, cache_images=True
)

testloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=N_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(N_WORKERS > 0),
    prefetch_factor=4 if N_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
)

len(dataset_test)



## === cell 8
USE_FOLDS_FOR_TRAINING = (
    1  # minimal to unblock and produce a valid submission within time constraints
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(train_df, y_single))

fold_indices = list(range(USE_FOLDS_FOR_TRAINING))
fold_indices




## === cell 9
def make_loaders_for_fold(fold_idx: int):
    tr_idx, va_idx = splits[fold_idx]
    df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
    df_va = train_df.iloc[va_idx].reset_index(drop=True)

    ds_tr = PlantDataset(
        df=df_tr, test_set=False, transforms=transforms_train, cache_images=True
    )
    ds_va = PlantDataset(
        df=df_va, test_set=False, transforms=transforms_valid, cache_images=True
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=_dl_generator,
        persistent_workers=(N_WORKERS > 0),
        prefetch_factor=4 if N_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(N_WORKERS > 0),
        prefetch_factor=4 if N_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
    )
    return dl_tr, dl_va


@torch.inference_mode()
def eval_loss(model, loader, criterion):
    model.eval()
    running = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if targets.ndim == 2:
            targets_ce = targets.argmax(dim=1)
        else:
            targets_ce = targets
        with torch.cuda.amp.autocast(enabled=_USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets_ce)
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)




## === cell 10
models = []
start = time.perf_counter()


def _init_best_state_cpu(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _update_best_state_cpu_inplace(best_state_cpu, model: nn.Module):
    sd = model.state_dict()
    for k, v in sd.items():
        best_state_cpu[k].copy_(v.detach().cpu())


for fold_idx in fold_indices:
    seed_everything(SEED + fold_idx)
    dl_tr, dl_va = make_loaders_for_fold(fold_idx)

    model = PlantModel(num_classes=4).to(device)

    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_va = float("inf")
    best_state = _init_best_state_cpu(model)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss = train_one_epoch(model, dl_tr, optimizer, criterion)
        va_loss = eval_loss(model, dl_va, criterion)
        if va_loss < best_va:
            best_va = va_loss
            _update_best_state_cpu_inplace(best_state, model)

        if epoch in (1, 5, 10, 15, 20):
            print(
                f"Fold {fold_idx} Epoch {epoch}/{N_EPOCHS} - train_loss={tr_loss:.4f} valid_loss={va_loss:.4f}"
            )

    model.load_state_dict(best_state)
    models.append(model)

print(f"Finished Training in {(time.perf_counter() - start):.2f} seconds")



## === cell 11
test_probs_folds = []
start = time.perf_counter()
for m in models:
    test_probs_folds.append(predict_proba(m, testloader))

test_probs_mean = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
print(f"Finished Inference in {(time.perf_counter() - start):.2f} seconds")
test_probs_mean.shape



## === cell 12
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

submission_df = submission_df.merge(test_df[["image_id"]], on="image_id", how="right")

submission_df[target_cols] = test_probs_mean.astype(np.float32)
submission_df.to_csv("submission.csv", index=False)

submission_df.head()



## === cell 13
assert os.path.exists("submission.csv")
assert submission_df.shape[0] == test_df.shape[0]
assert list(submission_df.columns) == ["image_id"] + target_cols
submission_df.describe()



## === cell 14
submission_df
