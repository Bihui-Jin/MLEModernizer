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
import torch.optim as optim

import timm

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")




## === cell 1
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device




## === cell 2
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"


def resolve_images_dir():
    candidates = [
        os.path.join(DIR_INPUT, "images"),
        "/kaggle/input/images",
        "/kaggle/input/plant-pathology-2020-fgvc7/images",
    ]
    for c in candidates:
        if os.path.isdir(c) and len(os.listdir(c)) > 0:
            return c
    for root in ["/kaggle/input", "/kaggle/data"]:
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == "images" and any(
                f.lower().endswith(".jpg") for f in filenames
            ):
                return dirpath
    raise FileNotFoundError(
        "Could not locate images directory under /kaggle/input or /kaggle/data."
    )


IMAGES_DIR = resolve_images_dir()
IMAGES_DIR




## === cell 3
N_FOLDS = 5
N_EPOCHS = 15
BATCH_SIZE = 8
NUM_CLASSES = 4

train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(c in train_df.columns for c in ["image_id"] + target_cols)
assert list(sample_sub.columns) == ["image_id"] + target_cols

train_df["stratify_label"] = train_df[target_cols].values.argmax(1)

train_df.shape, test_df.shape, IMAGES_DIR




## === cell 4
CACHE_H, CACHE_W = (
    273,
    409,
)  # fixed size used by valid/test; train crops from this fixed canvas


def build_image_cache(image_ids, images_dir, cache_size=(CACHE_H, CACHE_W)):
    cache = {}
    cache_rs = {}
    h, w = cache_size
    for image_id in tqdm(image_ids, desc="Caching images", leave=False):
        image_src = os.path.join(images_dir, f"{image_id}.jpg")
        img = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_src}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cache[image_id] = img
        cache_rs[image_id] = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return cache, cache_rs


all_image_ids = pd.concat([train_df["image_id"], test_df["image_id"]]).unique().tolist()
image_cache, image_cache_rs = build_image_cache(all_image_ids, IMAGES_DIR)
len(image_cache), len(image_cache_rs)




## === cell 5
class PlantDataset(Dataset):
    def __init__(
        self,
        df,
        images_dir,
        transforms=None,
        test_set=False,
        image_cache=None,
        image_cache_rs=None,
        use_resized_cache=False,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transforms = transforms
        self.test_set = test_set
        self.image_cache = image_cache
        self.image_cache_rs = image_cache_rs
        self.use_resized_cache = use_resized_cache
        if self.transforms is None:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]

        if self.use_resized_cache and (self.image_cache_rs is not None):
            image = self.image_cache_rs[image_id]
        elif self.image_cache is not None:
            image = self.image_cache[image_id]
        else:
            image_src = os.path.join(self.images_dir, f"{image_id}.jpg")
            image = cv2.imread(image_src, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Image not found or unreadable: {image_src}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            labels = self.df.loc[idx, target_cols].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            return image, labels
        else:
            return image




## === cell 6
def trim_network_at_index(network, index=-1):
    assert index < 0, f"Param index must be negative. Received {index}"
    return nn.Sequential(*list(network.children())[:index])


class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model("resnest269e", pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone = trim_network_at_index(self.backbone, -1)
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.logit(x)
        return x




## === cell 7
transforms_train = A.Compose(
    [
        A.RandomResizedCrop(
            size=(273, 409), scale=(0.85, 1.0), ratio=(0.75, 1.3333), p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_valid = A.Compose(
    [
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)




## === cell 8
def make_loader(ds, batch_size, shuffle, num_workers, seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=g,
    )
    if num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    return DataLoader(ds, **kwargs)


def _maybe_compile(model: nn.Module) -> nn.Module:
    if not torch.cuda.is_available():
        return model
    try:
        return torch.compile(model, mode="max-autotune", fullgraph=False)
    except Exception:
        return model


def train_one_fold(fold, tr_idx, va_idx, out_dir="/kaggle/working"):
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    ds_tr = PlantDataset(
        tr_df,
        images_dir=IMAGES_DIR,
        transforms=transforms_train,
        test_set=False,
        image_cache=image_cache,
        image_cache_rs=image_cache_rs,
        use_resized_cache=True,  # train from fixed-size cached canvas
    )
    ds_va = PlantDataset(
        va_df,
        images_dir=IMAGES_DIR,
        transforms=transforms_valid,
        test_set=False,
        image_cache=image_cache,
        image_cache_rs=image_cache_rs,
        use_resized_cache=True,  # valid: no resize needed now
    )

    num_workers = min(4, os.cpu_count() or 2)
    dl_tr = make_loader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        seed=SEED + fold,
    )
    dl_va = make_loader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        seed=SEED + fold,
    )

    model = PlantModel(num_classes=NUM_CLASSES).to(device)
    model = _maybe_compile(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_va_loss = float("inf")
    best_path = os.path.join(out_dir, f"modelF{fold}.pth")

    for epoch in range(N_EPOCHS):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(ds_tr)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(ds_va)

        if va_loss < best_va_loss:
            best_va_loss = va_loss
            sd = model.state_dict()
            torch.save(sd, best_path)

        print(
            f"Fold {fold} | Epoch {epoch+1}/{N_EPOCHS} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | best_val={best_va_loss:.4f}"
        )

    return best_path




## === cell 9
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_paths = []

start = time.perf_counter()
for fold, (tr_idx, va_idx) in enumerate(
    skf.split(train_df, train_df["stratify_label"].values)
):
    path = os.path.join("/kaggle/working", f"modelF{fold}.pth")
    if not os.path.exists(path):
        fold_paths.append(
            train_one_fold(fold, tr_idx, va_idx, out_dir="/kaggle/working")
        )
    else:
        fold_paths.append(path)

print(f"Finished training in {(time.perf_counter() - start):.2f} seconds")
fold_paths




## === cell 10
dataset_test = PlantDataset(
    df=test_df,
    images_dir=IMAGES_DIR,
    test_set=True,
    transforms=transforms_valid,
    image_cache=image_cache,
    image_cache_rs=image_cache_rs,
    use_resized_cache=True,  # test: no resize needed now
)

num_workers = min(4, os.cpu_count() or 2)
testloader = make_loader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    seed=SEED,
)




## === cell 11
def test_model_reuse(model, model_path, testloader):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    use_amp = torch.cuda.is_available()

    test_probs = []
    with torch.no_grad():
        for batch in tqdm(testloader, total=len(testloader), leave=False):
            xb = batch.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                probs = torch.sigmoid(logits)
            test_probs.append(probs.detach().cpu().numpy())
    return np.concatenate(test_probs, axis=0)




## === cell 12
test_probs_folds = []
start = time.perf_counter()

model = PlantModel(num_classes=NUM_CLASSES).to(device)
model = _maybe_compile(model)

for fold in range(N_FOLDS):
    model_path = os.path.join("/kaggle/working", f"modelF{fold}.pth")
    test_probs_fold = test_model_reuse(model, model_path, testloader)
    test_probs_folds.append(test_probs_fold)

print(f"Finished inference in {(time.perf_counter() - start):.2f} seconds")
np.array(test_probs_folds).shape




## === cell 13
test_probs_mean = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
test_probs_mean.shape




## === cell 14
submission_df = pd.DataFrame({"image_id": test_df["image_id"].values})
submission_df[target_cols] = test_probs_mean.astype(np.float32)

submission_df = submission_df[["image_id"] + target_cols]

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head(), submission_path




## === cell 15
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id"] + target_cols
assert len(sub_check) == len(test_df)
sub_check.describe(include="all")
