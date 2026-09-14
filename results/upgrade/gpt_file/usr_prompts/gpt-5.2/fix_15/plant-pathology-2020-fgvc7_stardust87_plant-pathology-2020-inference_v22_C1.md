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

# 5. Target score

0.9694673783378412

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

device



## === cell 2
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"


def resolve_images_dir():
    candidates = [
        os.path.join(DIR_INPUT, "images"),
        "/kaggle/input/plant-pathology-2020-fgvc7/images",
        "/kaggle/input/images",
        "/kaggle/data/plant-pathology-2020-fgvc7/images",
        "/kaggle/data/images",
    ]
    for c in candidates:
        if os.path.isdir(c):
            try:
                for f in os.listdir(c):
                    if f.lower().endswith(".jpg"):
                        return c
            except Exception:
                pass
    raise FileNotFoundError(
        "Could not locate images directory under expected Kaggle competition paths."
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
try:
    cv2.setNumThreads(0)
except Exception:
    pass

CACHE_H, CACHE_W = (273, 409)  # fixed canvas used for caching


def build_resized_image_cache_rgb_uint8_hwc(
    image_ids, images_dir, cache_size=(CACHE_H, CACHE_W)
):
    h, w = cache_size
    cache = {}
    for image_id in tqdm(
        image_ids, desc="Caching resized RGB uint8 images", leave=False
    ):
        image_src = os.path.join(images_dir, f"{image_id}.jpg")
        data = None
        try:
            data = np.fromfile(image_src, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            img = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_src}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        cache[image_id] = img  # numpy uint8 HWC RGB
    return cache


def build_image_cache_rgb_uint8_hwc(image_ids, images_dir):
    cache = {}
    for image_id in tqdm(
        image_ids, desc="Caching original RGB uint8 images", leave=False
    ):
        image_src = os.path.join(images_dir, f"{image_id}.jpg")
        data = None
        try:
            data = np.fromfile(image_src, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            img = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_src}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cache[image_id] = img  # numpy uint8 HWC RGB (original size)
    return cache


image_cache = None
image_cache_rs = None

len(train_df), len(test_df)



## === cell 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(
    1, 3, 1, 1
)
_STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)


def normalize_batch_imagenet(x: torch.Tensor) -> torch.Tensor:
    return (x - _MEAN_T) / _STD_T




## === cell 6
class IdentityTensorTransform:
    def __call__(self, image):
        return {"image": image}


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
        image_tensor_cache=None,
        lazy_tensor_cache=False,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transforms = transforms
        self.test_set = test_set
        self.image_cache = image_cache
        self.image_cache_rs = image_cache_rs
        self.use_resized_cache = use_resized_cache
        self.image_tensor_cache = image_tensor_cache
        self.lazy_tensor_cache = lazy_tensor_cache

        if self.transforms is None:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

        self.image_ids = self.df["image_id"].values
        if not self.test_set:
            self.labels = self.df[target_cols].values.astype(np.float32)
        else:
            self.labels = None

        if self.lazy_tensor_cache and (self.image_tensor_cache is None):
            self.image_tensor_cache = {}

    def __len__(self):
        return self.df.shape[0]

    def _read_from_disk_rgb(self, image_id):
        image_src = os.path.join(self.images_dir, f"{image_id}.jpg")
        data = None
        try:
            data = np.fromfile(image_src, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None:
            image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        if self.image_tensor_cache is not None and image_id in self.image_tensor_cache:
            image = self.image_tensor_cache[image_id]
        else:
            if self.use_resized_cache and (self.image_cache_rs is not None):
                img = self.image_cache_rs[image_id]
                transformed = (
                    self.transforms(image=img)
                    if self.transforms is not None
                    else {"image": img}
                )
                image = transformed["image"]
            elif self.image_cache is not None:
                img = self.image_cache[image_id]
                transformed = self.transforms(image=img)
                image = transformed["image"]
            else:
                img = self._read_from_disk_rgb(image_id)
                transformed = self.transforms(image=img)
                image = transformed["image"]

            if self.lazy_tensor_cache and (self.image_tensor_cache is not None):
                self.image_tensor_cache[image_id] = image

        if not self.test_set:
            labels = torch.from_numpy(self.labels[idx])
            return image, labels
        else:
            return image




## === cell 7
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




## === cell 8
transforms_train = A.Compose(
    [
        A.RandomResizedCrop(
            size=(273, 409), scale=(0.85, 1.0), ratio=(0.75, 1.3333), p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0),  # uint8 -> float32 in [0,1], CHW
    ]
)

transforms_valid = A.Compose(
    [
        ToTensorV2(p=1.0),  # uint8 -> float32 in [0,1], CHW
    ]
)




## === cell 9
def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _fast_collate(batch):
    if isinstance(batch[0], tuple):
        xs, ys = zip(*batch)
        x = torch.stack(xs, 0)
        y = torch.stack(ys, 0)
        if torch.cuda.is_available():
            x = x.pin_memory()
            y = y.pin_memory()
        return x, y
    else:
        x = torch.stack(batch, 0)
        if torch.cuda.is_available():
            x = x.pin_memory()
        return x


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
        worker_init_fn=_seed_worker,
        collate_fn=_fast_collate,
    )
    if torch.cuda.is_available():
        kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=8))
    return DataLoader(ds, **kwargs)


def _maybe_compile(model: nn.Module) -> nn.Module:
    if hasattr(torch, "compile") and torch.cuda.is_available():
        try:
            return torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            return model
    return model


ALL_IMAGE_IDS = (
    pd.concat([train_df["image_id"], test_df["image_id"]], axis=0).unique().tolist()
)

image_cache = build_image_cache_rgb_uint8_hwc(ALL_IMAGE_IDS, IMAGES_DIR)
image_cache_rs_all = build_resized_image_cache_rgb_uint8_hwc(ALL_IMAGE_IDS, IMAGES_DIR)


def precompute_valid_tensors(va_df, image_cache_rs_local):
    ids = va_df["image_id"].values
    labels = va_df[target_cols].values.astype(np.float32)
    X = torch.empty((len(ids), 3, CACHE_H, CACHE_W), dtype=torch.float32)
    Y = torch.from_numpy(labels)
    for i, image_id in enumerate(ids):
        img = image_cache_rs_local[image_id]  # uint8 HWC RGB already resized
        X[i] = transforms_valid(image=img)["image"]
    X = (
        X - torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    ) / torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return X, Y


class PrecomputedTensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_one_fold(
    fold,
    tr_idx,
    va_idx,
    out_dir="/kaggle/working",
    image_cache_rs_local=None,
):
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    if image_cache_rs_local is None:
        image_cache_rs_local = image_cache_rs_all

    ds_tr = PlantDataset(
        tr_df,
        images_dir=IMAGES_DIR,
        transforms=transforms_train,
        test_set=False,
        image_cache=image_cache,  # CHANGE: use original-size cache to avoid disk I/O
        image_cache_rs=image_cache_rs_local,
        use_resized_cache=False,  # CHANGE: keep RandomResizedCrop semantics (crop from original size)
        image_tensor_cache=None,  # keep None: training augmentations must stay random each access
        lazy_tensor_cache=False,
    )

    X_va, Y_va = precompute_valid_tensors(va_df, image_cache_rs_local)
    ds_va = PrecomputedTensorDataset(X_va, Y_va)

    cpu = os.cpu_count() or 2
    num_workers = 6 if cpu >= 16 else (4 if cpu >= 8 else (2 if cpu >= 4 else 0))

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
        num_workers=0,
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
        tr_loss_sum = 0.0

        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb = normalize_batch_imagenet(xb)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss_sum += loss.item() * xb.size(0)

        tr_loss = tr_loss_sum / len(ds_tr)

        model.eval()
        va_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                va_loss_sum += loss.item() * xb.size(0)

        va_loss = va_loss_sum / len(ds_va)

        if va_loss < best_va_loss:
            best_va_loss = va_loss
            torch.save(model.state_dict(), best_path)

        print(
            f"Fold {fold} | Epoch {epoch+1}/{N_EPOCHS} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | best_val={best_va_loss:.4f}"
        )

    return best_path




## === cell 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_paths = []

start = time.perf_counter()
for fold, (tr_idx, va_idx) in enumerate(
    skf.split(train_df, train_df["stratify_label"].values)
):
    path = os.path.join("/kaggle/working", f"modelF{fold}.pth")
    if not os.path.exists(path):
        fold_paths.append(
            train_one_fold(
                fold,
                tr_idx,
                va_idx,
                out_dir="/kaggle/working",
                image_cache_rs_local=image_cache_rs_all,
            )
        )
    else:
        fold_paths.append(path)

print(f"Finished training in {(time.perf_counter() - start):.2f} seconds")
fold_paths



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3932710371.py in <cell line: 0>()
      9     if not os.path.exists(path):
     10         fold_paths.append(
---> 11             train_one_fold(
     12                 fold,
     13                 tr_idx,

/tmp/ipykernel_55/732852513.py in train_one_fold(fold, tr_idx, va_idx, out_dir, image_cache_rs_local)
    158         tr_loss_sum = 0.0
    159 
--> 160         for xb, yb in dl_tr:
    161             xb = xb.to(device, non_blocking=True)
    162             yb = yb.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/732852513.py", line 17, in _fast_collate
    x = x.pin_memory()
        ^^^^^^^^^^^^^^
RuntimeError: CUDA error: initialization error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.



## === cell 11
dataset_test = PlantDataset(
    df=test_df,
    images_dir=IMAGES_DIR,
    test_set=True,
    transforms=transforms_valid,
    image_cache=image_cache,
    image_cache_rs=image_cache_rs_all,
    use_resized_cache=True,
    image_tensor_cache=None,
    lazy_tensor_cache=True,
)

cpu = os.cpu_count() or 2
num_workers = 6 if cpu >= 16 else (4 if cpu >= 8 else (2 if cpu >= 4 else 0))
testloader = make_loader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=max(0, num_workers // 2),
    seed=SEED,
)




## === cell 12
def test_model_reuse(model, model_path, testloader):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model checkpoint: {model_path}. Training likely failed earlier."
        )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    use_amp = torch.cuda.is_available()

    test_probs = []
    with torch.inference_mode():
        for batch in tqdm(testloader, total=len(testloader), leave=False):
            xb = batch.to(device, non_blocking=True)
            xb = normalize_batch_imagenet(xb)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                probs = torch.sigmoid(logits)
            test_probs.append(probs.detach().cpu().numpy())
    return np.concatenate(test_probs, axis=0)




## === cell 13
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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1846735546.py in <cell line: 0>()
      7 for fold in range(N_FOLDS):
      8     model_path = os.path.join("/kaggle/working", f"modelF{fold}.pth")
----> 9     test_probs_fold = test_model_reuse(model, model_path, testloader)
     10     test_probs_folds.append(test_probs_fold)
     11 

/tmp/ipykernel_55/2562796607.py in test_model_reuse(model, model_path, testloader)
      1 def test_model_reuse(model, model_path, testloader):
      2     if not os.path.exists(model_path):
----> 3         raise FileNotFoundError(
      4             f"Missing model checkpoint: {model_path}. Training likely failed earlier."
      5         )

FileNotFoundError: Missing model checkpoint: /kaggle/working/modelF0.pth. Training likely failed earlier.

## === cell 14
test_probs_mean = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
test_probs_mean.shape



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2681119021.py in <cell line: 0>()
----> 1 test_probs_mean = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
      2 test_probs_mean.shape
      3 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 15
assert len(test_probs_mean) == len(
    test_df
), "Prediction row count must match test.csv rows."

submission_df = pd.DataFrame({"image_id": test_df["image_id"].values})
submission_df[target_cols] = test_probs_mean.astype(np.float32)
submission_df = submission_df[["image_id"] + target_cols]

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head(), submission_path



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/274022586.py in <cell line: 0>()
----> 1 assert len(test_probs_mean) == len(
      2     test_df
      3 ), "Prediction row count must match test.csv rows."
      4 
      5 submission_df = pd.DataFrame({"image_id": test_df["image_id"].values})

NameError: name 'test_probs_mean' is not defined

## === cell 16
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id"] + target_cols
assert len(sub_check) == len(test_df)
assert (sub_check["image_id"].values == test_df["image_id"].values).all()
sub_check.describe(include="all")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/2404215315.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 sub_check = pd.read_csv("submission.csv")
      3 assert list(sub_check.columns) == ["image_id"] + target_cols
      4 assert len(sub_check) == len(test_df)
      5 assert (sub_check["image_id"].values == test_df["image_id"].values).all()

AssertionError:
