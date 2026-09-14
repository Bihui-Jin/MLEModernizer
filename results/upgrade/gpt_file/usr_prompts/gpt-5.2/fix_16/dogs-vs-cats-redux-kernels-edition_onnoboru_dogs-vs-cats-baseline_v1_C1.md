# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

albumentations==2.0.8
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
sentence-transformers==4.1.0
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
transformers==4.53.3

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

# 5. Code solution

## === cell 0
import os
import gc
import re
import sys
import time
import copy
import random
import glob

import zipfile
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import transformers

from tqdm import tqdm

from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

import torchvision
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

from functools import lru_cache

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False




## === cell 1
class CFG:
    debug_one_epoch = False
    debug_one_fold = False

    only_infer = True

    num_workers = 16
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    early_stopping_round = 5
    random_seed = 42
    n_splits = 5
    model_name = "resnet18"  # timm model name
    pretrained_path = None
    train_dir = None
    test_dir = None
    optimizer = torch.optim.AdamW
    criterion = nn.BCEWithLogitsLoss()
    scheduler = transformers.get_linear_schedule_with_warmup
    input_imgsize = 224

    data_dir = "../input/dogs-vs-cats-redux-kernels-edition/"
    kaggle_working_dir = "/kaggle/working/"

    infer_num_folds = 2  # <= n_splits

    prob_smooth_alpha = 0.08

    use_torch_compile = False

    max_decode_cache = 50000  # enough for full train+test in most cases


def seed_torch(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CFG.random_seed)

if CFG.debug_one_epoch:
    CFG.num_epochs = 1

print("KAGGLE_URL_BASE" in set(os.environ.keys()))




## === cell 2
def resolve_data_root():
    candidates = [
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/dogs-vs-cats-redux-kernels-edition",
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition",
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition/dogs-vs-cats-redux-kernels-edition",
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition",
        "/kaggle/working",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "sample_submission.csv")):
            return c
    return CFG.data_dir


CFG.data_dir = resolve_data_root()
print("Resolved CFG.data_dir =", CFG.data_dir)

submission = pd.read_csv(os.path.join(CFG.data_dir, "sample_submission.csv"))
submission.head(3)




## === cell 3
def maybe_unpack_archives(data_root: str, working_root: str):
    train_zip = os.path.join(data_root, "train.zip")
    test_zip = os.path.join(data_root, "test.zip")

    def has_any_images_fast(root: str) -> bool:
        candidates = [
            os.path.join(root, "train"),
            os.path.join(root, "test"),
            os.path.join(root, "train", "train"),
            os.path.join(root, "test", "unknown"),
            os.path.join(root, "test", "test", "unknown"),
        ]
        for d in candidates:
            if os.path.isdir(d):
                try:
                    with os.scandir(d) as it:
                        for e in it:
                            if e.is_file() and e.name.lower().endswith(".jpg"):
                                return True
                except FileNotFoundError:
                    pass
        return False

    if (os.path.exists(train_zip) or os.path.exists(test_zip)) and (
        not has_any_images_fast(data_root)
    ):
        os.makedirs(working_root, exist_ok=True)
        if os.path.exists(train_zip):
            shutil.unpack_archive(train_zip, working_root)
        if os.path.exists(test_zip):
            shutil.unpack_archive(test_zip, working_root)
        return working_root
    return data_root


CFG.data_dir = maybe_unpack_archives(CFG.data_dir, CFG.kaggle_working_dir)

CFG.train_dir = os.path.join(CFG.data_dir, "train")
CFG.test_dir = os.path.join(CFG.data_dir, "test")
print("CFG.train_dir =", CFG.train_dir)
print("CFG.test_dir  =", CFG.test_dir)




## === cell 4
def find_train_images(root):
    patterns = [
        os.path.join(root, "train", "*.jpg"),
        os.path.join(root, "train", "*", "*.jpg"),
        os.path.join(root, "train", "train", "*.jpg"),
        os.path.join(root, "train", "train", "*", "*.jpg"),
    ]
    files_set = set()
    for p in patterns:
        files_set.update(glob.glob(p))
    return sorted(files_set)


def find_test_images(root):
    patterns_prefer_unknown = [
        os.path.join(root, "test", "unknown", "*.jpg"),
        os.path.join(root, "test", "test", "unknown", "*.jpg"),
        os.path.join(
            root, "dogs-vs-cats-redux-kernels-edition", "test", "unknown", "*.jpg"
        ),
        os.path.join(
            root,
            "dogs-vs-cats-redux-kernels-edition",
            "test",
            "test",
            "unknown",
            "*.jpg",
        ),
    ]
    files_unknown = set()
    for p in patterns_prefer_unknown:
        files_unknown.update(glob.glob(p))
    if len(files_unknown) > 0:
        return sorted(files_unknown)

    patterns = [
        os.path.join(root, "test", "*.jpg"),
        os.path.join(root, "test", "*", "*.jpg"),
        os.path.join(root, "test", "test", "*.jpg"),
        os.path.join(root, "test", "test", "*", "*.jpg"),
    ]
    files_set = set()
    for p in patterns:
        files_set.update(glob.glob(p))
    return sorted(files_set)


train_list = find_train_images(CFG.data_dir)
test_list = find_test_images(CFG.data_dir)

if len(train_list) == 0 or len(test_list) == 0:
    nested = os.path.join(CFG.data_dir, "dogs-vs-cats-redux-kernels-edition")
    if os.path.isdir(nested):
        train_list = find_train_images(nested)
        test_list = find_test_images(nested)
        if len(train_list) > 0 and len(test_list) > 0:
            CFG.data_dir = nested
            CFG.train_dir = os.path.join(CFG.data_dir, "train")
            CFG.test_dir = os.path.join(CFG.data_dir, "test")

print(f"train data : {len(train_list)}")
print(f"test data : {len(test_list)}")

if len(train_list) == 0 or len(test_list) == 0:
    raise FileNotFoundError(
        f"Could not find train/test images under CFG.data_dir={CFG.data_dir}. "
        f"Found train={len(train_list)} test={len(test_list)}."
    )




## === cell 5
dog_cnt = 0
cat_cnt = 0
for p in train_list:
    b = os.path.basename(p)
    if "dog" in b:
        dog_cnt += 1
    elif "cat" in b:
        cat_cnt += 1
print("the number of dog : ", dog_cnt)
print("the number of cat : ", cat_cnt)




## === cell 6
if False:
    from PIL import Image

    random_img = random.choice(train_list)
    img = Image.open(random_img).convert("RGB")
    print(random_img)
    print(img.size)
    plt.imshow(img)




## === cell 7
if False:
    img_array = np.array(img)
    print(img_array.shape)




## === cell 8
if False:
    print(img_array[:, :, 0])




## === cell 9
if False:
    transform_tmp = A.Compose([A.Resize(CFG.input_imgsize, CFG.input_imgsize)])
    img_transformed_tmp = transform_tmp(image=np.array(img))
    print(img_transformed_tmp.keys())
    img_transformed_tmp = Image.fromarray(img_transformed_tmp["image"])
    print(img_transformed_tmp.size)
    plt.imshow(img_transformed_tmp)




## === cell 10
if False:
    transform_tmp = A.Compose(
        [A.Resize(CFG.input_imgsize, CFG.input_imgsize), A.HorizontalFlip(p=1.0)]
    )
    img_transformed_tmp = transform_tmp(image=np.array(img))
    img_transformed_tmp = Image.fromarray(img_transformed_tmp["image"])
    plt.imshow(img_transformed_tmp)




## === cell 11
if False:
    transform_tmp = A.Compose(
        [A.Resize(CFG.input_imgsize, CFG.input_imgsize), A.Normalize()]
    )
    img_transformed_tmp = transform_tmp(image=np.array(img))["image"]
    plt.imshow(img_transformed_tmp)




## === cell 12
if False:
    transform_tmp = A.Compose(
        [A.Resize(CFG.input_imgsize, CFG.input_imgsize), ToTensorV2()]
    )
    img_transformed_tmp = transform_tmp(image=np.array(img))
    print(img_transformed_tmp.keys())
    print(type(img_transformed_tmp["image"]))
    print(img_transformed_tmp["image"].shape)




## === cell 13
def extract_label_from_path(p: str):
    base = os.path.basename(p).lower()
    if base.startswith("dog"):
        return 1
    if base.startswith("cat"):
        return 0
    parts = os.path.normpath(p).split(os.sep)
    if "dog" in parts:
        return 1
    if "cat" in parts:
        return 0
    m = re.match(r"^(dog|cat)\.", base)
    if m:
        return 1 if m.group(1) == "dog" else 0
    raise ValueError(f"Cannot infer label from path: {p}")


def extract_test_id(p: str):
    base = os.path.basename(p)
    return int(os.path.splitext(base)[0])


train_df = pd.DataFrame(train_list, columns=["path"])
train_df["class"] = train_df["path"].apply(extract_label_from_path).astype(np.int64)

test_df = pd.DataFrame(test_list, columns=["path"])
test_df["class"] = -1
test_df["id"] = test_df["path"].apply(extract_test_id).astype(np.int64)
test_df = test_df.sort_values("id").reset_index(drop=True)

train_df.head(3)




## === cell 14
test_df.head(3)




## === cell 15
train_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)
test_transform = A.Compose(
    [A.Resize(CFG.input_imgsize, CFG.input_imgsize), A.Normalize(), ToTensorV2()]
)




## === cell 16
@lru_cache(maxsize=CFG.max_decode_cache)
def _read_rgb_uint8_hwc_cached(path: str) -> np.ndarray:
    if _HAS_CV2:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            t = read_image(path, mode=ImageReadMode.RGB)
            return t.permute(1, 2, 0).contiguous().numpy()
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = read_image(path, mode=ImageReadMode.RGB)  # uint8 [3,H,W] on CPU
    return t.permute(1, 2, 0).contiguous().numpy()


def _read_rgb_uint8_hwc(path: str) -> np.ndarray:
    return _read_rgb_uint8_hwc_cached(path)


class DogsCatsDataset(Dataset):
    def __init__(self, df, transform=None):
        df = df.reset_index(drop=True)
        self.paths = df["path"].tolist()
        self.labels = df["class"].astype(np.float32).values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = _read_rgb_uint8_hwc(path)
        img = self.transform(image=img)["image"]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


class DogsCatsTestDataset(Dataset):
    def __init__(self, df, transform=None):
        df = df.reset_index(drop=True)
        self.paths = df["path"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = _read_rgb_uint8_hwc(self.paths[idx])
        img = self.transform(image=img)["image"]
        return img




## === cell 17
_TQDM_DISABLE = True


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    for img, label in tqdm(dataloader, leave=False, disable=_TQDM_DISABLE):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(img)
        loss = criterion(output.squeeze(-1), label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return float(np.mean(losses))




## === cell 18
def eval_one_epoch(model, dataloader, criterion):
    model.eval()
    losses = []
    n = len(dataloader.dataset)
    all_labels = np.empty((n,), dtype=np.float64)
    all_outputs = np.empty((n,), dtype=np.float64)

    offset = 0
    with torch.inference_mode():
        for img, label in tqdm(dataloader, leave=False, disable=_TQDM_DISABLE):
            bs = img.size(0)
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            output = model(img)
            loss = criterion(output.squeeze(-1), label)
            losses.append(loss.item())

            all_labels[offset : offset + bs] = (
                label.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            all_outputs[offset : offset + bs] = (
                torch.sigmoid(output)
                .detach()
                .cpu()
                .numpy()
                .flatten()
                .astype(np.float64, copy=False)
            )
            offset += bs

    if offset != n:
        all_labels = all_labels[:offset]
        all_outputs = all_outputs[:offset]

    return {
        "bce_loss": float(np.mean(losses)),
        "log_loss": float(log_loss(all_labels, all_outputs)),
        "labels": all_labels,
        "outputs": all_outputs,
    }




## === cell 19
def infer(model, dataloader, test=False):
    model.eval()
    n = len(dataloader.dataset)
    all_outputs = np.empty((n,), dtype=np.float64)
    offset = 0
    with torch.inference_mode():
        if test:
            for img in tqdm(dataloader, leave=False, disable=_TQDM_DISABLE):
                bs = img.size(0)
                img = img.to(device, non_blocking=True)
                output = model(img)
                all_outputs[offset : offset + bs] = (
                    torch.sigmoid(output)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                    .astype(np.float64, copy=False)
                )
                offset += bs
        else:
            for img, label in tqdm(dataloader, leave=False, disable=_TQDM_DISABLE):
                bs = img.size(0)
                img = img.to(device, non_blocking=True)
                output = model(img)
                all_outputs[offset : offset + bs] = (
                    torch.sigmoid(output)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                    .astype(np.float64, copy=False)
                )
                offset += bs

    if offset != n:
        all_outputs = all_outputs[:offset]
    return all_outputs




## === cell 20
def _seed_worker(worker_id: int):
    worker_seed = CFG.random_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _dl_kwargs(train: bool):
    kw = dict(
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )
    if CFG.num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4 if train else 8
    return kw




## === cell 21
def _maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    if not CFG.use_torch_compile:
        return model
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="reduce-overhead")
        except Exception:
            return model
    return model


def run_train_cv(train, test):
    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    oof = np.zeros((len(train),), dtype=np.float64)
    predictions = []

    test_dataset_shared = DogsCatsTestDataset(test_df, transform=test_transform)
    test_loader_shared = DataLoader(
        test_dataset_shared,
        batch_size=CFG.batch_size,
        shuffle=False,
        **_dl_kwargs(train=False),
    )

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"====================fold : {fold}====================")
        tr_df = train.iloc[train_idx].reset_index(drop=True)
        va_df = train.iloc[valid_idx].reset_index(drop=True)

        train_dataset = DogsCatsDataset(tr_df, transform=train_transform)
        valid_dataset = DogsCatsDataset(va_df, transform=test_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            drop_last=True,
            **_dl_kwargs(train=True),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            **_dl_kwargs(train=False),
        )

        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.to(device)
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)
        model = _maybe_compile(model)

        optimizer = CFG.optimizer(model.parameters(), lr=CFG.lr)
        training_steps = len(train_loader) * CFG.num_epochs
        warmup_steps = int(training_steps * 0.1)
        scheduler = CFG.scheduler(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

        best_loss = np.inf
        early_stopping_round = 0

        for epoch in range(CFG.num_epochs):
            start_time = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, CFG.criterion
            )
            valid_result = eval_one_epoch(model, valid_loader, CFG.criterion)
            print(
                f"epoch : {epoch} - train loss : {train_loss:.6f} - "
                f"valid loss : {valid_result['bce_loss']:.6f} - valid log loss : {valid_result['log_loss']:.6f}"
            )

            if valid_result["bce_loss"] < best_loss:
                best_loss = valid_result["bce_loss"]
                early_stopping_round = 0
                torch.save(model.state_dict(), f"{CFG.model_name}_fold{fold}.pth")
            else:
                early_stopping_round += 1
                if early_stopping_round > CFG.early_stopping_round:
                    break

            print(f"spend time for epoch {epoch} : {time.time() - start_time:.1f}s")

        best_path = f"{CFG.model_name}_fold{fold}.pth"
        model.load_state_dict(torch.load(best_path, map_location=device))

        oof[valid_idx] = infer(model, valid_loader)

        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            train_dataset,
            valid_dataset,
        )
        gc.collect()
        torch.cuda.empty_cache()

        if CFG.debug_one_fold:
            break

    max_folds_available = 1 if CFG.debug_one_fold else CFG.n_splits
    max_folds = int(min(max_folds_available, max(1, CFG.infer_num_folds)))

    for fold in range(max_folds):
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.load_state_dict(
            torch.load(f"{CFG.model_name}_fold{fold}.pth", map_location=device)
        )
        model.to(device)
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)
        model = _maybe_compile(model)
        predictions.append(infer(model, test_loader_shared, test=True))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    predictions = np.mean(np.stack(predictions, axis=0), axis=0)

    del test_loader_shared, test_dataset_shared
    gc.collect()
    torch.cuda.empty_cache()

    return {"oof": oof, "predictions": predictions}




## === cell 22
def smooth_probs_toward_half(p: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return np.clip(p, 1e-7, 1.0 - 1e-7)
    p2 = (1.0 - alpha) * p + alpha * 0.5
    return np.clip(p2, 1e-7, 1.0 - 1e-7)




## === cell 23
def _checkpoints_exist(num_folds: int) -> bool:
    for fold in range(num_folds):
        if not os.path.exists(f"{CFG.model_name}_fold{fold}.pth"):
            return False
    return True


def main():
    need_ckpt_folds = int(min(CFG.n_splits, max(1, CFG.infer_num_folds)))
    if CFG.only_infer and not _checkpoints_exist(need_ckpt_folds):
        print(
            f"[WARN] only_infer=True but checkpoints for folds 0..{need_ckpt_folds-1} are missing. "
            f"Falling back to training (CFG.only_infer=False)."
        )
        CFG.only_infer = False

    sub = submission.copy()
    sub["id"] = sub["id"].astype(int)
    sub = sub.sort_values("id").reset_index(drop=True)
    sub_ids = sub["id"].values
    sub_id_set = set(sub_ids.tolist())

    if "id" not in test_df.columns:
        raise ValueError(
            "test_df missing id column; cannot align to sample_submission.csv"
        )
    test_df_aligned = test_df[test_df["id"].isin(sub_id_set)].copy()
    test_df_aligned = test_df_aligned.sort_values("id").reset_index(drop=True)
    if len(test_df_aligned) != len(sub):
        raise ValueError(
            f"After aligning to sample_submission.csv, test rows={len(test_df_aligned)} "
            f"but submission rows={len(sub)}. Check test image discovery."
        )
    globals()["test_df"] = test_df_aligned  # keep downstream code unchanged

    if CFG.only_infer:
        test_dataset = DogsCatsTestDataset(test_df, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            **_dl_kwargs(train=False),
        )

        preds_folds = []
        for fold in range(need_ckpt_folds):
            ckpt = f"{CFG.model_name}_fold{fold}.pth"
            model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state)
            model.to(device)
            if torch.cuda.is_available():
                model = model.to(memory_format=torch.channels_last)
            model = _maybe_compile(model)
            preds_folds.append(infer(model, test_loader, test=True))
            del model, state
            gc.collect()
            torch.cuda.empty_cache()

        predictions = np.mean(np.stack(preds_folds, axis=0), axis=0)
        predictions = smooth_probs_toward_half(
            predictions.astype(np.float64), CFG.prob_smooth_alpha
        )

        if len(predictions) != len(sub):
            raise ValueError(
                f"Pred length {len(predictions)} != submission length {len(sub)}"
            )
        sub["label"] = predictions.astype(np.float64)
        sub.to_csv("submission.csv", index=False)
        print("Wrote submission.csv with", len(sub), "rows")
    else:
        result = run_train_cv(train_df, test_df)
        oof_preds = result["oof"]
        predictions = result["predictions"]

        predictions = smooth_probs_toward_half(
            predictions.astype(np.float64), CFG.prob_smooth_alpha
        )

        if len(predictions) != len(sub):
            raise ValueError(
                f"Pred length {len(predictions)} != submission length {len(sub)}"
            )
        sub["label"] = predictions.astype(np.float64)
        sub.to_csv("submission.csv", index=False)
        print("Wrote submission.csv with", len(sub), "rows")

        train_out = train_df.copy()
        train_out["oof_preds"] = smooth_probs_toward_half(
            oof_preds.astype(np.float64), CFG.prob_smooth_alpha
        )
        train_out.to_csv("oof_preds.csv", index=False)

        if CFG.debug_one_fold is False:
            print(
                f"oof log loss : {log_loss(train_out['class'].values, train_out['oof_preds'].values)}"
            )


if __name__ == "__main__":
    main()
