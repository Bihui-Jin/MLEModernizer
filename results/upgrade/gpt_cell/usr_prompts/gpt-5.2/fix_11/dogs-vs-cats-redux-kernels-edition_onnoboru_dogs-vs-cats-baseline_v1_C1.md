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
import time
import random
import glob
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import transformers

from tqdm import tqdm

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 1
class CFG:
    debug_one_epoch = False
    debug_one_fold = False
    only_infer = False

    num_workers = 4
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

    pin_memory = True
    persistent_workers = True
    prefetch_factor = 4


def seed_torch(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch(CFG.random_seed)

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

if CFG.debug_one_epoch:
    CFG.num_epochs = 1

print("KAGGLE_URL_BASE" in set(os.environ.keys()))



## === cell 2
candidate_roots = [
    CFG.data_dir,
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition/",
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/",
    "/kaggle/data/",
    "/kaggle/input/",
]
submission_path = None
for r in candidate_roots:
    p = os.path.join(r, "sample_submission.csv")
    if os.path.exists(p):
        submission_path = p
        break

if submission_path is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")

submission = pd.read_csv(submission_path)
print("Loaded sample_submission from:", submission_path)
submission.head(3)



## === cell 3
dataset_roots = [
    CFG.data_dir,
    os.path.dirname(submission_path),
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition/",
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/",
]
data_root = None
for r in dataset_roots:
    if (
        r
        and os.path.exists(os.path.join(r, "train.zip"))
        and os.path.exists(os.path.join(r, "test.zip"))
    ):
        data_root = r
        break
    if (
        r
        and os.path.isdir(os.path.join(r, "train"))
        and os.path.isdir(os.path.join(r, "test"))
    ):
        data_root = r
        break

if data_root is None:
    raise FileNotFoundError(
        "Could not locate dataset root containing train/test or train.zip/test.zip."
    )

CFG.data_dir = data_root
print("Using dataset root:", CFG.data_dir)

if "KAGGLE_URL_BASE" in set(os.environ.keys()):
    kaggle_train_dir = os.path.join(CFG.kaggle_working_dir, "train")
    if not os.path.exists(kaggle_train_dir) and os.path.exists(
        os.path.join(CFG.data_dir, "train.zip")
    ):
        shutil.unpack_archive(
            os.path.join(CFG.data_dir, "train.zip"), CFG.kaggle_working_dir
        )

    kaggle_test_dir = os.path.join(CFG.kaggle_working_dir, "test")
    if not os.path.exists(kaggle_test_dir) and os.path.exists(
        os.path.join(CFG.data_dir, "test.zip")
    ):
        shutil.unpack_archive(
            os.path.join(CFG.data_dir, "test.zip"), CFG.kaggle_working_dir
        )

    if os.path.isdir(os.path.join(CFG.kaggle_working_dir, "train")) and os.path.isdir(
        os.path.join(CFG.kaggle_working_dir, "test")
    ):
        CFG.data_dir = CFG.kaggle_working_dir

CFG.train_dir = os.path.join(CFG.data_dir, "train")
CFG.test_dir = os.path.join(CFG.data_dir, "test")
print("Resolved train_dir:", CFG.train_dir)
print("Resolved test_dir :", CFG.test_dir)



## === cell 4
train_list = glob.glob(os.path.join(CFG.train_dir, "*.jpg"))
test_list = glob.glob(os.path.join(CFG.test_dir, "*.jpg"))

if len(train_list) == 0:
    train_list = glob.glob(os.path.join(CFG.train_dir, "**", "*.jpg"), recursive=True)

if len(test_list) == 0:
    test_list = glob.glob(os.path.join(CFG.test_dir, "**", "*.jpg"), recursive=True)

print(f"train data : {len(train_list)}")
print(f"test data : {len(test_list)}")



## === cell 5
print(
    "the number of dog : ", len([i for i in train_list if "dog" in os.path.basename(i)])
)
print(
    "the number of cat : ", len([i for i in train_list if "cat" in os.path.basename(i)])
)



## === cell 6
if len(train_list) == 0:
    raise FileNotFoundError(
        f"No training images found. Searched under '{CFG.train_dir}'."
    )



## === cell 7
train_df = pd.DataFrame(train_list, columns=["path"])


def extract_label(path: str):
    base = os.path.basename(path).lower()
    if base.startswith("dog."):
        return 1
    if base.startswith("cat."):
        return 0
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent == "dog":
        return 1
    if parent == "cat":
        return 0
    return np.nan


train_df["class"] = train_df["path"].apply(extract_label).astype("float32")

if train_df["class"].isna().any():
    bad = train_df[train_df["class"].isna()].head(5)["path"].tolist()
    raise ValueError(
        f"Some train labels could not be inferred from filenames/dirs. Examples: {bad}"
    )

test_df = pd.DataFrame(test_list, columns=["path"])
test_df["class"] = -1
test_df["id"] = test_df["path"].apply(
    lambda x: int(os.path.splitext(os.path.basename(x))[0])
)
test_df = test_df.sort_values("id").reset_index(drop=True)

train_df.head(3)



## === cell 8
test_df.head(3)



## === cell 9
train_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)
test_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.Normalize(),
        ToTensorV2(),
    ]
)



## === cell 10
try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


def _read_rgb_uint8(path: str) -> np.ndarray:
    if _HAS_CV2:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im)


class DogsCatsDataset(Dataset):
    def __init__(self, df, transform=None):
        df = df.reset_index(drop=True)
        self.paths = df["path"].to_numpy()
        self.labels = df["class"].to_numpy(dtype=np.float32, copy=False)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_np = _read_rgb_uint8(path)
        img = self.transform(image=img_np)["image"]
        label = np.float32(self.labels[idx])
        return img, label




## === cell 11
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    for img, label in tqdm(dataloader, leave=False):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(img)
        loss = criterion(output.squeeze(-1), label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if len(losses) else np.nan




## === cell 12
def eval_one_epoch_with_preds(model, dataloader, criterion):
    model.eval()
    losses = []
    n = len(dataloader.dataset)
    all_labels = np.empty((n,), dtype=np.float32)
    all_outputs = np.empty((n,), dtype=np.float32)
    offset = 0

    with torch.no_grad():
        for img, label in tqdm(dataloader, leave=False):
            bsz = img.size(0)
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(img)
            loss = criterion(output.squeeze(-1), label)
            losses.append(loss.item())

            lbl = (
                label.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            )
            pred = (
                torch.sigmoid(output)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32, copy=False)
            )
            all_labels[offset : offset + bsz] = lbl
            all_outputs[offset : offset + bsz] = pred
            offset += bsz

    if offset != n:
        all_labels = all_labels[:offset]
        all_outputs = all_outputs[:offset]

    return {
        "bce_loss": float(np.mean(losses)) if len(losses) else np.nan,
        "log_loss": log_loss(all_labels, all_outputs),
        "labels": all_labels,
        "outputs": all_outputs,
    }




## === cell 13
def infer(model, dataloader, test=False):
    model.eval()
    n = len(dataloader.dataset)
    all_outputs = np.empty((n,), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for img, label in tqdm(dataloader, leave=False):
            if test:
                if isinstance(label, torch.Tensor):
                    assert float(label[0].item()) == -1.0
                else:
                    assert float(label[0]) == -1.0
            bsz = img.size(0)
            img = img.to(device, non_blocking=True)
            output = model(img)
            pred = (
                torch.sigmoid(output)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32, copy=False)
            )
            all_outputs[offset : offset + bsz] = pred
            offset += bsz

    if offset != n:
        all_outputs = all_outputs[:offset]
    return all_outputs




## === cell 14
def _make_loader(dataset, batch_size, shuffle, drop_last):
    nw = CFG.num_workers
    if nw is None:
        nw = 0
    try:
        cpu_cnt = os.cpu_count() or 2
    except Exception:
        cpu_cnt = 2
    if nw > cpu_cnt:
        nw = cpu_cnt

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=CFG.pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )
    if torch.cuda.is_available():
        kwargs["pin_memory_device"] = "cuda"
    if nw > 0:
        kwargs["persistent_workers"] = CFG.persistent_workers
        kwargs["prefetch_factor"] = CFG.prefetch_factor
    return DataLoader(dataset, **kwargs)


def _maybe_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            return model
    return model


def run_train_cv(train, test):
    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    oof = np.zeros((len(train), 1), dtype=np.float32)

    test_dataset = DogsCatsDataset(test, transform=test_transform)
    test_loader = _make_loader(
        test_dataset, batch_size=CFG.batch_size, shuffle=False, drop_last=False
    )

    test_pred_sum = None
    n_pred_folds = 0

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"====================fold : {fold}====================")
        train_df_fold = train.iloc[train_idx].reset_index(drop=True)
        valid_df_fold = train.iloc[valid_idx].reset_index(drop=True)

        train_dataset = DogsCatsDataset(train_df_fold, transform=train_transform)
        valid_dataset = DogsCatsDataset(valid_df_fold, transform=test_transform)

        train_loader = _make_loader(
            train_dataset, batch_size=CFG.batch_size, shuffle=True, drop_last=True
        )
        valid_loader = _make_loader(
            valid_dataset, batch_size=CFG.batch_size, shuffle=False, drop_last=False
        )

        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)
        model.to(device)
        model = _maybe_compile_model(model)

        optimizer = CFG.optimizer(model.parameters(), lr=CFG.lr)
        training_steps = len(train_loader) * CFG.num_epochs
        warmup_steps = int(training_steps * 0.1)
        scheduler = CFG.scheduler(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

        best_loss = np.inf
        early_stopping_round = 0
        best_path = f"{CFG.model_name}_fold{fold}.pth"
        best_valid_outputs = None  # store best fold OOF preds to avoid extra eval pass

        for epoch in range(CFG.num_epochs):
            start_time = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, CFG.criterion
            )

            valid_result = eval_one_epoch_with_preds(model, valid_loader, CFG.criterion)
            print(
                f"epoch : {epoch} - train loss : {train_loss:.6f} - valid loss : {valid_result['bce_loss']:.6f} - valid log loss : {valid_result['log_loss']:.6f}"
            )

            if valid_result["bce_loss"] < best_loss:
                best_loss = valid_result["bce_loss"]
                early_stopping_round = 0
                best_valid_outputs = valid_result["outputs"].astype(
                    np.float32, copy=False
                )
                torch.save(model.state_dict(), best_path)
            else:
                early_stopping_round += 1
                if early_stopping_round > CFG.early_stopping_round:
                    break

            print(f"spend time for epoch {epoch} : {time.time() - start_time:.1f}s")

            if CFG.debug_one_epoch:
                break

        if best_valid_outputs is None:
            model.load_state_dict(torch.load(best_path, map_location=device))
            valid_best = eval_one_epoch_with_preds(model, valid_loader, CFG.criterion)
            best_valid_outputs = valid_best["outputs"].astype(np.float32)

        oof[valid_idx, 0] = best_valid_outputs

        model.load_state_dict(torch.load(best_path, map_location=device))
        fold_test_pred = infer(model, test_loader, test=True).astype(
            np.float32, copy=False
        )
        if test_pred_sum is None:
            test_pred_sum = fold_test_pred.astype(np.float64, copy=True)
        else:
            test_pred_sum += fold_test_pred
        n_pred_folds += 1

        del (
            model,
            optimizer,
            scheduler,
            train_dataset,
            valid_dataset,
            train_loader,
            valid_loader,
            fold_test_pred,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if CFG.debug_one_fold:
            break

    if n_pred_folds == 0 or test_pred_sum is None:
        raise RuntimeError("No folds were run; cannot generate predictions.")

    predictions = (test_pred_sum / float(n_pred_folds)).astype(np.float32, copy=False)
    return {"oof": oof, "predictions": predictions}




## === cell 15
def main():
    global test_df

    sub_ids = submission["id"].astype(int).to_numpy()

    if (test_df is None) or (len(test_df) == 0):
        test_list_fix = glob.glob(os.path.join(CFG.test_dir, "*.jpg"))
        if len(test_list_fix) == 0:
            test_list_fix = glob.glob(
                os.path.join(CFG.test_dir, "**", "*.jpg"), recursive=True
            )
        if len(test_list_fix) == 0:
            raise FileNotFoundError(
                f"No test images found. Searched recursively under '{CFG.test_dir}'."
            )
        test_df_fix = pd.DataFrame(test_list_fix, columns=["path"])
        test_df_fix["class"] = -1
        test_df_fix["id"] = test_df_fix["path"].apply(
            lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        test_df = test_df_fix

    test_df = test_df[test_df["id"].isin(sub_ids)].copy()
    test_df = test_df.sort_values("id").reset_index(drop=True)

    if len(test_df) != len(submission):
        raise ValueError(
            f"Test images after filtering do not match sample_submission rows: "
            f"found={len(test_df)} vs submission={len(submission)}. "
            f"Check CFG.test_dir='{CFG.test_dir}'."
        )

    if CFG.only_infer:
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)
        test_loader = _make_loader(
            test_dataset, batch_size=CFG.batch_size, shuffle=False, drop_last=False
        )
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)
        model.to(device)
        model = _maybe_compile_model(model)
        predictions = infer(model, test_loader, test=True)
        submission["label"] = np.asarray(predictions).reshape(-1)
        submission.to_csv("submission.csv", index=False)
        print("Wrote submission.csv")
        return

    result = run_train_cv(train_df, test_df)
    oof_preds = result["oof"].reshape(-1)
    predictions = np.asarray(result["predictions"]).reshape(-1)

    if len(predictions) != len(submission):
        raise ValueError(
            f"Prediction length mismatch: predictions={len(predictions)} vs submission={len(submission)}. "
            f"Check test image discovery under CFG.test_dir='{CFG.test_dir}'."
        )

    submission["label"] = predictions
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")

    train_df_out = train_df.copy()
    train_df_out["oof_preds"] = oof_preds
    train_df_out.to_csv("oof_preds.csv", index=False)

    if CFG.debug_one_fold is False:
        print(f"oof log loss : {log_loss(train_df_out['class'].values, oof_preds)}")


if __name__ == "__main__":
    main()
