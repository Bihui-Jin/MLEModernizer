# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

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
pydicom==3.0.1
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
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy
from torch.optim import lr_scheduler
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)


g = torch.Generator()
g.manual_seed(SEED)



## === cell 1
CACHE_DIR = "/kaggle/working/rsna_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path_for(dcm_path: str, img_size: int) -> str:
    key = str(abs(hash((dcm_path, img_size, "v2"))))
    return os.path.join(CACHE_DIR, f"{key}.npy")


def _meta_fallback_image(
    dicom: "pydicom.dataset.FileDataset | None", img_size: int
) -> np.ndarray:
    H = W = int(img_size)
    vals = []
    if dicom is not None:
        for attr in (
            "PatientAge",
            "Rows",
            "Columns",
            "WindowCenter",
            "WindowWidth",
            "RescaleSlope",
            "RescaleIntercept",
            "BitsStored",
            "BitsAllocated",
            "HighBit",
        ):
            v = getattr(dicom, attr, 0)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                v = v[0]
            if isinstance(v, str):
                digits = "".join([c for c in v if c.isdigit()])
                v = int(digits) if digits else 0
            try:
                v = float(v)
            except Exception:
                v = 0.0
            vals.append(v)
    if not vals:
        vals = [0.0]

    v = np.asarray(vals, dtype=np.float32)
    v = v - v.min()
    denom = float(v.max()) if float(v.max()) > 0 else 1.0
    v = v / denom  # [0,1]
    base = np.zeros((H, W), dtype=np.float32)
    for i, val in enumerate(v[: min(len(v), 32)]):
        r0 = (i * 7) % H
        c0 = (i * 13) % W
        base[r0 : min(H, r0 + 8), c0 : min(W, c0 + 8)] = val
    base = cv2.GaussianBlur(base, (7, 7), 0)
    base = base[np.newaxis]
    return base.astype(np.float32, copy=False)


def read_xray(file_path, img_size=None):
    dicom = pydicom.dcmread(
        file_path,
        force=False,
        stop_before_pixels=False,
    )

    try:
        img = dicom.pixel_array.astype(np.float32)
        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = np.max(img) - img

        img = img - np.min(img)
        mx = np.max(img)
        if mx > 0:
            img = img / mx
        else:
            img = np.zeros_like(img, dtype=np.float32)

        if img_size:
            img_size = (img_size, img_size)
            img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_AREA)

        img = img[np.newaxis].astype("float32")
        return img
    except Exception:
        if img_size is None:
            img_size = 256
        return _meta_fallback_image(dicom, img_size)


def read_xray_cached(file_path: str, img_size: int):
    cp = _cache_path_for(file_path, img_size)
    try:
        arr = np.load(cp, allow_pickle=False, mmap_mode="r")
        if arr.dtype == np.float32 and arr.ndim == 3 and arr.shape[0] == 1:
            return np.asarray(arr)
    except Exception:
        pass

    arr = read_xray(file_path, img_size=img_size)

    tmp = cp + ".tmp.npy"
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, cp)
    return arr




## === cell 2
DATA_ROOT = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
TEST_CSV = f"{DATA_ROOT}/test.csv"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"

train_df_full = pd.read_csv(TRAIN_CSV)

patient = train_df_full["patient_id"].astype(str)
image = train_df_full["image_id"].astype(str)
train_df_full["path"] = TRAIN_IMG_DIR + "/" + patient + "/" + image + ".dcm"

rng = np.random.default_rng(SEED)

N_SAMPLES = min(2000, len(train_df_full))
train_df_small = train_df_full.sample(n=N_SAMPLES, random_state=SEED).reset_index(
    drop=True
)

unique_patients = train_df_small["patient_id"].unique()
rng.shuffle(unique_patients)
val_patient_count = max(1, int(0.16 * len(unique_patients)))
val_patients = set(unique_patients[:val_patient_count])

train_df = train_df_small[~train_df_small["patient_id"].isin(val_patients)].reset_index(
    drop=True
)
val_df = train_df_small[train_df_small["patient_id"].isin(val_patients)].reset_index(
    drop=True
)

print(
    "train rows:",
    len(train_df),
    "val rows:",
    len(val_df),
    "pos rate train:",
    train_df["cancer"].mean(),
)



## === cell 3
IMG_SIZE = 256  # small enough for speed, large enough for signal


def build_arrays(df):
    from concurrent.futures import ThreadPoolExecutor

    paths = df["path"].to_numpy()
    labels = df["cancer"].to_numpy(dtype=np.int64, copy=False)

    n = len(paths)
    if n == 0:
        raise RuntimeError("Empty dataframe passed to build_arrays().")

    X = np.empty((n, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)

    def _load_one(i: int):
        p = paths[i]
        try:
            arr = read_xray_cached(p, img_size=IMG_SIZE)
            if arr.dtype != np.float32 or arr.ndim != 3 or arr.shape[0] != 1:
                return None
            return i, arr
        except Exception:
            return None

    max_workers = min(8, (os.cpu_count() or 1))
    write_pos = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_load_one, range(n), chunksize=64):
            if res is None:
                continue
            i, arr = res
            X[write_pos] = arr
            y[write_pos] = int(labels[i])
            write_pos += 1

    if write_pos == 0:
        raise RuntimeError(
            "No training images could be read. Check dataset paths/availability."
        )

    X = X[:write_pos]
    y = y[:write_pos]
    return X, y


X_train, y_train = build_arrays(train_df)
X_validation, y_validation = build_arrays(val_df)

perm = np.random.default_rng(SEED).permutation(len(y_train))
X_train = X_train[perm]
y_train = y_train[perm]

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_validation.shape, "y_val:", y_validation.shape)




## === cell 4
class RSNA_dataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.feature[idx])  # float32
        y = int(self.label[idx])
        return x, torch.tensor(y, dtype=torch.long)




## === cell 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pin = device.type == "cuda"

cpu_cnt = os.cpu_count() or 1
_NUM_WORKERS = 0 if len(X_train) < 1024 else min(4, cpu_cnt)

train_dataset = RSNA_dataset(X_train, y_train)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=_NUM_WORKERS,
    pin_memory=pin,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if _NUM_WORKERS > 0 else None,
    generator=g,
)

validation_dataset = RSNA_dataset(X_validation, y_validation)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=pin,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if _NUM_WORKERS > 0 else None,
    generator=g,
)

dataloaders = {"train": train_dataloader, "val": validation_dataloader}
dataset_sizes = {"train": len(train_dataset), "val": len(validation_dataset)}
print("device:", device, "num_workers:", _NUM_WORKERS)




## === cell 6
def train_model(
    model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += float(loss.detach().cpu()) * inputs.size(0)
                running_corrects += int((preds == labels).sum().detach().cpu())

            if phase == "train":
                scheduler.step()

            epoch_acc = running_corrects / max(1, dataset_sizes[phase])

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model




## === cell 7
epochs = 2

model_ft = models.resnet18(weights=None)
model_ft.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)
model_ft = model_ft.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        model_ft = torch.compile(model_ft, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)



## === cell 8
model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=epochs,
)



## === cell 9
test_df = pd.read_csv(TEST_CSV)

patient = test_df["patient_id"].astype(str)
image = test_df["image_id"].astype(str)
test_df["path"] = TEST_IMG_DIR + "/" + patient + "/" + image + ".dcm"

print(test_df.head())




## === cell 10
class RSNATestDataset(Dataset):
    def __init__(self, paths, img_size):
        self.paths = list(paths)
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            x = read_xray_cached(p, img_size=self.img_size)  # [1,H,W] float32
        except Exception:
            x = _meta_fallback_image(None, self.img_size).astype(np.float32, copy=False)
        return torch.from_numpy(x), idx


model_ft.eval()

batch_paths = test_df["path"].tolist()
test_dataset = RSNATestDataset(batch_paths, IMG_SIZE)

test_num_workers = min(4, (os.cpu_count() or 1))
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=test_num_workers,
    pin_memory=pin,
    persistent_workers=(test_num_workers > 0),
    prefetch_factor=2 if test_num_workers > 0 else None,
    worker_init_fn=_seed_worker if test_num_workers > 0 else None,
    generator=g,
)

probs = np.zeros(len(test_df), dtype=np.float32)

with torch.no_grad():
    for xb, idxb in test_loader:
        xb = xb.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        logits = model_ft(xb)
        pb = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs[idxb.numpy()] = pb.astype(np.float32, copy=False)

test_pred = test_df[["prediction_id"]].copy()
test_pred["cancer"] = probs.tolist()

submission = test_pred.groupby("prediction_id", as_index=False)["cancer"].mean()

sample_sub = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
submission = sample_sub[["prediction_id"]].merge(
    submission, on="prediction_id", how="left"
)
submission["cancer"] = submission["cancer"].fillna(0.0).astype(float)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
