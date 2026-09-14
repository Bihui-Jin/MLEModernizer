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
from sklearn.utils import shuffle
import pydicom
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler
import time
import copy

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(min(8, os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(8, os.cpu_count() or 1)))

torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)



## === cell 1
from pydicom.pixels import pixel_array as pyd_pixel_array

pydicom.config.convert_wrong_length_to_UN = True


def read_xray(file_path, img_size=None):
    try:
        dcm = pydicom.dcmread(
            file_path,
            force=True,
            stop_before_pixels=False,
            specific_tags=[
                "PixelData",
                "PhotometricInterpretation",
                "BitsAllocated",
                "BitsStored",
                "HighBit",
                "PixelRepresentation",
                "SamplesPerPixel",
                "PlanarConfiguration",
                "Rows",
                "Columns",
                "NumberOfFrames",
                "TransferSyntaxUID",
                "RescaleIntercept",
                "RescaleSlope",
                "WindowCenter",
                "WindowWidth",
            ],
        )
        img = pyd_pixel_array(dcm).astype(np.float32)

        if getattr(dcm, "PhotometricInterpretation", None) == "MONOCHROME1":
            img = np.max(img) - img

        if img.ndim > 2:
            img = img.squeeze()
        if img.ndim != 2:
            raise ValueError(f"Unexpected DICOM pixel array shape: {img.shape}")

        if img_size:
            img = cv2.resize(
                img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA
            )

        img = img[np.newaxis].astype("float32")  # (1,H,W)

        mx = float(np.max(img))
        if mx > 0:
            img = img / mx
        else:
            img = img * 0.0

        return np.ascontiguousarray(img)
    except Exception:
        if img_size is None:
            img_size = 224
        return np.zeros((1, img_size, img_size), dtype=np.float32)




## === cell 2
DATA_ROOT = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df_full = pd.read_csv(TRAIN_CSV)

pos_df = train_df_full[train_df_full["cancer"] == 1]
neg_df = train_df_full[train_df_full["cancer"] == 0]

n_total = 2000
n_pos = min(len(pos_df), max(50, int(0.10 * n_total)))
n_neg = min(len(neg_df), n_total - n_pos)

rng = np.random.default_rng(SEED)
pos_idx = (
    rng.choice(pos_df.index.values, size=n_pos, replace=False)
    if n_pos > 0
    else np.array([], dtype=int)
)
neg_idx = rng.choice(neg_df.index.values, size=n_neg, replace=False)

train_df = train_df_full.loc[np.concatenate([pos_idx, neg_idx])].copy()
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

IMG_SIZE = 224

train_paths = np.array(
    [
        os.path.join(TRAIN_IMG_DIR, str(pid), f"{iid}.dcm")
        for pid, iid in zip(
            train_df["patient_id"].to_numpy(), train_df["image_id"].to_numpy()
        )
    ],
    dtype=object,
)
train_labels = train_df["cancer"].to_numpy(dtype=np.int64)

exists_mask = np.fromiter(
    (os.path.exists(p) for p in train_paths), dtype=np.bool_, count=len(train_paths)
)
missing = int((~exists_mask).sum())
train_paths = train_paths[exists_mask]
train_labels = train_labels[exists_mask]

train_paths, train_labels = shuffle(train_paths, train_labels, random_state=SEED)

val_size = min(316, max(1, int(0.15 * len(train_labels))))
val_paths = train_paths[-val_size:]
y_validation = train_labels[-val_size:]
train_paths = train_paths[:-val_size]
y_train = train_labels[:-val_size]

print(
    f"Prepared train N={len(y_train)}, val N={len(y_validation)}, missing_paths={missing}"
)



## === cell 3
TRAIN_CACHE_DIR = f"/kaggle/working/rsna_train_cache_{IMG_SIZE}"
os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)


def _cache_path_from_dcm(dcm_path: str, img_size: int, cache_dir: str) -> str:
    base = os.path.basename(dcm_path).replace(".dcm", "")
    pid = os.path.basename(os.path.dirname(dcm_path))
    return os.path.join(cache_dir, f"{pid}_{base}_{img_size}.npy")


def _atomic_save_npy(dst_path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp.npy"
    np.save(tmp_path, array, allow_pickle=False)
    os.replace(tmp_path, dst_path)


def precompute_npy_cache(
    dcm_paths: np.ndarray, img_size: int, cache_dir: str
) -> np.ndarray:
    cached = np.empty((len(dcm_paths),), dtype=object)
    for i, p in enumerate(dcm_paths):
        cp = _cache_path_from_dcm(p, img_size, cache_dir)
        cached[i] = cp
        if not os.path.exists(cp):
            x = read_xray(p, img_size=img_size)
            _atomic_save_npy(cp, x)
    return cached


train_cached_paths = precompute_npy_cache(train_paths, IMG_SIZE, TRAIN_CACHE_DIR)
val_cached_paths = precompute_npy_cache(val_paths, IMG_SIZE, TRAIN_CACHE_DIR)


class RSNA_dataset(Dataset):
    def __init__(self, cached_paths, label):
        self.cached_paths = cached_paths
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = np.load(self.cached_paths[idx], mmap_mode="r")
        y = self.label[idx]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor(
            int(y), dtype=torch.long
        )




## === cell 4
train_dataset = RSNA_dataset(train_cached_paths, y_train)

num_workers_train = min(4, (os.cpu_count() or 2))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers_train,
    pin_memory=True,
    persistent_workers=(num_workers_train > 0),
    prefetch_factor=4 if num_workers_train > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

validation_dataset = RSNA_dataset(val_cached_paths, y_validation)
num_workers_val = min(2, (os.cpu_count() or 2))
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers_val,
    pin_memory=True,
    persistent_workers=(num_workers_val > 0),
    prefetch_factor=4 if num_workers_val > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

dataloaders = {"train": train_dataloader, "val": validation_dataloader}
dataset_sizes = {"train": len(train_dataset), "val": len(validation_dataset)}




## === cell 5
def train_model(
    model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = torch.zeros((), device=device)
            running_corrects = torch.zeros((), device=device, dtype=torch.long)

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

                running_loss += loss.detach() * inputs.size(0)
                running_corrects += (preds == labels).sum()

            if phase == "train":
                scheduler.step()

            epoch_loss = (running_loss / max(1, dataset_sizes[phase])).item()
            epoch_acc = (running_corrects.float() / max(1, dataset_sizes[phase])).item()

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    _ = time.time() - since
    return model




## === cell 6
epochs = 2

model_ft = models.resnet18(weights=None)  # keep pretrained=False equivalent
model_ft.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device).to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)



## === cell 7
model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=epochs,
)



## === cell 8
test_df = pd.read_csv(TEST_CSV)

TEST_CACHE_DIR = "/kaggle/working/rsna_test_cache_224"
os.makedirs(TEST_CACHE_DIR, exist_ok=True)

_test_patient = test_df["patient_id"].to_numpy()
_test_image = test_df["image_id"].to_numpy()
_test_dcm_paths = np.array(
    [
        os.path.join(TEST_IMG_DIR, str(pid), f"{iid}.dcm")
        for pid, iid in zip(_test_patient, _test_image)
    ],
    dtype=object,
)
_test_exists = np.fromiter(
    (os.path.exists(p) for p in _test_dcm_paths),
    dtype=np.bool_,
    count=len(_test_dcm_paths),
)

test_cached_paths = np.empty((len(_test_dcm_paths),), dtype=object)
test_ok = np.zeros((len(_test_dcm_paths),), dtype=np.bool_)

for i, (p, ex) in enumerate(zip(_test_dcm_paths, _test_exists)):
    cp = _cache_path_from_dcm(p, IMG_SIZE, TEST_CACHE_DIR)
    test_cached_paths[i] = cp

    if not ex:
        test_ok[i] = False
        if not os.path.exists(cp):
            _atomic_save_npy(cp, np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32))
        continue

    if os.path.exists(cp):
        x = np.load(cp, mmap_mode="r")
        test_ok[i] = bool(float(np.max(x)) > 0.0)
    else:
        x = read_xray(p, img_size=IMG_SIZE)
        test_ok[i] = bool(float(x.max()) > 0.0)
        _atomic_save_npy(cp, x)


class RSNATestNpyDataset(Dataset):
    def __init__(self, cached_paths, ok_mask):
        self.cached_paths = cached_paths
        self.ok_mask = ok_mask

    def __len__(self):
        return len(self.cached_paths)

    def __getitem__(self, idx):
        x = np.load(self.cached_paths[idx], mmap_mode="r")
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), bool(
            self.ok_mask[idx]
        )


def _test_collate(batch):
    imgs, ok = zip(*batch)
    return torch.stack(imgs, 0), torch.tensor(ok, dtype=torch.bool)


test_dataset = RSNATestNpyDataset(test_cached_paths, test_ok)

num_workers_test = min(4, (os.cpu_count() or 2))
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers_test,
    pin_memory=True,
    persistent_workers=(num_workers_test > 0),
    prefetch_factor=4 if num_workers_test > 0 else None,
    collate_fn=_test_collate,
    worker_init_fn=seed_worker,
    generator=g,
)

model_ft.eval()
probs = np.zeros((len(test_df),), dtype=np.float32)
skipped_test = int((~test_ok).sum())

offset = 0
with torch.no_grad():
    for imgs, ok_mask in test_loader:
        bsz = imgs.size(0)
        imgs = imgs.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        out = model_ft(imgs)
        p = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
        ok_np = ok_mask.numpy()
        if not ok_np.all():
            p = p.copy()
            p[~ok_np] = 0.0
        probs[offset : offset + bsz] = p
        offset += bsz

test_df["cancer"] = probs
test_df["cancer"] = test_df["cancer"].fillna(0.0).clip(0.0, 1.0)
print(f"Test rows={len(test_df)}, skipped_test={skipped_test}")



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB)

pred_by_pid = test_df.groupby("prediction_id", as_index=False)["cancer"].mean()

submission = sample_sub[["prediction_id"]].merge(
    pred_by_pid, on="prediction_id", how="left"
)
submission["cancer"] = (
    submission["cancer"].fillna(0.0).astype(np.float32).clip(0.0, 1.0)
)

print("Submission shape:", submission.shape)
print(submission.head())



## === cell 10
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
