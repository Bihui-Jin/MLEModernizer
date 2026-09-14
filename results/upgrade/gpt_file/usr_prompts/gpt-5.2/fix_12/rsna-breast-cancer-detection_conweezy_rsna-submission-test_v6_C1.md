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

# 5. Target score

0.037966513167087

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I first fix the root cause of the crash: DICOM loading fails for JPEG2000-compressed images because the required pixel-data handlers (e.g., pylibjpeg/gdcm) aren’t available, so I switch the image reader to a safe OpenCV fallback that reads the DICOM file bytes (supports the competition’s JP2 DICOMs) while keeping the same normalization and 1×224×224 tensor shape. Then I make execution robust so later cells don’t error when earlier cells fail by ensuring `device`, training arrays, and `cancer` predictions are always defined, and by disabling `persistent_workers` when `num_workers=0`. Finally, I ensure the submission matches `sample_submission.csv` exactly (same `prediction_id` rows/order) and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'You’re spending most of the wall time decoding thousands of DICOMs and writing/reading per-image `.npy` files; the I/O and process-pool overhead dominates and pushes you past 600s. The fastest equivalent fix is to remove the disk cache and instead decode DICOMs on-the-fly inside DataLoader workers with a small in-worker LRU cache, while also using faster pydicom reads (`defer_size`, `reading_validation_mode`) and skipping redundant `os.path.exists` scans. We keep the exact same model, loss, optimizer, epochs, and train/val split; we only change how images are loaded (still via the same `read_xray` logic) and keep determinism via seeded workers. Test-time “missing file => zeros => prob=0” behavior is preserved without precomputing caches.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model producing near-constant probabilities that don’t align with pF1’s sensitivity to calibration/positives, which is expected when training ResNet18 from scratch for only 2 epochs on a tiny subset. To move the score upward toward the target with minimal semantic change, I keep the same architecture/training loop/loss, but (1) enable ImageNet pretrained weights (same model, just better initialization), (2) train on a slightly larger but still bounded subset so the model learns a usable signal within the 600s budget, and (3) use a pF1-aligned post-processing step: choose a single global probability scaling factor on the validation set (not labels leakage; it’s standard calibration) and apply it to test predictions. These are small, targeted changes that typically lift pF1 from ~0 to a low-but-nonzero range without rewriting the approach.'

# 9. Code solution

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
            defer_size="1 KB",
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
            reading_validation_mode="IGNORE",
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

n_total = 8000
n_pos = min(len(pos_df), max(200, int(0.10 * n_total)))
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

train_paths, train_labels = shuffle(train_paths, train_labels, random_state=SEED)

val_size = min(1000, max(1, int(0.15 * len(train_labels))))
val_paths = train_paths[-val_size:]
y_validation = train_labels[-val_size:]
train_paths = train_paths[:-val_size]
y_train = train_labels[:-val_size]

print(f"Prepared train N={len(y_train)}, val N={len(y_validation)}")




## === cell 3
class RSNA_dataset(Dataset):
    def __init__(self, dcm_paths, label, img_size: int):
        self.dcm_paths = dcm_paths
        self.label = label
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = read_xray(self.dcm_paths[idx], img_size=self.img_size)
        y = self.label[idx]
        return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long)




## === cell 4
class _LRUCache:
    __slots__ = ("maxsize", "d")

    def __init__(self, maxsize: int = 64):
        self.maxsize = int(maxsize)
        self.d = {}

    def get(self, k):
        v = self.d.get(k, None)
        if v is None:
            return None
        self.d.pop(k, None)
        self.d[k] = v
        return v

    def put(self, k, v):
        if k in self.d:
            self.d.pop(k, None)
        self.d[k] = v
        if len(self.d) > self.maxsize:
            oldest = next(iter(self.d))
            self.d.pop(oldest, None)


_WORKER_IMG_CACHE = None


class RSNA_dataset_cached(RSNA_dataset):
    def __getitem__(self, idx):
        global _WORKER_IMG_CACHE
        if _WORKER_IMG_CACHE is None:
            _WORKER_IMG_CACHE = _LRUCache(maxsize=64)

        p = self.dcm_paths[idx]
        x = _WORKER_IMG_CACHE.get(p)
        if x is None:
            x = read_xray(p, img_size=self.img_size)
            _WORKER_IMG_CACHE.put(p, x)

        y = self.label[idx]
        return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long)


train_dataset = RSNA_dataset_cached(train_paths, y_train, IMG_SIZE)

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

validation_dataset = RSNA_dataset_cached(val_paths, y_validation, IMG_SIZE)
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

model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
def pf1_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    y_true = y_true.astype(np.float32)
    y_prob = np.clip(y_prob.astype(np.float32), 0.0, 1.0)
    pTP = float(np.sum(y_true * y_prob))
    pFP = float(np.sum((1.0 - y_true) * y_prob))
    TP = float(np.sum(y_true))
    FN = float(np.sum(y_true * (1.0 - y_prob)))
    pPrecision = pTP / (pTP + pFP + eps)
    pRecall = pTP / (TP + FN + eps)
    return float(2.0 * pPrecision * pRecall / (pPrecision + pRecall + eps))


def predict_probs(loader, model) -> np.ndarray:
    model.eval()
    out_probs = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            logits = model(imgs)
            p = (
                torch.softmax(logits, dim=1)[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            out_probs.append(p)
    return np.concatenate(out_probs, axis=0)


val_probs = predict_probs(validation_dataloader, model_ft)
y_val = y_validation.astype(np.int64)

scales = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0], dtype=np.float32)
best_s = 1.0
best_pf1 = -1.0
for s in scales:
    p_adj = np.clip(val_probs * float(s), 0.0, 1.0)
    score = pf1_score(y_val, p_adj)
    if score > best_pf1:
        best_pf1 = score
        best_s = float(s)

print(f"Selected val scaling best_s={best_s}, val_pf1={best_pf1:.6f}")



## === cell 9
test_df = pd.read_csv(TEST_CSV)

_test_patient = test_df["patient_id"].to_numpy()
_test_image = test_df["image_id"].to_numpy()
_test_dcm_paths = np.array(
    [
        os.path.join(TEST_IMG_DIR, str(pid), f"{iid}.dcm")
        for pid, iid in zip(_test_patient, _test_image)
    ],
    dtype=object,
)

_TEST_WORKER_IMG_CACHE = None


class RSNATestDicomDataset(Dataset):
    def __init__(self, dcm_paths, img_size: int):
        self.dcm_paths = dcm_paths
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.dcm_paths)

    def __getitem__(self, idx):
        global _TEST_WORKER_IMG_CACHE
        if _TEST_WORKER_IMG_CACHE is None:
            _TEST_WORKER_IMG_CACHE = _LRUCache(maxsize=64)

        p = self.dcm_paths[idx]
        x = _TEST_WORKER_IMG_CACHE.get(p)
        if x is None:
            if not os.path.exists(p):
                x = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
            else:
                x = read_xray(p, img_size=self.img_size)
            _TEST_WORKER_IMG_CACHE.put(p, x)

        ok = bool(float(np.max(x)) > 0.0)
        return torch.from_numpy(x), ok


def _test_collate(batch):
    imgs, ok = zip(*batch)
    return torch.stack(imgs, 0), torch.tensor(ok, dtype=torch.bool)


test_dataset = RSNATestDicomDataset(_test_dcm_paths, IMG_SIZE)
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
skipped_test = 0

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
            skipped_test += int((~ok_np).sum())

        probs[offset : offset + bsz] = p
        offset += bsz

probs = np.clip(probs * best_s, 0.0, 1.0)

test_df["cancer"] = probs
test_df["cancer"] = test_df["cancer"].fillna(0.0).clip(0.0, 1.0)
print(f"Test rows={len(test_df)}, skipped_test={skipped_test}")



## === cell 10
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



## === cell 11
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
