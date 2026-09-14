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

No external packages required in the script and installed.

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
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import cv2
except Exception as e:
    raise ImportError(
        "cv2 is required in this solution but is not available in the environment."
    ) from e

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import dicomsdl  # noqa: F401

    _HAVE_DICOMSDL = True
except Exception:
    _HAVE_DICOMSDL = False

try:
    import pydicom
    from pydicom.encaps import generate_pixel_data_frame

    _HAVE_PYDICOM = True
except Exception:
    _HAVE_PYDICOM = False

print("dicomsdl available:", _HAVE_DICOMSDL)
print("pydicom available:", _HAVE_PYDICOM)
print("torch:", torch.__version__)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    cv2.setNumThreads(0)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_CPU_COUNT = os.cpu_count() or 2

if torch.cuda.is_available():
    _NUM_WORKERS = min(8, _CPU_COUNT)
    _PREFETCH_FACTOR = 4
else:
    _NUM_WORKERS = min(8, _CPU_COUNT)
    _PREFETCH_FACTOR = 4




## === cell 1
data_dir = "/kaggle/input/rsna-breast-cancer-detection"
assert os.path.exists(data_dir), f"Expected data_dir not found: {data_dir}"

target_size = [216, 216]
batch_size = 64
num_epochs = 5




## === cell 2
train_df = pd.read_csv(f"{data_dir}/train.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")
sample_sub = pd.read_csv(f"{data_dir}/sample_submission.csv")

train_df["dcm_path"] = (
    data_dir
    + "/train_images/"
    + train_df["patient_id"].astype(str)
    + "/"
    + train_df["image_id"].astype(str)
    + ".dcm"
)

test_df["dcm_path"] = (
    data_dir
    + "/test_images/"
    + test_df["patient_id"].astype(str)
    + "/"
    + test_df["image_id"].astype(str)
    + ".dcm"
)

print(
    "train_df:",
    train_df.shape,
    "test_df:",
    test_df.shape,
    "sample_sub:",
    sample_sub.shape,
)
train_df.head(2)




## === cell 3
from functools import lru_cache


def _decode_embedded_frame_with_opencv(frame_bytes: bytes):
    """
    Fallback decode for encapsulated (compressed) pixel data frames using OpenCV.
    Returns grayscale ndarray or raises.
    """
    buf = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cv2.imdecode failed on encapsulated frame bytes.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _read_dicom_pixels_uncached(path):
    """
    Returns: (pixel_array, photometric_interpretation_str_or_None, is_uint8_display)
    """
    if _HAVE_DICOMSDL:
        dcm = dicomsdl.open(path)
        pi = getattr(dcm, "PhotometricInterpretation", None)
        storedvalue = pi == "MONOCHROME1"

        try:
            arr_u8 = dcm.pixelData(storedvalue=storedvalue, dtype=np.uint8)
            if arr_u8 is not None:
                return arr_u8, pi, True
        except Exception:
            pass

        arr = dcm.pixelData(storedvalue=storedvalue)
        return arr.astype(np.float32, copy=False), pi, False

    if not _HAVE_PYDICOM:
        raise ImportError("No DICOM reader available (dicomsdl or pydicom).")

    ds = pydicom.dcmread(path, force=True)

    try:
        arr = ds.pixel_array.astype(np.float32, copy=False)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        pi = str(getattr(ds, "PhotometricInterpretation", "") or "")
        return arr, pi, False
    except Exception:
        try:
            if not hasattr(ds, "PixelData"):
                raise RuntimeError("DICOM has no PixelData to decode.")
            frame0 = next(generate_pixel_data_frame(ds.PixelData))
            img = _decode_embedded_frame_with_opencv(frame0).astype(
                np.float32, copy=False
            )
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            img = img * slope + intercept
            pi = str(getattr(ds, "PhotometricInterpretation", "") or "")
            return img, pi, False
        except Exception as e2:
            raise RuntimeError(f"Failed to decode DICOM pixels for: {path}") from e2


@lru_cache(maxsize=8192)
def _read_dicom_pixels_cached(path):
    return _read_dicom_pixels_uncached(path)


def _read_dicom_pixels(path):
    if _NUM_WORKERS == 0:
        return _read_dicom_pixels_cached(path)
    return _read_dicom_pixels_uncached(path)


_cv2_resize = cv2.resize
_cv2_inter = cv2.INTER_LINEAR


def normalize_xray(path, fix_monochrome=True):
    try:
        dicom_arr, photometric, is_u8 = _read_dicom_pixels(path)
    except Exception:
        return np.zeros((target_size[0], target_size[1]), dtype=np.uint8)

    if fix_monochrome and photometric == "MONOCHROME1":
        if is_u8:
            dicom_arr = np.uint8(255) - dicom_arr
        else:
            dicom_arr = np.max(dicom_arr) - dicom_arr

    if is_u8:
        mn = int(dicom_arr.min())
        mx = int(dicom_arr.max())
        if mx > mn:
            arr = dicom_arr.astype(np.float32, copy=False)
            arr = (arr - mn) * (255.0 / (mx - mn))
            return arr.astype(np.uint8)
        return np.zeros_like(dicom_arr, dtype=np.uint8)

    mn = float(np.min(dicom_arr))
    mx = float(np.max(dicom_arr))
    if mx > mn:
        dicom_arr = (dicom_arr - mn) / (mx - mn)
    else:
        dicom_arr = dicom_arr * 0.0

    dicom_arr = (dicom_arr * 255.0).astype(np.uint8)
    return dicom_arr


def crop_and_resize(image, crop_size=5):
    if (
        crop_size > 0
        and image.shape[0] > 2 * crop_size
        and image.shape[1] > 2 * crop_size
    ):
        image = image[crop_size:-crop_size, crop_size:-crop_size]
    image = _cv2_resize(image, target_size[::-1], interpolation=_cv2_inter)
    return image


def preprocess_image(file_path):
    image = normalize_xray(file_path)
    image = crop_and_resize(image)
    return image


def pfbeta_torch(labels, preds, beta=1):
    preds = np.clip(preds, 0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    if (ctp + cfp) == 0 or y_true_count == 0:
        return 0.0
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        return (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
    return 0.0




## === cell 4
def _collate_train(batch):
    images_u8 = np.stack([b["images"] for b in batch], axis=0)  # (B,H,W) uint8
    t = torch.from_numpy(images_u8).unsqueeze(1)  # (B,1,H,W) uint8
    t = t.expand(-1, 3, -1, -1).contiguous().to(dtype=torch.float32).div_(255.0)
    cancer = torch.tensor([float(b["cancer"]) for b in batch], dtype=torch.float32)
    return {"images": t, "cancer": cancer}


def _collate_test(batch):
    pred_ids = [b[0] for b in batch]
    images_u8 = np.stack([b[1] for b in batch], axis=0)  # (B,H,W) uint8
    t = torch.from_numpy(images_u8).unsqueeze(1)
    t = t.expand(-1, 3, -1, -1).contiguous().to(dtype=torch.float32).div_(255.0)
    return pred_ids, t


class MyDataset(Dataset):
    def __init__(self, df, transform=None, cache_images=False):
        df = df.reset_index(drop=True)
        self.transform = transform

        self._paths = df["dcm_path"].to_numpy()
        self._cancer = df["cancer"].astype(np.int64).to_numpy()

        self.cache_images = bool(cache_images)
        self._cache = [None] * len(self._paths) if self.cache_images else None

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        cancer = float(self._cancer[index])
        full_path = self._paths[index]

        if self.cache_images:
            img = self._cache[index]
            if img is None:
                img = preprocess_image(full_path)
                self._cache[index] = img
        else:
            img = preprocess_image(full_path)

        if self.transform:
            pass

        return {"cancer": cancer, "images": img}




## === cell 5
class PretrainedBinaryClassifier(nn.Module):
    def __init__(self):
        super(PretrainedBinaryClassifier, self).__init__()
        self.model = models.resnet50(pretrained=False)

        weights_path = "/kaggle/input/my-requirement-files/resnet50-0676ba61.pth"
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x




## === cell 6
train_subset_0 = train_df[train_df.cancer == 0].iloc[:55, :]
train_subset_1 = train_df[train_df.cancer == 1].iloc[:45, :]
train_combined = pd.concat([train_subset_0, train_subset_1], axis=0).reset_index(
    drop=True
)

print("train_combined:", train_combined.shape)
print(train_combined["cancer"].value_counts())

training_set, validation_set = train_test_split(
    train_combined, test_size=0.2, random_state=42, stratify=train_combined["cancer"]
)
print("training_set:", training_set.shape, "validation_set:", validation_set.shape)




## === cell 7
def _precompute_u8_images(paths: np.ndarray) -> np.ndarray:
    out = np.empty((len(paths), target_size[0], target_size[1]), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = preprocess_image(str(p))
    return out


class InMemoryTrainDataset(Dataset):
    def __init__(self, images_u8: np.ndarray, cancer: np.ndarray):
        self.images_u8 = images_u8
        self.cancer = cancer.astype(np.float32, copy=False)

    def __len__(self):
        return self.images_u8.shape[0]

    def __getitem__(self, idx):
        return {"cancer": float(self.cancer[idx]), "images": self.images_u8[idx]}


train_paths = training_set["dcm_path"].to_numpy()
val_paths = validation_set["dcm_path"].to_numpy()

train_images_u8 = _precompute_u8_images(train_paths)
val_images_u8 = _precompute_u8_images(val_paths)

train_dataset = InMemoryTrainDataset(train_images_u8, training_set["cancer"].to_numpy())
val_dataset = InMemoryTrainDataset(val_images_u8, validation_set["cancer"].to_numpy())

transform = None
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=_collate_train,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=_collate_train,
)

print("num_workers (train/val):", 0)
print("len(train_loader):", len(train_loader), "len(val_loader):", len(val_loader))




## === cell 8
model = PretrainedBinaryClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("device:", device)




## === cell 9
train_losses, train_acc_metric, train_pf1_metric = [], [], []
val_losses, val_acc_metric, val_pf1_metric = [], [], []

for epoch in range(num_epochs):
    running_train_loss = 0.0
    running_train_acc = 0.0

    train_all_targets = []
    train_all_outputs = []

    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data["images"], data["cancer"]
        targets = targets.view(-1, 1)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        predicted = torch.round(outputs)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        running_train_acc += accuracy

        train_all_targets.append(targets.detach().cpu())
        train_all_outputs.append(outputs.detach().cpu())

    avg_train_loss = running_train_loss / max(1, len(train_loader))
    avg_train_acc = running_train_acc / max(1, len(train_loader))

    train_targets_np = torch.cat(train_all_targets, dim=0).numpy()
    train_outputs_np = torch.cat(train_all_outputs, dim=0).numpy()
    epoch_train_pf1 = pfbeta_torch(train_targets_np, train_outputs_np)

    train_losses.append(avg_train_loss)
    train_acc_metric.append(avg_train_acc)
    train_pf1_metric.append(epoch_train_pf1)

    print(
        f"Epoch {epoch+1}, avg training loss: {avg_train_loss:.3f}, avg training accuracy: {avg_train_acc:.3f}, avg training pf1: {epoch_train_pf1:.3f}"
    )

    model.eval()
    running_val_loss = 0.0
    running_val_acc = 0.0
    val_all_targets = []
    val_all_outputs = []

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, targets = data["images"], data["cancer"]
            targets = targets.view(-1, 1)

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            predicted = torch.round(outputs)
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0)

            running_val_loss += loss.item()
            running_val_acc += accuracy

            val_all_targets.append(targets.detach().cpu())
            val_all_outputs.append(outputs.detach().cpu())

    avg_val_loss = running_val_loss / max(1, len(val_loader))
    avg_val_acc = running_val_acc / max(1, len(val_loader))

    val_targets_np = torch.cat(val_all_targets, dim=0).numpy()
    val_outputs_np = torch.cat(val_all_outputs, dim=0).numpy()
    epoch_val_pf1 = pfbeta_torch(val_targets_np, val_outputs_np)

    val_losses.append(avg_val_loss)
    val_acc_metric.append(avg_val_acc)
    val_pf1_metric.append(epoch_val_pf1)

    print(
        f"Epoch {epoch+1}, avg validation loss: {avg_val_loss:.3f}, avg validation accuracy: {avg_val_acc:.3f}, avg validation pf1: {epoch_val_pf1:.3f}"
    )




## === cell 10
class TestDataset(Dataset):
    def __init__(self, df, transform=None, cache_images=False):
        df = df.reset_index(drop=True)
        self.transform = transform
        self._paths = df["dcm_path"].to_numpy()
        self._pred_ids = df["prediction_id"].astype(str).to_numpy()

        self.cache_images = bool(cache_images)
        self._cache = [None] * len(self._paths) if self.cache_images else None

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        full_path = self._paths[index]
        img = preprocess_image(full_path)

        if self.transform:
            pass
        return (self._pred_ids[index], img)




## === cell 11
test_transform = None
test_dataset = TestDataset(test_df, transform=test_transform, cache_images=False)

test_batch_size = 128
test_num_workers = _NUM_WORKERS  # use available CPUs
test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=test_num_workers,
    pin_memory=True,
    persistent_workers=(test_num_workers > 0),
    prefetch_factor=_PREFETCH_FACTOR if test_num_workers > 0 else None,
    collate_fn=_collate_test,
)

pred_predid = []
pred_prob = []

model.eval()
with torch.no_grad():
    for prediction_ids, images in test_dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = outputs.squeeze(1).detach().cpu().numpy()
        pred_predid.extend(prediction_ids)  # list of strings
        pred_prob.extend(probs.tolist())  # keep same final python list type

print(
    "num test rows:",
    len(test_df),
    "num preds:",
    len(pred_prob),
    "num pred_ids:",
    len(pred_predid),
    "test_num_workers:",
    test_num_workers,
)




## === cell 12
pred_df = pd.DataFrame({"prediction_id": pred_predid, "cancer": pred_prob})
sub = pred_df.groupby("prediction_id", as_index=False)["cancer"].mean()

sub = sample_sub[["prediction_id"]].merge(sub, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(0.0).clip(0.0, 1.0)

print(sub.head())
print("submission rows:", len(sub), "expected:", len(sample_sub))




## === cell 13
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path))
