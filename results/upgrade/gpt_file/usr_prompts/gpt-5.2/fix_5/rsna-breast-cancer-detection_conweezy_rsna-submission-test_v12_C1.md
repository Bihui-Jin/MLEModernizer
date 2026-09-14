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

0.0421190355210599

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The timeout is dominated by per-image DICOM decoding and per-row GPU inference in a Python loop. I keep the exact same model, training loop, loss, and image preprocessing, but I (1) speed up DICOM reads by avoiding unnecessary parsing, (2) preallocate and fill the training tensors instead of growing Python lists, and (3) batch test-time inference with a DataLoader so the GPU runs efficiently and Python overhead is minimized. These changes are equivalent in outputs (same pixels, same normalization, same softmax) and only remove overhead and redundant work.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the “no training images could be loaded” failure by correcting the dataset path fallback logic so it works in both `/kaggle/input/...` and the provided `/kaggle/data/...` layout, without changing the model/training core. Then I make DICOM reading robust by requesting the minimal tags needed for MONOCHROME inversion and guarding pixel decoding so images reliably load. Finally, I ensure the pipeline always reaches submission writing and that `submission.csv` matches `sample_submission.csv` ordering and columns, which should move the score up from 0.0 by producing non-degenerate predictions instead of failing early.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the 0.0 score by making `resolve_data_root()` actually find the dataset in your environment (`/kaggle/data/...`), so training images load instead of raising “No training images could be loaded”. I also make DICOM pixel decoding more robust for JPEG2000-compressed mammograms by using `pydicom.pixels.pixel_array()` with a safe fallback, which prevents silent skip-all failures on many files. Finally, I keep the model/training loop unchanged, but ensure inference always produces valid probabilities and writes a correctly aligned `submission.csv` with the required columns and ordering.'

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
import time
import copy
from torch.optim import lr_scheduler

try:
    from pydicom.pixels import pixel_array as pydicom_pixel_array
except Exception:
    pydicom_pixel_array = None




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 2
def resolve_data_root(preferred="/kaggle/input/rsna-breast-cancer-detection"):
    candidates = [
        preferred,
        "/kaggle/data/rsna-breast-cancer-detection",
        "/kaggle/input/rsna-breast-cancer-detection/rsna-breast-cancer-detection",
        "/kaggle/data/rsna-breast-cancer-detection/rsna-breast-cancer-detection",
        "/kaggle/data",
        "/kaggle/input",
    ]

    def _is_valid_root(root: str) -> bool:
        return os.path.exists(os.path.join(root, "train.csv")) and os.path.exists(
            os.path.join(root, "train_images")
        )

    for c in candidates:
        if _is_valid_root(c):
            return c

    for base in candidates:
        cand = os.path.join(base, "rsna-breast-cancer-detection")
        if _is_valid_root(cand):
            return cand
        cand2 = os.path.join(
            base, "rsna-breast-cancer-detection", "rsna-breast-cancer-detection"
        )
        if _is_valid_root(cand2):
            return cand2

    return preferred


DATA_ROOT = resolve_data_root("/kaggle/input/rsna-breast-cancer-detection")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_ROOT = os.path.join(DATA_ROOT, "test_images")

print("Resolved DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TRAIN_IMG_ROOT exists:", os.path.exists(TRAIN_IMG_ROOT))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("TEST_IMG_ROOT exists:", os.path.exists(TEST_IMG_ROOT))




## === cell 3
def read_xray(file_path, img_size=None):
    dicom = pydicom.dcmread(
        file_path,
        stop_before_pixels=False,
        force=True,
        specific_tags=[
            "PhotometricInterpretation",
            "PixelData",
            "BitsStored",
            "BitsAllocated",
            "SamplesPerPixel",
            "PixelRepresentation",
            "RescaleIntercept",
            "RescaleSlope",
        ],
    )

    if pydicom_pixel_array is not None:
        try:
            img = pydicom_pixel_array(dicom)
        except Exception:
            img = dicom.pixel_array
    else:
        img = dicom.pixel_array

    if getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1":
        img = np.max(img) - img

    if img_size:
        img_size = (img_size, img_size)
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_AREA)

    img = img[np.newaxis]  # (1, H, W)

    maxv = np.max(img)
    if maxv > 0:
        img = img / maxv
    img = img.astype("float32")

    return img




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
train_df_small = train_df.iloc[:2000].copy()

IMG_SIZE = 224  # compatible with ResNet default stem; minimal and standard

n_max = len(train_df_small)
X_train_arr = np.empty((n_max, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
y_train_arr = np.empty((n_max,), dtype=np.int64)

missing = 0
n_ok = 0

train_img_root = TRAIN_IMG_ROOT
for r in train_df_small.itertuples(index=False):
    dcm_path = os.path.join(train_img_root, str(r.patient_id), f"{r.image_id}.dcm")
    if not os.path.exists(dcm_path):
        missing += 1
        continue
    try:
        img = read_xray(dcm_path, img_size=IMG_SIZE)  # (1, 224, 224)
        X_train_arr[n_ok] = img
        y_train_arr[n_ok] = int(r.cancer)
        n_ok += 1
    except Exception:
        missing += 1
        continue

if n_ok == 0:
    raise RuntimeError(
        f"No training images could be loaded from {TRAIN_IMG_ROOT}. "
        f"Check mount/path; DATA_ROOT={DATA_ROOT}"
    )

X_train = X_train_arr[:n_ok]
y_train = y_train_arr[:n_ok]

X_train, y_train = shuffle(X_train, y_train, random_state=42)

print(
    f"Loaded train subset: X_train={X_train.shape}, positives={int((y_train==1).sum())}, missing/skipped={missing}"
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4078835920.py in <cell line: 0>()
     29 
     30 if n_ok == 0:
---> 31     raise RuntimeError(
     32         f"No training images could be loaded from {TRAIN_IMG_ROOT}. "
     33         f"Check mount/path; DATA_ROOT={DATA_ROOT}"

RuntimeError: No training images could be loaded from /kaggle/input/rsna-breast-cancer-detection/train_images. Check mount/path; DATA_ROOT=/kaggle/input/rsna-breast-cancer-detection

## === cell 5
val_size = min(316, len(y_train) // 5 if len(y_train) >= 5 else 0)
if val_size < 1:
    val_size = max(1, len(y_train) // 10)

X_validation = X_train[-val_size:]
y_validation = y_train[-val_size:]

X_train = X_train[:-val_size]
y_train = y_train[:-val_size]

print(f"Train split: {X_train.shape}, Val split: {X_validation.shape}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3104569195.py in <cell line: 0>()
----> 1 val_size = min(316, len(y_train) // 5 if len(y_train) >= 5 else 0)
      2 if val_size < 1:
      3     val_size = max(1, len(y_train) // 10)
      4 
      5 X_validation = X_train[-val_size:]

NameError: name 'y_train' is not defined

## === cell 6
class RSNA_dataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.feature[idx]).float()
        y = torch.tensor(self.label[idx]).long()
        return x, y




## === cell 7
_nw = min(4, (os.cpu_count() or 2))
train_dataset = RSNA_dataset(X_train, y_train)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=_nw,
    pin_memory=True,
    persistent_workers=(_nw > 0),
)

validation_dataset = RSNA_dataset(X_validation, y_validation)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=_nw,
    pin_memory=True,
    persistent_workers=(_nw > 0),
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/687649440.py in <cell line: 0>()
      1 _nw = min(4, (os.cpu_count() or 2))
----> 2 train_dataset = RSNA_dataset(X_train, y_train)
      3 train_dataloader = DataLoader(
      4     train_dataset,
      5     batch_size=32,

NameError: name 'X_train' is not defined

## === cell 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloaders = {"train": train_dataloader, "val": validation_dataloader}
dataset_sizes = {"train": len(train_dataset), "val": len(validation_dataset)}
print("Device:", device)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1742739132.py in <cell line: 0>()
      1 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
----> 2 dataloaders = {"train": train_dataloader, "val": validation_dataloader}
      3 dataset_sizes = {"train": len(train_dataset), "val": len(validation_dataset)}
      4 print("Device:", device)
      5 

NameError: name 'train_dataloader' is not defined

## === cell 9
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

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / max(1, dataset_sizes[phase])
            epoch_acc = running_corrects.double() / max(1, dataset_sizes[phase])

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    _ = time.time() - since
    model.load_state_dict(best_model_wts)
    return model




## === cell 10
epochs = 2

model_ft = models.resnet18(weights=None)
model_ft.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)



## === cell 11
model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=epochs,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3841457220.py in <cell line: 0>()
      4     optimizer_ft,
      5     exp_lr_scheduler,
----> 6     dataloaders,
      7     dataset_sizes,
      8     num_epochs=epochs,

NameError: name 'dataloaders' is not defined

## === cell 12
test_df = pd.read_csv(TEST_CSV)


class RSNATestDicomDataset(Dataset):
    def __init__(self, df, img_root, img_size):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        dcm_path = os.path.join(self.img_root, str(r.patient_id), f"{r.image_id}.dcm")
        if not os.path.exists(dcm_path):
            return r.prediction_id, None
        try:
            img = read_xray(dcm_path, img_size=self.img_size)  # (1,H,W) float32
            x = torch.from_numpy(img)  # (1,H,W)
            return r.prediction_id, x
        except Exception:
            return r.prediction_id, None


def _test_collate(batch):
    pred_ids = [b[0] for b in batch]
    xs = [b[1] for b in batch]
    keep = [i for i, x in enumerate(xs) if x is not None]
    if len(keep) == 0:
        return pred_ids, None, keep
    x = torch.stack([xs[i] for i in keep], dim=0)  # (B,1,H,W)
    return pred_ids, x, keep


model_ft.eval()
test_ds = RSNATestDicomDataset(test_df, TEST_IMG_ROOT, IMG_SIZE)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=_nw,
    pin_memory=True,
    persistent_workers=(_nw > 0),
    collate_fn=_test_collate,
)

probs = np.full((len(test_df),), np.nan, dtype=np.float32)

row_base = 0
missing_test = 0
with torch.no_grad():
    for pred_ids, x, keep in test_loader:
        bsz = len(pred_ids)
        if x is None:
            missing_test += bsz
            row_base += bsz
            continue
        x = x.to(device, non_blocking=True).float()
        logits = model_ft(x)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)

        for j, k in enumerate(keep):
            probs[row_base + k] = p[j]
        missing_test += bsz - len(keep)
        row_base += bsz

test_df_pred = test_df[["prediction_id"]].copy()
test_df_pred["cancer"] = probs

nanmean = np.nanmean(test_df_pred["cancer"].values)
global_mean = float(nanmean) if np.isfinite(nanmean) else 0.0
test_df_pred["cancer"] = test_df_pred["cancer"].fillna(global_mean)

submission = test_df_pred.groupby("prediction_id", as_index=False)["cancer"].mean()

sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
submission = sample_sub[["prediction_id"]].merge(
    submission, on="prediction_id", how="left"
)
submission["cancer"] = submission["cancer"].fillna(global_mean).astype(np.float32)

print(
    "Test rows:",
    len(test_df),
    "Unique prediction_id:",
    submission.shape[0],
    "Missing test images:",
    missing_test,
)



## === cell 13
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
