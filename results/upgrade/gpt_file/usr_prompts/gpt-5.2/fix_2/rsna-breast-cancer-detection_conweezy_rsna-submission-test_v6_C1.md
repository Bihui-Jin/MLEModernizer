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




## === cell 1
def read_xray(file_path, img_size=None):
    dicom = pydicom.dcmread(file_path)
    img = dicom.pixel_array

    if getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1":
        img = np.max(img) - img

    if img_size:
        img_size = (img_size, img_size)
        img = cv2.resize(img, dsize=img_size)

    img = img[np.newaxis].astype("float32")

    mx = float(np.max(img))
    if mx > 0:
        img = img / mx
    else:
        img = img * 0.0

    return img




## === cell 2
DATA_ROOT = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

train_df_full = pd.read_csv(TRAIN_CSV)

pos_df = train_df_full[train_df_full["cancer"] == 1]
neg_df = train_df_full[train_df_full["cancer"] == 0]

n_total = 2000
n_pos = min(
    len(pos_df), max(50, int(0.10 * n_total))
)  # small positive share for learning signal
n_neg = min(len(neg_df), n_total - n_pos)

rng = np.random.default_rng(42)
pos_idx = (
    rng.choice(pos_df.index.values, size=n_pos, replace=False)
    if n_pos > 0
    else np.array([], dtype=int)
)
neg_idx = rng.choice(neg_df.index.values, size=n_neg, replace=False)

train_df = train_df_full.loc[np.concatenate([pos_idx, neg_idx])].copy()
train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

IMG_SIZE = 224

X_list, y_list = [], []
missing = 0
for row in train_df.itertuples(index=False):
    dcm_path = os.path.join(TRAIN_IMG_DIR, str(row.patient_id), f"{row.image_id}.dcm")
    if not os.path.exists(dcm_path):
        missing += 1
        continue
    try:
        x = read_xray(dcm_path, img_size=IMG_SIZE)  # (1,224,224)
        X_list.append(x)
        y_list.append(int(row.cancer))
    except Exception:
        missing += 1
        continue

if len(X_list) == 0:
    raise RuntimeError(
        "No training images could be loaded. Check DICOM availability/codecs/paths."
    )

X_train = np.stack(X_list, axis=0)  # (N,1,H,W)
y_train = np.asarray(y_list, dtype=np.int64)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

val_size = min(316, max(1, int(0.15 * len(y_train))))
X_validation = X_train[-val_size:]
y_validation = y_train[-val_size:]
X_train = X_train[:-val_size]
y_train = y_train[:-val_size]

print(f"Loaded train N={len(y_train)}, val N={len(y_validation)}, skipped={missing}")




## === cell 3
class RSNA_dataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.feature[idx]
        y = self.label[idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)




## === cell 4
train_dataset = RSNA_dataset(X_train, y_train)
train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)

validation_dataset = RSNA_dataset(X_validation, y_validation)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / max(1, dataset_sizes[phase])
            epoch_acc = running_corrects / max(1, dataset_sizes[phase])

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

model_ft = model_ft.to(device)

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



## === cell 9
model_ft.eval()

probs = []
batch_imgs = []
batch_rows = []

BATCH = 32
skipped_test = 0

with torch.no_grad():
    for row in test_df.itertuples(index=False):
        dcm_path = os.path.join(
            TEST_IMG_DIR, str(row.patient_id), f"{row.image_id}.dcm"
        )
        if not os.path.exists(dcm_path):
            skipped_test += 1
            probs.append(np.nan)
            continue
        try:
            x = read_xray(dcm_path, img_size=IMG_SIZE)  # (1,224,224)
            batch_imgs.append(torch.from_numpy(x).float())
            batch_rows.append(row)
        except Exception:
            skipped_test += 1
            probs.append(np.nan)
            continue

        if len(batch_imgs) == BATCH:
            inp = torch.stack(batch_imgs, dim=0).to(
                device, non_blocking=True
            )  # (B,1,224,224)
            out = model_ft(inp)
            p = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
            batch_imgs, batch_rows = [], []

    if len(batch_imgs) > 0:
        inp = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        out = model_ft(inp)
        p = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
        probs.extend(p.tolist())

if len(probs) != len(test_df):
    probs = []
    with torch.no_grad():
        for row in test_df.itertuples(index=False):
            dcm_path = os.path.join(
                TEST_IMG_DIR, str(row.patient_id), f"{row.image_id}.dcm"
            )
            if not os.path.exists(dcm_path):
                probs.append(0.0)
                continue
            try:
                x = read_xray(dcm_path, img_size=IMG_SIZE)
                inp = torch.from_numpy(x).unsqueeze(0).float().to(device)
                out = model_ft(inp)
                p = torch.softmax(out, dim=1)[:, 1].item()
                probs.append(p)
            except Exception:
                probs.append(0.0)

test_df["cancer"] = np.asarray(probs, dtype=np.float32)

test_df["cancer"] = test_df["cancer"].fillna(0.0).clip(0.0, 1.0)

print(f"Test rows={len(test_df)}, skipped_test={skipped_test}")



## === cell 10
submission_df = test_df[["prediction_id", "cancer"]]
submission = submission_df.groupby(["prediction_id"])["cancer"].mean().reset_index()



## === cell 11
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
