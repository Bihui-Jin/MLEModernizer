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
import gc
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models

from PIL import Image

from sklearn.model_selection import train_test_split



## === cell 1
data_dir = "/kaggle/input/rsna-breast-cancer-detection"

print("data_dir exists:", os.path.exists(data_dir))
print("train.csv exists:", os.path.exists(f"{data_dir}/train.csv"))
print("test.csv exists:", os.path.exists(f"{data_dir}/test.csv"))
print(
    "sample_submission.csv exists:", os.path.exists(f"{data_dir}/sample_submission.csv")
)



## === cell 2
target_size = [216, 216]
batch_size = 64
num_epochs = 5



## === cell 3
train_df = pd.read_csv(f"{data_dir}/train.csv")
train_df["dcm_path"] = (
    data_dir
    + "/train_images/"
    + train_df["patient_id"].astype(str)
    + "/"
    + train_df["image_id"].astype(str)
    + ".dcm"
)
print(train_df.shape)
train_df.head(2)



## === cell 4
test_df = pd.read_csv(f"{data_dir}/test.csv")
test_df["dcm_path"] = (
    data_dir
    + "/test_images/"
    + test_df["patient_id"].astype(str)
    + "/"
    + test_df["image_id"].astype(str)
    + ".dcm"
)
print(test_df.shape)
test_df.head(2)




## === cell 5
def normalize_xray(path, fix_monochrome=True):
    ds = pydicom.dcmread(path, force=True)

    try:
        arr = ds.pixel_array
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
    except Exception:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise

    arr = arr.astype(np.float32)

    if fix_monochrome and getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = np.max(arr) - arr

    arr = arr - np.min(arr)
    mx = np.max(arr)
    if mx > 0:
        arr = arr / mx
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    return arr


def crop_and_resize(image, crop_size=5):
    if (
        crop_size > 0
        and image.shape[0] > 2 * crop_size
        and image.shape[1] > 2 * crop_size
    ):
        image = image[crop_size:-crop_size, crop_size:-crop_size]
    image = cv2.resize(image, target_size[::-1], cv2.INTER_LINEAR)
    return image


def preprocess_and_save(file_path):
    try:
        image = normalize_xray(file_path)
        h, w = image.shape[:2]
        image = crop_and_resize(image)
    except Exception:
        h, w = target_size[0], target_size[1]
        image = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)

    parts = file_path.split("/")
    pid = parts[-2]
    iid = os.path.splitext(parts[-1])[0]
    return pid, iid, h, w, image


def pfbeta_torch(labels, preds, beta=1):
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    preds = np.asarray(preds).astype(np.float32).reshape(-1)

    preds = np.clip(preds, 0, 1)
    y_true_count = labels.sum()
    if y_true_count == 0:
        return 0.0
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-12)
    c_recall = ctp / (y_true_count + 1e-12)
    if c_precision > 0 and c_recall > 0:
        return (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall + 1e-12)
        )
    return 0.0




## === cell 6
class MyDataset(Dataset):
    def __init__(self, df, transform=None, cache_images=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.paths = self.df["dcm_path"].to_numpy()
        self.cancers = self.df["cancer"].to_numpy(dtype=np.float32)

        self.cache_images = cache_images
        self._img_cache = [None] * len(self.df) if cache_images else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cancer = torch.tensor(self.cancers[index], dtype=torch.float32)
        path = self.paths[index]

        if self.cache_images and self._img_cache[index] is not None:
            img = self._img_cache[index]
        else:
            _, _, _, _, img = preprocess_and_save(path)
            if self.cache_images:
                self._img_cache[index] = img

        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"cancer": cancer, "images": img}




## === cell 7
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.paths = self.df["dcm_path"].to_numpy()
        self.prediction_ids = self.df["prediction_id"].astype(str).to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        _, _, _, _, img = preprocess_and_save(path)
        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return self.prediction_ids[index], img




## === cell 8
class PretrainedBinaryClassifier(nn.Module):
    def __init__(self):
        super(PretrainedBinaryClassifier, self).__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
        except Exception:
            self.model = models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x




## === cell 9
train_subset_0 = train_df[train_df.cancer == 0].iloc[:55, :]
train_subset_1 = train_df[train_df.cancer == 1].iloc[:45, :]
train_combined = pd.concat([train_subset_0, train_subset_1])
print(train_combined.shape)
print(train_combined.laterality.value_counts())
print(train_combined.cancer.value_counts())
train_combined.reset_index(drop=True, inplace=True)



## === cell 10
training_set, validation_set = train_test_split(
    train_combined, test_size=0.2, random_state=42, stratify=train_combined["cancer"]
)
print(training_set.shape)
print(validation_set.shape)



## === cell 11
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MyDataset(training_set, transform=transform, cache_images=True)
val_dataset = MyDataset(validation_set, transform=transform, cache_images=True)

print("The training dataset contains", len(train_dataset), "samples.")
print("The val dataset contains", len(val_dataset), "samples.")



## === cell 12
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)

print("train batches:", len(train_loader))
print("val batches:", len(val_loader))



## === cell 13
print(torch.cuda.is_available())
print(torch.cuda.device_count())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PretrainedBinaryClassifier().to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)



## === cell 14
train_losses = []
train_acc_metric = []
train_pf1_metric = []

val_losses = []
val_acc_metric = []
val_pf1_metric = []

for epoch in range(num_epochs):
    running_train_loss = 0.0
    running_train_acc = 0.0
    running_train_pf1 = 0.0

    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data["images"], data["cancer"]
        targets = targets.view(-1, 1)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        predicted = torch.round(outputs)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)
        pf1 = pfbeta_torch(
            targets.detach().cpu().numpy(), outputs.detach().cpu().numpy()
        )

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        running_train_acc += accuracy
        running_train_pf1 += pf1

    avg_train_loss = running_train_loss / max(1, len(train_loader))
    avg_train_acc = running_train_acc / max(1, len(train_loader))
    avg_train_pf1 = running_train_pf1 / max(1, len(train_loader))

    train_losses.append(avg_train_loss)
    train_acc_metric.append(avg_train_acc)
    train_pf1_metric.append(avg_train_pf1)

    print(
        f"Epoch {epoch+1}, avg training loss: {avg_train_loss:.3f}, avg training accuracy: {avg_train_acc:.3f}, avg training pf1: {avg_train_pf1:.3f}"
    )

    model.eval()
    running_val_loss = 0.0
    running_val_acc = 0.0
    running_val_pf1 = 0.0

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
            pf1 = pfbeta_torch(
                targets.detach().cpu().numpy(), outputs.detach().cpu().numpy()
            )

            running_val_loss += loss.item()
            running_val_acc += accuracy
            running_val_pf1 += pf1

    avg_val_loss = running_val_loss / max(1, len(val_loader))
    avg_val_acc = running_val_acc / max(1, len(val_loader))
    avg_val_pf1 = running_val_pf1 / max(1, len(val_loader))

    val_losses.append(avg_val_loss)
    val_acc_metric.append(avg_val_acc)
    val_pf1_metric.append(avg_val_pf1)

    print(
        f"Epoch {epoch+1}, avg validation loss: {avg_val_loss:.3f}, avg validation accuracy: {avg_val_acc:.3f}, avg validation pf1: {avg_val_pf1:.3f}"
    )



## === cell 15
test_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = TestDataset(test_df, transform=test_transform)

num_workers = min(8, (os.cpu_count() or 2))
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

print("test rows:", len(test_dataset))
print("test batches:", len(test_dataloader))
print("num_workers:", num_workers)



## === cell 16
model.eval()
pred_rows = []

with torch.no_grad():
    for batch in test_dataloader:
        prediction_ids, images = batch
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = outputs.detach().cpu().numpy().reshape(-1)
        pred_rows.extend(zip(prediction_ids, probs.astype(np.float32).tolist()))

pred_df = pd.DataFrame(pred_rows, columns=["prediction_id", "cancer"])
print(pred_df.shape)
pred_df.head()



## === cell 17
sub = pred_df.groupby("prediction_id", as_index=False)["cancer"].mean()

sample_sub = pd.read_csv(f"{data_dir}/sample_submission.csv")
sub = sample_sub[["prediction_id"]].merge(sub, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(0.0).clip(0.0, 1.0)

print(sub.shape)
sub.head()



## === cell 18
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub.columns))
print("submission.csv head:")
print(pd.read_csv("submission.csv").head())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
