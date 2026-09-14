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
import math
import random
import numpy as np
import pandas as pd

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import pydicom


def seed_everything(seed: int = 534):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(534)



## === cell 1
data_dir = "/kaggle/input/rsna-breast-cancer-detection"
tmp_output_dir = "/kaggle/tmp/output"

train_dir = os.path.join(tmp_output_dir, "train_images")
test_dir = os.path.join(tmp_output_dir, "test_images")



## === cell 2
target_size = [224, 224]
batch_size = 16
num_epochs = 6



## === cell 3
train_df = pd.read_csv(f"{data_dir}/train.csv")

train_df["dcm_path"] = train_df.apply(
    lambda i: os.path.join(
        f"{data_dir}", "train_images", str(i["patient_id"]), str(i["image_id"]) + ".dcm"
    ),
    axis=1,
)
print(train_df.shape)
train_df.head(2)



## === cell 4
test_df = pd.read_csv(f"{data_dir}/test.csv")

test_df["dcm_path"] = test_df.apply(
    lambda i: os.path.join(
        f"{data_dir}", "test_images", str(i["patient_id"]), str(i["image_id"]) + ".dcm"
    ),
    axis=1,
)
print(test_df.shape)
test_df.head(2)




## === cell 5
def normalize_xray(path, fix_monochrome=True):
    """
    Returns uint8 2D image in [0,255].
    If the DICOM can't be decoded (e.g., missing JPEG2000 codec), returns a blank image.
    """
    try:
        dicom = pydicom.dcmread(path, force=True)

        arr = dicom.pixel_array.astype(np.float32)

        photo = getattr(dicom, "PhotometricInterpretation", None)
        if fix_monochrome and photo == "MONOCHROME1":
            arr = arr.max() - arr

        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        arr = arr - np.min(arr)
        mx = np.max(arr)
        if mx > 0:
            arr = arr / mx
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return arr
    except Exception:
        return np.zeros((target_size[0], target_size[1]), dtype=np.uint8)


def crop_and_resize(image, crop_size=5):
    if image.ndim == 3:
        image = image[..., 0]
    h, w = image.shape[:2]
    if h > 2 * crop_size and w > 2 * crop_size:
        image = image[crop_size:-crop_size, crop_size:-crop_size]
    image = cv2.resize(image, target_size[::-1], cv2.INTER_LINEAR)
    return image


def preprocess_and_save(file_path):
    image = normalize_xray(file_path)
    h, w = image.shape[:2]
    image = crop_and_resize(image)
    sub_path = file_path.split("/", 4)[-1].split(".dcm")[0] + ".png"
    infos = sub_path.split("/")
    pid = infos[-2]
    iid = infos[-1]
    iid = iid.replace(".png", "")
    return pid, iid, h, w, image


def pfbeta_torch(labels, preds, beta=1):
    preds = np.clip(preds, 0, 1)
    labels = labels.astype(np.float32)
    y_true_count = labels.sum()
    if y_true_count == 0:
        return 0.0
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-12)
    c_recall = ctp / (y_true_count + 1e-12)
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall + 1e-12)
        )
        return float(result)
    else:
        return 0.0




## === cell 6
class MyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        cancer = row["cancer"]
        cancer = torch.tensor(cancer, dtype=torch.long)
        path = row["dcm_path"]
        pid, iid, h, w, img = preprocess_and_save(path)
        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"cancer": cancer, "images": img}




## === cell 7
class PretrainedBinaryClassifier(nn.Module):
    def __init__(self):
        super(PretrainedBinaryClassifier, self).__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x




## === cell 8
random_seed = 534
train_subset_0 = train_df[train_df.cancer == 0].sample(n=20, random_state=random_seed)
train_subset_1 = train_df[train_df.cancer == 1].sample(n=10, random_state=random_seed)
train_combined = pd.concat([train_subset_0, train_subset_1])

print(train_combined.shape)
print(train_combined.laterality.value_counts())
print(train_combined.cancer.value_counts())
train_combined.reset_index(inplace=True, drop=True)



## === cell 9
training_set, validation_set = train_test_split(
    train_combined, test_size=0.2, random_state=276, stratify=train_combined["cancer"]
)
print(training_set.shape)
print(validation_set.shape)




## === cell 10
class ToTensorWithErasing(object):
    def __call__(self, img):
        img_tensor = transforms.functional.to_tensor(img)
        img_tensor = transforms.RandomErasing(
            p=0.5, scale=(0.1, 0.5), ratio=(0.3, 3.3)
        )(img_tensor)
        return img_tensor


train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=45, translate=(0.3, 0.3), scale=(0.7, 1.3)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.7),
        ToTensorWithErasing(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = MyDataset(training_set, transform=train_transform)
val_dataset = MyDataset(validation_set, transform=val_transform)



## === cell 11
if len(train_dataset) > 0:
    print("The training dataset contains", len(train_dataset), "samples.")
else:
    print("The training dataset is empty.")

if len(val_dataset) > 0:
    print("The val dataset contains", len(val_dataset), "samples.")
else:
    print("The val dataset is empty.")



## === cell 12
print("checking if transformation is applied")
print("Training dataset transform is: ", train_dataset.transform)
print("Validation dataset transform is:", val_dataset.transform)



## === cell 13
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)



## === cell 14
print("Number of training batches", len(train_loader))
print("Number of validation batches", len(val_loader))



## === cell 15
print("Number of training examples before augmentation:", len(training_set))
print("Number of training examples after augmentation:", len(train_dataset))



## === cell 16
print(torch.cuda.is_available())
print(torch.cuda.device_count())



## === cell 17
model = PretrainedBinaryClassifier()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

learning_rate = 0.0001
weight_decay = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



## === cell 18
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
        outputs = model(inputs)  # logits

        loss = criterion(outputs, targets.float())

        probs = torch.sigmoid(outputs)
        predicted = torch.round(probs)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)

        pf1_metric = pfbeta_torch(
            targets.detach().cpu().numpy().reshape(-1),
            probs.detach().cpu().numpy().reshape(-1),
        )

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        running_train_acc += accuracy
        running_train_pf1 += pf1_metric

    avg_train_loss = running_train_loss / max(1, len(train_loader))
    avg_train_acc = running_train_acc / max(1, len(train_loader))
    avg_train_pf1 = running_train_pf1 / max(1, len(train_loader))
    train_losses.append(avg_train_loss)
    train_acc_metric.append(avg_train_acc)
    train_pf1_metric.append(avg_train_pf1)
    print(
        "Epoch %d, avg training loss: %.3f, avg training accuracy: %.3f, avg training pf1: %.3f"
        % (epoch + 1, avg_train_loss, avg_train_acc, avg_train_pf1)
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

            outputs = model(inputs)  # logits
            loss = criterion(outputs, targets.float())

            probs = torch.sigmoid(outputs)
            predicted = torch.round(probs)
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0)

            pf1_metric = pfbeta_torch(
                targets.detach().cpu().numpy().reshape(-1),
                probs.detach().cpu().numpy().reshape(-1),
            )

            running_val_loss += loss.item()
            running_val_acc += accuracy
            running_val_pf1 += pf1_metric

    avg_val_loss = running_val_loss / max(1, len(val_loader))
    avg_val_acc = running_val_acc / max(1, len(val_loader))
    avg_val_pf1 = running_val_pf1 / max(1, len(val_loader))
    val_losses.append(avg_val_loss)
    val_acc_metric.append(avg_val_acc)
    val_pf1_metric.append(avg_val_pf1)
    print(
        "Epoch %d, avg validation loss: %.3f, avg validation accuracy: %.3f, avg validation pf1: %.3f"
        % (epoch + 1, avg_val_loss, avg_val_acc, avg_val_pf1)
    )



## === cell 19
print(len(train_losses))
print(len(train_acc_metric))
print(len(train_pf1_metric))
print(train_losses)
print(train_acc_metric)
print(train_pf1_metric)



## === cell 20
print(len(val_losses))
print(len(val_acc_metric))
print(len(val_pf1_metric))
print(val_losses)
print(val_acc_metric)
print(val_pf1_metric)



## === cell 21
plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(train_acc_metric, label="Training Accuracy")
plt.plot(val_acc_metric, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(train_pf1_metric, label="Training Probabilistic F1 Score")
plt.plot(val_pf1_metric, label="Validation Probabilistic F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Probabilistic F1 Score")
plt.legend()
plt.show()



## === cell 22
test_df.head()




## === cell 23
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row["dcm_path"]
        patient_id = str(row["patient_id"])
        pid, iid, h, w, img = preprocess_and_save(path)
        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return patient_id, img




## === cell 24
test_batch_size = 32
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_dataset = TestDataset(test_df, transform=test_transform)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)



## === cell 25
model.eval()
predictions = []

with torch.no_grad():
    for batch in test_dataloader:
        pids, images = batch
        images = images.to(device, non_blocking=True)

        outputs = model(images)  # logits
        probabilities = torch.sigmoid(outputs)
        predictions.extend(probabilities.detach().cpu().numpy().reshape(-1).tolist())

print("Num predictions:", len(predictions), "Num test rows:", len(test_df))



## === cell 26
test_df = test_df.copy()
test_df["cancer"] = predictions

sub = test_df.groupby("prediction_id", as_index=False)["cancer"].mean()

sample_sub = pd.read_csv(f"{data_dir}/sample_submission.csv")
sub = sample_sub[["prediction_id"]].merge(sub, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(0.0).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
