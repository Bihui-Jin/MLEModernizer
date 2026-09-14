# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import glob
import matplotlib.pyplot as plt
import gc
import albumentations as A
import torchmetrics
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torchvision.models as models



## === cell 1
torch.cuda.empty_cache()
gc.collect()
DEBUG = False
DIMENTION = (256, 171)  # image size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_data = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images_names = sorted(glob.glob(test_images_path + "*.jpg"))
train_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"

train_data["labels"] = train_data["labels"].apply(lambda string: string.split(" "))
s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)
labels_size = len(train_labels.columns)

train_df = pd.concat([train_data[["image"]], train_labels], axis=1)




## === cell 2
class PlantDataSet(Dataset):
    def __init__(self, dataset, images_path, transform=None):
        super(PlantDataSet, self).__init__()
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform

    def __getitem__(self, idx):
        if self.images_path is not None:
            image = cv2.imread(self.images_path + self.dataset.image.iloc[idx])
            labels = torch.tensor(
                self.dataset.iloc[idx].loc[self.dataset.columns != "image"].tolist(),
                dtype=torch.float32,
            )
        else:
            image = cv2.imread(self.dataset[idx])
            labels = np.array([])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, DIMENTION)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, labels

    def __len__(self):
        return len(self.dataset)




## === cell 3
transform = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=42,
    shuffle=True,
)

train_ds = PlantDataSet(
    train_df.iloc[train_idx].reset_index(drop=True), train_images_path, transform
)
val_ds = PlantDataSet(
    train_df.iloc[val_idx].reset_index(drop=True), train_images_path, transform
)

BS = 30
train_loader = DataLoader(
    train_ds,
    batch_size=BS,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=BS,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_dataset = PlantDataSet(test_images_names, None, transform)
plants_test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 4
def test(test_dataloader, model):
    predictions = None
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, (images, _) in enumerate(test_dataloader):
            images = images.float().to(device)

            output = model(images)
            probabilities = torch.sigmoid(output)

            batch_preds = probabilities.detach().cpu().numpy()
            if i == 0:
                predictions = batch_preds
            else:
                predictions = np.concatenate((predictions, batch_preds), axis=0)

            del images, output, probabilities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return np.array(predictions)




## === cell 5
def create_submission(test_images_path, predictions):
    rows = []
    for image_name, prediction in zip(test_images_path, predictions):
        name = image_name.split("/")[-1]
        arr = [
            lbl for pred, lbl in zip(prediction, train_labels.columns) if pred > 0.75
        ]
        if len(arr) == 0:
            arr = ["healthy"]
        prediction_labels = " ".join(arr)
        rows.append((name, prediction_labels))

    submission_df = pd.DataFrame(rows, columns=["image", "labels"]).reset_index(
        drop=True
    )
    submission_df.to_csv("submission.csv", index=False)
    return submission_df




## === cell 6
labels_size = int(labels_size)

weights_path = "../input/resnext50-32x4d-final/resnext50_32x4d_final.pth"
use_external = False
try:
    with open(weights_path, "rb"):
        use_external = True
except FileNotFoundError:
    use_external = False

if use_external:
    resnext50 = models.resnext50_32x4d(pretrained=False, num_classes=labels_size)
    state = torch.load(weights_path, map_location="cpu")
    resnext50.load_state_dict(state)
else:
    try:
        w = models.ResNeXt50_32X4D_Weights.DEFAULT
        resnext50 = models.resnext50_32x4d(weights=w)
    except Exception:
        resnext50 = models.resnext50_32x4d(pretrained=True)
    in_features = resnext50.fc.in_features
    resnext50.fc = torch.nn.Linear(in_features, labels_size)

resnext50 = resnext50.to(device)




## === cell 7
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for images, targets in loader:
        images = images.float().to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs

        del images, targets, logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return total_loss / max(n, 1)


@torch.no_grad()
def validate_f1(model, loader, threshold=0.75):
    model.eval()
    preds_all = []
    targs_all = []
    for images, targets in loader:
        images = images.float().to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > threshold).int().numpy()
        targs = targets.int().numpy()
        preds_all.append(preds)
        targs_all.append(targs)
        del images, targets, logits, probs
    preds_all = np.concatenate(preds_all, axis=0)
    targs_all = np.concatenate(targs_all, axis=0)
    f1_per_class = []
    eps = 1e-9
    for j in range(preds_all.shape[1]):
        p = preds_all[:, j]
        y = targs_all[:, j]
        tp = (p & y).sum()
        fp = (p & (1 - y)).sum()
        fn = ((1 - p) & y).sum()
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        f1_per_class.append(f1)
    return float(np.mean(f1_per_class))


if not use_external:
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(resnext50.parameters(), lr=2e-4, weight_decay=1e-2)

    EPOCHS = 2  # minimal change to move score upward toward target without over-tuning
    for epoch in range(EPOCHS):
        tr_loss = train_one_epoch(resnext50, train_loader, optimizer, criterion)
        val_f1 = validate_f1(resnext50, val_loader, threshold=0.75)
        print(
            f"epoch {epoch+1}/{EPOCHS} | train_loss={tr_loss:.4f} | val_mean_f1@0.75={val_f1:.4f}"
        )



## === cell 8
test_predictions = test(plants_test_data_loader, resnext50)
print(test_predictions.shape)
submission_df = create_submission(test_images_names, test_predictions)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
