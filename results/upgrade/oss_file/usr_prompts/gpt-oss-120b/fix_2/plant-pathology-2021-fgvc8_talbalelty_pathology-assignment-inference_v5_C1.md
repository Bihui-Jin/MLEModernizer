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

# 5. Target score

0.6083841181902121

# 6. Current score

0.24456

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.24456) has done: 'The fix loads a pretrained ResNeXt‑50 model (instead of the missing checkpoint) and replaces its final fully‑connected layer with the correct number of classes. The submission builder is rewritten to avoid the removed `DataFrame.append` method by collecting rows in a list and converting them once. With these changes the script runs end‑to‑end and writes a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
import gc
import albumentations as A
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



## === cell 1
torch.cuda.empty_cache()
gc.collect()
DEBUG = False
DIMENTION = (256, 171)  # image size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
train_data = pd.read_csv(train_csv_path)
test_images_names = glob.glob(test_images_path + "*.jpg")

train_data["labels"] = train_data["labels"].apply(lambda s: s.split(" "))
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(train_data["labels"].tolist()),
    columns=mlb.classes_,
    index=train_data.index,
)
labels_size = len(train_labels.columns)




## === cell 2
class PlantDataSet(Dataset):
    def __init__(self, dataset, images_path, transform=None):
        super().__init__()
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform

    def __getitem__(self, idx):
        if self.images_path is not None:
            img_path = self.images_path + self.dataset.iloc[idx]["image"]
            image = cv2.imread(img_path)
            labels = torch.tensor(
                self.dataset.iloc[idx].loc[self.dataset.columns != "image"].tolist(),
                dtype=torch.float32,
            )
        else:
            img_path = self.dataset[idx]
            image = cv2.imread(img_path)
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
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
test_dataset = PlantDataSet(test_images_names, None, transform)
BS = 30
plants_test_data_loader = DataLoader(dataset=test_dataset, batch_size=BS, shuffle=False)




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

            if i == 0:
                predictions = probabilities.detach().cpu().numpy()
            else:
                predictions = np.concatenate(
                    (predictions, probabilities.detach().cpu().numpy()), axis=0
                )

            del images
            torch.cuda.empty_cache()
            gc.collect()
    return np.array(predictions)




## === cell 5
def create_submission(test_images_path_list, predictions):
    rows = []
    for img_path, pred_vec in zip(test_images_path_list, predictions):
        name = img_path.split("/")[-1]
        selected = [
            lbl for prob, lbl in zip(pred_vec, train_labels.columns) if prob > 0.75
        ]
        if not selected:
            selected = ["healthy"]
        prediction_labels = " ".join(selected)
        rows.append([name, prediction_labels])

    submission_df = pd.DataFrame(rows, columns=["image", "labels"])
    if DEBUG:
        display(submission_df.head())
    submission_df.to_csv("submission.csv", index=False)




## === cell 6
resnext50 = models.resnext50_32x4d(pretrained=True)
in_features = resnext50.fc.in_features
resnext50.fc = nn.Linear(in_features, labels_size)



## === cell 7
test_predictions = test(plants_test_data_loader, resnext50)
print(test_predictions.shape)
create_submission(test_images_names, test_predictions)
