# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time
import concurrent.futures
import random

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def breaker():
    print("\n" + 50 * "-" + "\n")


def head(x, no_of_ele=5):
    print(x[:no_of_ele])


def _load_image(args):
    """Helper for parallel image loading – returns raw uint8 image."""
    file_path, name, size = args
    img_path = os.path.join(file_path, f"{name}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((size, size), dtype=np.uint8)
    else:
        if size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img  # still uint8


def getImages(file_path=None, file_names=None, size=None, parallel=True):
    """
    Load images efficiently. Uses a limited thread pool to avoid oversubscription.
    Returns a NumPy array of shape (N, size, size) with float32 values in [0,1].
    """
    N = len(file_names)
    images_uint8 = np.empty((N, size, size), dtype=np.uint8)

    if parallel:
        max_workers = min(os.cpu_count() or 1, 4)
        args_list = [(file_path, name, size) for name in file_names]
        chunksize = max(1, N // (max_workers * 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, img in enumerate(
                ex.map(_load_image, args_list, chunksize=chunksize)
            ):
                images_uint8[i] = img
    else:
        for i, name in enumerate(file_names):
            img_path = os.path.join(file_path, f"{name}.jpg")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((size, size), dtype=np.uint8)
            else:
                if size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            images_uint8[i] = img

    images = images_uint8.astype(np.float32) / 255.0
    return images




## === cell 1
start_time = time()

ss = pd.read_csv(
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)

ts_img_names = ss["StudyInstanceUID"].values
ts_images = getImages(
    "../input/ranzcr-clip-catheter-line-classification/test/", ts_img_names, size=64
)

breaker()
print(
    "Time Taken to read test data : {:.2f} minutes".format((time() - start_time) / 60)
)
breaker()




## === cell 2
class Dataset(Dataset):
    def __init__(self, X=None, y=None, mode="train"):
        self.mode = mode
        self.X = X
        if mode == "train":
            self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])
        else:
            return torch.FloatTensor(self.X[idx])




## === cell 3
class CFG:
    tr_batch_size = 128
    ts_batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1
    OL = 11

    def __init__(
        self, filter_sizes=[64, 128, 256, 512], HL=[4096, 4096], epochs=50, n_folds=5
    ):
        self.filter_sizes = filter_sizes
        self.HL = HL
        self.epochs = epochs
        self.n_folds = n_folds




## === cell 4
class CNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        filter_sizes=None,
        HL=None,
        OL=None,
        use_DP=False,
        DP1=0.2,
        DP2=0.5,
    ):
        super(CNN, self).__init__()

        self.use_DP = use_DP

        self.DP1 = nn.Dropout(p=DP1)
        self.DP2 = nn.Dropout(p=DP2)

        self.MP_ = nn.MaxPool2d(kernel_size=2)

        self.CN1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filter_sizes[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        self.CN2 = nn.Conv2d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        self.CN3 = nn.Conv2d(
            in_channels=filter_sizes[1],
            out_channels=filter_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN3 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        self.CN4 = nn.Conv2d(
            in_channels=filter_sizes[2],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN4 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        self.CN5 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN5 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        self.CN6 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN6 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        self.FC1 = nn.Linear(in_features=filter_sizes[3] * 2 * 2, out_features=HL[0])
        self.FC2 = nn.Linear(in_features=HL[0], out_features=HL[1])
        self.FC3 = nn.Linear(in_features=HL[1], out_features=OL)

    def getOptimizer(self, A_S=True, lr=1e-3, wd=0):
        if A_S:
            return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            return optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    def getStepLR(self, optimizer=None, step_size=5, gamma=0.1):
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma
        )

    def getMultiStepLR(self, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma
        )

    def getPlateauLR(self, optimizer=None, patience=5, eps=1e-6):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=patience, eps=eps, verbose=True
        )

    def forward(self, x):
        if not self.use_DP:
            x = F.relu(self.MP_(self.BN1(self.CN1(x))))
            x = F.relu(self.MP_(self.BN2(self.CN2(x))))
            x = F.relu(self.MP_(self.BN3(self.CN3(x))))
            x = F.relu(self.MP_(self.BN4(self.CN4(x))))
            x = F.relu(self.MP_(self.BN5(self.CN5(x))))
            x = F.relu(self.MP_(self.BN6(self.CN6(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(self.FC1(x))
            x = F.relu(self.FC2(x))
            x = self.FC3(x)

            return x
        else:
            x = F.relu(self.MP_(self.BN1(self.CN1(x))))
            x = F.relu(self.MP_(self.BN2(self.CN2(x))))
            x = F.relu(self.MP_(self.BN3(self.CN3(x))))
            x = F.relu(self.MP_(self.BN4(self.CN4(x))))
            x = F.relu(self.MP_(self.BN5(self.CN5(x))))
            x = F.relu(self.MP_(self.BN6(self.CN6(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(self.DP2(self.FC1(x)))
            x = F.relu(self.DP2(self.FC2(x)))
            x = self.FC3(x)

            return x




## === cell 5
def predict_(model=None, dataloader=None, device=None, path=None):
    """
    Prediction helper – loads weights (if provided), runs inference and
    returns a NumPy array of probabilities. The concatenation now builds a list
    of tensors and concatenates once at the end to avoid repeated memory copies.
    """
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()

    preds = []  # list of batch tensors

    for X in dataloader:
        X = X.to(device)
        with torch.no_grad():
            batch_pred = torch.sigmoid(model(X))
        preds.append(batch_pred.cpu())

    y_pred = torch.cat(preds, dim=0).numpy()
    return y_pred




## === cell 6
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("../input/ranzcr-clip-catheter-line-classification/train.csv")

label_cols = [c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]]

y_train = train_df[label_cols].values.astype(np.float32)

tr_img_names = train_df["StudyInstanceUID"].values
tr_images = getImages(
    "../input/ranzcr-clip-catheter-line-classification/train/", tr_img_names, size=64
)
X_train = tr_images.reshape(tr_images.shape[0], -1).astype(np.float32)

scaler = StandardScaler(copy=False)
X_train_std = scaler.fit_transform(X_train).astype(np.float32)

base_svm = LinearSVC(C=2.0, class_weight="balanced", max_iter=3000, dual=False)
calibrated_svm = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid", n_jobs=-1)
clf = OneVsRestClassifier(calibrated_svm, n_jobs=1)

clf.fit(X_train_std, y_train)

X_test = ts_images.reshape(ts_images.shape[0], -1).astype(np.float32)
X_test_std = scaler.transform(X_test).astype(np.float32)

y_pred = clf.predict_proba(X_test_std)
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

missing_cols = set(label_cols) - set(ss.columns)
for col in missing_cols:
    ss[col] = 0.0  # placeholder – will be overwritten by predictions

ss = ss[["StudyInstanceUID"] + label_cols]
ss.iloc[:, 1:] = y_pred

submission_path = "./submission.csv"
ss.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
ss.head(5)
