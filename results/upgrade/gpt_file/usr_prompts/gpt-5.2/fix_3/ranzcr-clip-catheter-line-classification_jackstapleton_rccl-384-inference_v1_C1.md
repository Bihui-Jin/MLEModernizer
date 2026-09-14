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

# 5. Target score

0.9088197075841224

# 6. Current score

0.54725

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.54725) has done: 'I fix the immediate runtime blocker by making the weight-loading robust: if the referenced checkpoint isn’t available in this Kaggle environment, the code automatically search common `/kaggle/input/**` locations for a matching `.pt` file and load it if found; otherwise it still run end-to-end and produce a valid `submission.csv`. I also fix the submission shape/columns mismatch: `sample_submission.csv` here has only 9 target columns, while the model outputs 11, so we align predictions to the exact submission columns (using the intersection in the correct order). Finally, I prevent a common namespace bug caused by overwriting `transforms` (module) with a variable, keeping the core model/inference logic unchanged while ensuring the CSV is correctly written.'
- What this solution (achieved 0.54725) has done: 'Your score is far below the target, and the biggest likely cause is that the referenced checkpoint isn’t present so the model is running with random (untrained) weights; I make checkpoint discovery stricter and more targeted so it reliably finds a real `Epoch_14.pt` (or the closest available epoch) under `/kaggle/input`. Next, your environment’s `sample_submission.csv` is missing two target columns, which can cause either invalid submissions or unintentionally dropping predictions; I instead use `train.csv` to define the full 11-label target set and always output all 11 columns in the required order. Finally, I keep the model and inference identical, but ensure the submission rows align to the sample submission’s `StudyInstanceUID` ordering exactly, and I write `submission.csv` with the correct header.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torchvision
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

import gc
import os
import cv2
from time import time
from glob import glob
import re

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



## === cell 1
INPUT_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification"
TRAIN_IMG_DIR = f"{INPUT_DIR}/train/"
TEST_IMG_DIR = f"{INPUT_DIR}/test/"
SAMPLE_SUB_PATH = f"{INPUT_DIR}/sample_submission.csv"
TRAIN_CSV_PATH = f"{INPUT_DIR}/train.csv"




## === cell 2
def breaker():
    print("\n" + 50 * "-" + "\n")


def head(x, no_of_ele=5):
    print(x[:no_of_ele])


def getImages(file_path=None, file_names=None, size=None):
    images = []
    for name in file_names:
        image_path = os.path.join(file_path, f"{name}.jpg")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        if size:
            image = cv2.resize(
                image, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4
            )
            images.append(image.reshape(size, size, 1))
        else:
            h, w = image.shape[:2]
            images.append(image.reshape(h, w, 1))
    return np.array(images)




## === cell 3
def _extract_epoch_num(path: str):
    m = re.search(r"Epoch[_\- ]?(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def find_checkpoint(preferred_path: str):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    base = os.path.basename(preferred_path) if preferred_path else None
    candidates = []
    if base:
        candidates = glob(f"/kaggle/input/**/{base}", recursive=True)
        if candidates:
            return sorted(candidates, key=lambda p: len(p))[0]

    desired_epoch = _extract_epoch_num(preferred_path) if preferred_path else None
    epoch_candidates = glob("/kaggle/input/**/Epoch_*.pt", recursive=True)
    if epoch_candidates:
        if desired_epoch is not None:

            def rank(p):
                e = _extract_epoch_num(p)
                return (
                    0 if e is not None else 1,
                    abs(e - desired_epoch) if e is not None else 10**9,
                    len(p),
                )

            return sorted(epoch_candidates, key=rank)[0]
        return sorted(epoch_candidates, key=lambda p: len(p))[0]

    any_pt = glob("/kaggle/input/**/*.pt", recursive=True)
    return sorted(any_pt, key=lambda p: len(p))[0] if any_pt else None




## === cell 4
def load_state_dict_safely(model, ckpt_path, device):
    if ckpt_path is None:
        return False, "No checkpoint found."
    try:
        state = torch.load(ckpt_path, map_location=device)
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]

        if isinstance(state, dict):
            keys = list(state.keys())
            if keys and all(k.startswith("model.") for k in keys):
                state = {k[len("model.") :]: v for k, v in state.items()}

        model.load_state_dict(state, strict=True)
        return True, f"Loaded checkpoint: {ckpt_path}"
    except Exception as e:
        return False, f"Failed to load checkpoint ({ckpt_path}): {repr(e)}"




## === cell 5
start_time = time()

ss = pd.read_csv(SAMPLE_SUB_PATH)

train_df = pd.read_csv(TRAIN_CSV_PATH)
ALL_TARGET_COLS = [
    c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]
]

ts_img_names = ss["StudyInstanceUID"].values
ts_images = getImages(TEST_IMG_DIR, ts_img_names, size=384)

breaker()
print("Time Taken to read data : {:.2f} minutes".format((time() - start_time) / 60))
print(f"Sample submission shape: {ss.shape}")
print(
    f"Using target columns from train.csv ({len(ALL_TARGET_COLS)}): {ALL_TARGET_COLS}"
)
breaker()




## === cell 6
class DS(Dataset):
    def __init__(this, X=None, y=None, transform=None, mode="train"):
        this.mode = mode
        this.transform = transform
        this.X = X
        if mode == "train":
            this.y = y

    def __len__(this):
        return this.X.shape[0]

    def __getitem__(this, idx):
        img = this.transform(this.X[idx])
        if this.mode == "train":
            return img, torch.FloatTensor(this.y[idx])
        else:
            return img




## === cell 7
tfm = T.Compose([T.ToTensor()])

ts_data_setup = DS(X=ts_images, y=None, transform=tfm, mode="test")
ts_data = DL(
    ts_data_setup,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 8
class CFG:
    tr_batch_size = 64
    ts_batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1
    OL = 11

    def __init__(
        this, filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=50, n_folds=5
    ):
        this.filter_sizes = filter_sizes
        this.HL = HL
        this.epochs = epochs
        this.n_folds = n_folds




## === cell 9
class CNN(nn.Module):
    def __init__(
        this, in_channels=1, filter_sizes=None, HL=None, OL=None, use_DP=True, DP=0.50
    ):

        super(CNN, this).__init__()

        this.use_DP = use_DP

        this.DP_ = nn.Dropout(p=DP)
        this.MP_ = nn.MaxPool2d(kernel_size=2)

        this.CN1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filter_sizes[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        this.CN2 = nn.Conv2d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        this.CN3 = nn.Conv2d(
            in_channels=filter_sizes[1],
            out_channels=filter_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN3 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        this.CN4 = nn.Conv2d(
            in_channels=filter_sizes[2],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN4 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN5 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN5 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN6 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN6 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN7 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN7 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.FC1 = nn.Linear(in_features=filter_sizes[3] * 3 * 3, out_features=HL[0])
        this.FC2 = nn.Linear(in_features=HL[0], out_features=OL)

    def getOptimizer(this, lr=1e-3, wd=0):
        return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)

    def getStepLR(this, optimizer=None, step_size=5, gamma=0.1):
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma
        )

    def getMultiStepLR(this, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma
        )

    def getPlateauLR(this, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=patience, eps=1e-8, verbose=True
        )

    def forward(this, x):
        if this.use_DP:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))
            x = F.relu(this.MP_(this.BN7(this.CN7(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(this.DP_(this.FC1(x)))
            x = this.FC2(x)

            return x
        else:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))
            x = F.relu(this.MP_(this.BN7(this.CN7(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(this.FC1(x))
            x = this.FC2(x)

            return x




## === cell 10
def predict_(model=None, dataloader=None, device=None, path=None):
    if path:
        ckpt = find_checkpoint(path)
        ok, msg = load_state_dict_safely(model, ckpt, device)
        print(msg)
        if not ok:
            print(
                "Proceeding with uninitialized weights (submission will likely score poorly)."
            )

    model.to(device)
    model.eval()

    y_pred = torch.zeros(1, 11, device=device)

    for X in dataloader:
        X = X.to(device, non_blocking=True)
        with torch.no_grad():
            Pred = torch.sigmoid(model(X))
        y_pred = torch.cat((y_pred, Pred), dim=0)

    return y_pred[1:].detach().cpu().numpy()




## === cell 11
cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=30, n_folds=5)

model = CNN(
    in_channels=cfg.in_channels, filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL
)

ckpt_path = "../input/rccl-384-train/Epoch_14.pt"

y_pred = predict_(model=model, dataloader=ts_data, device=cfg.device, path=ckpt_path)
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

model_out_cols = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]
col_to_idx = {c: i for i, c in enumerate(model_out_cols)}

sub = pd.DataFrame({"StudyInstanceUID": ts_img_names})
for col in ALL_TARGET_COLS:
    if col in col_to_idx:
        sub[col] = y_pred[:, col_to_idx[col]].astype(np.float32)
    else:
        sub[col] = np.float32(0.5)

sub = sub[["StudyInstanceUID"] + ALL_TARGET_COLS]
sub.to_csv("./submission.csv", index=False)

print(sub.head(5))
print(f"Saved submission.csv with shape {sub.shape} and columns {list(sub.columns)}")
print(
    "Note: If the printed message above says no checkpoint found/loaded, score will remain low; "
    "with a real trained checkpoint, score should move significantly toward the target."
)
