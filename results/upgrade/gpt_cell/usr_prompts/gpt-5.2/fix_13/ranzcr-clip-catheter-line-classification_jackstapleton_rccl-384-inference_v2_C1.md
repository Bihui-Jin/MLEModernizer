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

0.75525

# 6. Current score

0.51183

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4982) has done: 'Diagnosis: The crash happens when assigning `y_pred` into `ss.iloc[:, 1:]` because `y_pred` has 11 columns (model output `OL=11`), while the sample submission has only 9 target columns (10 total columns including `StudyInstanceUID`). This mismatch triggers pandas’ “equal len keys and value” error. The root cause is a label-set mismatch between training (11 labels) and the provided submission format (9 labels). We must align predictions to the submission columns without changing the model or inference logic.

Patch summary: In cell 7, keep the model output as-is, but slice `y_pred` to the exact number of submission target columns (`ss.shape[1]-1`) right before assignment. Add a defensive check to fail early if the model outputs fewer columns than required.

Updated cells: only cell 7.

Compatibility notes for cell k+1: `ss` remains a DataFrame with the same columns as read from `sample_submission.csv`, and `submission.csv` is written in the same path/name as before. `y_pred` remains a numpy array; only its column count is aligned to `ss.iloc[:, 1:]`.

Assumptions: The required submission targets are exactly the columns present in `sample_submission.csv` (all columns except `StudyInstanceUID`), and the first `n_targets` logits from the 11 model outputs correspond to those targets.'
- What this solution (achieved 0.47424) has done: 'Your current score is far below the target, so we should make a small, low-risk change that legitimately improves AUC without changing the CNN, loss, or training loop (you’re doing inference-only). The biggest likely issue is a silent label/column mismatch: you’re slicing “first N columns” from an 11-logit model output, which may not correspond to the 9 labels present in the sample submission, producing near-random predictions and an AUC around 0.5. I keep the exact same model/inference, but add a minimal mapping step that aligns the 11 model outputs to the submission columns by name (using the standard RANZCR label order), and only fall back to slicing if we can’t verify the mapping. This should move the score upward toward the target while preserving core logic and producing a valid submission.csv.'
- What this solution (achieved 0.47733) has done: 'Your score is far below the target, so the most likely cause is still a label/column alignment issue: the provided `sample_submission.csv` in your environment is missing two target columns, and slicing/mapping must exactly match the submission columns in the right order. I keep the same model and inference, but (1) enforce deterministic ordering based on `sample_submission.csv`, (2) align model outputs to those columns by name using the known 11-label order, and (3) add a safe fallback that fills any missing required columns with a neutral prior (mean predicted probability) rather than silently slicing the wrong logits. This should move AUC upward toward the target while keeping the core logic unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved 0.47733) has done: 'Your current score is far below the target, so the most likely remaining issue is still label/output misalignment, but now at the dataset level: you are training/inferencing an 11-logit model while your environment’s `sample_submission.csv` only has 9 target columns, which can cause systematically wrong columns to be submitted (and AUC ~0.5). I keep the model/inference unchanged, but (1) detect and use the full 11-label submission schema from `train.csv` (which matches the competition metric) instead of relying on the truncated sample submission, and (2) build the submission by explicitly ordering columns to the canonical 11-label set, ensuring your 11 outputs map 1:1 to the required columns. This is a minimal change focused purely on correct evaluation semantics and should move AUC upward toward the target without changing the CNN, transforms, or prediction routine. The code still write a valid `submission.csv` with the correct header and row alignment.'
- What this solution (achieved 0.47733) has done: 'Your score is still near-random AUC, which strongly suggests the submission columns are not the competition’s full 11 targets (your environment’s `sample_submission.csv` is truncated to 9 targets). I keep the exact same model and inference, but rebuild the submission schema from `train.csv` (the true 11-label set used for evaluation) and enforce the canonical 11-label order so your 11 logits map 1:1 to the correct columns. This fixes evaluation semantics/alignment without changing architecture, transforms, or prediction logic, and should move AUC up toward the target. I also add a strict shape check to fail early if the checkpoint/model output doesn’t match 11.'
- What this solution (achieved 0.56043) has done: 'Your current AUC is still near-random relative to the target, so the smallest likely “real” fix is to ensure the checkpoint is actually being loaded and that test images are normalized similarly to what the checkpoint was trained on (without changing the CNN or prediction loop). I (1) make checkpoint loading robust to common Kaggle formats (plain `state_dict` vs `{"state_dict":...}` and possible `module.` prefixes), which prevents silent non-loading that yields ~0.5 AUC, and (2) add a minimal grayscale normalization in the existing transform (just `Normalize(mean=0.5,std=0.5)`), which often materially improves transfer of a pretrained CNN on X-rays. I keep your 11-output-to-11-column alignment logic and still write `./submission.csv` with the required header. These are narrowly scoped changes aimed at moving your score upward toward the target without altering architecture, loss, or training loops.'
- What this solution (achieved 0.56043) has done: 'Your score gap is large (0.56043 vs target 0.75525), so the most likely “minimal but real” improvement is fixing an input distribution mismatch: you are feeding raw 0–255 grayscale into `ToTensor()+Normalize(0.5,0.5)`, which assumes inputs were first scaled to 0–1 (but `ToTensor` does not scale for numpy arrays). I keep your model, checkpoint loading, and inference loop identical, but ensure the test images are converted to float32 in [0,1] before the transform so normalization behaves as intended. This change is small, legitimate, and commonly moves AUC substantially upward for pretrained checkpoints without changing architecture or training. The submission schema/alignment logic stays the same and still writes `./submission.csv` end-to-end.'
- What this solution (achieved 0.56043) has done: 'Your score is still far below the target, so we should make a small change that legitimately improves AUC without changing the CNN, loss, or inference loop. The most likely remaining issue is an input preprocessing mismatch with the checkpoint: you currently use `ToTensor()` on a numpy array that is already float in `[0,1]`, which can unintentionally rescale again (depending on torchvision behavior) and shift the distribution away from what the model was trained on. I keep the same data loading and model, but replace `ToTensor()` with an explicit, deterministic `Lambda` that converts HWC numpy `[0,1]` to CHW tensor without any extra scaling, then apply the same normalization. This should move predictions closer to what the checkpoint expects and push AUC upward toward your target while keeping everything else identical.'
- What this solution (achieved 0.56043) has done: 'Your score is still far below the target, so the most likely remaining “minimal but real” fix is that the checkpoint is not being loaded at all (or is being loaded partially), which yields near-random predictions. I make checkpoint discovery robust by searching common input locations for `Epoch_14.pt` and, if found, enforce `strict=True` loading so we don’t silently run with a mostly-random model. To avoid a submission/metric mismatch, I also keep your 11-column submission schema but ensure it’s ordered exactly as the canonical 11 targets. These are narrow changes that preserve your model, transforms, and inference loop while pushing AUC upward toward the target.'
- What this solution (achieved 0.4933) has done: 'Your score is still well below the target, so the most likely remaining minimal fix is a test-image preprocessing mismatch with what the checkpoint expects. I keep your CNN, checkpoint loading, and prediction loop unchanged, but adjust the transform so the model sees standardized ImageNet-like normalization replicated to 3 channels (a common expectation for shared checkpoints), while still using the same grayscale image content. I also make the Dataset return contiguous tensors and ensure the submission columns are built from `train.csv` (full 11 targets) in the canonical order, so inference-to-submission mapping is correct and stable.'
- What this solution (achieved 0.51183) has done: 'Your score is far below the target, so we should make a minimal, high-impact correctness fix rather than tuning. The most likely remaining issue is a test/train preprocessing mismatch: your current ImageNet normalization on replicated grayscale can easily be incompatible with the provided checkpoint, pushing AUC toward random. I keep the exact same CNN, checkpoint loading, and inference loop, but switch the transform to the simpler grayscale normalization (`mean=0.5,std=0.5`) that this style of X-ray checkpoint commonly expects. I also ensure the submission schema matches the full 11-label target set from `train.csv` in canonical order (so evaluation uses the intended columns).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

import gc
import os
import cv2
from time import time

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)




## === cell 1
def breaker():
    print("\n" + 50 * "-" + "\n")


def head(x, no_of_ele=5):
    print(x[:no_of_ele])


def getImages(file_path=None, file_names=None, size=None):
    images = []
    for name in file_names:
        try:
            image = cv2.imread(file_path + name + ".jpg", cv2.IMREAD_GRAYSCALE)
        except AttributeError:
            print(file_path + name)
        if size:
            image = cv2.resize(
                image, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4
            )
        images.append(image.reshape(size, size, 1))
    return np.array(images)




## === cell 2
start_time = time()

ss = pd.read_csv(
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)

ts_img_names = ss["StudyInstanceUID"].values
ts_images = getImages(
    "../input/ranzcr-clip-catheter-line-classification/test/", ts_img_names, size=384
)

ts_images = ts_images.astype(np.float32) / 255.0

breaker()
print("Time Taken to read data : {:.2f} minutes".format((time() - start_time) / 60))
breaker()




## === cell 3
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
        img = this.transform(np.ascontiguousarray(this.X[idx]))
        if this.mode == "train":
            return img, torch.FloatTensor(this.y[idx])
        else:
            return img




## === cell 4
class CFG:
    tr_batch_size = 64
    ts_batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 3
    OL = 11

    def __init__(
        this, filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=50, n_folds=5
    ):
        this.filter_sizes = filter_sizes
        this.HL = HL
        this.epochs = epochs
        this.n_folds = n_folds




## === cell 5
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




## === cell 6
def _load_checkpoint_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)
    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state_dict = ckpt["state_dict"]
    elif (
        isinstance(ckpt, dict)
        and "model_state_dict" in ckpt
        and isinstance(ckpt["model_state_dict"], dict)
    ):
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format at: {path}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict


def _find_checkpoint_epoch14():
    candidates = [
        "../input/rccl-384-train/Epoch_14.pt",
        "../input/ranzcr-clip-catheter-line-classification/Epoch_14.pt",
        "../input/Epoch_14.pt",
    ]
    base = "../input"
    if os.path.isdir(base):
        try:
            for d in os.listdir(base):
                p = os.path.join(base, d, "Epoch_14.pt")
                candidates.append(p)
        except Exception:
            pass

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def predict_(model=None, dataloader=None, device=None, path=None):
    if path:
        state_dict = _load_checkpoint_state_dict(path, device)
        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    y_pred = torch.zeros(1, 11).to(device)

    for X in dataloader:
        X = X.to(device)
        with torch.no_grad():
            Pred = torch.sigmoid(model(X))
        y_pred = torch.cat((y_pred, Pred), dim=0)

    return y_pred[1:].detach().cpu().numpy()




## === cell 7
cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[4096], epochs=None, n_folds=None)

transforms = transforms.Compose(
    [
        transforms.Lambda(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(x.transpose(2, 0, 1))
            ).float()
        ),  # (1,H,W) in [0,1]
        transforms.Lambda(lambda t: t.repeat(3, 1, 1)),  # (3,H,W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

ts_data_setup = DS(X=ts_images, y=None, transform=transforms, mode="test")
ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

model = CNN(
    in_channels=cfg.in_channels, filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL
)

ckpt_path = _find_checkpoint_epoch14()
if ckpt_path is None:
    print(
        "[WARN] Checkpoint not found; predictions will be from randomly initialized model (very low AUC expected)."
    )
else:
    print(f"[INFO] Using checkpoint: {ckpt_path}")

y_pred = predict_(model=model, dataloader=ts_data, device=cfg.device, path=ckpt_path)
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

train_df = pd.read_csv("../input/ranzcr-clip-catheter-line-classification/train.csv")
train_target_cols = [
    c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]
]

canonical_model_output_cols = [
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

if (
    set(train_target_cols) == set(canonical_model_output_cols)
    and len(train_target_cols) == 11
):
    target_cols = canonical_model_output_cols
else:
    target_cols = train_target_cols

if y_pred.shape[1] != 11:
    raise ValueError(
        f"Model output has {y_pred.shape[1]} columns, expected 11 to match competition targets."
    )

out_idx = {c: i for i, c in enumerate(canonical_model_output_cols)}
y_pred_aligned = np.zeros((y_pred.shape[0], len(target_cols)), dtype=np.float32)
for j, c in enumerate(target_cols):
    if c in out_idx:
        y_pred_aligned[:, j] = y_pred[:, out_idx[c]]
    else:
        y_pred_aligned[:, j] = 0.5

submission = pd.DataFrame(y_pred_aligned, columns=target_cols)
submission.insert(0, "StudyInstanceUID", ts_img_names)
submission[target_cols] = np.clip(submission[target_cols].values, 1e-15, 1 - 1e-15)

submission.to_csv("./submission.csv", index=False)
submission.head(5)
