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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
fastai==2.8.5
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

17.74031641015639

# 6. Current score

22.33989

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 21.80521) has done: 'I fix the shape-mismatch crash by making the Swin backbone always return a fixed-length feature vector (using `forward_features` + global pooling) and by building the regression head input size from a dummy forward pass, so the MLP always matches the actual backbone output. I also ensure the tabular feature vector is exactly the 12 competition metadata columns (the current code accidentally passes 18 columns via slicing), which was a hidden cause of the `64x19` input shape. After that, the training loop can run, TTA inference work, and the notebook always write a valid `submission.csv` with the required `Id,Pawpularity` columns. These fixes are execution/correctness fixes and should move you toward a sensible RMSE rather than failing to submit.'
- What this solution (achieved 21.84455) has done: 'Your current RMSE (21.805) is worse than the target (17.740), so we should make small, low-risk changes that improve generalization without changing the model’s core structure. I (1) add proper tabular feature normalization computed from the training split only (to stabilize the MLP that consumes metadata), (2) fix TTA so it actually applies test-time augmentation (right now it repeats identical inference 4x), and (3) keep everything else (backbone, head layers, loss, training loop) the same to stay within the “minimal change” constraint. These changes typically reduce RMSE a bit and should move you closer to the target band while keeping runtime under control.'
- What this solution (achieved 21.85107) has done: 'Your current RMSE (21.84455) is worse than the target (17.7403), so we want small, low-risk improvements that keep the same backbone+MLP and training loop semantics. I (1) fix the augmentation pipeline bug where you resize to 448 then immediately pad/crop to 224 (effectively discarding the intended resize); this should improve signal quality without changing the model, (2) ensure test-time augmentation is actually different per pass by varying the random seed per TTA step (right now it can be overly correlated under deterministic settings), and (3) keep everything else (architecture, loss, fit_one_cycle, epochs) unchanged. These are minimal changes aimed at improving generalization and moving RMSE downward toward the target. The script still run end-to-end and write `/kaggle/working/submission.csv` with `Id,Pawpularity`.'
- What this solution (achieved 21.76858) has done: 'Your current RMSE (21.851) is worse than the target (17.740), so we should make one small, low-risk change that improves generalization without altering the model architecture or training loop semantics. The main issue is a metric/target mismatch: your model output is sigmoid-scaled to 0–100, but the target is trained in raw 0–100 space, which can make optimization harder and often yields underfit predictions around the mean. I keep the same backbone + tabular concat + MLP and the same `fit_one_cycle`, but change the loss/metric to operate on a normalized target in [0,1] (and only rescale back to [0,100] for submission). This typically reduces RMSE for this competition with minimal code changes and no change in evaluation meaning.'
- What this solution (achieved 22.10235) has done: 'Your current RMSE (21.7686) is worse than the target (17.7403), so we want a small, low-risk improvement that keeps the same backbone+tabular-concat+MLP and the same training loop semantics. The biggest generalization gap here is likely that you train on only 90% of data and validate on 10%, but then you submit using that single model; for Kaggle, it’s usually beneficial (and minimal-change) to retrain the exact same model on 100% of the training data after you’ve verified the pipeline works. I keep your architecture, loss, transforms, and `fit_one_cycle` intact, but add a second training phase that reloads the best weights from the 90/10 run and fine-tunes on the full dataset for the same number of epochs, then run the same TTA inference. This typically reduces RMSE without changing evaluation semantics, and it preserves your deterministic setup and still writes a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 22.18803) has done: 'Your current RMSE (22.10) is still far from the target (17.74), so we should make a small change that improves generalization without altering your model architecture, loss, or training loop. The biggest low-risk lever here is to include the provided 12 metadata features more effectively by applying the same (train-split-derived) normalization to the full-data fine-tuning phase and by ensuring the test dataset uses those same stats consistently (right now the full phase recomputes stats, slightly shifting the tabular distribution seen at inference). I also add a simple validation-based best-checkpoint reload before the full-data phase so the second phase starts from the best generalizing weights rather than the final epoch weights. These are minimal wiring changes that typically reduce RMSE a bit while preserving your backbone+concat+MLP and `fit_one_cycle` setup and still writing a valid `submission.csv`.'
- What this solution (achieved 22.11433) has done: 'Your current RMSE (22.188) is worse than the target (17.740), so we want small, low-risk improvements that reduce error without changing the model architecture, loss, or training loop. The biggest likely issue is that tabular features are normalized for the 90/10 split, but the full-data fine-tuning continues to use those split-derived stats, which is slightly mismatched for the final model trained on 100% of the data; we recompute tab_mean/tab_std on the full training set for the full-data phase and use the same full-data stats consistently for test/TTA inference. To keep things stable, we still start the full-data phase from the best 90/10 checkpoint and keep epochs/lr unchanged. This is a wiring/statistics alignment fix (not a modeling change) and is expected to move RMSE downward toward your target while preserving end-to-end submission generation.'
- What this solution (achieved 22.13813) has done: 'Your current RMSE (22.114) is worse than the target (17.740), so we make the smallest change that is likely to improve generalization without changing your model architecture, loss, or training loop semantics. The main low-risk lever here is that your training uses random crop/rotate/color jitter, but your “TTA” is a different augmentation distribution (random crop) than the training distribution (shift/scale/rotate + color jitter), which can make TTA average worse rather than better. I (1) align TTA to be “validation + mild horizontal flip” (same resize/pad/center-crop pipeline as inference, with only flip as augmentation), and (2) keep the rest identical so runtime and core logic remain unchanged. This should move RMSE downward toward your target while preserving end-to-end submission generation.'
- What this solution (achieved 22.28795) has done: 'Your current RMSE (22.138) is worse than the target (17.740), so we want a small, low-risk improvement that usually reduces error without changing your model’s architecture, loss, or training loop. The most impactful minimal change here is to make the 90/10 split stratified by the target (binning Pawpularity), which stabilizes validation selection and typically yields a model that generalizes better when you later fine-tune on 100% of the data. This keeps the same training procedure (fit_one_cycle, same epochs/lr, same transforms, same head) but improves the representativeness of the checkpoint you start the full-data phase from. Everything else (including submission writing and column names) stays the same.'
- What this solution (achieved 22.35498) has done: 'We need to lower RMSE from 22.29 toward 17.74 (lower is better), so we make the smallest change that typically improves this specific competition without changing your backbone+concat+MLP or training loop structure. The biggest low-risk issue is that your loss is MSE on sigmoid-normalized targets, which under-penalizes large absolute errors at the high end; switching to RMSE (still on the same [0,1] normalized target) aligns the optimization objective with the leaderboard metric while preserving the same output scaling and inference semantics. We keep the same model, augmentations, fit_one_cycle calls, epochs, and TTA; only the loss function changes from MSE to RMSE (metric stays the same). This should move the score downward (better) toward the target with minimal code disruption.'
- What this solution (achieved 22.33989) has done: 'Your current RMSE (22.35498) is worse than the target (17.7403), so we want a small, low-risk change that improves generalization while preserving the same backbone+tabular-concat+MLP and the same training loops. The most likely issue is a train/inference mismatch: the model is trained with `ShiftScaleRotate` but never sees that distribution at test time (even with your current flip-only TTA), which can hurt performance. I minimally adjust TTA to include a mild `ShiftScaleRotate` (same operator as training but lower limits) so ensembling averages over plausible test-time perturbations aligned with training, while keeping everything else (loss, epochs, LR, architecture, normalization, split, submission format) unchanged. This should move RMSE downward toward the target without changing evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import gc
import sys
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

from PIL import Image

import albumentations as alb
from albumentations.pytorch import ToTensorV2

import timm

from fastai.vision.all import *
from fastai.data.core import *

seed = 999
set_seed(seed, reproducible=True)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)




## === cell 1
def petfinder_rmse_unit(input, target):
    pred = torch.sigmoid(input.flatten())
    return torch.sqrt(F.mse_loss(pred, target.float().flatten()))




## === cell 2
base_dir = "/kaggle/input"
comp_dir = os.path.join(base_dir, "petfinder-pawpularity-score")

train_folder = os.path.join(comp_dir, "train")
test_folder = os.path.join(comp_dir, "test")
train_file = os.path.join(comp_dir, "train.csv")
test_file = os.path.join(comp_dir, "test.csv")

model_weights = os.path.join("/kaggle/working", "swin_fastai(final).pth")

for p, kind in [(train_file, "Train CSV"), (test_file, "Test CSV")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{kind} not found at: {p}")
for p, kind in [
    (train_folder, "Train images folder"),
    (test_folder, "Test images folder"),
]:
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{kind} not found at: {p}")



## === cell 3
input_shape = (224, 224, 3)
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
model_name = "swin_base_patch4_window7_224"
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
output_categories = 11  # kept as in original code (not used directly)
n_epochs = 2  # keep runtime safe; core loop unchanged (train+infer)
num_of_hidden = 2
hidden_dimension = [256, 64]
save_name = "/kaggle/working"  # fastai Learner model_dir must be a directory



## === cell 4
train_csv = pd.read_csv(train_file)
test_csv = pd.read_csv(test_file)

train_csv["path_img"] = train_csv["Id"].map(
    lambda x: os.path.join(train_folder, f"{x}.jpg")
)
test_csv["path_img"] = test_csv["Id"].map(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)

missing_train = (
    train_csv.loc[~train_csv["path_img"].map(os.path.exists), "Id"].head(5).tolist()
)
missing_test = (
    test_csv.loc[~test_csv["path_img"].map(os.path.exists), "Id"].head(5).tolist()
)
if missing_train:
    raise FileNotFoundError(
        f"Some train images are missing. Example missing Ids: {missing_train}"
    )
if missing_test:
    raise FileNotFoundError(
        f"Some test images are missing. Example missing Ids: {missing_test}"
    )




## === cell 5
class PetsDataset(Dataset):
    def __init__(self, df, transform=None, other=False, tab_mean=None, tab_std=None):
        self.transform = transform
        self.df = df.reset_index(drop=True)
        self.other = other
        self.cat = [
            "Subject Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]
        self.tab_mean = (
            None if tab_mean is None else np.asarray(tab_mean, dtype=np.float32)
        )
        self.tab_std = (
            None if tab_std is None else np.asarray(tab_std, dtype=np.float32)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path_img"]

        if "Pawpularity" in self.df.columns:
            label_1 = np.float32(self.df.loc[idx, "Pawpularity"]) / np.float32(100.0)
        else:
            label_1 = np.float32(0.5)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            album = self.transform(image=np.array(img))
            img = album["image"]

        df_data = self.df.loc[idx, self.cat].values.astype(np.float32)

        if (self.tab_mean is not None) and (self.tab_std is not None):
            df_data = (df_data - self.tab_mean) / self.tab_std

        if self.other:
            return img, label_1, self.df.loc[idx, self.cat].to_dict()

        return (img, df_data, np.float32(label_1))




## === cell 6
train_transform = alb.Compose(
    [
        alb.LongestMaxSize(max_size=input_shape[0], interpolation=1),
        alb.PadIfNeeded(
            min_height=input_shape[0],
            min_width=input_shape[1],
            border_mode=0,
            value=(0, 0, 0),
        ),
        alb.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
        ),
        alb.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.6),
        alb.RandomCrop(height=input_shape[0], width=input_shape[1]),
        alb.HorizontalFlip(p=0.6),
        alb.Normalize(mean, std_dev),
        ToTensorV2(),
    ]
)

valid_transform = alb.Compose(
    [
        alb.LongestMaxSize(max_size=input_shape[0], interpolation=1),
        alb.PadIfNeeded(
            min_height=input_shape[0],
            min_width=input_shape[1],
            border_mode=0,
            value=(0, 0, 0),
        ),
        alb.CenterCrop(height=input_shape[0], width=input_shape[1]),
        alb.Normalize(mean, std_dev),
        ToTensorV2(),
    ]
)

test_transform = valid_transform

tta_transform = alb.Compose(
    [
        alb.LongestMaxSize(max_size=input_shape[0], interpolation=1),
        alb.PadIfNeeded(
            min_height=input_shape[0],
            min_width=input_shape[1],
            border_mode=0,
            value=(0, 0, 0),
        ),
        alb.CenterCrop(height=input_shape[0], width=input_shape[1]),
        alb.ShiftScaleRotate(
            shift_limit=0.03, scale_limit=0.03, rotate_limit=10, p=0.7
        ),
        alb.HorizontalFlip(p=0.5),
        alb.Normalize(mean, std_dev),
        ToTensorV2(),
    ]
)



## === cell 7
idx = np.arange(len(train_csv))
rng = np.random.default_rng(seed)

y = train_csv["Pawpularity"].astype(np.float32).values
n_bins = 10
bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
bins = np.unique(bins)
if len(bins) < 3:
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
else:
    y_bin = np.digitize(y, bins[1:-1], right=True)
    perm = rng.permutation(len(train_csv))
    tr_idx_list, va_idx_list = [], []
    for b in np.unique(y_bin):
        b_idx = perm[y_bin[perm] == b]
        k = int(np.floor(0.9 * len(b_idx)))
        tr_idx_list.append(b_idx[:k])
        va_idx_list.append(b_idx[k:])
    tr_idx = np.concatenate(tr_idx_list)
    va_idx = np.concatenate(va_idx_list)

train_df = train_csv.iloc[tr_idx].reset_index(drop=True)
valid_df = train_csv.iloc[va_idx].reset_index(drop=True)

cat_cols = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]

tab_mean_90 = train_df[cat_cols].astype(np.float32).mean(axis=0).values
tab_std_90 = train_df[cat_cols].astype(np.float32).std(axis=0).values
tab_std_90 = np.where(tab_std_90 < 1e-6, 1.0, tab_std_90).astype(np.float32)

train_ds = PetsDataset(
    train_df, train_transform, tab_mean=tab_mean_90, tab_std=tab_std_90
)
valid_ds = PetsDataset(
    valid_df, valid_transform, tab_mean=tab_mean_90, tab_std=tab_std_90
)

test_ds = PetsDataset(
    test_csv.assign(Pawpularity=50.0),
    test_transform,
    tab_mean=tab_mean_90,
    tab_std=tab_std_90,
)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)
testloader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

dls = DataLoaders(train_dl, valid_dl, device=device)




## === cell 8
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Network(nn.Module):
    """
    Backbone + tabular concat + MLP head preserved.
    """

    def __init__(
        self,
        base,
        number_of_hidden,
        hidden,
        regression_out,
        output_categories,
        freeze_layer,
        image_size=(224, 224),
        tabular_dim=12,
    ):
        super().__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of hidden layers and length of hidden dim must be same"
            )

        self.base = base
        self.tabular_dim = tabular_dim

        if hasattr(self.base, "reset_classifier"):
            self.base.reset_classifier(0)

        self.p = 0.5
        self.network = self.__freeze_layer(self.base, freeze_layer)

        self._in_feats = self._infer_in_feats(image_size=image_size)

        hidden_dim = hidden[:]
        hidden_dim.insert(0, self._in_feats + self.tabular_dim)

        self.regression = self.__fully_connected(
            number_of_hidden, hidden_dim, regression_out
        )

        self.__initialise_weights()

    def _backbone_features(self, x):
        if hasattr(self.network, "forward_features"):
            x1 = self.network.forward_features(x)
        else:
            x1 = self.network(x)

        if x1.ndim == 4:
            x1 = x1.mean(dim=(2, 3))
        elif x1.ndim == 3:
            x1 = x1.mean(dim=1)
        elif x1.ndim != 2:
            raise RuntimeError(f"Unexpected base output shape: {tuple(x1.shape)}")
        return x1

    def _infer_in_feats(self, image_size=(224, 224)):
        self.network.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, image_size[0], image_size[1], dtype=torch.float32)
            x1 = self._backbone_features(dummy)
        return int(x1.shape[1])

    def __initialise_weights(self):
        for m in self.regression:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __freeze_layer(self, base, freeze_layer):
        cnt = 0
        for child in base.children():
            cnt += 1
            if cnt > freeze_layer:
                break
            for param in child.parameters():
                param.requires_grad = False
        return base

    def __fully_connected(self, number_of_hidden, hidden_dim, output_categories):
        layers = []
        for i in range(number_of_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
            if i != (number_of_hidden - 1):
                layers.append(nn.Dropout(self.p))
        layers.append(nn.Linear(hidden_dim[-1], output_categories))
        return nn.Sequential(*layers)

    def forward(self, x, tab):
        x1 = self._backbone_features(x)

        if not torch.is_tensor(tab):
            tab = torch.tensor(tab, device=x.device)
        tab = tab.float()

        if tab.ndim != 2:
            tab = tab.view(tab.shape[0], -1)
        if tab.shape[1] != self.tabular_dim:
            raise RuntimeError(
                f"Tabular feature width mismatch: got {tab.shape[1]}, expected {self.tabular_dim}"
            )

        xcat = torch.cat([x1, tab], dim=1)
        reg = self.regression(xcat)
        return reg


network = timm.create_model(model_name, pretrained=True)
model = Network(
    network,
    num_of_hidden,
    hidden_dimension,
    1,
    11,
    0,
    image_size=(input_shape[0], input_shape[1]),
    tabular_dim=12,
)




## === cell 9
def get_learner(dls, model, loss, metric, save_path):
    dls = dls.to(device)
    model = model.to(device)
    learn = Learner(dls, model, loss_func=loss, metrics=metric, model_dir=save_path)
    if torch.cuda.is_available():
        learn = learn.to_fp16()
    return learn


class SigmoidRMSELossUnit(nn.Module):
    def forward(self, input, target):
        pred = torch.sigmoid(input.flatten())
        return torch.sqrt(F.mse_loss(pred, target.float().flatten()))


learn = get_learner(dls, model, SigmoidRMSELossUnit(), petfinder_rmse_unit, save_name)

cbs = [
    SaveModelCallback(monitor="petfinder_rmse_unit", fname="best_90_10", comp=np.less)
]
learn.fit_one_cycle(n_epochs, lr_max=3e-4, cbs=cbs)

best_path_90_10 = os.path.join(save_name, "best_90_10.pth")
if os.path.exists(best_path_90_10):
    learn.model.load_state_dict(torch.load(best_path_90_10, map_location=device))

torch.save(learn.model.state_dict(), model_weights)

learn.model.eval()
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None



## === cell 10
tab_mean_full = train_csv[cat_cols].astype(np.float32).mean(axis=0).values
tab_std_full = train_csv[cat_cols].astype(np.float32).std(axis=0).values
tab_std_full = np.where(tab_std_full < 1e-6, 1.0, tab_std_full).astype(np.float32)

full_train_ds = PetsDataset(
    train_csv.reset_index(drop=True),
    train_transform,
    tab_mean=tab_mean_full,
    tab_std=tab_std_full,
)
full_train_dl = DataLoader(
    full_train_ds,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)

full_dls = DataLoaders(full_train_dl, valid_dl, device=device)

learn_full = get_learner(
    full_dls, learn.model, SigmoidRMSELossUnit(), petfinder_rmse_unit, save_name
)
learn_full.model.load_state_dict(torch.load(model_weights, map_location=device))

learn_full.fit_one_cycle(n_epochs, lr_max=3e-4)

torch.save(learn_full.model.state_dict(), model_weights)

learn_full.model.eval()
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None



## === cell 11
tta_outputs = []
tta_steps = 4

for t in range(tta_steps):
    set_seed(seed + 12345 + t, reproducible=True)
    random.seed(seed + 12345 + t)
    np.random.seed(seed + 12345 + t)
    torch.manual_seed(seed + 12345 + t)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 12345 + t)

    tta_ds = PetsDataset(
        test_csv.assign(Pawpularity=50.0),
        transform=tta_transform,
        tab_mean=tab_mean_full,
        tab_std=tab_std_full,
    )
    tta_loader = DataLoader(
        tta_ds,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    final_outputs = []
    with torch.no_grad():
        for images, tabular, _ in tta_loader:
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True).float()

            reg_output = learn_full.model(images, tabular)
            reg_output = 100.0 * torch.sigmoid(reg_output)

            output = reg_output.detach().float().cpu().numpy().reshape(-1).tolist()
            final_outputs.extend(output)

    tta_outputs.append(final_outputs)

tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0)



## === cell 12
tta_outputs_arr = np.clip(tta_outputs_arr, 1.0, 100.0)

submission = pd.DataFrame(
    {
        "Id": test_csv["Id"].astype(str).values,
        "Pawpularity": tta_outputs_arr.astype(float),
    }
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print(submission.head())
print(f"Saved submission to: {submission_path} with shape {submission.shape}")
print("CSV check columns:", submission.columns.tolist())
print("Submission file exists:", os.path.exists(submission_path))
