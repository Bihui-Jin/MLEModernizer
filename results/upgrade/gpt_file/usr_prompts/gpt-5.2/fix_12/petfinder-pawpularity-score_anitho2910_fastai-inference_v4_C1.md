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

17.756818644265625

# 6. Current score

30.09482

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fix the fastai mixed-precision call that’s crashing by using `learn.to_fp16()` only when it exists (newer fastai/torch combos can lack it) and otherwise keeping pure-fp32 so inference runs. I also correct the dataset tabular feature list to match the actual CSV columns (use `"Subject Focus"` rather than `"Focus"`) so tabular tensors have the expected 12 features. Then I make the TTA aggregation robust (ensure we always average a 2D array of shape `[tta_steps, n_test]`) and clamp predictions strictly inside (1, 100) to satisfy Kaggle’s submission validator. Finally, I write `submission.csv` with exactly `Id,Pawpularity` columns.'
- What this solution (achieved 42.24644) has done: 'I fix the missing-weights crash by loading the checkpoint from the competition dataset path and falling back to a safe glob search within `/kaggle/input` if the exact filename differs. I also fix the tensor dimension mismatch in the model forward pass by correctly extracting the Swin feature vector (pooling/flattening the 4D output when the head is replaced by `Identity`), which is the root cause of the inference crash and empty TTA outputs. These changes keep the same model architecture/training semantics, but make inference run end-to-end and produce a valid `submission.csv`. With the proper pretrained weights loaded and correct feature extraction, the score should move substantially toward the target RMSE.'
- What this solution (achieved 42.24644) has done: 'I fix the checkpoint discovery so weights are actually found in the provided dataset (the filename differs), and load them in a way that matches fastai’s saved formats. Then I fix the regression head’s input dimension calculation: for Swin in timm 1.x the feature dim must come from `num_features` (not `head.in_features`), which currently causes the 48x19 vs 1036x256 matmul crash. With those two fixes, TTA produce non-empty predictions and the script write a valid `submission.csv` with the required `Id,Pawpularity` columns; these are correctness fixes and should also move the score substantially toward the target because you finally be using the intended trained weights and correct feature extraction.'
- What this solution (achieved 42.24644) has done: 'I fix the checkpoint discovery so it actually finds the `.pth` file in this dataset layout (it’s inside the nested `petfinder-pawpularity-score/petfinder-pawpularity-score/` folder), instead of throwing `FileNotFoundError`. Then I fix the Swin feature-dimension mismatch that causes the `48x19 and 1036x256` matmul crash by constructing the backbone in a way that guarantees the output feature size matches the regression head (use `timm.create_model(..., num_classes=0, global_pool='avg')` so the backbone returns a flat `[B, num_features]` vector). These are correctness fixes (not training changes) and should also improve RMSE substantially versus the broken/inaccurate inference path, moving toward the target score. Finally, I keep the existing TTA + clamping + submission writing, but make TTA robust so it always produces a `[tta_steps, n_test]` array.'
- What this solution (achieved 42.24644) has done: 'I fix the checkpoint discovery so it can actually find the provided `.pth` weights in this Kaggle dataset layout (the filename often differs from `swin_fastai(final).pth`), and load it robustly. I also fix the regression-head input dimension mismatch by ensuring the Swin backbone always outputs a flat feature vector and by inferring the true feature dimension with a single dummy forward pass (so the first Linear layer matches the actual backbone output). These are correctness fixes that unblock inference and also move RMSE substantially toward your target because the model use the intended trained weights and correct feature extraction. Finally, I make TTA aggregation safe so it always produces exactly `len(test)` predictions and write a valid `submission.csv` with `Id,Pawpularity`.'
- What this solution (achieved 27.19232) has done: 'I fix two blocking issues so inference can run end-to-end: (1) robustly locate/load the provided trained checkpoint (your current search finds none, so we fall back to a safe “no-weights” path but still produce a submission), and (2) fix the Swin backbone/regression head feature-dimension mismatch that causes the `48x19 and 1036x256` matmul error by ensuring the backbone output is a flat `[B, num_features]` vector and building the regression head to match. These are correctness fixes that keep the same overall architecture (Swin + tabular concat + MLP head + sigmoid-to-0..100 scaling) while making the model actually usable, which should substantially reduce RMSE toward the target versus the currently broken/near-random outputs. Finally, I make the DataLoaders compatible with fastai (proper `after_item/after_batch` not required since we do manual inference) and ensure a valid `submission.csv` is always written with the exact required columns.'
- What this solution (achieved 23.75997) has done: 'Your current RMSE (27.19) is worse than the target (17.76), so we should make small, low-risk changes that improve generalization without changing the core model/training semantics. The biggest issue is that “TTA” currently repeats identical predictions because the test transform is deterministic; I make TTA actually apply random, mild augmentations (horizontal flip + small shift/scale/rotate) while keeping the same normalization/size pipeline. I also switch the Swin backbone to `pretrained=True` only when no checkpoint is found (so we don’t override your trained weights path), which typically improves RMSE substantially versus random init while preserving the same architecture. Finally, I keep submission formatting identical and ensure predictions remain within [1, 100] as before.'
- What this solution (achieved 38.11438) has done: 'Your current RMSE (23.76) is worse than the target (17.76), so we should make a small, low-risk improvement without changing the model or training semantics. The biggest easy win here is to make inference deterministic and more stable: right now your “TTA” is stochastic (random flip/rotate) but you never reset seeds, so results vary and the average can drift; we instead keep the same augmentations but use fixed seeds per TTA step and enable true eval-mode inference (disable grad, cudnn determinism where helpful). We also fix a subtle fastai `model_dir` misuse (it should be a directory, not a filename) that can interfere with learner internals, while keeping your checkpoint loading exactly the same. Finally, we keep the same submission formatting and clamping, but ensure prediction order stays aligned with `sample_submission.csv` Id order (a safe, often-material RMSE improvement if any ordering mismatch ever occurs).'
- What this solution (achieved 27.40569) has done: 'Your RMSE (38.11) is much worse than the target (17.76), so the most likely minimal improvement is to remove the main inference-time distribution shift: you are currently doing “TTA” with random flips/rotations, but this model was typically trained with deterministic center-crop eval, so stochastic TTA can hurt on this competition. I keep your same model, weights loading, and sigmoid*100 post-processing, but switch test-time transforms to deterministic (no random aug) and replace multi-step TTA with a single deterministic pass (still using the same averaging code path with `tta_steps=1`). This is a minimal semantic change (pure inference-time), should be more stable, and is expected to move RMSE down toward your target. Submission writing and Id alignment remain unchanged.'
- What this solution (achieved 30.09482) has done: 'Your current RMSE (27.41) is still well above the target (17.76), so we should make a small, low-risk improvement without changing the model architecture or training loop. The biggest likely gap is input distribution mismatch: Swin backbones in timm are typically trained with ImageNet mean/std, while your code uses custom normalization; switching to the model’s default `timm.data` mean/std usually improves RMSE materially while keeping inference semantics identical. I also add a very small, deterministic test-time augmentation (horizontal flip) and average the two passes (still pure inference-time, no randomness), which often helps a bit on this competition. Finally, I keep the same checkpoint loading, tabular features, post-processing, and submission alignment to ensure a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.losses import BCEWithLogitsLossFlat
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()), target))




## === cell 2
base_dir = "/kaggle/input"

dataset_roots = [
    os.path.join(base_dir, "petfinder-pawpularity-score"),
    os.path.join(
        base_dir, "petfinder-pawpularity-score", "petfinder-pawpularity-score"
    ),
]


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


test_file = _first_existing([os.path.join(r, "test.csv") for r in dataset_roots])
train_file = _first_existing([os.path.join(r, "train.csv") for r in dataset_roots])
test_folder = _first_existing([os.path.join(r, "test") for r in dataset_roots])
train_folder = _first_existing([os.path.join(r, "train") for r in dataset_roots])

default_weights_candidates = [
    os.path.join(r, "swin_fastai(final).pth") for r in dataset_roots
]
model_weights = _first_existing(default_weights_candidates)



## === cell 3
input_shape = (224, 224, 3)

model_name = "swin_base_patch4_window7_224"
_model_tmp = timm.create_model(
    model_name, pretrained=False, num_classes=0, global_pool="avg"
)
_cfg = getattr(_model_tmp, "default_cfg", {}) or {}
mean = _cfg.get("mean", (0.485, 0.456, 0.406))
std_dev = _cfg.get("std", (0.229, 0.224, 0.225))
del _model_tmp

batch_size = 48
device = "cuda" if torch.cuda.is_available() else "cpu"
output_categories = 11  # kept for compatibility with original config
n_epochs = 15
num_of_hidden = 2
hidden_dimension = [256, 64]
save_name = "/kaggle/working/swin_fastai(final).pth"



## === cell 4
print("device:", device)
print("test_file exists:", test_file, os.path.exists(test_file) if test_file else None)
print(
    "test_folder exists:",
    test_folder,
    os.path.exists(test_folder) if test_folder else None,
)
print(
    "train_file exists:",
    train_file,
    train_file is not None and os.path.exists(train_file),
)
print(
    "default model_weights exists:",
    model_weights,
    os.path.exists(model_weights) if model_weights else None,
)
print("Using normalization mean/std:", mean, std_dev)

if test_file is None or test_folder is None:
    raise FileNotFoundError(
        "Could not locate test.csv or test/ folder under /kaggle/input/petfinder-pawpularity-score"
    )



## === cell 5
test_csv = pd.read_csv(test_file)
test_csv.head()



## === cell 6
test_csv["path_img"] = list(
    map(lambda x: os.path.join(test_folder, x + ".jpg"), test_csv["Id"])
)
test_csv["Pawpularity"] = [1.0] * len(test_csv)




## === cell 7
class PetsDataset(Dataset):
    def __init__(self, df, transform=None, other=False):
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path_img"]
        label_1 = self.df.loc[idx, "Pawpularity"]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            album = self.transform(image=np.array(img))
            img = album["image"]

        df_data = self.df.loc[idx, self.cat].values.astype(np.float32)
        df_data = torch.tensor(df_data, dtype=torch.float32)

        if self.other:
            return img, label_1, self.df.iloc[idx, 1:-3].to_dict()

        return (img, df_data, label_1)




## === cell 8
def make_test_transform(hflip: bool = False):
    t = [
        A.LongestMaxSize(max_size=448, interpolation=1),
        A.PadIfNeeded(
            min_height=input_shape[0],
            min_width=input_shape[1],
            border_mode=0,
            value=(0, 0, 0),
        ),
        A.CenterCrop(height=input_shape[0], width=input_shape[1]),
    ]
    if hflip:
        t.append(A.HorizontalFlip(p=1.0))
    t += [
        A.Normalize(mean, std_dev),
        ToTensorV2(),
    ]
    return A.Compose(t)


test_transform = make_test_transform(hflip=False)
test_dataset = PetsDataset(test_csv, test_transform)
testloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=(device == "cuda"),
)



## === cell 9
dls = DataLoaders.from_dsets(test_dataset, test_dataset, bs=batch_size, device=device)



## === cell 10
b = next(iter(testloader))
print(type(b[0]), b[0].shape, b[1].shape)




## === cell 11
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Network(nn.Module):
    def __init__(
        self,
        base,
        number_of_hidden,
        hidden,
        regression_out,
        output_categories,
        freeze_layer,
        tab_dim=12,
    ):
        super(Network, self).__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        self.network = self.__freeze_layer(base, freeze_layer)

        with torch.no_grad():
            self.network.eval()
            dummy = torch.zeros(1, 3, input_shape[0], input_shape[1])
            out = self.network(dummy)
            if out.ndim == 4:
                out = out.mean(dim=(2, 3))
            elif out.ndim == 3:
                out = out.mean(dim=1)
            elif out.ndim == 2:
                pass
            else:
                out = out.view(out.size(0), -1)
            feature_dim = int(out.shape[1])

        hidden_dim = hidden[:]
        hidden_dim.insert(0, feature_dim + tab_dim)

        self.p = 0.5
        self.regression = self.__fully_connected(
            number_of_hidden, hidden_dim, regression_out
        )
        self.__initialise_weights()

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
        x1 = self.network(x)

        if x1.ndim == 4:
            x1 = x1.mean(dim=(2, 3))
        elif x1.ndim == 3:
            x1 = x1.mean(dim=1)
        elif x1.ndim == 2:
            pass
        else:
            x1 = x1.view(x1.size(0), -1)

        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg




## === cell 12
_have_ckpt = (model_weights is not None) and os.path.exists(model_weights)
network = timm.create_model(
    model_name, pretrained=(not _have_ckpt), num_classes=0, global_pool="avg"
)
model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0, tab_dim=12)




## === cell 13
def get_learner(dls, model, loss, metric, save_path):
    model = model.to(device)
    model_dir = os.path.dirname(save_path) if isinstance(save_path, str) else save_path
    if model_dir == "":
        model_dir = "."
    learn = Learner(dls, model, loss_func=loss, metrics=metric, model_dir=model_dir)
    if device == "cuda" and hasattr(learn, "to_fp16"):
        learn = learn.to_fp16()
    return learn


learn = get_learner(dls, model, BCEWithLogitsLossFlat(), petfinder_rmse, save_name)

if model_weights is None or (not os.path.exists(model_weights)):
    candidates = []
    for r in dataset_roots:
        candidates += glob.glob(os.path.join(r, "**", "*.pth"), recursive=True)
        candidates += glob.glob(os.path.join(r, "**", "*.pt"), recursive=True)
    preferred = [
        p
        for p in candidates
        if ("swin" in os.path.basename(p).lower())
        and (
            "fastai" in os.path.basename(p).lower()
            or "paw" in os.path.basename(p).lower()
        )
    ]
    if len(preferred) > 0:
        model_weights = preferred[0]
    elif len(candidates) > 0:
        model_weights = candidates[0]

print("Loading model weights from:", model_weights)

if model_weights is not None and os.path.exists(model_weights):
    state = torch.load(model_weights, map_location="cpu")

    loaded = False
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        learn.model.load_state_dict(state["model"], strict=False)
        loaded = True
    elif isinstance(state, dict) and any(
        k.startswith(("network.", "regression.")) for k in state.keys()
    ):
        learn.model.load_state_dict(state, strict=False)
        loaded = True
    elif (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        sd = state["state_dict"]
        if any(k.startswith("model.") for k in sd.keys()):
            sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        learn.model.load_state_dict(sd, strict=False)
        loaded = True
    elif isinstance(state, dict):
        learn.model.load_state_dict(state, strict=False)
        loaded = True

    if not loaded:
        raise ValueError(
            "Unknown checkpoint format (not a dict / state_dict-like object)"
        )
else:
    print("WARNING: No checkpoint found; using pretrained backbone weights only.")

learn.model.eval()
learn.model.to(device)




## === cell 14
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


tta_transforms = [make_test_transform(hflip=False), make_test_transform(hflip=True)]

tta_outputs = []
base_seed = 202401  # kept for reproducibility

for step, tfm in enumerate(tta_transforms):
    seed_everything(base_seed + step)

    _ds = PetsDataset(test_csv, tfm)
    _dl = DataLoader(
        _ds,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    final_outputs = []
    with torch.no_grad():
        for images, tabular, _ in _dl:
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)

            reg_output = learn.model(images, tabular)
            reg_output = 100 * torch.sigmoid(reg_output)

            output = reg_output.detach().float().cpu().numpy().reshape(-1).tolist()
            final_outputs.extend(output)

    tta_outputs.append(final_outputs)

n_test = len(test_csv)
for i, o in enumerate(tta_outputs):
    if len(o) != n_test:
        raise RuntimeError(f"TTA step {i} produced {len(o)} preds; expected {n_test}")



## === cell 15
tta_outputs_arr = np.asarray(tta_outputs, dtype=np.float32)  # [tta_steps, n_test]
if tta_outputs_arr.ndim != 2:
    raise ValueError(f"Unexpected TTA output shape: {tta_outputs_arr.shape}")
tta_outputs_arr = tta_outputs_arr.mean(axis=0)



## === cell 16
print(tta_outputs_arr[:10], len(tta_outputs_arr), "expected:", len(test_csv))



## === cell 17
pred = np.array(tta_outputs_arr, dtype=np.float32)
eps = 1e-3
pred = np.clip(pred, 1.0 + eps, 100.0 - eps)
test_csv["Pawpularity"] = pred



## === cell 18
sample_sub = None
for r in dataset_roots + [os.path.join(base_dir, "")]:
    cand = os.path.join(r, "sample_submission.csv")
    if os.path.exists(cand):
        sample_sub = cand
        break

if sample_sub is not None and os.path.exists(sample_sub):
    sub_ids = pd.read_csv(sample_sub)[["Id"]]
    test_csv = sub_ids.merge(test_csv[["Id", "Pawpularity"]], on="Id", how="left")
else:
    test_csv = test_csv[["Id", "Pawpularity"]]

test_csv.head()



## === cell 19
out_path = "submission.csv"
test_csv.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(test_csv))
print(
    "min/max:",
    float(test_csv["Pawpularity"].min()),
    float(test_csv["Pawpularity"].max()),
)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
