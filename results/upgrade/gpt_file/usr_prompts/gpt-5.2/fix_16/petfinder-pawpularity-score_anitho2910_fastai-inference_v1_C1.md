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

17.70938217053753

# 6. Current score

20.08411

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fix the Albumentations name collision that turns `A` into a `functools.partial`, which is why `A.Compose` crashes and prevents `test_dataset/dls/learn` from being created. I also fix model weight loading: fastai’s `Learner.load()` expects a filename stem inside `model_dir`, so we robustly fall back to `torch.load` when given a full `.pth` path (score-neutral, but unblocks inference). Finally, I make the submission post-processing guarantee the required [1, 100] range even if TTA fails or outputs NaNs, and ensure the CSV has the exact required columns and `.csv` suffix.'
- What this solution (achieved 23.31015) has done: 'I fix the fastai mixed-precision call that currently crashes because `to_fp16` is no longer imported/used that way in fastai 2.8, by using the supported `learn.to_fp16()` method instead of chaining `.to_fp16()` on the `Learner` constructor result. I also make the `DataLoaders` creation explicit (using the PyTorch `testloader`) so inference doesn’t depend on a partially-initialized fastai dataloader, which avoids the downstream “learn is not defined” and length-mismatch errors. Finally, I keep your existing inference/TTA and post-processing logic intact but add a small guard to ensure the prediction vector length always matches the test dataframe length before writing `submission.csv`. These changes are execution-critical and should also improve RMSE versus the current broken/degenerate path that yielded 42.24644.'
- What this solution (achieved 23.31015) has done: 'I fix the import error by using `torch.nn.functional as F` instead of `from fastai.imports import F`, which restores the RMSE metric function without changing its behavior. Then I fix the fastai `DataLoaders` device transfer crash by wrapping the PyTorch `DataLoader` in a small adapter that implements `.to()` (and forwards iteration/length), so `dls.to(device)` works while keeping your existing inference loop unchanged. Finally, I keep the model, transforms, TTA, and post-processing logic intact, ensuring the script runs end-to-end and writes a valid `submission.csv` with the correct columns.'
- What this solution (achieved 23.31015) has done: 'I fix the `to_fp16()` crash by importing and using fastai’s mixed-precision callback correctly (`learn.to_fp16()` requires `fastai.callback.fp16` to be imported), which currently prevents the learner from being created and therefore blocks inference. I also make `model_dir` a directory (not a `.pth` file path) so `learn.load()` uses the intended fastai semantics and doesn’t silently fail due to an invalid model directory. Finally, I keep your model/inference/TTA logic unchanged but add a robust weight-loading fallback that also handles common checkpoint key prefixes (`module.`, `model.`), which should improve score versus partially-loaded weights while remaining within the same core approach.'
- What this solution (achieved 23.31015) has done: 'I fix the missing-weights crash by locating the checkpoint file robustly within `/kaggle/input` (including the common nested dataset folder) without changing the model itself. I also fix the runtime shape error in `forward()` by ensuring the backbone output is pooled to a 2D tensor before concatenating tabular features, which preserves the intended “image-embedding + 12 features → MLP regression” core logic. Finally, I keep your existing TTA/inference and post-processing, but make sure the prediction length matches `test.csv` exactly and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 23.31015) has done: 'I fix the two execution blockers: first, make checkpoint discovery robust by searching for any plausible `.pth`/`.pkl` weight file under `/kaggle/input` (including the nested competition folder) and fall back gracefully if only a fastai-exported file exists. Second, I fix the regression head input-dimension mismatch causing the `mat1 and mat2 shapes cannot be multiplied` error by deriving the backbone feature dimension from a real forward pass (instead of relying on `base.head.in_features`, which is not valid for Swin in timm 1.0+), preserving the same “image embedding + 12 tabular features → MLP” architecture. These changes should unblock inference and, by correctly wiring the head to the backbone, move RMSE down toward the target versus the current broken inference path. The submission writing stays the same but is kept robust to NaNs and length mismatches.'
- What this solution (achieved 27.5775) has done: 'I (1) make checkpoint loading non-fatal by falling back to running with randomly initialized weights when the `.pth` file is not present in this Kaggle environment, so the notebook always completes and writes `submission.csv`. Then (2) I fix the regression head input dimension bug that causes `mat1 and mat2 shapes cannot be multiplied` by ensuring Swin outputs a consistent pooled feature vector (using `forward_features`) and inferring the true feature dimension after the head is replaced. These fixes preserve your core “Swin backbone + 12 metadata features → MLP regression” logic and keep the same inference/TTA and post-processing semantics, while unblocking execution and restoring correct tensor shapes. If the weights are available, they be loaded as before; if not, the script still produce a valid submission file.'
- What this solution (achieved 27.5775) has done: 'Your current score (27.5775 RMSE) is far worse than the target (17.7094), and the code indicates you may be running with randomly initialized weights when the checkpoint isn’t found, which strongly degrade RMSE. I make the smallest change that improves score: ensure the script actually finds and loads the checkpoint if it exists anywhere under `/kaggle/input`, and stop using random weights silently by raising a clear error if no weights are found (so you don’t accidentally submit garbage). I also make inference deterministic by removing stochastic test-time augmentations (ShiftScaleRotate/ColorJitter/RandomCrop/Flip) from the *test* pipeline while preserving the same model and sigmoid→[0,100] post-processing; this typically reduces RMSE (less noisy predictions) and should move you toward the target. Finally, I keep the submission formatting/length guards intact and still write a valid `submission.csv`.'
- What this solution (achieved 20.08411) has done: 'I fix the execution blocker by making checkpoint loading non-fatal: if no weights are found under `/kaggle/input`, the script proceed using the untrained model and still write a valid `submission.csv` (so you always get a submission). To avoid producing an obviously terrible submission in that case, I add a minimal, score-stabilizing fallback that predicts the train-set mean Pawpularity (a standard baseline) when weights are missing, which should reduce RMSE versus random predictions while keeping the model/inference core logic unchanged when weights do exist. I also correct the hardcoded checkpoint search to avoid relying on a non-existent `saved-weights` root dataset path, but still keep your preferred filename and robust walk-based search. All other modeling, transforms, and post-processing remain the same.'
- What this solution (achieved 20.08411) has done: 'Your current RMSE (20.084) is worse than the target (17.709), so we should make small, low-risk changes that improve accuracy without changing the core Swin+tabular→MLP inference logic. The biggest likely issue is that you’re normalizing with custom mean/std that may not match the Swin weights you’re loading; switching to the model’s expected preprocessing (timm default) usually improves RMSE materially while keeping the model/forward/loss unchanged. I also ensure the checkpoint is loaded in a way that can’t silently miss due to fastai `model_dir` semantics (but still falls back to `torch.load` exactly as you already do). Finally, I keep your single-pass “TTA” at 1, but make inference use `autocast` consistently to match fp16 learner behavior and avoid small numeric drift.'
- What this solution (achieved 20.08411) has done: 'Your current RMSE (20.084) is worse than the target (17.709), so we should make a small, low-risk change that improves predictions without changing your model, loss, or inference loop. The biggest likely score drag is a train/test preprocessing mismatch: you currently resize to 448 then center-crop to 224, which changes the image content distribution versus the standard 224 pipeline most Swin checkpoints are trained with. I adjust only the test-time resize/crop to the standard timm inference preprocessing (resize shortest side to 256 then center-crop 224) while keeping normalization and everything else identical, which typically reduces RMSE. I also bump `num_workers` modestly to reduce dataloader overhead (score-neutral, helps runtime) and keep submission writing unchanged.'
- What this solution (achieved 20.08411) has done: 'Your current RMSE (20.084) is worse than the target (17.709), so we make a minimal, low-risk change that typically improves inference accuracy without changing the model, loss, or training approach: use the exact timm-recommended preprocessing pipeline for the chosen Swin backbone (resize/crop interpolation + normalization) derived from `timm.data.create_transform`. This addresses common score loss from subtle preprocessing mismatches (mean/std, interpolation, crop logic) while preserving your core Swin+tabular→MLP forward and your existing sigmoid→[0,100] post-processing. I keep your dataloader/inference loop identical, and only swap the test transform construction to timm’s resolved config so it matches what the checkpoint expects. Submission writing remains unchanged and still produces `submission.csv`.'
- What this solution (achieved 20.08411) has done: 'We need to move RMSE down from 20.084 toward 17.709 (lower is better), so we should make a small, low-risk inference-only change that improves calibration without changing your model, loss, or dataloader logic. The most reliable minimal gain here is to apply a lightweight “shrink-to-mean” calibration on the final predictions (a common RMSE reducer when a model is overconfident/noisy), with the shrink factor estimated from the training label distribution only (no leakage). This preserves your core Swin+tabular forward and sigmoid→[0,100] semantics; it only post-processes predictions to reduce squared error on average. If weights are missing, the baseline behavior remains effectively unchanged (already mean).'
- What this solution (achieved 20.08411) has done: 'Your current RMSE (20.084) is worse than the target (17.709), so we should make a small, low-risk improvement that tends to reduce squared error without changing your model, loss, or inference loop. The most likely remaining issue is prediction miscalibration: the fixed shrink factor (0.92) may be suboptimal; I replace it with a data-driven shrink computed from the training label variance and an estimated noise level, using only `train.csv` (no leakage). This keeps identical evaluation semantics (still outputs in [1,100]) and only adjusts the final post-processing step to move RMSE downward toward the target. All paths, model definition, and inference remain unchanged, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import sys



## === cell 1
sys.path.append(
    "../input/d/anitho2910/saved-weights/pytorch-image-models/pytorch-image-models"
)



## === cell 2
import os
import gc
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F  # use torch.nn.functional

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
import torchvision  # kept
import matplotlib.pyplot as plt  # kept

import fastai
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.losses import BCEWithLogitsLossFlat

from fastai.callback.fp16 import *  # noqa: F401,F403




## === cell 3
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))




## === cell 4
base_dir = "/kaggle/input"
model_weights = os.path.join(base_dir, "saved-weights", "swin_fastai(final).pth")

test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")



## === cell 5
input_shape = (224, 224, 3)

model_name = "swin_base_patch4_window7_224"
timm_data_cfg = timm.data.resolve_model_data_config(
    timm.create_model(model_name, pretrained=False)
)
mean = list(timm_data_cfg.get("mean", (0.485, 0.456, 0.406)))
std_dev = list(timm_data_cfg.get("std", (0.229, 0.224, 0.225)))

batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
output_categories = 11  # kept
n_epochs = 15
num_of_hidden = 2
hidden_dimension = [256, 64]

save_name = "/kaggle/working"



## === cell 6
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 7
test_csv = pd.read_csv(test_file)
test_csv.head()



## === cell 8
test_csv["path_img"] = list(
    map(lambda x: os.path.join(test_folder, x + ".jpg"), test_csv["Id"])
)
test_csv["Pawpularity"] = [1] * len(test_csv)




## === cell 9
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
        img_path = self.df["path_img"].iloc[idx]
        label_1 = self.df["Pawpularity"].iloc[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            album = self.transform(image=np.array(img))
            img = album["image"]

        df_data = self.df.loc[idx, self.cat].values.astype(np.float32)
        df_data = torch.from_numpy(df_data)

        if self.other:
            return img, label_1, self.df.iloc[idx, 1:-3].to_dict()

        return (img, df_data, label_1)




## === cell 10
from timm.data import create_transform


def _albumentations_from_timm_inference_cfg(data_cfg: dict):
    input_size = data_cfg.get("input_size", (3, 224, 224))
    size = int(input_size[-1])
    crop_pct = float(data_cfg.get("crop_pct", 0.9))
    interpolation = str(data_cfg.get("interpolation", "bicubic")).lower()
    mean_ = list(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    std_ = list(data_cfg.get("std", (0.229, 0.224, 0.225)))

    interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "bicubic": 2,
        "lanczos": 4,
        "lanczos4": 4,
        "area": 3,
    }
    interp_code = interp_map.get(interpolation, 2)

    resize_short = int(round(size / crop_pct))
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=resize_short, interpolation=interp_code),
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean_, std_),
            ToTensorV2(),
        ]
    )


mean = list(timm_data_cfg.get("mean", mean))
std_dev = list(timm_data_cfg.get("std", std_dev))

test_transform = _albumentations_from_timm_inference_cfg(timm_data_cfg)

test_dataset = PetsDataset(test_csv, test_transform)

testloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)




## === cell 11
class _FastaiDLAdapter:
    def __init__(self, dl):
        self.dl = dl
        self._device = None

    def to(self, device):
        self._device = device
        return self

    def __iter__(self):
        return iter(self.dl)

    def __len__(self):
        return len(self.dl)

    @property
    def dataset(self):
        return self.dl.dataset

    def __getattr__(self, name):
        return getattr(self.dl, name)


dls = DataLoaders(
    _FastaiDLAdapter(testloader), _FastaiDLAdapter(testloader), device=device
)



## === cell 12
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()




## === cell 13
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def _infer_backbone_feat_dim(backbone: nn.Module, device: str) -> int:
    """
    Bugfix: infer feature dimension after classifier head is removed, and do so
    using backbone.forward_features when available (timm Swin).
    """
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_shape[0], input_shape[1], device=device)
        if hasattr(backbone, "forward_features"):
            out = backbone.forward_features(dummy)
        else:
            out = backbone(dummy)

        if out.ndim == 4:
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        elif out.ndim == 3:
            out = out.mean(dim=1)
        elif out.ndim == 2:
            pass
        else:
            out = out.view(out.size(0), -1)
        return int(out.shape[1])


class Network(nn.Module):
    def __init__(
        self,
        base,
        number_of_hidden,
        hidden,
        regression_out,
        output_categories,
        freeze_layer,
    ):
        super(Network, self).__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        if hasattr(base, "head"):
            base.head = Identity()
        if hasattr(base, "fc"):
            base.fc = Identity()
        if hasattr(base, "classifier"):
            base.classifier = Identity()

        feat_dim = _infer_backbone_feat_dim(base.to(device), device)

        hidden_dim = hidden[:]
        hidden_dim.insert(0, feat_dim + 12)

        self.p = 0.5

        self.regression = self.__fully_connected(
            number_of_hidden, hidden_dim, regression_out
        )

        self.network = self.__freeze_layer(base, freeze_layer)

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
        if hasattr(self.network, "forward_features"):
            x1 = self.network.forward_features(x)
        else:
            x1 = self.network(x)

        if x1.ndim == 4:
            x1 = F.adaptive_avg_pool2d(x1, 1).flatten(1)
        elif x1.ndim == 3:
            x1 = x1.mean(dim=1)
        elif x1.ndim == 2:
            pass
        else:
            x1 = x1.view(x1.size(0), -1)

        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg


network = timm.create_model(model_name, pretrained=False)
model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)




## === cell 14
def get_learner(dls, model, loss, metric, save_path):
    dls = dls.to(device)
    model = model.to(device)
    os.makedirs(save_path, exist_ok=True)
    learn = Learner(dls, model, loss_func=loss, metrics=metric, model_dir=save_path)
    learn = learn.to_fp16()
    return learn


def _clean_state_dict(sd):
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        return sd

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _find_checkpoint_path(preferred_path: str) -> str | None:
    """
    Bugfix: robustly find the checkpoint if present under /kaggle/input.
    Keep the original preferred name, but don't assume a particular dataset root.
    """
    if os.path.isfile(preferred_path):
        return preferred_path

    preferred_base = os.path.basename(preferred_path)
    exts = (".pth", ".pt", ".pkl")

    candidates = [
        os.path.join("/kaggle/input", preferred_base),
        os.path.join("/kaggle/input", "petfinder-pawpularity-score", preferred_base),
        os.path.join(
            "/kaggle/input",
            "petfinder-pawpularity-score",
            "petfinder-pawpularity-score",
            preferred_base,
        ),
        os.path.join("/kaggle/input", "models", preferred_base),
        os.path.join(
            "/kaggle/input", "petfinder-pawpularity-score", "models", preferred_base
        ),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    found_exact = []
    found_any = []
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            full = os.path.join(root, f)
            if f == preferred_base:
                found_exact.append(full)
            if f.lower().endswith(exts) and ("swin" in f.lower()):
                found_any.append(full)

    def _depth(p: str) -> int:
        return p.count(os.sep)

    if found_exact:
        found_exact = sorted(found_exact, key=lambda p: (_depth(p), len(p), p))
        return found_exact[0]
    if found_any:
        found_any = sorted(found_any, key=lambda p: (_depth(p), len(p), p))
        return found_any[0]
    return None




## === cell 15
learn = get_learner(dls, model, BCEWithLogitsLossFlat(), petfinder_rmse, save_name)

model_weights_found = _find_checkpoint_path(model_weights)
weights_loaded = False

if model_weights_found is not None:
    print("Using model weights:", model_weights_found)
    try:
        import shutil

        stem = os.path.splitext(os.path.basename(model_weights_found))[0]
        dst = os.path.join(save_name, f"{stem}.pth")
        if os.path.abspath(model_weights_found) != os.path.abspath(dst):
            shutil.copyfile(model_weights_found, dst)

        learn.load(stem)
        weights_loaded = True
    except Exception:
        sd = torch.load(model_weights_found, map_location="cpu")
        sd = _clean_state_dict(sd)
        missing, unexpected = learn.model.load_state_dict(sd, strict=False)
        print(
            "Loaded via torch.load; missing keys:",
            len(missing),
            "unexpected:",
            len(unexpected),
        )
        weights_loaded = True
else:
    print(
        "WARNING: Model weights not found under /kaggle/input. "
        "Will fall back to a mean-baseline prediction to ensure a usable submission."
    )

learn.model.eval()

train_file = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")
if not os.path.isfile(train_file):
    train_file = os.path.join(base_dir, "train.csv")
train_mean_pawpularity = None
train_std_pawpularity = None
try:
    train_df = pd.read_csv(train_file, usecols=["Pawpularity"])
    train_mean_pawpularity = float(train_df["Pawpularity"].mean())
    train_std_pawpularity = float(train_df["Pawpularity"].std(ddof=0))
except Exception:
    train_mean_pawpularity = 50.0  # safe fallback
    train_std_pawpularity = 20.0  # safe-ish fallback scale



## === cell 16
tta_outputs = []
tta_steps = 1

use_amp = torch.cuda.is_available()

if weights_loaded:
    for _ in range(tta_steps):
        final_outputs = []
        learn.model.eval()
        with torch.no_grad():
            for images, tabular, _ in testloader:
                images = images.to(device, non_blocking=True)
                tabular = tabular.to(device, non_blocking=True).float()

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    reg_output = learn.model(images, tabular)
                    reg_output = 100 * torch.sigmoid(reg_output)

                output = reg_output.detach().to("cpu").numpy().reshape(-1).tolist()
                final_outputs.extend(output)

        tta_outputs.append(final_outputs)
else:
    tta_outputs = [([train_mean_pawpularity] * len(test_csv))]



## === cell 17
tta_outputs_arr = np.mean(np.array(tta_outputs, dtype=np.float32), axis=0)
tta_outputs_arr = np.asarray(tta_outputs_arr, dtype=np.float32).reshape(-1)
tta_outputs_arr = np.nan_to_num(tta_outputs_arr, nan=50.0, posinf=100.0, neginf=1.0)

n = len(test_csv)
if tta_outputs_arr.shape[0] != n:
    if tta_outputs_arr.shape[0] > n:
        tta_outputs_arr = tta_outputs_arr[:n]
    else:
        pad_val = float(np.nanmean(tta_outputs_arr)) if tta_outputs_arr.size else 50.0
        tta_outputs_arr = np.pad(
            tta_outputs_arr, (0, n - tta_outputs_arr.shape[0]), constant_values=pad_val
        )

tta_outputs_arr[:10], len(tta_outputs_arr)



## === cell 18
if train_mean_pawpularity is None:
    train_mean_pawpularity = 50.0
if (
    train_std_pawpularity is None
    or not np.isfinite(train_std_pawpularity)
    or train_std_pawpularity <= 1e-6
):
    train_std_pawpularity = 20.0

label_var = float(train_std_pawpularity**2)
noise_var = float(16.0**2)
shrink_alpha = label_var / (label_var + noise_var)
shrink_alpha = float(np.clip(shrink_alpha, 0.80, 0.97))

tta_outputs_arr = (shrink_alpha * tta_outputs_arr) + (
    (1.0 - shrink_alpha) * train_mean_pawpularity
)

test_csv["Pawpularity"] = np.clip(tta_outputs_arr, 1.0, 100.0)



## === cell 19
test_csv = test_csv[["Id", "Pawpularity"]]



## === cell 20
test_csv.head()



## === cell 21
test_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_csv.shape)
print(
    "Pawpularity range:",
    float(test_csv["Pawpularity"].min()),
    float(test_csv["Pawpularity"].max()),
)
print("Weights loaded:", weights_loaded)
print("Baseline mean used (if no weights):", train_mean_pawpularity)
print("Train std used:", train_std_pawpularity)
print("Shrink alpha used:", shrink_alpha)
print("timm data cfg used:", timm_data_cfg)
print("Normalization mean/std used:", mean, std_dev)
