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
cuml-cu12==25.2.1
fastai==2.8.5
geopandas==0.14.4
libcuml-cu12==25.2.1
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

17.00575828788845

# 6. Current score

20.08411

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fix the runtime crash by removing the GPU-only `cuml` dependency (it fails due to insufficient CUDA driver) and loading the saved SVR models with standard `pickle` while keeping the same SVR blending logic. I also fix the Albumentations import collision that turned `A` into a `functools.partial`, by avoiding `from fastai.vision.all import *` and importing only what’s needed from fastai so `A.Compose` works again. Finally, I make the submission valid by clipping predictions to the required [1, 100] range (a score-neutral formatting fix that also prevents Kaggle rejection) and ensure the code writes `submission.csv` end-to-end.'
- What this solution (achieved 42.24644) has done: 'I fix the runtime error by removing the incompatible `learn.to_fp16()` call (FastAI v2 no longer exposes it on `Learner` the same way, and it triggers an attribute lookup that fails on your model). I keep the exact model/weights/SVR+TTA blending logic unchanged, and only add safe, score-neutral robustness fixes: correct the Network constructor call (it currently has a missing argument) and switch the test-time augmentation pipeline to deterministic (no random augmentations during inference), which should materially improve RMSE toward your target without changing the core approach. Finally, I ensure the weights are loaded correctly with FastAI’s expected `load()` naming and that `submission.csv` is always written with the required columns.'
- What this solution (achieved 42.24644) has done: 'I fix the crash by making the weight/SVR filename resolution robust to the actual files present in `/kaggle/input/saved-weights` instead of assuming a specific naming pattern (`swin_large_384_fold_0_full.pth`), which is currently missing. The core model/learner/tta/SVR blending logic stays the same; only the mapping from `model_name/fold` to on-disk filenames is corrected. To keep score moving toward your target (lower RMSE), I also remove the ineffective “TTA loop” randomness (your transforms are deterministic), and instead reuse a single deterministic pass per fold (this is score-neutral to slightly better, and much faster/stabler). Finally, the code always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 42.24644) has done: 'I fix the crash by making the weight/SVR filename matching consistent with timm’s full model names (your current `m_name` truncation to `swin_large` / `swin_large_384` doesn’t match the actual saved filenames). To keep the core inference/blending logic identical, I only adjust how we derive the `must_contain` tokens used to locate the correct `.pth` and `.pkl` per fold. I also make `learn.load()` robust to both fastai-style “stem-only” loading and direct state-dict `.pth` loading (some saved weights aren’t fastai-exported), without changing the model architecture or predictions beyond negligible floating-point differences. The rest of the pipeline (dataset, transforms, SVR blending, clipping, and submission writing) remains the same.'
- What this solution (achieved 42.24644) has done: 'I fix the fold weight/SVR file discovery so it matches the actual filenames in `/kaggle/input/saved-weights` by (1) indexing files recursively and (2) using a more flexible fold-regex that handles common variants like `fold0`, `fold-0`, `fold_0`, or `Fold 0`. This is directly targeted at the current `FileNotFoundError` and keeps the same model/SVR blending and inference logic unchanged. I also make the loader use the exact resolved path (rather than re-deriving from just the filename) so it works whether files are nested or flat. These changes are score-neutral (they only unblock correct weight loading); the score improvement comes from actually using the intended trained weights rather than failing/forcing fallbacks.'
- What this solution (achieved 42.24644) has done: 'I fix the `FileNotFoundError` by making fold/filename discovery robust to common weight naming conventions that don’t include the full timm model string, and by falling back to “best match by model tokens + fold” (rather than failing hard immediately). I also make the fold regex handle `fold0`, `fold-0`, `fold_0`, and similar patterns, and add a safe fallback that can run end-to-end even if a specific model’s weight/SVR pair is missing (it skip that model so you still get a valid submission). These are execution-unblocking changes; they should also improve your score versus the current run because you actually load and use the intended trained artifacts instead of crashing or implicitly using random-init. The core model, inference, SVR blending, and submission logic are unchanged.'
- What this solution (achieved 42.24644) has done: 'I fix the artifact discovery so it can actually find the saved `.pth` weights and `.pkl` SVR models under `/kaggle/input/saved-weights`, since right now none match and the pipeline aborts. Concretely, I broaden the matching logic to (1) not require the fold string to be present (falls back to best token match), and (2) treat common “oof/svr” files as global per model when fold-specific ones aren’t available. This is a minimal change that keeps the same model architectures, the same SVR+NN blending, and the same inference flow, but unblocks using the intended trained artifacts—expected to reduce RMSE substantially toward the target. The rest of the code (transforms, Network, loss/metric, clipping, and submission writing) is kept intact.'
- What this solution (achieved 23.31015) has done: 'I fix the “no models produced predictions” failure by making artifact discovery deterministic and model-specific (so we don’t accidentally pick unrelated `.pth/.pkl` files) and by adding a safe, score-neutral fallback path that still produces a valid submission if no saved artifacts exist in `/kaggle/input/saved-weights`. This keeps your core inference/blending logic intact (same Network, same SVR+NN blend, same preprocessing), but unblocks end-to-end execution. If artifacts are present but naming differs, the improved matching uses stricter token coverage plus fold matching so the intended weights are much more likely to be found—this should materially improve RMSE toward your target versus the current “no predictions” crash. Finally, I ensure the submission is always written as `submission.csv` with the correct columns and aligned row order.'
- What this solution (achieved 20.08411) has done: 'Your current RMSE (23.31; lower is better) is still above the target (17.01), so we should make small, low-risk changes that improve predictive quality without changing the core architecture or blending approach. The biggest score win that fits your constraints is fixing an inference-mode mismatch: you’re using `BCEWithLogitsLoss` and sigmoid scaling, which expects the target to be in `[0,1]`, but your SVR and final predictions are on `[1,100]`; this mismatch typically harms calibration and RMSE. I keep the same model, SVR+NN blending, and 100*sigmoid output, but I (a) scale the SVR predictions onto the same `[1,100]` range before blending and (b) apply a tiny, validation-free linear calibration that reduces systematic bias (shrink toward the training mean), which is a minimal post-processing adjustment for RMSE. I also ensure deterministic inference (`learn.model.eval()` already exists) and add a strict alignment check so the submission row order always matches `sample_submission.csv` order.'
- What this solution (achieved 20.08411) has done: 'I make two minimal, score-relevant fixes that should reduce RMSE from 20.08 toward your 17.01 target without changing your core model/SVR blending logic. First, your fastai `DataLoaders` currently build with `num_workers=0` and no pinned memory, while the manual `testloader` uses workers/pin; this mismatch can subtly affect throughput and determinism—so I make both loaders consistent and deterministic. Second, your SVR predictions are only rescaled when they look like `[0,1]`; if the SVR was trained on raw logits or standardized targets, it be mis-scaled and harm the blend—so I add a safe, monotonic per-run calibration of SVR outputs to match the NN prediction distribution on test (rank-preserving), which typically improves RMSE while keeping the same blending approach. Everything else (architecture, forward, 100*sigmoid head, SVR+NN weighted blend, shrink-to-mean, submission merge) stays intact and it still writes a valid `submission.csv`.'
- What this solution (achieved 20.08411) has done: 'We make two score-focused but minimal adjustments that keep your core model/SVR blending intact: (1) set Albumentations resize interpolation explicitly to a high-quality mode (`cv2.INTER_CUBIC`) to better match common training pipelines and reduce distribution shift, and (2) apply a very light, deterministic “blend shrink” that pulls each model’s SVR prediction slightly toward the NN prediction before the SVR+NN weighted blend, which tends to reduce SVR outlier harm without changing the architecture or training semantics. We keep the existing artifact discovery, Network, and 100*sigmoid head unchanged, and still write `submission.csv` in the exact required format. These changes are small and should move RMSE down from 20.08 toward your 17.01 target without risky refactors.'
- What this solution (achieved 20.08411) has done: 'We need to move RMSE down from 20.084 toward 17.006 (lower is better), so we make only small inference/post-processing tweaks that preserve your existing model/SVR blending and architecture. The main score drag in this pipeline is almost always systematic bias/scale mismatch across folds/models; we can reduce that with a very light, deterministic affine calibration of the final predictions (fit on the training target distribution only, no leakage) plus a slightly better “shrink-to-mean” that accounts for your prediction variance. We also make SVR rank-matching use a monotone quantile map to the NN outputs (still deterministic, same blend logic) and ensure the tabular tensor is created without an extra copy. These changes keep the same models, same weights, same SVR+NN weighted blend, and still produce a valid `submission.csv`.'
- What this solution (achieved 20.08411) has done: 'We need to move RMSE down from 20.084 toward 17.006 (lower is better), so the safest gains come from fixing a likely inference mismatch rather than changing architectures. Your model head outputs `100*sigmoid(logit)` but you’re loading checkpoints that may have been trained for a *scaled target in [0,1]* (common for this competition when using `BCEWithLogitsLoss`), which would make your current `100*sigmoid` systematically mis-scaled. I add a minimal, deterministic “auto-scale” check per fold: compare the NN output distribution in both modes (scaled vs 100x) and choose the one that better matches the training target mean/std (no label leakage). I also replace the current SVR “quantile map to NN” fallback with a symmetric map to the *chosen NN scale* (so the SVR blend is on the same scale), while keeping your SVR+NN blending weights and the rest of the pipeline unchanged.'
- What this solution (achieved 20.08411) has done: 'Your current RMSE (20.084, lower is better) is still above the target (17.006), so the safest way to move toward the target without changing the model/blend logic is to remove a likely source of systematic error: the per-fold “auto scale” chooser that can flip the NN/SVR scale inconsistently and distort the final blend. I make scale handling deterministic and consistent by always treating the NN head as producing Pawpularity in `[1,100]` via `1+99*sigmoid`, and then map SVR predictions onto the same `[1,100]` scale with a minimal, monotone distribution match when needed (preserves ranking; keeps the same SVR+NN weighted blend). I also remove the redundant re-choose step after averaging (it was using a heuristic inverse transform that can be wrong) and keep your existing light post-hoc calibration/clipping/submission alignment unchanged. These are small, inference-only changes aimed at reducing avoidable scale mismatch noise, which commonly improves RMSE on this competition.'
- What this solution (achieved 20.08411) has done: 'Your current RMSE (20.084, lower is better) is still above the target (17.006), so we make the smallest inference-only changes that reduce avoidable prediction noise without changing the model/SVR blending core. The main low-risk issue is that `PetsDataset` returns tabular features as NumPy arrays, but you then call `torch.from_numpy(tabular)` in inference—when DataLoader batches, `tabular` is already a Tensor, so this can silently create incorrect tensors or crash depending on PyTorch version; fixing this makes the SVR embedding/features consistent and typically improves RMSE. I also make the SVR “distribution repair” decision more robust by always applying the monotone quantile map when SVR scale is suspicious (std/range mismatch), which preserves ranking but aligns it to the NN output scale—still the same SVR+NN blend, just safer. Everything else (architecture, 1+99*sigmoid scaling, weights discovery/loading, fold averaging, final light calibration, submission alignment) stays the same and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import random
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.losses import BCEWithLogitsLossFlat

import cv2

sys.path.append("../input/saved-weights/pytorch-image-models/pytorch-image-models")




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()), target))




## === cell 2
base_dir = "/kaggle/input"
model_weights_dir = os.path.join(base_dir, "saved-weights")
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")
train_file = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")
sample_sub_file = os.path.join(
    base_dir, "petfinder-pawpularity-score", "sample_submission.csv"
)



## === cell 3
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
device = "cuda" if torch.cuda.is_available() else "cpu"

N_FOLDS = 5
embed_dim = 128
num_of_hidden = 2
batch_size = 64
hidden_dimension = [256, 64]

input_shape_224 = (224, 224, 3)
input_shape_384 = (384, 384, 3)
max_size_384 = 480
max_size_224 = 384

models_list = {
    "swin_large_patch4_window12_384_in22k": 64,
    "swin_base_patch4_window7_224_in22k": 128,
    "swin_large_patch4_window7_224_in22k": 64,
}
save_name = "/kaggle/working/"

weights_svr = {
    "swin_large_patch4_window12_384_in22k": [0.1, 0.9],
    "swin_base_patch4_window7_224_in22k": [0.5, 0.5],
    "swin_large_patch4_window7_224_in22k": [0.4, 0.6],
}

model_weights = {
    "swin_large_patch4_window12_384_in22k": 0.5,
    "swin_base_patch4_window7_224_in22k": 0.2,
    "swin_large_patch4_window7_224_in22k": 0.3,
}




## === cell 4
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 5
test_csv = pd.read_csv(test_file)
test_csv.head()



## === cell 6
train_csv = pd.read_csv(train_file, usecols=["Pawpularity"])
train_mean = float(train_csv["Pawpularity"].mean())
train_mean



## === cell 7
test_csv["path_img"] = list(
    map(lambda x: os.path.join(test_folder, x + ".jpg"), test_csv["Id"])
)
test_csv["Pawpularity"] = [1] * len(test_csv)




## === cell 8
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df
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

        df_data = torch.tensor(self.df[self.cat].iloc[idx].values, dtype=torch.float32)
        return (img, df_data, label_1)




## === cell 9
def get_data(batch_size, max_size, input_shape):
    test_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=max_size, interpolation=cv2.INTER_CUBIC),
            A.PadIfNeeded(
                min_height=input_shape[0],
                min_width=input_shape[1],
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.CenterCrop(height=input_shape[0], width=input_shape[1]),
            A.Normalize(mean, std_dev),
            ToTensorV2(),
        ]
    )

    test_dataset = PetsDataset(test_csv, test_transform)

    n_workers = 4
    pin = torch.cuda.is_available()

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=pin,
        persistent_workers=(n_workers > 0),
    )

    dls = DataLoaders.from_dsets(
        test_dataset,
        test_dataset,
        bs=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin,
        persistent_workers=(n_workers > 0),
    )
    return dls, testloader




## === cell 10
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

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
    ):
        super().__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        hidden_dim = hidden[:]
        hidden_dim.insert(0, embed_dim + 12)

        base.head = nn.Linear(base.head.in_features, embed_dim)
        nn.init.kaiming_normal_(base.head.weight)
        nn.init.constant_(base.head.bias, 0)

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
        layers = [nn.Dropout(self.p)]
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
        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg, x




## === cell 11
def get_learner(model_name, batch_size, loss, metric, max_size, input_size, save_path):
    dls, testloader = get_data(batch_size, max_size, input_size)
    dls = dls.to(device)
    network = timm.create_model(model_name, pretrained=False)
    model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0).to(device)
    learn = Learner(dls, model, loss_func=loss, metrics=metric, model_dir=save_path)
    return learn, testloader




## === cell 12
def _index_available_files(weights_dir: str):
    p = Path(weights_dir)
    if not p.exists():
        return [], []
    pths = [x for x in p.rglob("*.pth") if x.is_file()]
    pkls = [x for x in p.rglob("*.pkl") if x.is_file()]
    return pths, pkls


def _fold_patterns(fold: int):
    return [
        re.compile(rf"(?i)\bfold\s*[_\-\s]*{fold}\b"),
        re.compile(rf"(?i)\bf\s*[_\-\s]*{fold}\b"),
        re.compile(rf"(?i)fold{fold}\b"),
        re.compile(rf"(?i)f{fold}\b"),
    ]


def _match_fold(filename: str, fold: int) -> bool:
    for pat in _fold_patterns(fold):
        if pat.search(filename):
            return True
    return False


def _derive_must_contain_tokens(model_name: str):
    toks = []
    toks.append(model_name)
    base = model_name.replace("_in22k", "")
    toks.append(base)
    toks.append(base.replace("_", "-"))
    toks.append(base.replace("_", ""))
    toks.append(base.split("_in22k")[0])
    if "swin" in model_name:
        toks.append("swin")
        if "large" in model_name:
            toks.append("large")
            toks.append("swin_large")
            toks.append("swinlarge")
        if "base" in model_name:
            toks.append("base")
            toks.append("swin_base")
            toks.append("swinbase")
        if "384" in model_name:
            toks.append("384")
        if "224" in model_name:
            toks.append("224")
        if "window12" in model_name:
            toks.append("window12")
        if "window7" in model_name:
            toks.append("window7")
        if "patch4" in model_name:
            toks.append("patch4")
    return list(dict.fromkeys([t for t in toks if t]))


def _token_coverage_score(filename: str, tokens) -> float:
    fn = filename.lower()
    hits = 0
    for t in tokens:
        tl = t.lower()
        if tl and tl in fn:
            hits += 1
    return hits / max(1, len(tokens))


def _score_candidate(
    path: Path, fold: int, tokens, prefer_fold: bool, kind: str
) -> float:
    fn = path.name.lower()

    coverage = _token_coverage_score(path.name, tokens)
    score = 100.0 * coverage

    if prefer_fold:
        score += 40.0 if _match_fold(path.name, fold) else -20.0

    if "full" in fn:
        score += 2.0
    if "best" in fn:
        score += 2.0

    if kind == "svr":
        if "svr" in fn:
            score += 8.0
        if "oof" in fn:
            score += 2.0
        if "model" in fn or "weight" in fn:
            score -= 5.0

    score -= min(5.0, len(fn) / 200.0)
    return score


def _find_best_file(paths, fold: int, tokens, exts, prefer_fold: bool, kind: str):
    cand = [p for p in paths if (not exts or p.suffix in exts)]
    if len(cand) == 0:
        return None

    scored = sorted(
        cand,
        key=lambda x: (
            -_score_candidate(x, fold, tokens, prefer_fold=prefer_fold, kind=kind),
            x.name,
        ),
    )
    best = scored[0]
    if _token_coverage_score(best.name, tokens) < 0.12:
        return None
    return best


def _load_weights_robust(learn: Learner, weight_stem_or_path: str):
    try:
        learn.load(weight_stem_or_path)
        return
    except Exception:
        pass

    path = weight_stem_or_path
    if not os.path.isfile(path) and os.path.isfile(path + ".pth"):
        path = path + ".pth"
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Weight file not found for loading: {weight_stem_or_path}"
        )

    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v

    missing, unexpected = learn.model.load_state_dict(new_state, strict=False)
    if len(unexpected) > 0:
        print(
            "Warning: unexpected keys when loading:",
            unexpected[:5],
            "..." if len(unexpected) > 5 else "",
        )
    if len(missing) > 0:
        print(
            "Warning: missing keys when loading:",
            missing[:5],
            "..." if len(missing) > 5 else "",
        )


pth_files, pkl_files = _index_available_files(model_weights_dir)
print(
    f"Found {len(pth_files)} .pth and {len(pkl_files)} .pkl files under {model_weights_dir}"
)
if len(pth_files) > 0:
    print("Example .pth files:", sorted([p.name for p in pth_files])[:10])
if len(pkl_files) > 0:
    print("Example .pkl files:", sorted([p.name for p in pkl_files])[:10])




## === cell 13
def _svr_to_pawpularity_scale(preds: np.ndarray) -> np.ndarray:
    preds = preds.astype(np.float32)
    pmin = float(np.nanmin(preds))
    pmax = float(np.nanmax(preds))
    if pmin >= -0.05 and pmax <= 1.05:
        preds = 1.0 + 99.0 * np.clip(preds, 0.0, 1.0)
    return preds


def _quantile_map_to_reference(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    n = len(src)
    if n == 0:
        return src

    order = np.argsort(src, kind="mergesort")
    src_sorted = src[order]
    ref_sorted = np.sort(ref)

    ranks = np.linspace(0.0, 1.0, n, dtype=np.float32)

    if float(src_sorted[0]) == float(src_sorted[-1]):
        return np.full_like(src, float(np.mean(ref_sorted)), dtype=np.float32)

    q = np.interp(src, src_sorted, ranks).astype(np.float32)
    out = np.interp(q, ranks, ref_sorted).astype(np.float32)
    return out


def _svr_scale_suspicious(svr_preds: np.ndarray, nn_preds: np.ndarray) -> bool:
    svr_preds = np.asarray(svr_preds, dtype=np.float32)
    nn_preds = np.asarray(nn_preds, dtype=np.float32)

    if len(svr_preds) == 0:
        return False

    s_std = float(np.std(svr_preds) + 1e-6)
    n_std = float(np.std(nn_preds) + 1e-6)
    std_ratio = s_std / n_std

    s_min, s_max = float(np.min(svr_preds)), float(np.max(svr_preds))
    if (s_min < -5.0) or (s_max > 105.0):
        return True
    if (std_ratio < 0.25) or (std_ratio > 4.0):
        return True
    return False


def tta(
    testloader,
    learn,
    svr,
    svr_weight,
    y_mean: float,
    y_std: float,
    tta_steps=1,
):
    tta_outputs = []
    learn.model.eval()

    for _ in range(tta_steps):
        final_outputs_sigmoid = []
        svr_data = []

        with torch.no_grad():
            for images, tabular, _ in testloader:
                images = images.to(device, non_blocking=True)
                tabular = tabular.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                reg_output, embed = learn.model(images, tabular)

                reg_sig = torch.sigmoid(reg_output).detach().cpu().numpy().reshape(-1)
                final_outputs_sigmoid.extend(reg_sig.tolist())
                svr_data.extend(embed.detach().cpu().numpy())

        final_outputs_sigmoid = np.array(final_outputs_sigmoid, dtype=np.float32)
        svr_data = np.array(svr_data, dtype=np.float32)

        final_outputs = 1.0 + 99.0 * np.clip(final_outputs_sigmoid, 0.0, 1.0)
        final_outputs = final_outputs.astype(np.float32)

        svr_preds = svr.predict(svr_data).astype(np.float32)
        svr_preds = _svr_to_pawpularity_scale(svr_preds)

        if _svr_scale_suspicious(svr_preds, final_outputs):
            svr_preds = _quantile_map_to_reference(svr_preds, final_outputs)

        svr_shrink = 0.08
        svr_preds = (1.0 - svr_shrink) * svr_preds + svr_shrink * final_outputs

        final_outputs = svr_weight[0] * svr_preds + svr_weight[1] * final_outputs
        tta_outputs.append(final_outputs)

    tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0).astype(np.float32)
    return tta_outputs_arr




## === cell 14
y = train_csv["Pawpularity"].to_numpy(dtype=np.float32)
y_mean = float(np.mean(y))
y_std = float(np.std(y) + 1e-6)

final_predictions = []
used_model_weights = []
used_any_artifacts = False

for model_name, bs in models_list.items():
    fold_tta = []
    svr_weights = weights_svr[model_name]
    w = model_weights[model_name]

    m_384 = "384" in model_name

    print("#################################")
    print(f"Testing Model: {model_name}")
    print("#################################")

    tokens = _derive_must_contain_tokens(model_name)

    global_svr_p = _find_best_file(
        pkl_files, fold=0, tokens=tokens, exts={".pkl"}, prefer_fold=False, kind="svr"
    )
    if global_svr_p is None:
        print(f"Warning: no SVR .pkl found for model '{model_name}'. Skipping model.")
        continue

    model_ok = True
    for fold in range(N_FOLDS):
        print(f"Fold: {fold}")

        weight_p = _find_best_file(
            pth_files,
            fold=fold,
            tokens=tokens,
            exts={".pth"},
            prefer_fold=True,
            kind="weights",
        )

        svr_p = _find_best_file(
            pkl_files,
            fold=fold,
            tokens=tokens,
            exts={".pkl"},
            prefer_fold=True,
            kind="svr",
        )
        if svr_p is None:
            svr_p = global_svr_p

        if weight_p is None or svr_p is None:
            print(
                f"Warning: missing artifacts for model '{model_name}' fold {fold}. "
                f"weight_p={None if weight_p is None else weight_p.name}, "
                f"svr_p={None if svr_p is None else svr_p.name}. "
                f"Skipping entire model to keep pipeline running."
            )
            model_ok = False
            break

        used_any_artifacts = True

        if m_384:
            learn, testloader = get_learner(
                model_name,
                bs,
                BCEWithLogitsLossFlat(),
                petfinder_rmse,
                max_size_384,
                input_shape_384,
                save_name,
            )
        else:
            learn, testloader = get_learner(
                model_name,
                bs,
                BCEWithLogitsLossFlat(),
                petfinder_rmse,
                max_size_224,
                input_shape_224,
                save_name,
            )

        os.makedirs(save_name, exist_ok=True)

        weight_path = str(weight_p)
        weight_fn = weight_p.name
        local_weight_path = os.path.join(save_name, weight_fn)
        if (not os.path.exists(local_weight_path)) and (
            local_weight_path != weight_path
        ):
            import shutil

            shutil.copy2(weight_path, local_weight_path)

        _load_weights_robust(
            learn, os.path.join(save_name, weight_fn).replace(".pth", "")
        )

        if not any(p.requires_grad for p in learn.model.parameters()):
            for p in learn.model.parameters():
                p.requires_grad = True

        with open(str(svr_p), "rb") as f:
            svr = pickle.load(f)

        fold_tta.append(
            tta(
                testloader,
                learn,
                svr,
                svr_weights,
                y_mean=y_mean,
                y_std=y_std,
                tta_steps=1,
            )
        )

        del learn, svr, testloader
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if not model_ok:
        continue

    fold_tta = np.array(fold_tta, dtype=np.float32)
    fold_tta = np.mean(fold_tta, axis=0)
    final_predictions.append(w * fold_tta)
    used_model_weights.append(w)

if len(final_predictions) == 0:
    print(
        "Warning: No models produced predictions (no matching weights/SVR found). "
        "Falling back to a constant prediction to produce a valid submission."
    )
    final_predictions = np.full((len(test_csv),), train_mean, dtype=np.float32)
    used_model_weights = [1.0]
else:
    final_predictions = np.array(final_predictions, dtype=np.float32)



## === cell 15
if isinstance(final_predictions, np.ndarray) and final_predictions.ndim == 2:
    final_predictions = np.sum(final_predictions, axis=0)



## === cell 16
if len(used_model_weights) != len(model_weights):
    denom = float(np.sum(np.array(used_model_weights, dtype=np.float32)))
    if denom > 0:
        final_predictions = final_predictions / denom



## === cell 17
p_mean = float(np.mean(final_predictions))
p_std = float(np.std(final_predictions) + 1e-6)

alpha = y_std / p_std
alpha = float(np.clip(alpha, 0.90, 1.10))  # very light calibration only
final_predictions = alpha * (final_predictions - p_mean) + p_mean

shrink = 0.06
final_predictions = (1.0 - shrink) * final_predictions + shrink * y_mean

final_predictions = np.clip(final_predictions, 1.0, 100.0)



## === cell 18
test_csv["Pawpularity"] = final_predictions



## === cell 19
sample_sub = pd.read_csv(sample_sub_file, usecols=["Id"])
submission = sample_sub.merge(test_csv[["Id", "Pawpularity"]], on="Id", how="left")
if submission["Pawpularity"].isna().any():
    submission["Pawpularity"] = submission["Pawpularity"].fillna(train_mean)

submission = submission[["Id", "Pawpularity"]].copy()
submission.head()



## === cell 20
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe(include="all"))
