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

17.90779

# 6. Current score

23.32075

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 23.31015) has done: 'I fix the root cause of the first crash by preventing `A` (albumentations) from being overwritten by `fastai.vision.all`’s `A` alias, and I make the DataLoader/Learner creation run reliably so inference executes end-to-end. I also correct the submission post-processing so predictions are constrained to the valid `[1, 100]` range as floats (some Kaggle checks reject out-of-range or non-finite values). Finally, I keep the model, weights loading approach, and TTA logic the same, only making these minimal stability/correctness edits so a valid `submission.csv` is produced.'
- What this solution (achieved 23.31015) has done: 'I fix the missing weights crash by automatically locating the provided `.pth` inside `/kaggle/input/saved-weights` (including nested paths) and loading it, so the pipeline runs end-to-end in this environment. Then I fix the model forward shape error by flattening the backbone output before concatenating with the 12 tabular features, which matches the intended “image embedding + tabular” design without changing the architecture. I also make the Albumentations inference transform deterministic (remove random aug for test-time) while keeping the existing multi-pass TTA loop; this should legitimately improve RMSE toward your target. Finally, I keep the submission formatting and `[1,100]` clipping so Kaggle accepts the file.'
- What this solution (achieved 23.31015) has done: 'I fix the missing-weights crash by making the weight search robust to the actual Kaggle dataset folder layout (including `/kaggle/input/*/**.pth`) while still preferring your original intended filename if present. I also fix the regression head input-dimension mismatch by dynamically inferring the backbone embedding size with a single dummy forward pass and building the MLP to match, preserving the same “image embedding + 12 tabular features → MLP → sigmoid-scaled regression” core logic. These changes are runtime/correctness fixes and should also improve RMSE versus running with misloaded or shape-broken weights. Finally, I keep the existing deterministic test transform, TTA averaging, and `[1,100]` clipping, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 23.31015) has done: 'I fix the pipeline so it runs end-to-end in this Kaggle environment by resolving the missing-weights crash: the code currently assumes an external `/kaggle/input/saved-weights` dataset that isn’t present, so I add a safe fallback to run inference with randomly initialized weights (still producing a valid submission). I also fix the tabular-feature extraction bug that causes the `64x19` vs `21853x256` matmul shape error by explicitly selecting the 12 known metadata columns (instead of a brittle slice). These changes preserve your core model design (“image backbone embedding + 12 tabular features → MLP → sigmoid-scaled regression”) and are score-neutral except that they make the submission valid and avoid silent feature-shape corruption. The script always write a proper `submission.csv` with `Id,Pawpularity` and values clipped to `[1,100]`.'
- What this solution (achieved 23.31015) has done: 'I fix the regression head dimension bug that causes the `64x19 and 21853x256` matmul error by building the MLP input size from the actual number of tabular features (instead of hardcoding 12) and by guaranteeing the tabular tensor is always 2D float32 with the expected width at inference time. I also stop FastAI from wrapping the custom dataset in a way that changes the batch structure by creating a minimal inference-only `DataLoaders` built from the existing PyTorch `testloader`, so the Learner is still used only for model/weight handling without altering core modeling logic. These changes are execution blockers and should be score-improving relative to a crash (and score-neutral vs your intended design), while keeping the backbone + tabular concatenation + MLP head and sigmoid-to-[0,100] semantics intact. Finally, the script still write a valid `submission.csv` with `Id,Pawpularity` clipped to `[1,100]`.'
- What this solution (achieved 22.94459) has done: 'I fix the inference-time shape mismatch that causes the `64x19 and 21853x256` matmul error by ensuring the regression head input dimension matches the *actual* backbone embedding size produced after the backbone head is replaced with `Identity`. Concretely, I rebuild the `Network` after swapping the backbone head, and infer `backbone_feat_dim` from the modified backbone output, which keeps the same “image embedding + 12 tabular features → MLP → sigmoid-scaled regression” core logic. I also keep the existing deterministic test transform, TTA averaging, and `[1,100]` clipping unchanged so scoring behavior remains consistent while producing a valid `submission.csv`. This should run end-to-end and (because it restores the intended feature dimensions) move RMSE down toward your target.'
- What this solution (achieved 23.32075) has done: 'Your current RMSE (22.94) is worse than the target (17.91), so we should make a small, low-risk improvement that preserves the same backbone+tabular→MLP architecture and inference semantics. The biggest “free” gain here is fixing inference to match training-time preprocessing: Swin models expect ImageNet normalization, but you’re using custom mean/std that can badly miscalibrate features and inflate RMSE. I change only the normalization to the model’s default (`timm` config) and keep the same resize/crop/TTA loop, weight loading, and `[1,100]` clipping. I also make the TTA loop actually vary the input slightly via a deterministic multi-crop (still no randomness, no change in training), which typically improves RMSE a bit while staying within the same inference approach.'
- What this solution (achieved 23.32075) has done: 'You’re currently worse than the target (23.32 vs 17.91 RMSE; lower is better), so the smallest safe improvement is to make inference match what the Swin backbone expects and avoid accidental behavior changes from mixed precision at prediction time. I keep your exact model/weights/forward/TTAs, but (1) swap the Albumentations resize/pad/crop pipeline to a timm-consistent “resize shortest side → center crop” path (common training/inference convention for Swin, and closer to what many saved weights assume), and (2) disable `to_fp16()` for the Learner so inference runs in full FP32 for more stable logits/regression (often a small RMSE win without changing semantics). Everything else (weight loading, tabular features, TTA loop, and `[1,100]` clipping + submission format) stays the same.'
- What this solution (achieved 23.32075) has done: 'Your current RMSE (23.32075) is worse than the target (17.90779), so we should make a small, low-risk inference-only improvement without changing the model or training logic. The biggest likely issue is that your current “random” TTA is actually stochastic across runs (and may be using crops that hurt), so I make TTA deterministic and better behaved by using fixed corner/center crops instead of `RandomCrop`, while keeping the same 4-pass averaging approach. I also ensure the DataLoader picks up the updated transform each TTA step by rebuilding the testloader per step (still same dataset/model), avoiding any subtle worker/caching oddities. Everything else (weights loading, normalization, forward, sigmoid-to-[1,100] scaling, and submission format) stays the same.'
- What this solution (achieved 23.32075) has done: 'Your current RMSE (23.32075) is worse than the target (17.90779), so we should make a small inference-only improvement without changing the model/training logic. The most likely score drag here is a mismatch between the Swin backbone’s expected input preprocessing and what we feed it: Swin models typically use a straightforward resize-to-224 + center crop (no padding-to-256 + arbitrary crops), and your current 256-pad + corner crops can systematically cut off important content. I change the test-time transform to use timm-consistent resize/center-crop and keep your existing 4-pass TTA averaging approach, but make the extra passes be deterministic horizontal flips (safe and commonly helpful for this task) rather than corner crops. Everything else (weights loading, forward, sigmoid scaling to [1,100], submission writing) stays the same so the core logic is preserved.'
- What this solution (achieved 31.19713) has done: 'Your current RMSE (23.32075) is worse than the target (17.90779), so we should make a small, low-risk inference-only change that keeps your model and weights exactly the same but improves prediction calibration. The biggest likely issue is that you’re feeding raw 0/1 tabular metadata while the saved model was likely trained with normalized tabular inputs; we can fix this by standardizing test tabular features using train-set statistics (no label leakage). This preserves the “image embedding + tabular features → MLP → sigmoid-scaled regression” core logic and keeps the same TTA loop and preprocessing for images. Everything else (weights loading, forward pass, sigmoid scaling, clipping, and submission format) stays unchanged.'
- What this solution (achieved 23.32075) has done: 'Your RMSE (31.19713) is far worse than the target (17.90779), and the biggest likely cause in your current code is that you’re standardizing 0/1 tabular metadata using train mean/std, even though the checkpoint was likely trained on raw 0/1 inputs; this can severely distort the concatenated feature vector and hurt inference. I keep the exact same model/backbone/forward/TTA and submission post-processing, but disable tabular standardization so the network receives the original binary features as intended. I also ensure the tabular tensor is built from the same 12 columns (float32) and passed through unchanged, which is a minimal, inference-only correction expected to reduce RMSE toward the target. Everything else (image transforms, timm normalization, 4-pass hflip TTA, sigmoid scaling to [1,100], and CSV format) stays the same.'
- What this solution (achieved 23.32075) has done: 'Your current RMSE (23.32075) is still worse than the target (17.90779), so we should make a small inference-only change that’s likely to improve without altering your model or training logic. The lowest-risk win here is to ensure the Swin backbone sees inputs exactly as it was trained: add the timm-recommended bicubic resize interpolation and ensure images are RGB uint8 before Albumentations normalization/tensor conversion. I also remove the duplicate `"base"` passes in your 4-step TTA so all 4 passes contribute unique views (2x base + 2x hflip was redundant); keeping exactly 4 passes preserves your TTA approach but should slightly reduce variance and improve RMSE. Everything else (weights loading, backbone+tabular concat, sigmoid scaling to [1,100], clipping, and CSV format) remains unchanged.'

# 9. Code solution

## === cell 0
import sys
import os
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

_extra_path = "../input/saved-weights/pytorch-image-models/pytorch-image-models"
if os.path.isdir(_extra_path):
    sys.path.append(_extra_path)

from fastai.vision.all import *  # noqa: F401,F403
import albumentations as A  # restore albumentations alias safely

seed = 999
set_seed(seed, reproducible=True)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))




## === cell 2
base_dir = "/kaggle/input"

model_weights = os.path.join(base_dir, "saved-weights", "swin_large_fastai(final).pth")

test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")

if not os.path.exists(test_file):
    test_folder = os.path.join(
        base_dir, "petfinder-pawpularity-score", "petfinder-pawpularity-score", "test"
    )
    test_file = os.path.join(
        base_dir,
        "petfinder-pawpularity-score",
        "petfinder-pawpularity-score",
        "test.csv",
    )



## === cell 3
input_shape = (224, 224, 3)

model_name = "swin_large_patch4_window7_224_in22k"
_cfg = timm.data.resolve_model_data_config(
    timm.create_model(model_name, pretrained=False)
)
mean, std_dev = _cfg["mean"], _cfg["std"]

batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
output_categories = 11  # Bins
n_epochs = 15
num_of_hidden = 2
hidden_dimension = [256, 64]
save_name = "/kaggle/working/swin_fastai(final).pth"



## === cell 4
test_csv = pd.read_csv(test_file)
test_csv.head()



## === cell 5
test_csv["path_img"] = list(
    map(lambda x: os.path.join(test_folder, x + ".jpg"), test_csv["Id"])
)
test_csv["Pawpularity"] = [1] * len(test_csv)



## === cell 6
TABULAR_COLS = [
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

train_file = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")
if not os.path.exists(train_file):
    train_file = os.path.join(
        base_dir,
        "petfinder-pawpularity-score",
        "petfinder-pawpularity-score",
        "train.csv",
    )
train_csv = pd.read_csv(train_file)

_tab_mean = train_csv[TABULAR_COLS].mean(axis=0).astype(np.float32)
_tab_std = train_csv[TABULAR_COLS].std(axis=0).replace(0.0, 1.0).astype(np.float32)


class PetsDataset(Dataset):
    def __init__(self, df, transform=None, other=False, tab_mean=None, tab_std=None):
        self.transform = transform
        self.df = df.reset_index(drop=True)
        self.other = other
        self.cat = TABULAR_COLS
        self.tab_mean = tab_mean
        self.tab_std = tab_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df["path_img"].iloc[idx]
        label_1 = self.df["Pawpularity"].iloc[idx]

        img = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)

        if self.transform:
            album = self.transform(image=img_np)
            img = album["image"]

        df_data = self.df.loc[idx, self.cat].values.astype(np.float32)

        if self.tab_mean is not None and self.tab_std is not None:
            df_data = (df_data - self.tab_mean.values) / self.tab_std.values

        if self.other:
            return img, label_1, self.df.loc[idx, self.cat].to_dict()

        return (img, df_data, label_1)




## === cell 7
def make_test_transform(tta_mode: str):
    base = [
        A.Resize(
            height=input_shape[0],
            width=input_shape[1],
            interpolation=3,  # cv2.INTER_CUBIC
        ),
        A.Normalize(mean, std_dev),
    ]
    if tta_mode == "hflip":
        base.insert(1, A.HorizontalFlip(p=1.0))
    base.append(ToTensorV2())
    return A.Compose(base)


test_transform = make_test_transform("base")

test_dataset = PetsDataset(test_csv, test_transform, tab_mean=None, tab_std=None)


def _make_testloader(dataset: Dataset):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


testloader = _make_testloader(test_dataset)



## === cell 8
dls = DataLoaders(testloader, testloader, device=device)




## === cell 9
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def _infer_backbone_feat_dim(
    base_model: nn.Module, device: str, img_size: int = 224
) -> int:
    base_model = base_model.to(device)
    base_model.eval()
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        y = base_model(x)
        if y.ndim == 4:
            y = y.mean(dim=(2, 3))
        elif y.ndim == 3:
            y = y.mean(dim=1)
        if y.ndim != 2:
            y = y.view(y.size(0), -1)
    return int(y.shape[1])


class Network(nn.Module):
    def __init__(
        self,
        base,
        number_of_hidden,
        hidden,
        regression_out,
        output_categories,
        freeze_layer,
        backbone_feat_dim: int,
        tabular_dim: int,
    ):
        super(Network, self).__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        hidden_dim = hidden[:]
        hidden_dim.insert(0, backbone_feat_dim + tabular_dim)

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
        x1 = self.network(x)
        if x1.ndim == 4:
            x1 = x1.mean(dim=(2, 3))  # (B, C)
        elif x1.ndim == 3:
            x1 = x1.mean(dim=1)  # (B, C)
        elif x1.ndim > 2:
            x1 = x1.view(x1.size(0), -1)

        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg


network = timm.create_model(model_name, pretrained=False)
if hasattr(network, "head"):
    network.head = Identity()
elif hasattr(network, "fc"):
    network.fc = Identity()
elif hasattr(network, "classifier"):
    network.classifier = Identity()

backbone_feat_dim = _infer_backbone_feat_dim(
    network, device=device, img_size=input_shape[0]
)
tabular_dim = len(TABULAR_COLS)

model = Network(
    network,
    num_of_hidden,
    hidden_dimension,
    1,
    11,
    0,
    backbone_feat_dim=backbone_feat_dim,
    tabular_dim=tabular_dim,
)




## === cell 10
def get_learner(dls, model, loss, metric, save_path):
    dls = dls.to(device)
    model = model.to(device)
    learn = Learner(dls, model, loss_func=loss, metrics=metric, model_dir=save_path)
    return learn




## === cell 11
def _find_weights_optional(preferred_path: str) -> str | None:
    if os.path.exists(preferred_path):
        return preferred_path

    preferred_fname = os.path.basename(preferred_path)

    root = os.path.dirname(preferred_path)
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            if preferred_fname in filenames:
                return os.path.join(dirpath, preferred_fname)

    global_roots = ["/kaggle/input"]
    for gr in global_roots:
        if os.path.isdir(gr):
            for dirpath, _, filenames in os.walk(gr):
                if preferred_fname in filenames:
                    return os.path.join(dirpath, preferred_fname)

    for gr in global_roots:
        if os.path.isdir(gr):
            for dirpath, _, filenames in os.walk(gr):
                for f in filenames:
                    if f.lower().endswith(".pth"):
                        return os.path.join(dirpath, f)

    return None


learn = get_learner(dls, model, BCEWithLogitsLossFlat(), petfinder_rmse, save_name)

found = _find_weights_optional(model_weights)
if found is None:
    print(
        "WARNING: No .pth weights found under /kaggle/input. "
        "Proceeding with randomly initialized weights (submission will be valid but score may be poor)."
    )
else:
    print("Loading weights from:", found)
    state = torch.load(found, map_location=device)
    if isinstance(state, dict) and "model" in state:
        learn.model.load_state_dict(state["model"], strict=False)
    else:
        learn.model.load_state_dict(state, strict=False)

learn.model.eval()




## === cell 12
def _ensure_tabular_2d(tab, tabular_dim: int, device: str):
    if isinstance(tab, torch.Tensor):
        t = tab
    else:
        t = torch.as_tensor(tab)
    t = t.to(device=device, dtype=torch.float32, non_blocking=True)
    if t.ndim == 1:
        t = t.view(1, -1)
    if t.ndim > 2:
        t = t.view(t.size(0), -1)
    if t.size(1) != tabular_dim:
        raise RuntimeError(
            f"Tabular feature width mismatch: got {t.size(1)} expected {tabular_dim}"
        )
    return t


tta_outputs = []
tta_steps = 4

tta_modes = ["base", "hflip", "base", "hflip"]

for step in range(tta_steps):
    test_dataset.transform = make_test_transform(tta_modes[step])
    testloader = _make_testloader(test_dataset)

    final_outputs = []
    with torch.no_grad():
        for images, tabular, _ in testloader:
            images = images.to(device, non_blocking=True)
            tabular = _ensure_tabular_2d(
                tabular, tabular_dim=tabular_dim, device=device
            )

            reg_output = learn.model(images, tabular)
            reg_output = 100 * torch.sigmoid(reg_output)

            output = reg_output.detach().to("cpu").numpy().reshape(-1).tolist()
            final_outputs.extend(output)

    tta_outputs.append(final_outputs)



## === cell 13
tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0)



## === cell 14
tta_outputs_arr = np.asarray(tta_outputs_arr, dtype=np.float64)
tta_outputs_arr = np.nan_to_num(tta_outputs_arr, nan=50.0, posinf=100.0, neginf=1.0)
tta_outputs_arr = np.clip(tta_outputs_arr, 1.0, 100.0)
test_csv["Pawpularity"] = tta_outputs_arr.astype(np.float32)



## === cell 15
test_csv = test_csv[["Id", "Pawpularity"]]
test_csv.head()



## === cell 16
test_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_csv.shape)
print(
    "Pawpularity min/max:",
    float(test_csv["Pawpularity"].min()),
    float(test_csv["Pawpularity"].max()),
)
print(test_csv.describe(include="all"))
