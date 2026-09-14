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

21.80521

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 21.80521) has done: 'I fix the shape-mismatch crash by making the Swin backbone always return a fixed-length feature vector (using `forward_features` + global pooling) and by building the regression head input size from a dummy forward pass, so the MLP always matches the actual backbone output. I also ensure the tabular feature vector is exactly the 12 competition metadata columns (the current code accidentally passes 18 columns via slicing), which was a hidden cause of the `64x19` input shape. After that, the training loop can run, TTA inference work, and the notebook always write a valid `submission.csv` with the required `Id,Pawpularity` columns. These fixes are execution/correctness fixes and should move you toward a sensible RMSE rather than failing to submit.'

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
def petfinder_rmse(input, target):
    pred = 100.0 * torch.sigmoid(input.flatten())
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
        label_1 = (
            self.df.loc[idx, "Pawpularity"] if "Pawpularity" in self.df.columns else 1.0
        )

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            album = self.transform(image=np.array(img))
            img = album["image"]

        df_data = self.df.loc[idx, self.cat].values.astype(np.float32)

        if self.other:
            return img, label_1, self.df.loc[idx, self.cat].to_dict()

        return (img, df_data, np.float32(label_1))




## === cell 6
train_transform = alb.Compose(
    [
        alb.LongestMaxSize(max_size=448, interpolation=1),
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
        alb.LongestMaxSize(max_size=448, interpolation=1),
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



## === cell 7
idx = np.arange(len(train_csv))
rng = np.random.default_rng(seed)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_df = train_csv.iloc[tr_idx].reset_index(drop=True)
valid_df = train_csv.iloc[va_idx].reset_index(drop=True)

train_ds = PetsDataset(train_df, train_transform)
valid_ds = PetsDataset(valid_df, valid_transform)
test_ds = PetsDataset(test_csv.assign(Pawpularity=1.0), test_transform)

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
    Bug fix: resolve matmul shape mismatch by:
      1) For timm models, using forward_features() and a consistent global pooling to a 2D vector.
      2) Inferring the true feature dimension via a dummy forward pass.
      3) Concatenating exactly 12 tabular features.
    Core architecture (backbone + MLP head) is preserved.
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


class SigmoidScaledMSELoss(nn.Module):
    def forward(self, input, target):
        pred = 100.0 * torch.sigmoid(input.flatten())
        return F.mse_loss(pred, target.float().flatten())


learn = get_learner(dls, model, SigmoidScaledMSELoss(), petfinder_rmse, save_name)

learn.fit_one_cycle(n_epochs, lr_max=3e-4)

torch.save(learn.model.state_dict(), model_weights)

learn.model.eval()
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None



## === cell 10
tta_outputs = []
tta_steps = 4

for _ in range(tta_steps):
    final_outputs = []
    with torch.no_grad():
        for images, tabular, _ in testloader:
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True).float()

            reg_output = learn.model(images, tabular)
            reg_output = 100 * torch.sigmoid(reg_output)

            output = reg_output.detach().float().cpu().numpy().reshape(-1).tolist()
            final_outputs.extend(output)

    tta_outputs.append(final_outputs)

tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0)



## === cell 11
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
