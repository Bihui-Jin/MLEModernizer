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

20.08403

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 23.68267) has done: 'I replace the fixed‑size linear regressor with a lazy‑linear version that determines the required input dimension at runtime, eliminating the shape mismatch that caused the RuntimeError. This change preserves the overall architecture while ensuring the model can concatenate image and tabular features correctly. The rest of the pipeline stays unchanged, so the script now run end‑to‑end and output a valid `submission.csv` within the required value range.'
- What this solution (achieved 20.08403) has done: 'I add a lightweight calibration step that fits a simple linear regression on the validation (train) data to correct systematic bias in the model’s raw sigmoid‑scaled outputs. The calibrated mapping is then applied to the test‑time TTA predictions before clipping, which should lower the RMSE and move the score closer to the target while keeping the original architecture and inference pipeline intact.'
- What this solution (achieved 20.08403) has done: 'I add a slight random horizontal‑flip to the test transform so each TTA step produces a different view, increase the number of TTA steps from 4 to 8 to average more predictions, and relax the lower clipping bound from 1 to 0 (the true scores can be 0). These minimal adjustments keep the model architecture unchanged while providing a modest improvement toward the target RMSE.'
- What this solution (achieved 20.08405) has done: 'I add a small vertical flip to the test augmentation and increase the TTA count slightly, then apply a modest temperature scaling (×1.2) to the model logits before the sigmoid. These lightweight tweaks keep the original architecture intact while likely improving calibration and reducing the RMSE, moving the score closer to the target.'
- What this solution (achieved 20.08403) has done: 'I slightly simplify the test‑time augmentations (remove the random vertical flip) and restore the model’s original logit scale by setting the temperature factor back to 1.0. I also increase the TTA count a bit (12 passes) to smooth predictions without altering the network architecture. These minimal adjustments keep the calibration step unchanged but are expected to reduce the RMSE, moving the score closer to the target.'
- What this solution (achieved 20.08403) has done: 'I keep the overall model and inference pipeline unchanged but improve calibration by using a deterministic transform (no random flip) for the calibration set and replace the simple LinearRegression with a Ridge regression (α=1.0) to reduce over‑fitting. These minimal tweaks should lower the RMSE toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 20.08403) has done: 'I keep the overall architecture and training pipeline unchanged, but add a tiny post‑calibration scaling factor that is fitted on the training‑set predictions to better align the calibrated outputs with the true scores. I also increase the TTA passes slightly (from 12 to 16) to smooth the predictions a bit more. These minimal tweaks are expected to lower the RMSE and move the score toward the target without altering the core model logic.'
- What this solution (achieved 20.08403) has done: 'I remove the extra multiplicative calibration scale (which can over‑correct the Ridge predictions) and set a slight temperature scaling < 1.0 so the sigmoid logits are modestly shrunk before the final calibration, keeping all other architecture and data handling unchanged. These minimal tweaks are expected to lower the RMSE toward the target without altering the core model.'

# 9. Code solution

## === cell 0
import os, random, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.linear_model import Ridge  # use Ridge for more stable calibration

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()), target))




## === cell 2
base_dir = "/kaggle/input"
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")
train_file = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "swin_base_patch4_window7_224"
batch_size = 64
input_shape = (224, 224)




## === cell 3
test_csv = pd.read_csv(test_file)
test_csv["path_img"] = test_csv["Id"].apply(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)




## === cell 4
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
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
        row = self.df.iloc[idx]
        img_path = row["path_img"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)  # returns a Tensor
        tab = torch.tensor(row[self.cat].astype(float).values, dtype=torch.float32)
        label = torch.tensor(0.0, dtype=torch.float32)
        return img, tab, label




## === cell 5
test_transform = transforms.Compose(
    [
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(input_shape),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5023, 0.4615, 0.4226], std=[0.2640, 0.2593, 0.2575]
        ),
    ]
)

deterministic_transform = transforms.Compose(
    [
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5023, 0.4615, 0.4226], std=[0.2640, 0.2593, 0.2575]
        ),
    ]
)

test_dataset = PetsDataset(test_csv, transform=test_transform)
testloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
)




## === cell 6
class Identity(nn.Module):
    def forward(self, x):
        return x


class Network(nn.Module):
    def __init__(self, base, hidden_dims, out_features, freeze_until):
        super().__init__()
        self.dropout_p = 0.5
        base.head = Identity()
        self.backbone = self._freeze_layers(base, freeze_until)

        layers = []
        layers.append(nn.LazyLinear(hidden_dims[0]))
        layers.append(nn.GELU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(self.dropout_p))

        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            if i != len(hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout_p))

        layers.append(nn.Linear(hidden_dims[-1], out_features))
        self.regressor = nn.Sequential(*layers)

    def _freeze_layers(self, model, freeze_until):
        for i, child in enumerate(model.children()):
            if i < freeze_until:
                for p in child.parameters():
                    p.requires_grad = False
        return model

    def forward(self, img, tab):
        feats = self.backbone(img)  # may be [B, C, H, W] or [B, C]
        if feats.dim() > 2:
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        x = torch.cat([feats, tab], dim=1)
        return self.regressor(x)


backbone = timm.create_model(model_name, pretrained=True)
model = Network(
    backbone,
    hidden_dims=[256, 64],
    out_features=1,
    freeze_until=0,
).to(device)
model.eval()




## === cell 7
train_csv = pd.read_csv(train_file)
train_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "train")
train_csv["path_img"] = train_csv["Id"].apply(
    lambda x: os.path.join(train_folder, f"{x}.jpg")
)


class CalibDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
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
        row = self.df.iloc[idx]
        img_path = row["path_img"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab = torch.tensor(row[self.cat].astype(float).values, dtype=torch.float32)
        label = torch.tensor(row["Pawpularity"], dtype=torch.float32)
        return img, tab, label


calib_dataset = CalibDataset(train_csv, transform=deterministic_transform)
calib_loader = DataLoader(
    calib_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
)

raw_preds = []
true_vals = []
with torch.no_grad():
    for imgs, tabs, labels in calib_loader:
        imgs = imgs.to(device, non_blocking=True)
        tabs = tabs.to(device, non_blocking=True)
        preds = model(imgs, tabs)  # logits
        preds = 100 * torch.sigmoid(preds)  # scale to [0,100]
        raw_preds.extend(preds.squeeze(1).cpu().numpy())
        true_vals.extend(labels.cpu().numpy())

calib_reg = Ridge(alpha=1.0)
calib_reg.fit(np.array(raw_preds).reshape(-1, 1), np.array(true_vals))

train_calibrated = calib_reg.predict(np.array(raw_preds).reshape(-1, 1))
numer = np.dot(train_calibrated, true_vals)
denom = np.dot(train_calibrated, train_calibrated) + 1e-12
calib_scale = numer / denom  # kept but not applied to final predictions




## === cell 8
tta_steps = 16  # keep moderate TTA passes
temperature = 0.98  # slight shrinking of logits before sigmoid
all_outputs = []

with torch.no_grad():
    for _ in range(tta_steps):
        step_outputs = []
        for imgs, tabs, _ in testloader:
            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            preds = model(imgs, tabs)  # logits
            preds = 100 * torch.sigmoid(
                preds * temperature
            )  # apply slight temperature scaling
            step_outputs.extend(preds.squeeze(1).cpu().numpy().tolist())
        all_outputs.append(step_outputs)

tta_mean = np.mean(np.array(all_outputs), axis=0)

tta_calibrated = calib_reg.predict(tta_mean.reshape(-1, 1))





## === cell 9
tta_clipped = np.clip(tta_calibrated, 0, 100)

test_csv["Pawpularity"] = tta_clipped
submission = test_csv[["Id", "Pawpularity"]]
submission.to_csv("submission.csv", index=False)
