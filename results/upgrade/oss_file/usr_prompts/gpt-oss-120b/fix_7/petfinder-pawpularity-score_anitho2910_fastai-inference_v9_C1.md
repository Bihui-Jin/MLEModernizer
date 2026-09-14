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

17.02184905599554

# 6. Current score

20.33496

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fixed the import errors (removed the unavailable cuml module), replaced the Albumentations pipeline with a torchvision‑based transform that works on PIL images, ensured the tabular data are returned as tensors, clipped the final predictions to the required [1, 100] range, and kept the overall model‑inference logic unchanged.'
- What this solution (achieved 65.01253) has done: 'Implemented a lightweight tabular‑only baseline to replace the failing image‑model pipeline.  
- Added sklearn imports and loaded the original train CSV.  
- Trained a GradientBoostingRegressor on the 12 binary metadata features, printing validation RMSE.  
- Generated predictions for the test set, stored them in `final_predictions`, then clipped to the required range and saved `submission.csv`.  
- Removed the problematic timm model loading loop, keeping the rest of the script unchanged.'
- What this solution (achieved 20.14059) has done: 'I fix the critical bug that collapses the prediction vector into a single constant by removing the erroneous sum operation, and I slightly improve the GradientBoostingRegressor hyper‑parameters (more estimators and a lower learning rate) to boost predictive performance while keeping the overall modelling approach unchanged. These minimal changes keep the core logic intact, ensure a proper per‑row prediction, and move the RMSE toward the target.'
- What this solution (achieved 20.16588) has done: 'I slightly improve the tabular‑only GradientBoostingRegressor by (1) adding a simple “total_flags” feature that captures the overall number of positive metadata annotations, and (2) tuning its hyper‑parameters (more trees, a lower learning rate and a slightly deeper tree). These minimal, yet relevant changes keep the original modelling pipeline intact while expectedly lowering the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.24534) has done: 'Improved the feature engineering and model pipeline: added an “any_flag” binary feature, applied standard scaling via a scikit‑learn Pipeline, and tweaked the GradientBoostingRegressor hyper‑parameters (more trees, slightly lower learning rate, deeper trees) to gain a modest RMSE reduction while keeping the original tabular‑only approach intact. The script now trains the scaled model, generates predictions, clips them to the required range, and writes a valid `submission.csv`.'
- What this solution (achieved 20.33496) has done: 'I added a few lightweight feature‑engineering steps (a flag‑ratio feature) and slightly strengthened the GradientBoostingRegressor (more trees, a lower learning rate, a bit higher subsample and using a Huber loss) to improve validation RMSE without changing the overall pipeline or model type. These adjustments keep the core logic intact while moving the RMSE closer to the target.'

# 9. Code solution

## === cell 0
import sys

sys.path.append("../input/saved-weights/pytorch-image-models/pytorch-image-models")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2  # kept for compatibility (not used)
import timm
import numpy as np
import pandas as pd
import os
import gc
import random
import pickle
from fastai.vision.all import *
from fastai.data.core import *
from torchmetrics import MeanSquaredError




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(torch.sigmoid(input).flatten(), target))


base_dir = "/kaggle/input"
model_weights_dir = os.path.join(base_dir, "saved-weights")
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")




## === cell 2
test_csv = pd.read_csv(test_file)
test_csv["path_img"] = test_csv["Id"].apply(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)
test_csv["Pawpularity"] = 1.0  # dummy column for dataset compatibility




## === cell 3
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
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
        img_path = self.df.iloc[idx]["path_img"]
        label = self.df.iloc[idx]["Pawpularity"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        tabular = self.df.iloc[idx][self.cat].values.astype(np.float32)
        tabular = torch.tensor(tabular, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.float32)
        return img, tabular, label




## === cell 4
def get_data(batch_size, max_size, input_shape):
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                max_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop((input_shape[0], input_shape[1])),
            transforms.RandomAffine(
                degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std_dev),
        ]
    )

    test_dataset = PetsDataset(test_csv, transform=test_transform)
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    dls = DataLoaders.from_dsets(
        test_dataset, test_dataset, bs=batch_size, shuffle=False
    )
    return dls, testloader




## === cell 5
class Identity(nn.Module):
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
            raise ValueError("Number of hidden layers must match length of hidden list")
        hidden_dim = hidden[:]
        hidden_dim.insert(0, embed_dim + 12)  # 12 tabular features

        base.head = nn.Linear(base.head.in_features, embed_dim)
        nn.init.kaiming_normal_(base.head.weight)
        nn.init.constant_(base.head.bias, 0)

        self.p = 0.5
        self.regression = self._fully_connected(
            number_of_hidden, hidden_dim, regression_out
        )
        self.network = self._freeze_layer(base, freeze_layer)
        self._init_weights()

    def _init_weights(self):
        for m in self.regression:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_layer(self, base, freeze_layer):
        for i, child in enumerate(base.children()):
            if i < freeze_layer:
                for param in child.parameters():
                    param.requires_grad = False
        return base

    def _fully_connected(self, num_hidden, hidden_dim, out_dim):
        layers = [nn.Dropout(self.p)]
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
            if i != num_hidden - 1:
                layers.append(nn.Dropout(self.p))
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        return nn.Sequential(*layers)

    def forward(self, x, tab):
        x1 = self.network(x)
        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg, x




## === cell 6
def get_learner(model_name, batch_size, loss, metric, max_size, input_size, save_path):
    dls, testloader = get_data(batch_size, max_size, input_size)
    dls = dls.to(device)
    base = timm.create_model(model_name, pretrained=False)
    model = Network(base, num_of_hidden, hidden_dimension, 1, 11, 0)
    model = model.to(device)
    learn = Learner(
        dls, model, loss_func=loss, metrics=metric, model_dir=save_path
    ).to_fp16()
    return learn, testloader




## === cell 7
def tta(testloader, learn, clf, svr_weight, tta_steps=4):
    learn.model.eval()
    all_outputs = []
    for _ in range(tta_steps):
        step_outputs = []
        svr_embeds = []
        with torch.no_grad():
            for images, tabular, _ in testloader:
                images, tabular = images.to(device), tabular.to(device)
                reg_out, embed = learn.model(images, tabular)
                reg_out = 100 * torch.sigmoid(reg_out)
                step_outputs.extend(reg_out.cpu().numpy().reshape(-1))
                svr_embeds.append(embed.cpu().numpy())
        step_outputs = np.array(step_outputs)
        svr_embeds = np.concatenate(svr_embeds, axis=0)
        svr_preds = clf.predict(svr_embeds)
        combined = svr_weight[0] * svr_preds + svr_weight[1] * step_outputs
        all_outputs.append(combined)
    return np.mean(np.array(all_outputs), axis=0)




## === cell 8
train_path = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")
train_df = pd.read_csv(train_path)

feature_cols = [
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

X = train_df[feature_cols].astype(np.float32)

X["total_flags"] = X.sum(axis=1)  # total number of positive flags
X["any_flag"] = (X["total_flags"] > 0).astype(np.int8)  # binary indicator
X["flag_ratio"] = X["total_flags"] / len(feature_cols)  # proportion of flags

y = train_df["Pawpularity"].astype(np.float32)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "gbr",
            GradientBoostingRegressor(
                random_state=42,
                n_estimators=1200,  # more trees for better fit
                learning_rate=0.02,  # lower LR to stabilise learning
                max_depth=5,  # keep depth similar
                subsample=0.9,  # slightly more data per tree
                loss="huber",  # robust loss improves RMSE
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)

val_pred = pipeline.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

test_features = test_csv[feature_cols].astype(np.float32)
test_features["total_flags"] = test_features.sum(axis=1)
test_features["any_flag"] = (test_features["total_flags"] > 0).astype(np.int8)
test_features["flag_ratio"] = test_features["total_flags"] / len(feature_cols)

final_predictions = pipeline.predict(test_features)




## === cell 9
final_predictions = np.clip(final_predictions, 1, 100)

test_csv["Pawpularity"] = final_predictions
submission = test_csv[["Id", "Pawpularity"]]
submission.to_csv("submission.csv", index=False)
