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

3.12

# 3. Installed packages

cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
tqdm==4.67.1
transformers==4.53.3

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

18.18046707404222

# 6. Current score

20.16675

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.77238) has done: 'I fixed the import errors (removed the CUDA‑only cuML import and the failing CLIP import), switched to the CPU‑based `sklearn.svm.SVR`, and simplified the pipeline to use only the provided metadata features (which are sufficient to train a model and create a valid submission). The script now runs end‑to‑end, computes a reasonable RMSE, and writes `submission.csv` with the correct columns.'
- What this solution (achieved 20.64754) has done: 'I add a polynomial feature expansion for the metadata (degree 2) to give the SVR more expressive inputs and slightly adjust the SVR hyper‑parameters (lower C and a small epsilon). These changes keep the overall pipeline and model type intact while aiming to reduce the RMSE toward the target value.'
- What this solution (achieved 20.68508) has done: 'We slightly strengthen the SVR by raising the regularization C and lowering epsilon, which gives the model more flexibility and typically reduces RMSE while keeping the same overall pipeline. Only the SVR initialization in cell 19 is changed.'
- What this solution (achieved 20.73424) has done: 'I slightly adjust the SVR hyper‑parameters to give the model more flexibility (increase C and reduce epsilon). This keeps the overall pipeline unchanged while aiming to lower the RMSE toward the target value.'
- What this solution (achieved 20.75044) has done: 'I tighten the preprocessing and SVR settings while keeping the overall pipeline unchanged. First, the scaler be fit only on the training features (removing the subtle leakage from including test data). Then I use a more flexible SVR by increasing the regularization C and lowering epsilon, and set gamma to “auto” so it adapts to the feature dimensionality. These small adjustments should lower the RMSE, moving the score closer to the target without altering the core model logic.'
- What this solution (achieved 20.73251) has done: 'I increase the polynomial feature degree to 3 to capture higher‑order interactions among the metadata, and adjust the SVR hyper‑parameters to a slightly less aggressive regularisation (C=100) and a larger epsilon (0.01) while using the default “scale” gamma. These minimal changes keep the same model type and preprocessing pipeline but give the regressor a better bias‑variance balance, which should lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.73251) has done: 'I add a quick train/validation split to evaluate a slightly stronger SVR configuration (higher C, lower epsilon, gamma = auto). The script pick the better‑performing hyper‑parameters and then retrain on the full training set before creating the submission, keeping the same model type and preprocessing pipeline while moving the RMSE toward the target.'
- What this solution (achieved 20.6428) has done: 'I reduce the polynomial expansion degree from 3 to 2 to lessen over‑fitting on the metadata features, and I add a more regularised SVR configuration (lower C and higher epsilon). These tiny adjustments keep the same SVR‑based pipeline while providing a smoother model that should lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.74753) has done: 'I add target‑value standardisation so the SVR works on a zero‑mean, unit‑variance y, and I broaden the hyper‑parameter sweep a little while keeping the same model type, polynomial features and scaling. This small change often improves SVR performance and should lower the validation RMSE, moving the final score closer to the target without altering the core pipeline.'
- What this solution (achieved 20.73614) has done: 'I raise the polynomial degree to 3 to give the SVR richer interaction features and add a stronger hyper‑parameter setting (C=1000, epsilon=0.0001, gamma='auto') to the grid. This keeps the overall SVR‑based pipeline unchanged while providing a better bias‑variance trade‑off, which should lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.74753) has done: 'I lower the polynomial expansion degree from 3 to 2 to reduce over‑fitting on the limited metadata, and I extend the SVR hyper‑parameter grid with a more strongly regularised setting (C = 5000, epsilon = 0.0001, gamma = "scale"). The validation loop automatically pick the configuration that yields the lowest RMSE, moving the score closer to the target while preserving the overall pipeline and model type.'
- What this solution (achieved 20.74753) has done: 'I extend the hyper‑parameter grid with a few additional SVR settings (including a larger C, smaller epsilon, and explicit numeric gamma values) so the validation loop can select a configuration that is likely to reduce over‑fitting and bring the RMSE closer to the target. This change preserves the overall pipeline (polynomial features, scaling, SVR) while giving the model a better chance to improve the score.'
- What this solution (achieved 20.74628) has done: 'I add a lightweight PCA step after scaling the polynomial features to reduce dimensionality and improve generalisation, keeping the same SVR model and overall pipeline. The PCA retains 95 % of variance, is fitted only on training data (no leakage), and is applied consistently to validation, full‑training and test data. This small change is expected to lower the validation RMSE, moving the score closer to the target while preserving the core logic.'
- What this solution (achieved 20.16675) has done: 'I remove the unnecessary PCA step (which can discard useful variance) and instead train and predict directly on the scaled polynomial features. After training on the full data, I fit a simple linear correction (slope + intercept) on the training predictions to better align the model output with the true target, then apply this correction to the test predictions. These minimal adjustments keep the SVR‑based pipeline intact while aiming to lower the validation RMSE toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import (
    PCA,
)  # retained import (unused after removal of PCA step)




## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"




## === cell 2
directory = "/kaggle/input/petfinder-pawpularity-score"
train_df = pd.read_csv(os.path.join(directory, "train.csv"))
test_df = pd.read_csv(os.path.join(directory, "test.csv"))

print("Train samples: ", len(train_df), "\nTest samples: ", len(test_df), "\n")




## === cell 3
x_meta = train_df.iloc[:, 1:13].values
x_test_meta = test_df.iloc[:, 1:13].values




## === cell 4
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(x_meta)
X_test = poly.transform(x_test_meta)

print("train_features shape after poly:", X.shape)
print("test_features shape after poly:", X_test.shape)




## === cell 5
y = train_df["Pawpularity"].values
print("Target shape:", y.shape)




## === cell 6
X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

y_mean = y_train_raw.mean()
y_std = y_train_raw.std()
y_train = (y_train_raw - y_mean) / y_std
y_val = (y_val_raw - y_mean) / y_std

configs = [
    {"C": 100.0, "epsilon": 0.01, "gamma": "scale"},
    {"C": 500.0, "epsilon": 0.001, "gamma": "auto"},
    {"C": 10.0, "epsilon": 0.05, "gamma": "scale"},
    {"C": 200.0, "epsilon": 0.005, "gamma": "auto"},
    {"C": 300.0, "epsilon": 0.01, "gamma": "scale"},
    {"C": 1000.0, "epsilon": 0.0001, "gamma": "auto"},
    {"C": 5000.0, "epsilon": 0.0001, "gamma": "scale"},
    {"C": 2000.0, "epsilon": 0.0005, "gamma": 0.01},
    {"C": 8000.0, "epsilon": 0.0001, "gamma": 0.001},
    {"C": 1500.0, "epsilon": 0.0002, "gamma": "auto"},
]

best_cfg = None
best_rmse = np.inf
for cfg in configs:
    reg = SVR(
        C=cfg["C"],
        epsilon=cfg["epsilon"],
        kernel="rbf",
        gamma=cfg["gamma"],
        max_iter=500000,
    )
    reg.fit(X_train_scaled, y_train)
    val_pred_scaled = reg.predict(X_val_scaled)
    val_pred = val_pred_scaled * y_std + y_mean
    rmse = np.sqrt(mean_squared_error(y_val_raw, val_pred))
    print(f"Config {cfg} => validation RMSE: {rmse:.5f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_cfg = cfg

print("Chosen config:", best_cfg, "with val RMSE:", best_rmse)




## === cell 7
scaler_full = StandardScaler()
scaler_full.fit(X)
X_scaled_full = scaler_full.transform(X)

y_full_mean = y.mean()
y_full_std = y.std()
y_scaled_full = (y - y_full_mean) / y_full_std

reg_final = SVR(
    C=best_cfg["C"],
    epsilon=best_cfg["epsilon"],
    kernel="rbf",
    gamma=best_cfg["gamma"],
    max_iter=500000,
)
reg_final.fit(X_scaled_full, y_scaled_full)

train_pred_scaled = reg_final.predict(X_scaled_full)
train_pred = train_pred_scaled * y_full_std + y_full_mean
train_rmse = np.sqrt(mean_squared_error(y, train_pred))
print("Training RMSE on full data:", train_rmse)

slope, intercept = np.polyfit(train_pred, y, 1)
print(f"Linear correction parameters: slope={slope:.6f}, intercept={intercept:.6f}")




## === cell 8
X_test_scaled = scaler_full.transform(X_test)
test_pred_scaled = reg_final.predict(X_test_scaled)
y_pred_raw = test_pred_scaled * y_full_std + y_full_mean

y_pred = y_pred_raw * slope + intercept




## === cell 9
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 10
print(submission.head())
