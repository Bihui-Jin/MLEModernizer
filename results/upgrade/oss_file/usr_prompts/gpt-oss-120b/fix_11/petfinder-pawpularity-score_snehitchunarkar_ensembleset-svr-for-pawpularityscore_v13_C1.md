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

17.660652904269853

# 6. Current score

20.05824

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.79322) has done: 'I replace the GPU‑dependent cuML imports with scikit‑learn equivalents, remove the unused feature‑extraction code, and build a simple pipeline that uses the provided metadata features. This fixes the import and variable‑name errors, creates a valid train/validation split to compute RMSE, trains an SVR model, and writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 20.15385) has done: 'I keep the existing data handling and scaling but add a GradientBoostingRegressor alongside the SVR, compare their validation RMSEs, and use the model (or their average) that gives the lowest validation error for the final test predictions. This modest ensemble should reduce the RMSE toward the target without altering the overall pipeline.'
- What this solution (achieved 20.30567) has done: 'I keep the same data handling and model types but give the GradientBoostingRegressor more capacity (more trees, lower learning rate) and combine its predictions with the SVR using a validation‑based weighted average instead of the binary “use average only if it beats both”. This small change is expected to lower the validation RMSE, moving the score toward the target while preserving the overall pipeline.'
- What this solution (achieved 20.27412) has done: 'I add a lightweight linear‑blending step that learns optimal combination weights from the validation set instead of using fixed inverse‑RMSE weights. By fitting a simple LinearRegression on the two model predictions (SVR and GBR) against the validation targets, we obtain a data‑driven ensemble that typically lowers the RMSE, moving the score toward the target while preserving the existing preprocessing and models.'
- What this solution (achieved 20.08066) has done: 'I keep the overall pipeline (metadata extraction, scaling, SVR + GBR models and a linear ensemble) but adjust a few hyper‑parameters that are inexpensive to change and can lower the validation RMSE. Increasing the GradientBoosting capacity, slightly strengthening the SVR, allowing an intercept in the linear blend, and clipping the final predictions to the valid 0‑100 range are all minimal tweaks that stay within the original logic while moving the score toward the target.'
- What this solution (achieved 20.06552) has done: 'I slightly adjust the SVR and GradientBoosting hyper‑parameters to give the models a bit more capacity (higher C for SVR, more trees, a slightly lower learning‑rate and deeper trees for GBR). These minimal tweaks stay within the original pipeline and are expected to lower the validation RMSE, moving the score closer to the target of 17.66 without changing any core logic.'
- What this solution (achieved 20.06628) has done: 'Implemented modest hyperparameter refinements to both SVR and GradientBoostingRegressor to give the models a bit more capacity while preserving the overall pipeline and linear ensemble.  
- SVR: increased C to 800 and reduced epsilon to 0.02 for finer fitting.  
- GBR: raised n_estimators to 2000, lowered learning_rate to 0.015, and deepened trees to max_depth=6.  
These adjustments are expected to lower the validation RMSE, moving the score closer to the target without altering core logic.'
- What this solution (achieved 20.06708) has done: 'I keep the original pipeline but make a few lightweight hyper‑parameter tweaks that are known to improve regression performance: increase GradientBoosting capacity slightly, tighten the SVR fit, and replace the plain linear blend with a lightly regularised Ridge regression. These changes stay within the existing logic and are expected to lower the validation RMSE, moving the score toward the target.'
- What this solution (achieved 20.06709) has done: 'I replace the ridge‑based blending with a plain LinearRegression ensemble, which usually fits the validation data more closely while keeping the original preprocessing and models unchanged. This minor change is expected to lower the validation RMSE and thus move the score nearer to the target.'
- What this solution (achieved 20.05824) has done: 'I add polynomial interaction features to capture relationships between metadata columns, modestly increase model capacity (higher SVR C and lower epsilon, more GBR trees with a smaller learning rate), and switch the blending step to a lightly‑regularized Ridge regression. These changes keep the overall pipeline intact while providing additional expressive power and a more stable ensemble, which should lower the validation RMSE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"



## === cell 2
directory = "/kaggle/input/petfinder-pawpularity-score"
train_df = pd.read_csv(os.path.join(directory, "train.csv"))
test_df = pd.read_csv(os.path.join(directory, "test.csv"))

print("Train samples:", len(train_df), "\nTest samples:", len(test_df), "\n")



## === cell 3
x_meta = train_df.iloc[:, 1:13].values
x_test_meta = test_df.iloc[:, 1:13].values

poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(x_meta)
X_test = poly.transform(x_test_meta)

print("Feature matrix shapes -> train:", X.shape, "test:", X_test.shape)



## === cell 4
y = train_df["Pawpularity"].values
print("Target vector shape:", y.shape)



## === cell 5
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)



## === cell 6
reg_svr = SVR(C=2000.0, kernel="rbf", epsilon=0.005, gamma="scale")
reg_svr.fit(X_train, y_train)

reg_gbr = GradientBoostingRegressor(
    n_estimators=4000,
    learning_rate=0.005,
    max_depth=6,
    subsample=0.8,
    max_features=0.8,
    random_state=42,
)
reg_gbr.fit(X_train, y_train)



## === cell 7
val_pred_svr = reg_svr.predict(X_val)
val_pred_gbr = reg_gbr.predict(X_val)

rmse_svr = np.sqrt(mean_squared_error(y_val, val_pred_svr))
rmse_gbr = np.sqrt(mean_squared_error(y_val, val_pred_gbr))

print(f"Validation RMSE SVR : {rmse_svr:.4f}")
print(f"Validation RMSE GBR : {rmse_gbr:.4f}")

val_ens_X = np.column_stack([val_pred_svr, val_pred_gbr])
ridge_ens = Ridge(alpha=1.0, fit_intercept=True)
ridge_ens.fit(val_ens_X, y_val)
weights = ridge_ens.coef_
intercept = ridge_ens.intercept_
print(
    f"Ridge ensemble weights -> SVR: {weights[0]:.3f}, GBR: {weights[1]:.3f}, Intercept: {intercept:.3f}"
)



## === cell 8
X_test_scaled = scaler.transform(X_test)

test_pred_svr = reg_svr.predict(X_test_scaled)
test_pred_gbr = reg_gbr.predict(X_test_scaled)

y_pred = test_pred_svr * weights[0] + test_pred_gbr * weights[1] + intercept
y_pred = np.clip(y_pred, 0, 100)



## === cell 9
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
