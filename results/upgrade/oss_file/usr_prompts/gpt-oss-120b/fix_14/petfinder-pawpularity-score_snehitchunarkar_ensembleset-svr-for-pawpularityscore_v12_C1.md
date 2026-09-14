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

17.726294775628872

# 6. Current score

20.11468

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.19058) has done: 'I fixed the runtime failures by removing the GPU‑only `cuml` import and the unused CLIP/EfficientNet image pipelines, and replaced them with a lightweight metadata‑only regression model using scikit‑learn. The script now loads the CSV files, builds features from the 12 metadata columns, evaluates a `RandomForestRegressor` on a validation split (reporting RMSE), refits the model on the full training set, predicts the test set, and writes a correctly‑formatted `submission.csv`. These changes keep the core logic (tabular regression) while ensuring the notebook runs end‑to‑end and produces a valid submission file.'
- What this solution (achieved 20.1957) has done: 'I add a simple engineered feature that sums all binary attribute columns, include it in the model, and increase the RandomForest size slightly (800 trees) to reduce variance. These minimal tweaks keep the same tabular‑regression approach while expected to lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.18913) has done: 'I add pairwise interaction features (binary AND of each attribute pair) to give the model more information, and tune the RandomForest hyper‑parameters (more trees and `max_features='sqrt'`) which usually lower over‑fitting and improve RMSE on validation. These changes keep the same tabular‑regression approach while aiming to bring the score closer to the target.'
- What this solution (achieved 20.17346) has done: 'I add a lightweight “attr_mean” feature (the average of the binary attributes) to give the model a normalized signal, and I slightly tighten the RandomForest (limit max depth and use a fractional max_features) while increasing the number of trees. These modest changes keep the same tabular‑regression pipeline but should improve generalisation enough to lower the RMSE toward the target.'
- What this solution (achieved 20.1889) has done: 'I tightened the RandomForest hyper‑parameters to give the model more capacity (removed the artificial depth limit) and used the classic “sqrt” feature‑subsampling, while slightly increasing the number of trees. These modest tweaks stay within the original tabular‑regression pipeline and are expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.09216) has done: 'I tighten the RandomForest to reduce over‑fitting, which should lower the validation RMSE and move the score closer to the target. In the model definition (cell 4) I set a modest `max_depth` and add `min_samples_leaf=2` while keeping the other settings unchanged. This small regularisation keeps the original tabular‑regression pipeline intact but is expected to improve generalisation and thus reduce the RMSE.'
- What this solution (achieved 20.19148) has done: 'I add a simple quadratic “attr_squared” feature (the square of the attribute sum) to give the model a non‑linear signal and relax the tree depth to let the forest capture richer interactions. I also increase the number of trees slightly and revert to the default leaf settings, which together should reduce validation RMSE and move the score below the target range while keeping the core RandomForest‑based tabular pipeline unchanged.'
- What this solution (achieved 20.07634) has done: 'I tighten the RandomForest (limit max depth and require a few samples per leaf) to reduce over‑fitting and add a simple linear model (Ridge) that captures any linear trend in the engineered features. Their average prediction usually gives a modest RMSE reduction, moving the score closer to the target while keeping the core tabular‑regression pipeline unchanged.'
- What this solution (achieved 20.09349) has done: 'I keep the existing tabular‑regression pipeline but replace the simple un‑weighted average of the RandomForest and Ridge predictions with a data‑driven weighted blend. By computing the optimal weight on the validation split (clipped to [0, 1]), the ensemble can better exploit the stronger model, which is expected to lower the validation RMSE and move the score closer to the target. The same weight is then applied to the test‑set predictions, and the rest of the script remains unchanged.'
- What this solution (achieved 20.10039) has done: 'I added a tiny but potentially useful feature – the standard deviation of the binary attributes (`attr_std`) – to give the models a sense of how mixed the attribute set is. In the RandomForest I removed the artificial depth limit and let the trees grow fully while keeping the other regularisation settings (min_samples_leaf = 2, max_features = sqrt). I also tightened the Ridge regularisation slightly (alpha = 0.5) to let it capture more signal from the richer feature set. All other logic, including the weighted blend, stays unchanged, and the script now writes a proper `submission.csv`.'
- What this solution (achieved 20.10514) has done: 'The changes add two simple binary features (`attr_any` and `attr_all`) that capture whether any attribute is present and whether all attributes are present, giving the models a clearer signal. The RandomForest is also given a modest maximum depth (20) to reduce over‑fitting while keeping the rest of the pipeline unchanged. These tweaks are minimal but expected to lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.11601) has done: 'I add a StandardScaler for the Ridge model (so its linear coefficients are properly regularized) and slightly loosen the RandomForest (remove the artificial depth limit and increase trees). These modest tweaks keep the same tabular‑regression pipeline but should improve validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.11468) has done: 'I tighten the RandomForest to reduce over‑fitting (limit depth and require a few samples per leaf) and increase ridge regularisation, then clip predictions to the valid [0, 100] range for both validation and test. These minimal tweaks keep the same tabular‑regression pipeline while targeting a lower RMSE, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler




## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"




## === cell 2
base_dir = "/kaggle/input/petfinder-pawpularity-score"

train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")




## === cell 3
base_attrs = [
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

train_df["attr_sum"] = train_df[base_attrs].sum(axis=1)
test_df["attr_sum"] = test_df[base_attrs].sum(axis=1)

train_df["attr_mean"] = train_df[base_attrs].mean(axis=1)
test_df["attr_mean"] = test_df[base_attrs].mean(axis=1)

train_df["attr_squared"] = train_df["attr_sum"] ** 2
test_df["attr_squared"] = test_df["attr_sum"] ** 2

train_df["attr_std"] = train_df[base_attrs].std(axis=1)
test_df["attr_std"] = test_df[base_attrs].std(axis=1)

train_df["attr_any"] = (train_df["attr_sum"] > 0).astype(int)
test_df["attr_any"] = (test_df["attr_sum"] > 0).astype(int)

train_df["attr_all"] = (train_df["attr_sum"] == len(base_attrs)).astype(int)
test_df["attr_all"] = (test_df["attr_sum"] == len(base_attrs)).astype(int)

from itertools import combinations

pair_features = []
for a, b in combinations(base_attrs, 2):
    col_name = f"{a}_{b}_and"
    train_df[col_name] = (train_df[a] & train_df[b]).astype(int)
    test_df[col_name] = (test_df[a] & test_df[b]).astype(int)
    pair_features.append(col_name)

feature_cols = (
    base_attrs
    + ["attr_sum", "attr_mean", "attr_squared", "attr_std", "attr_any", "attr_all"]
    + pair_features
)

X = train_df[feature_cols].values
y = train_df["Pawpularity"].values




## === cell 4
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=3000,
    max_depth=20,  # limit depth
    min_samples_leaf=2,  # require at least 2 samples per leaf
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

ridge = Ridge(alpha=1.0, random_state=42)  # increased regularisation
ridge.fit(X_train_scaled, y_train)

rf_val = rf.predict(X_val)
ridge_val = ridge.predict(X_val_scaled)

diff = rf_val - ridge_val
numer = np.dot(diff, y_val - ridge_val)
denom = np.dot(diff, diff)
blend_weight = numer / denom if denom != 0 else 0.5
blend_weight = np.clip(blend_weight, 0.0, 1.0)

val_pred = blend_weight * rf_val + (1.0 - blend_weight) * ridge_val
val_pred = np.clip(val_pred, 0, 100)

rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE (weighted blend, w={blend_weight:.4f}): {rmse:.4f}")




## === cell 5
rf.fit(X, y)

X_scaled_full = scaler.fit_transform(X)  # refit scaler on all data
ridge.fit(X_scaled_full, y)




## === cell 6
X_test = test_df[feature_cols].values
rf_test_pred = rf.predict(X_test)

X_test_scaled = scaler.transform(X_test)
ridge_test_pred = ridge.predict(X_test_scaled)

test_pred = blend_weight * rf_test_pred + (1.0 - blend_weight) * ridge_test_pred
test_pred = np.clip(test_pred, 0, 100)

submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
