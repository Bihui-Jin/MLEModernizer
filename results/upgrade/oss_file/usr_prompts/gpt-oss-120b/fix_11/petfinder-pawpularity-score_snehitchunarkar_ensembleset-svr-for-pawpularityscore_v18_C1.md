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

17.653730312654496

# 6. Current score

20.05977

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.19058) has done: 'I remove the GPU‑only `cuml` import and replace the failing image‑feature pipeline with a simple, fully CPU‑based model that uses only the tabular metadata. By scaling the 12 metadata columns and training a `RandomForestRegressor` (which is stable and fast on CPU), we avoid the CUDA errors and the NameError cascade, and we obtain an RMSE close to the target ≈ 17.6. The script now loads the data, builds the features, trains the model, evaluates it on a validation split, predicts the test set, and writes a valid `submission.csv` file.'
- What this solution (achieved 20.08441) has done: 'I replace the RandomForest with a GradientBoostingRegressor (which usually gives lower RMSE on this tabular data) and adjust its hyper‑parameters slightly. The import is added, the model definition in the validation split is updated, and the same model is reused for the final full‑data fit. No other logic is changed, so the script still loads data, scales features, evaluates on a validation split, trains on all data, predicts the test set and writes a correct `submission.csv`.'
- What this solution (achieved 20.1083) has done: 'I import `PolynomialFeatures` and expand the 12 metadata columns into degree‑2 interaction features before scaling and training. This adds modest expressive power to the existing GradientBoostingRegressor without changing its core architecture, and is expected to lower the validation RMSE toward the target ≈ 17.65.'
- What this solution (achieved 20.10629) has done: 'I remove the degree‑2 polynomial expansion (which was over‑fitting the metadata) and keep the raw scaled features, and I enable early‑stopping in the GradientBoostingRegressor to prevent over‑fitting. These minimal changes keep the overall pipeline intact while expectedly lowering the validation RMSE toward the target.'
- What this solution (achieved 20.07704) has done: 'I add a modest interaction‑only polynomial expansion of the 12 metadata columns to give the model a bit more expressive power, and I slightly adjust the GradientBoostingRegressor hyper‑parameters (more trees, a smaller learning rate and a deeper depth) which together tend to improve RMSE without changing the overall pipeline. These changes keep the same model type and training/evaluation flow, only enhancing the feature set and fine‑tuning the regressor to move the validation score closer to the target.'
- What this solution (achieved 20.08751) has done: 'I revert the feature generation to use only the original 12 metadata columns (removing the interaction‑only polynomial expansion) and adjust the GradientBoostingRegressor hyper‑parameters modestly (more trees, slightly lower learning rate and a shallower depth). These changes keep the overall pipeline intact while reducing over‑fitting, which should lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.08655) has done: 'I adjust the GradientBoostingRegressor hyper‑parameters slightly to give the model a bit more capacity while keeping early‑stopping enabled. By increasing the tree depth to 4, lowering the learning rate to 0.01 and raising n_estimators to 3000 (with a slightly smaller subsample), the model can fit the data better and is expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.41969) has done: 'I add interaction‑only polynomial features to give the GradientBoosting model more expressive power, and I adjust the regressor’s hyper‑parameters (use a slightly deeper tree, more estimators, a lower learning rate, and the robust “huber” loss) so that the validation RMSE moves closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 20.05977) has done: 'I correct the GradientBoostingRegressor loss parameter from the deprecated `'ls'` to the valid `'squared_error'`. This fixes the InvalidParameterError, allowing the model to train, evaluate, and generate predictions, resulting in a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor




## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"




## === cell 2
directory = "/kaggle/input/petfinder-pawpularity-score"
train_df = pd.read_csv(os.path.join(directory, "train.csv"))
test_df = pd.read_csv(os.path.join(directory, "test.csv"))

print("Train samples:", len(train_df), "\nTest samples:", len(test_df), "\n")




## === cell 3
"""
def ExtractModelFeature(Dataloader, model, Train_PCA=False):
    ...
"""




## === cell 4
"""
class dataset_EfficientNet: ...
...
"""




## === cell 5
"""
train_dataset = dataset_EfficientNet(...)
...
"""




## === cell 6
"""
X1 = ExtractModelFeature(...)
...
"""




## === cell 7
"""
from transformers import CLIPModel, CLIPProcessor
...
"""




## === cell 8
"""
class dataset_Clip: ...
"""




## === cell 9
"""
train_dataset = dataset_Clip(...)
...
"""




## === cell 10
"""
X2 = ExtractModelFeature(...)
...
"""




## === cell 11
"""
train_dataset = dataset_Clip(...)
...
"""




## === cell 12
"""
train_dataset = dataset_Clip(...)
...
"""




## === cell 13
x_meta = train_df.iloc[:, 1:13].values.astype(np.float32)  # shape (n_samples, 12)
x_test_meta = test_df.iloc[:, 1:13].values.astype(np.float32)

X = x_meta
X_test = x_test_meta

print(
    "Feature matrix shapes -> train:",
    X.shape,
    ", test:",
    X_test.shape,
)




## === cell 14
"""
class ImageExtract: ...
"""

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X = poly.fit_transform(x_meta)  # shape (n_samples, 12 + 66 = 78)
X_test = poly.transform(x_test_meta)  # same number of columns

print(
    "After polynomial interaction -> train:",
    X.shape,
    ", test:",
    X_test.shape,
)




## === cell 15
"""
train_dataset = ImageExtract(...)
...
"""




## === cell 16
"""
train_dataset = ImageExtract(...)
...
"""




## === cell 17
"""
X = np.hstack((X1,X2, X3, X4, x_meta))
X_test = np.hstack((X1_test, X2_test, X3_test, X4_test, x_test_meta))
"""




## === cell 18
X = X  # placeholder to keep cell numbering consistent
X_test = X_test




## === cell 19
y = train_df["Pawpularity"].values
print("Target vector shape:", y.shape)




## === cell 20
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(np.vstack((X_train, X_val)))

X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

model = GradientBoostingRegressor(
    loss="squared_error",  # corrected loss name
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=1e-4,
)

model.fit(X_train_sc, y_train)

val_pred = model.predict(X_val_sc)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 21
model.fit(scaler.transform(X), y)




## === cell 22
y_pred = model.predict(X_test_sc)
print("First 10 predictions:", y_pred[:10])




## === cell 23
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": y_pred})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## === cell 24
submission.head()
