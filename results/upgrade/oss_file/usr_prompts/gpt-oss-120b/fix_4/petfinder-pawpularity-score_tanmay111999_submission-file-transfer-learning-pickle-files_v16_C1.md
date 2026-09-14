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

No external packages required in the script and installed.

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

21.17819117493881

# 6. Current score

33.30027

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 32.35478) has done: 'I remove the TensorFlow dependency that caused the protobuf error and replace the VGG16 feature extractor with a simple raw‑pixel based extractor (flattened, scaled images). I also adjust the submission clipping to enforce Pawpularity values between 1 and 100, which fixes the invalid‑submission error. The rest of the pipeline (loading data, Ridge regression, and CSV output) is kept unchanged, ensuring the script runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved 33.30027) has done: 'I add a standard‑scaler to normalize the raw‑pixel features and introduce a quick validation split to pick a better Ridge regularisation strength. This small change often lowers RMSE without altering the core model or feature extraction, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_path = "../input/petfinder-pawpularity-score"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(base_path, "test.csv")
train_images_path = os.path.join(base_path, "train")
test_images_path = os.path.join(base_path, "test")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")




## === cell 1
def load_and_preprocess_image(img_path, size=(64, 64)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0  # normalize to [0,1]
    return img


def extract_flat_features(images):
    return images.reshape(images.shape[0], -1)




## === cell 2
train_images = []
train_ids = []
for _, row in tqdm(
    train_df.iterrows(), total=len(train_df), desc="Loading train images"
):
    img_id = row["Id"]
    img_path = os.path.join(train_images_path, f"{img_id}.jpg")
    train_images.append(load_and_preprocess_image(img_path))
    train_ids.append(img_id)

train_images = np.stack(train_images)  # (N, H, W, C)
train_features = extract_flat_features(train_images)  # (N, D)
train_targets = train_df["Pawpularity"].values.astype(np.float32)

print(f"Train features shape: {train_features.shape}")



## === cell 3
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)

X_tr, X_val, y_tr, y_val = train_test_split(
    train_features_scaled, train_targets, test_size=0.2, random_state=42
)

candidate_alphas = [0.1, 0.5, 1.0, 5.0, 10.0]
best_alpha = None
best_val_rmse = float("inf")

for a in candidate_alphas:
    model = Ridge(alpha=a, random_state=42)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    print(f"Alpha {a}: Validation RMSE = {rmse:.4f}")
    if rmse < best_val_rmse:
        best_val_rmse = rmse
        best_alpha = a

print(f"Selected Alpha = {best_alpha} with Validation RMSE = {best_val_rmse:.4f}")

regressor = Ridge(alpha=best_alpha, random_state=42)
regressor.fit(train_features_scaled, train_targets)

train_pred = regressor.predict(train_features_scaled)
train_rmse = mean_squared_error(train_targets, train_pred, squared=False)
print(f"Training RMSE (full data): {train_rmse:.4f}")



## === cell 4
test_images = []
test_ids = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Loading test images"):
    img_id = row["Id"]
    img_path = os.path.join(test_images_path, f"{img_id}.jpg")
    test_images.append(load_and_preprocess_image(img_path))
    test_ids.append(img_id)

test_images = np.stack(test_images)
test_features = extract_flat_features(test_images)

test_features_scaled = scaler.transform(test_features)

print(f"Test features shape: {test_features.shape}")



## === cell 5
test_predictions = regressor.predict(test_features_scaled)

submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_predictions})
submission["Pawpularity"] = submission["Pawpularity"].clip(
    1, 100
)  # ensure within 1‑100

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())
