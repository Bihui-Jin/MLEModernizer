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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

20.49703

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2, numpy as np, pandas as pd

BASE_DIR = os.path.abspath("../input/petfinder-pawpularity-score")
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.abspath("./data/petfinder-pawpularity-score")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")



## === cell 1
train_csv = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_csv = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))




## === cell 2
def load_images(img_dir, ids):
    """Load and resize images given a directory and list of ids."""
    imgs = []
    for img_id in ids:
        path = os.path.join(img_dir, f"{img_id}.jpg")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32) / 255.0)
    return np.stack(imgs)


train_ids_arr = train_csv["Id"].values
test_ids_arr = test_csv["Id"].values

train_imgs = load_images(TRAIN_DIR, train_ids_arr)
test_imgs = load_images(TEST_DIR, test_ids_arr)

train_img_feat = train_imgs.mean(axis=(1, 2)).astype(np.float32)
test_img_feat = test_imgs.mean(axis=(1, 2)).astype(np.float32)



## === cell 3
csv_feature_cols = [c for c in train_csv.columns if c not in ["Id", "Pawpularity"]]
train_csv_x = train_csv[csv_feature_cols].astype(np.float32).values
train_y = train_csv["Pawpularity"].astype(np.float32).values
test_csv_x = test_csv[csv_feature_cols].astype(np.float32).values
test_ids = test_csv["Id"].values

feat_mean = train_csv_x.mean(axis=0, keepdims=True)
feat_std = train_csv_x.std(axis=0, keepdims=True)
feat_std[feat_std == 0] = 1.0  # avoid division by zero
train_csv_x = (train_csv_x - feat_mean) / feat_std
test_csv_x = (test_csv_x - feat_mean) / feat_std

train_csv_x = np.hstack([train_csv_x, train_csv_x**2])
test_csv_x = np.hstack([test_csv_x, test_csv_x**2])

train_csv_x = np.hstack([train_csv_x, train_img_feat])
test_csv_x = np.hstack([test_csv_x, test_img_feat])




## === cell 4
def train_linear_regression(X, y, ridge_alpha=1e-2):
    """
    Solve (X^T X + alpha*I) w = X^T y on the bias‑augmented matrix.
    This keeps the linear‑regression spirit while adding tiny L2 regularisation.
    """
    X_bias = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
    A = X_bias.T @ X_bias + ridge_alpha * np.eye(X_bias.shape[1], dtype=np.float32)
    b = X_bias.T @ y
    w = np.linalg.solve(A, b)
    return w


def predict_linear_regression(X, w):
    X_bias = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
    return X_bias @ w




## === cell 5
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    train_csv_x, train_y, test_size=0.2, random_state=42
)

candidate_alphas = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 10.0]
best_alpha = candidate_alphas[0]
best_rmse = float("inf")
for a in candidate_alphas:
    w = train_linear_regression(X_tr, y_tr, ridge_alpha=a)
    val_pred = predict_linear_regression(X_val, w)
    rmse = np.sqrt(np.mean((val_pred.ravel() - y_val) ** 2))
    print(f"Alpha {a}: Validation RMSE = {rmse:.5f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = a

print(f"Chosen alpha: {best_alpha} with RMSE {best_rmse:.5f}")



## === cell 6
weights_full = train_linear_regression(train_csv_x, train_y, ridge_alpha=best_alpha)
test_pred = predict_linear_regression(test_csv_x, weights_full)

test_pred = np.clip(test_pred.ravel(), 0, 100)



## === cell 7
final_df = pd.DataFrame({"Id": test_ids, "Pawpularity": test_pred})



## === cell 8
output_path = "./working/submission.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
