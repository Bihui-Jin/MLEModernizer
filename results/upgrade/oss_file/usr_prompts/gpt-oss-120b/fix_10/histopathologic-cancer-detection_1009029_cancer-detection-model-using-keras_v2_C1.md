# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import concurrent.futures
import multiprocessing

print("Input root contents:", os.listdir("../input"))




## === cell 1
train_df = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv")
train_df.head()




## === cell 2
print(train_df.label.value_counts())
print("Positive ratio:", train_df.label.mean())




## === cell 3
base_tile_dir = "../input/histopathologic-cancer-detection/train/"
df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})
df["id"] = df.path.map(lambda x: os.path.basename(x).split(".")[0])
df = df.merge(train_df, on="id")
print("Total images found:", len(df))

df = df.sample(n=50000, random_state=42).reset_index(drop=True)
print("Using subset size:", len(df))




## === cell 4
def load_and_flatten(path):
    img = imread(path).astype(np.float32) / 255.0
    return img.ravel()


sample_img = imread(df.path.iloc[0]).astype(np.float32) / 255.0
feature_dim = sample_img.size
num_train = len(df)

X_train = np.empty((num_train, feature_dim), dtype=np.float16)

with concurrent.futures.ThreadPoolExecutor(
    max_workers=multiprocessing.cpu_count()
) as executor:
    for idx, flat in enumerate(executor.map(load_and_flatten, df.path, chunksize=500)):
        X_train[idx] = flat.astype(np.float16)

print("Training matrix shape:", X_train.shape)
gc.collect()




## === cell 5
x = X_train
y = df["label"].values
train_x, val_x, train_y, val_y = train_test_split(
    x, y, test_size=0.10, random_state=101, stratify=y
)




## === cell 6
logreg = LogisticRegression(
    max_iter=1000,  # allow more iterations for convergence
    solver="lbfgs",
    n_jobs=1,  # keep single‑core to avoid OOM (lbfgs ignores n_jobs anyway)
    class_weight="balanced",
    C=2.0,  # a bit less regularisation
)
logreg.fit(train_x, train_y)




## === cell 7
val_pred = logreg.predict_proba(val_x)[:, 1]
val_auc = roc_auc_score(val_y, val_pred)
print(f"Validation ROC‑AUC: {val_auc:.6f}")




## === cell 8
base_tile_dir_test = "../input/histopathologic-cancer-detection/test/"
test_df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir_test, "*.tif"))})
test_df["id"] = test_df.path.map(lambda x: os.path.basename(x).split(".")[0])

test_preds = []

with concurrent.futures.ThreadPoolExecutor(
    max_workers=multiprocessing.cpu_count()
) as executor:
    for flat in executor.map(load_and_flatten, test_df.path, chunksize=500):
        prob = logreg.predict_proba(flat.reshape(1, -1))[:, 1][0]
        test_preds.append(prob)

test_df["label"] = test_preds
print("Test predictions computed:", len(test_preds))
gc.collect()




## === cell 9
submission = test_df[["id", "label"]]
submission.head()




## === cell 10
submission.to_csv("submission.csv", index=False, header=True)
print("Submission saved to submission.csv")
