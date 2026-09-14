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

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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
import pandas as pd
import numpy as np
import torch
from fastai.tabular.all import *
from fastai.callback.all import *

try:
    from fastai.callback.schedule import ReduceLROnPlateau
except ImportError:
    from fastai.callback.tracker import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from pathlib import Path
from PIL import Image
from tqdm import tqdm



## === cell 1
train_labels_path = Path("../input/histopathologic-cancer-detection/train_labels.csv")
train_dir = Path("../input/histopathologic-cancer-detection/train")
test_dir = Path("../input/histopathologic-cancer-detection/test")

train_labels = pd.read_csv(train_labels_path)
pos_rate = train_labels["label"].mean()
print(f"Overall positive rate (baseline feature value): {pos_rate:.6f}")




## === cell 2
def compute_image_features(img_path):
    """Return simple numeric features for a given image."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # shape (H,W,3)
    h, w, _ = arr.shape
    ch, cw = 32, 32
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    patch = arr[y0 : y0 + ch, x0 : x0 + cw, :]
    mean_full = arr.mean()
    std_full = arr.std()
    mean_center = patch.mean()
    std_center = patch.std()
    return mean_full, std_full, mean_center, std_center


train_features = []
print("Computing training image features...")
for img_id in tqdm(train_labels["id"].values):
    img_path = train_dir / f"{img_id}.tif"
    feats = compute_image_features(img_path)
    train_features.append(feats)

train_features = np.array(train_features)
train_df = pd.DataFrame(
    {
        "mean_full": train_features[:, 0],
        "std_full": train_features[:, 1],
        "mean_center": train_features[:, 2],
        "std_center": train_features[:, 3],
        "y": train_labels["label"].astype("category"),
    }
)

test_ids = [p.stem for p in sorted(test_dir.glob("*.tif"))]
test_features = []
print("Computing test image features...")
for img_id in tqdm(test_ids):
    img_path = test_dir / f"{img_id}.tif"
    feats = compute_image_features(img_path)
    test_features.append(feats)

test_features = np.array(test_features)
test_df = pd.DataFrame(
    {
        "mean_full": test_features[:, 0],
        "std_full": test_features[:, 1],
        "mean_center": test_features[:, 2],
        "std_center": test_features[:, 3],
    }
)



## === cell 3
dep_var = "y"
cont_names = ["mean_full", "std_full", "mean_center", "std_center"]
dls = TabularDataLoaders.from_df(
    train_df,
    path=".",
    cat_names=[],  # no categorical predictors
    cont_names=cont_names,
    y_names=dep_var,
    y_block=CategoryBlock(),
    valid_pct=0.2,
    seed=47,
)




## === cell 4
def roc_score(inp, targ):
    probs = torch.nn.functional.softmax(inp, dim=1)[:, 1]
    return torch.tensor(roc_auc_score(targ.cpu().numpy(), probs.cpu().numpy()))


learn = tabular_learner(
    dls,
    layers=[200, 100],
    wd=1e-1,
    loss_func=CrossEntropyLossFlat(),
    metrics=[accuracy, roc_score],
)

if torch.cuda.is_available():
    learn = learn.to_fp16()

cbs = [
    EarlyStoppingCallback(monitor="roc_score", patience=5),
    ReduceLROnPlateau(monitor="roc_score", patience=2),
    SaveModelCallback(monitor="roc_score", fname="best"),
]
learn.fit_one_cycle(5, 1e-3, cbs=cbs)



## === cell 5
learn.load("best")
auc_val = learn.validate()[2].item()  # roc_score is the third metric
preds, _ = learn.get_preds(dl=learn.dls.test_dl(test_df))
preds = torch.softmax(preds, dim=1)[:, 1].numpy()

sub_path = Path("../input/histopathologic-cancer-detection/sample_submission.csv")
sub = pd.read_csv(sub_path)
sub["label"] = preds
submission_filename = f"submission_{auc_val:.6f}.csv"
sub.to_csv(submission_filename, index=False, header=True)



## === cell 6
print(f"Validation ROC‑AUC: {auc_val:.6f}")
print("Submission file created:", submission_filename)
