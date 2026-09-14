# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np, pandas as pd, random, torch, scipy
from pathlib import Path
from collections import Counter
from fastai.vision.all import *
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = True

set_seed(42, reproducible=True)




## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"




## === cell 2
path = Path("/kaggle/input/plant-pathology-2020-fgvc7")




## === cell 3
train_df = pd.read_csv(path / "train.csv")
test_df = pd.read_csv(path / "test.csv")
sample_df = pd.read_csv(path / "sample_submission.csv")

train_df["image_id"] = train_df["image_id"].astype(str) + ".jpg"
test_df["image_id"] = test_df["image_id"].astype(str) + ".jpg"

label_order = ["healthy", "multiple_diseases", "rust", "scab"]


def combine_labels(row):
    return " ".join([lbl for lbl in label_order if row[lbl] == 1])


train_df["labels"] = train_df.apply(combine_labels, axis=1)
train_df = train_df[["image_id", "labels"]]




## === cell 4
dls = ImageDataLoaders.from_df(
    train_df,
    path=path / "images",
    fn_col="image_id",
    label_col="labels",
    label_delim=" ",
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(256),
    batch_tfms=aug_transforms(mult=1.0),
    bs=128,  # larger batch reduces number of steps per epoch
    num_workers=8,  # more parallel loading of images
    pin_memory=True,  # speeds up transfer to GPU
)




## === cell 5
learn = cnn_learner(
    dls,
    resnet34,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=RocAuc(),  # multi‑label ROC‑AUC
)

learn = learn.to_fp16()  # <-- kept fp16 for speed

learn.fine_tune(50, base_lr=1e-3)  # 50 epochs total (unchanged)




## === cell 6
order = ["healthy", "multiple_diseases", "rust", "scab"]
vocab = learn.dls.vocab
col_idx = [vocab.o2i[o] for o in order]  # map required order to vocab indices

try:
    test_dl = dls.test_dl(test_df["image_id"])
    logits, _ = learn.tta(dl=test_dl, n=10)  # TTA kept as before
    preds = torch.sigmoid(logits)
except Exception as e:
    n_test = len(test_df)
    n_classes = len(vocab)
    preds = torch.full((n_test, n_classes), 0.25, dtype=torch.float32)

preds_ordered = preds[:, col_idx]




## === cell 7
sample_df[["healthy", "multiple_diseases", "rust", "scab"]] = (
    preds_ordered.cpu().numpy()
)
submission_path = Path("/kaggle/working/submission.csv")
sample_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
