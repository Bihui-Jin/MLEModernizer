# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

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
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from pathlib import Path
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import roc_auc_score

from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    aug_transforms,
    cnn_learner,
    resnet50,
    CrossEntropyLossFlat,
    error_rate,
    Metric,
    get_image_files,
)

base_path = Path("../input/aerial-cactus-identification")
train_img_path = base_path / "train"
test_img_path = base_path / "test"
train_csv_path = base_path / "train.csv"
sample_sub_path = base_path / "sample_submission.csv"

bs = 64

train_df_temp = pd.read_csv(train_csv_path)
counts = train_df_temp["has_cactus"].value_counts().sort_index()  # index 0,1
max_cnt = counts.max()
cls_weights = torch.tensor(
    [max_cnt / counts[0], max_cnt / counts[1]], dtype=torch.float
)



## === cell 1
train_df = pd.read_csv(train_csv_path)
train_df["has_cactus"] = train_df["has_cactus"].astype(int)



## === cell 2
dls = ImageDataLoaders.from_df(
    df=train_df,
    path=".",  # not used because we give full filenames via `folder`
    fn_col="id",
    label_col="has_cactus",
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),  # larger image size for richer features
    batch_tfms=aug_transforms(do_flip=False),
    bs=bs,
    folder=train_img_path,
    num_workers=4,  # <-- added: parallel data loading
)




## === cell 3
class RocAucBinary(Metric):
    "Computes ROC‑AUC for binary classification from logits"

    def __init__(self):
        self.preds, self.targs = [], []

    def reset(self):
        self.preds, self.targs = [], []

    def accumulate(self, learn):
        prob = torch.softmax(learn.pred, dim=1)[:, 1]
        self.preds.append(prob.detach().cpu())
        self.targs.append(learn.y.detach().cpu())

    @property
    def value(self):
        if not self.preds:
            return float("nan")
        preds = torch.cat(self.preds).numpy()
        targs = torch.cat(self.targs).numpy()
        return roc_auc_score(targs, preds)




## === cell 4
learn = cnn_learner(
    dls,
    resnet50,
    loss_func=CrossEntropyLossFlat(weight=cls_weights),
    metrics=[error_rate, RocAucBinary()],
    pretrained=True,
)
learn = learn.to_fp16()  # <-- added: mixed‑precision training for speed
learn.fine_tune(40)  # training for better performance



## === cell 5
test_files = get_image_files(test_img_path)  # Paths like …/test/xxxx.jpg
test_dl = dls.test_dl(test_files)

preds, _ = learn.get_preds(dl=test_dl, with_loss=False, cbs=[])
prob_cactus = preds[:, 1].numpy()  # probability of class 1



## === cell 6
sub = pd.read_csv(sample_sub_path)
sub["has_cactus"] = prob_cactus
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv at", datetime.now())
