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
numpy==1.26.4
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
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

set_seed(42069, reproducible=True)



## === cell 1
MODEL_PATH = "Resnet18_v1"
TRAIN = "../input/histopathologic-cancer-detection/train/"
TEST = "../input/histopathologic-cancer-detection/test/"
LABELS = "../input/histopathologic-cancer-detection/train_labels.csv"
SAMPLE_SUB = "../input/histopathologic-cancer-detection/sample_submission.csv"
SIZE = 48
BATCH_SIZE = 128

nw = 4



## === cell 2
train_df = pd.read_csv(LABELS)
train_names = train_df["id"].values
train_labels = train_df["label"].values.astype(int)

print(
    "Number of positive samples = {:.4f}%".format(
        np.count_nonzero(train_labels) * 100 / len(train_labels)
    )
)

test_names = [f.replace(".tif", "") for f in os.listdir(TEST) if f.endswith(".tif")]

tr_n, val_n = train_test_split(
    train_names, test_size=0.15, random_state=42069, stratify=train_labels
)
print(len(tr_n), len(val_n))

label_map = dict(zip(train_df["id"].values, train_df["label"].values.astype(int)))




## === cell 3
def id_to_path(x, base):
    return Path(base) / f"{x}.tif"


item_tfms = Resize(SIZE, method=ResizeMethod.Squish)
batch_tfms = [
    *aug_transforms(
        do_flip=True,
        flip_vert=True,
        max_rotate=20.0,
        max_zoom=1.0,
        max_lighting=0.0,
        max_warp=0.0,
        p_affine=1.0,
        p_lighting=0.0,
    )
]

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=[0, 1])),
    get_x=lambda o: id_to_path(o, TRAIN),
    get_y=lambda o: label_map[o],
    splitter=IndexSplitter(list(range(len(tr_n), len(tr_n) + len(val_n)))),
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)

items = list(tr_n) + list(val_n)
dls = dblock.dataloaders(items, bs=BATCH_SIZE, num_workers=nw, pin_memory=True)



## === cell 4
learn = vision_learner(
    dls, resnet18, metrics=[RocAucBinary()], loss_func=CrossEntropyLossFlat()
)
learn = learn.to_fp32()  # ensure stable CPU/GPU behavior



## === cell 5
learn.fine_tune(1, base_lr=3e-3)



## === cell 6
test_dl = dls.test_dl(
    [id_to_path(o, TEST) for o in test_names], with_labels=False, num_workers=nw
)

preds_t, _ = learn.tta(dl=test_dl, n=4, beta=0.0)

pred_pos = preds_t[:, 1].detach().cpu().numpy()
print(pred_pos.shape)



## === cell 7
sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = sample_df["id"].tolist()

pred_dic = dict(zip(test_names, pred_pos.astype(float)))
pred_list_cor = [pred_dic[_id] for _id in sample_list]

sub_df = pd.DataFrame({"id": sample_list, "label": pred_list_cor})
sub_path = "submission.csv"
sub_df.to_csv(sub_path, header=True, index=False)

print("Wrote:", sub_path)
print(sub_df.head())



## === cell 8
sub_df
