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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve

from fastai.vision.all import *

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
set_seed(42, reproducible=False)



## === cell 1
model_path = "."
path = "/kaggle/input/histopathologic-cancer-detection/"
train_folder = f"{path}train"
test_folder = f"{path}test"
train_lbl = f"{path}train_labels.csv"

bs = 64
num_workers = 2
sz = 96

train_folder, test_folder, train_lbl



## === cell 2
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)
print("torch:", torch.__version__)



## === cell 3
df_train = pd.read_csv(train_lbl)
print(f"Number of labels {len(df_train)}")
df_train.head()



## === cell 4
df_train["label"].value_counts(normalize=True)



## === cell 5
plt.figure(figsize=(4, 3))
sns.countplot(x="label", data=df_train)
plt.tight_layout()



## === cell 6
cancer_cell = df_train[df_train["label"] == 1].head()
non_cancer_cell = df_train[df_train["label"] == 0].head()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
img = np.asarray(plt.imread(f"{train_folder}/{cancer_cell.iloc[1][0]}.tif"))
plt.title("METASTATIC CELL TISSUE")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
img = np.asarray(plt.imread(f"{train_folder}/{non_cancer_cell.iloc[1][0]}.tif"))
plt.title("NON-METASTATIC CELL TISSUE")
plt.imshow(img)
plt.axis("off")
plt.tight_layout()



## === cell 7
test_list = os.listdir(test_folder)
train_list = os.listdir(train_folder)
len(test_list), len(train_list)



## === cell 8
item_tfms = Resize(90)
batch_tfms = [
    *aug_transforms(
        do_flip=True,
        flip_vert=True,
        max_rotate=0.0,
        max_zoom=1.1,
        max_lighting=0.05,
        max_warp=0.0,
    ),
    Normalize.from_stats(*imagenet_stats),
]

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader("id", pref=train_folder + "/", suff=".tif"),
    get_y=ColReader("label"),
    splitter=RandomSplitter(valid_pct=0.3, seed=42),
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)

dls = dblock.dataloaders(df_train, bs=bs, num_workers=num_workers)
dls.c, len(dls.train_ds), len(dls.valid_ds)



## === cell 9
dls.show_batch(max_n=9, figsize=(8, 8))



## === cell 10
model_dir = Path("/kaggle/working/tmp/models/")
model_dir.mkdir(parents=True, exist_ok=True)
model_dir



## === cell 11
learn = vision_learner(
    dls,
    resnet50,
    metrics=[accuracy, error_rate, RocAucBinary()],
    ps=0.5,
    model_dir=model_dir,
)

if torch.cuda.is_available():
    learn.to_fp32()
    learn.model = learn.model.cuda()

learn



## === cell 12
learn.fit_one_cycle(1, 1e-2)



## === cell 13
learn.save("stage-1")



## === cell 14
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-6, 1e-5), pct_start=0.8)



## === cell 15
learn.save("stage-2")



## === cell 16
preds, targs = learn.get_preds()
if preds.ndim == 2 and preds.shape[1] == 2:
    p1 = preds[:, 1].cpu().numpy()
else:
    p1 = preds.squeeze().cpu().numpy()

y_true = targs.cpu().numpy()
val_auc = roc_auc_score(y_true, p1)
val_acc = accuracy(preds, targs).item()
print("Validation AUC:", val_auc)
print("Validation accuracy:", val_acc)



## === cell 17
fpr, tpr, thresholds = roc_curve(y_true, p1, pos_label=1)
pred_score_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color="orange", label="ROC curve (area = %0.4f)" % pred_score_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.tight_layout()



## === cell 18
export_path = model_dir / "export.pkl"
learn.export(export_path)
export_path.exists(), export_path



## === cell 19
loaded_learner = load_learner(export_path)
loaded_learner



## === cell 20
sub_path = Path(path) / "sample_submission.csv"
sub = pd.read_csv(sub_path)
sub.head()



## === cell 21
test_files = [Path(test_folder) / f"{id_}.tif" for id_ in sub["id"].values]
test_dl = loaded_learner.dls.test_dl(test_files, bs=bs, num_workers=num_workers)
test_probs, _ = loaded_learner.get_preds(dl=test_dl)

if test_probs.ndim == 2 and test_probs.shape[1] == 2:
    test_p1 = test_probs[:, 1].cpu().numpy()
else:
    test_p1 = test_probs.squeeze().cpu().numpy()

len(test_p1), len(sub)



## === cell 22
sub_out = sub.copy()
sub_out["label"] = test_p1.astype(np.float32)

out_path = Path("/kaggle/working/submission.csv")
sub_out.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub_out.head())
