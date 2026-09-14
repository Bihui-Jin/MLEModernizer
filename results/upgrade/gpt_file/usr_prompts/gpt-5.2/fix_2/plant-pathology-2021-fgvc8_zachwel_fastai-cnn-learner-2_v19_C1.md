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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from fastai.vision.all import *
import matplotlib.pyplot as plt

plt.style.use("ggplot")

PATH = Path("/kaggle/input/plant-pathology-2021-fgvc8/")
TRAIN_CSV = PATH / "train.csv"
SAMPLE_SUB = PATH / "sample_submission.csv"
TRAIN_IMG_DIR = PATH / "train_images"
TEST_IMG_DIR = PATH / "test_images"

assert TRAIN_CSV.exists(), f"Missing: {TRAIN_CSV}"
assert SAMPLE_SUB.exists(), f"Missing: {SAMPLE_SUB}"
assert TRAIN_IMG_DIR.exists(), f"Missing: {TRAIN_IMG_DIR}"
assert TEST_IMG_DIR.exists(), f"Missing: {TEST_IMG_DIR}"



## === cell 1
df = pd.read_csv(TRAIN_CSV)
df.head()



## === cell 2
df.describe(include="all")



## === cell 3
df.labels.value_counts().plot(kind="bar", figsize=(16, 6))



## === cell 4
print("train_images:", len(get_image_files(TRAIN_IMG_DIR)))
print("test_images :", len(get_image_files(TEST_IMG_DIR)))



## === cell 5
PATH



## === cell 6
path = "../input/plant-pathology-2021-fgvc8/"
path



## === cell 7

set_seed(42, reproducible=True)


def split_labels(s):
    return s.split()


dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=ColReader("image", pref=str(TRAIN_IMG_DIR) + os.sep),
    get_y=ColReader("labels", label_delim=" "),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=384, min_scale=0.75),
)

dls = dblock.dataloaders(df, bs=32)



## === cell 8
learn = vision_learner(
    dls,
    resnet34,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[partial(accuracy_multi, thresh=0.5)],
).to_fp32()

learn.fine_tune(3, base_lr=3e-3)



## === cell 9
sub = pd.read_csv(SAMPLE_SUB)
test_files = [TEST_IMG_DIR / fn for fn in sub["image"].tolist()]

missing = [p for p in test_files if not Path(p).exists()]
if len(missing) > 0:
    print(
        f"Warning: {len(missing)} test files missing in this environment; using available files for preview."
    )
    test_files_available = get_image_files(TEST_IMG_DIR)
    test_dl = learn.dls.test_dl(test_files_available.sorted())
else:
    test_dl = learn.dls.test_dl(test_files)

preds, _ = learn.get_preds(dl=test_dl)
preds_np = preds.cpu().numpy()



## === cell 10
vocab = learn.dls.vocab
thresh = 0.5

labels_list = []
for row in preds_np:
    idxs = np.where(row >= thresh)[0].tolist()
    if len(idxs) == 0:
        idxs = [int(np.argmax(row))]
    labs = " ".join([vocab[i] for i in idxs])
    labels_list.append(labs)



## === cell 11
if len(labels_list) != len(sub):
    avail_files = get_image_files(TEST_IMG_DIR)
    avail_names = [p.name for p in avail_files]
    test_dl2 = learn.dls.test_dl(sorted(avail_files))
    preds2, _ = learn.get_preds(dl=test_dl2)
    preds2 = preds2.cpu().numpy()
    pred_map = {}
    for name, row in zip([p.name for p in sorted(avail_files)], preds2):
        idxs = np.where(row >= thresh)[0].tolist()
        if len(idxs) == 0:
            idxs = [int(np.argmax(row))]
        pred_map[name] = " ".join([vocab[i] for i in idxs])
    sub["labels"] = sub["image"].map(pred_map).fillna("healthy")
else:
    sub["labels"] = labels_list

sub.head()



## === cell 12
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.columns.tolist())
print(sub.head(3))
