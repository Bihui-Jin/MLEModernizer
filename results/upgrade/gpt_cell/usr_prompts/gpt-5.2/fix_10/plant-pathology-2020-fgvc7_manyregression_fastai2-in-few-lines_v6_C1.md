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

# 5. Target score

0.91678

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install -q fastai2


## === cell 1
from pathlib import Path



## === cell 2
path = Path("/kaggle/input/plant-pathology-2020-fgvc7")


## === cell 3
from IPython import get_ipython

ip = get_ipython()
if ip is not None:
    ip.kernel.do_shutdown(restart=True)
else:
    raise RuntimeError(
        "This fix requires a notebook/IPython kernel to restart cleanly."
    )


## === cell 4
train_df.query("image_id == 'Train_5'")


## === cell 5
get_image_files(path/"images")[5]


## === cell 6
train_df.iloc[0, 1:][train_df.iloc[0, 1:] == 1].index[0]


## === cell 7
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']


## === cell 8
def get_data(size=224):
    return DataBlock(blocks    = (ImageBlock, CategoryBlock),
                       get_x=ColReader(0, pref=path/"images", suff=".jpg"),
                       get_y=lambda o:o.iloc[1:][o.iloc[1:] == 1].index[0],
                       splitter=RandomSplitter(seed=42),
                       item_tfms=Resize(size),
                       batch_tfms=aug_transforms(flip_vert=True),
                      )


## === cell 9
dblock = get_data()


## === cell 10
dsets = dblock.datasets(train_df)
dsets.train[0]


## === cell 11
BS = (1024 - 256)//8


## === cell 12
dls = dblock.dataloaders(train_df, bs=BS)
dls.show_batch()


## === cell 13
from sklearn.metrics import roc_auc_score

def roc_auc(preds, targs, labels=range(4)):
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return roc_auc(*args, labels=[0])

def multiple_diseases_roc_auc(*args):
    return roc_auc(*args, labels=[1])

def rust_roc_auc(*args):
    return roc_auc(*args, labels=[2])

def scab_roc_auc(*args):
    return roc_auc(*args, labels=[3])


## === cell 14
metric = partial(AccumMetric, flatten=False)

learn = cnn_learner(dls, resnet152, metrics=[
            error_rate,
            metric(healthy_roc_auc),
            metric(multiple_diseases_roc_auc),
            metric(rust_roc_auc),
            metric(scab_roc_auc),
            metric(roc_auc)]
        ).to_fp16()


## === cell 19
lr = 3e-3


## === cell 20
learn.fine_tune(4, lr)


## === cell 21
test_df = pd.read_csv(path/"test.csv")
test_df.head()


## === cell 22
tst_dl = learn.dls.test_dl(test_df)


## === cell 23
preds, y = learn.get_preds(dl=tst_dl)


## === cell 24
preds


## === cell 25
subm = pd.read_csv(path/"sample_submission.csv")


## === cell 26
subm.iloc[:, 1:] = preds


## === cell 27
subm.to_csv("submission.csv", index=False, float_format='%.20f')


## === cell 28
pd.read_csv("submission.csv")
