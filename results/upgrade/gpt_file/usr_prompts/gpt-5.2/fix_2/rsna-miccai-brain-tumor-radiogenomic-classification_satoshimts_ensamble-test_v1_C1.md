# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

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
plotly==5.24.1
plotly-express==0.4.1
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import glob
import os

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot

np.random.seed(42)



## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train_labels.csv")
TEST_DIR = os.path.join(BASE_PATH, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)
train_labels["MGMT_value"] = pd.to_numeric(
    train_labels["MGMT_value"], errors="coerce"
).astype(float)

bad_ids = {"00109", "00123", "00709"}
train_labels_clean = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].copy()

listOfStudyPaths = glob.glob(os.path.join(TEST_DIR, "*"))
listOfStudies = sorted([os.path.basename(p) for p in listOfStudyPaths])

if len(listOfStudies) == 0:
    listOfStudies = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()

len(listOfStudies), listOfStudies[:5]



## === cell 2

global_mean = float(train_labels_clean["MGMT_value"].mean())
shrunk_mean = float(np.clip(0.95 * global_mean + 0.05 * 0.5, 0.0, 1.0))
const_half = 0.5

sub_746 = pd.DataFrame(
    {
        "BraTS21ID": listOfStudies,
        "MGMT_value": np.full(len(listOfStudies), global_mean, dtype=float),
    }
)
sub_674 = pd.DataFrame(
    {
        "BraTS21ID": listOfStudies,
        "MGMT_value": np.full(len(listOfStudies), shrunk_mean, dtype=float),
    }
)
sub_667 = pd.DataFrame(
    {
        "BraTS21ID": listOfStudies,
        "MGMT_value": np.full(len(listOfStudies), const_half, dtype=float),
    }
)

sub_746.head()



## === cell 3
hist_data = [sub_746.MGMT_value, sub_674.MGMT_value, sub_667.MGMT_value]
group_labels = ["sub_746", "sub_674", "sub_667"]

fig = ff.create_distplot(
    hist_data, group_labels, bin_size=0.2, show_hist=False, show_rug=False
)
fig.update_layout(width=700, height=400)
fig.show()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
LinAlgError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/scipy/stats/_kde.py in __init__(self, dataset, bw_method, weights)
    225         try:
--> 226             self.set_bandwidth(bw_method=bw_method)
    227         except linalg.LinAlgError as e:

/usr/local/lib/python3.11/dist-packages/scipy/stats/_kde.py in set_bandwidth(self, bw_method)
    573 
--> 574         self._compute_covariance()
    575 

/usr/local/lib/python3.11/dist-packages/scipy/stats/_kde.py in _compute_covariance(self)
    585                                                aweights=self.weights))
--> 586             self._data_cho_cov = linalg.cholesky(self._data_covariance,
    587                                                  lower=True)

/usr/local/lib/python3.11/dist-packages/scipy/linalg/_decomp_cholesky.py in cholesky(a, lower, overwrite_a, check_finite)
    100     """
--> 101     c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True,
    102                          check_finite=check_finite)

/usr/local/lib/python3.11/dist-packages/scipy/linalg/_decomp_cholesky.py in _cholesky(a, lower, overwrite_a, clean, check_finite)
     37     if info > 0:
---> 38         raise LinAlgError("%d-th leading minor of the array is not positive "
     39                           "definite" % info)

LinAlgError: 1-th leading minor of the array is not positive definite

The above exception was the direct cause of the following exception:

LinAlgError                               Traceback (most recent call last)
/tmp/ipykernel_11/488899474.py in <cell line: 0>()
      3 group_labels = ["sub_746", "sub_674", "sub_667"]
      4 
----> 5 fig = ff.create_distplot(
      6     hist_data, group_labels, bin_size=0.2, show_hist=False, show_rug=False
      7 )

/usr/local/lib/python3.11/dist-packages/plotly/figure_factory/_distplot.py in create_distplot(hist_data, group_labels, bin_size, curve_type, colors, rug_text, histnorm, show_hist, show_curve, show_rug)
    224                 show_hist,
    225                 show_curve,
--> 226             ).make_kde()
    227 
    228         data.append(curve)

/usr/local/lib/python3.11/dist-packages/plotly/figure_factory/_distplot.py in make_kde(self)
    359                 for x in range(500)
    360             ]
--> 361             self.curve_y[index] = scipy_stats.gaussian_kde(self.hist_data[index])(
    362                 self.curve_x[index]
    363             )

/usr/local/lib/python3.11/dist-packages/scipy/stats/_kde.py in __init__(self, dataset, bw_method, weights)
    233                    "analysis / dimensionality reduction and using "
    234                    "`gaussian_kde` with the transformed data.")
--> 235             raise linalg.LinAlgError(msg) from e
    236 
    237     def evaluate(self, points):

LinAlgError: The data appears to lie in a lower-dimensional subspace of the space in which it is expressed. This has resulted in a singular data covariance matrix, which cannot be treated using the algorithms implemented in `gaussian_kde`. Consider performing principal component analysis / dimensionality reduction and using `gaussian_kde` with the transformed data.

## === cell 4
dfs = {"sub_746": sub_746, "sub_674": sub_674, "sub_667": sub_667}

fig = go.Figure()
for i in dfs:
    fig = fig.add_trace(
        go.Scatter(
            x=dfs[i]["BraTS21ID"], y=dfs[i]["MGMT_value"], mode="markers", name=i
        )
    )
    fig = fig.add_trace(
        go.Scatter(
            x=[0, 1010],
            y=[0.5, 0.5],
            mode="lines",
            line=go.scatter.Line(color="gray"),
            showlegend=False,
        )
    )

fig.update_layout(width=700, height=500)
fig.show()



## === cell 5
sub_746["MGMT_value"].values.mean()



## === cell 6
Finalsubmission = sub_746.copy()
Finalsubmission["BraTS21ID"] = Finalsubmission["BraTS21ID"].astype(str).str.zfill(5)

Finalsubmission["MGMT_value"] = (
    sub_746["MGMT_value"].astype(float).values * 0.40
    + sub_674["MGMT_value"].astype(float).values * 0.30
    + sub_667["MGMT_value"].astype(float).values * 0.30
)

Finalsubmission["MGMT_value"] = (
    pd.to_numeric(Finalsubmission["MGMT_value"], errors="coerce")
    .fillna(0.5)
    .clip(0.0, 1.0)
)

Finalsubmission.head()



## === cell 7
Fsubmission = Finalsubmission.set_index("BraTS21ID")
FsubmissionDict = Fsubmission["MGMT_value"].to_dict()

predList = []
for eachStudy in listOfStudies:
    eachStudy = str(eachStudy).zfill(5)
    if eachStudy not in FsubmissionDict:
        predList.append(0.5)
    else:
        predList.append(float(FsubmissionDict[eachStudy]))

submission = pd.DataFrame({"BraTS21ID": listOfStudies, "MGMT_value": predList})
submission["BraTS21ID"] = submission["BraTS21ID"].astype(str).str.zfill(5)
submission["MGMT_value"] = (
    pd.to_numeric(submission["MGMT_value"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
)

submission.head(5)



## === cell 8
submission = submission.sort_values(by=["BraTS21ID"], ascending=True).reset_index(
    drop=True
)
submission.head(5)



## === cell 9
submission.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]
assert submission["MGMT_value"].between(0, 1).all()
submission.shape

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
