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

3.9

# 3. Installed packages

geopandas==0.14.4
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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read external “testsubmissions” CSV files that don’t exist in this environment, so all later variables are undefined and no `submission.csv` is written. I keep the ensemble/blending core logic intact, but add a safe fallback: if those external files are missing, we build a valid submission from `sample_submission.csv` with a constant probability (0.5) so the pipeline always completes. I also fix the paths to use the provided competition directory and ensure `BraTS21ID` formatting and row alignment are correct before writing the final `.csv`. This is primarily a correctness/stability fix so you can get a scored submission; without actual model prediction files, we cannot legitimately improve AUC beyond baseline.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
]
BASE_INPUT = next((p for p in BASE_INPUT_CANDIDATES if os.path.exists(p)), None)
if BASE_INPUT is None:
    for root in ["/kaggle/input", "/kaggle/data", "../input", "../data"]:
        if os.path.exists(root):
            cand = os.path.join(
                root, "rsna-miccai-brain-tumor-radiogenomic-classification"
            )
            if os.path.exists(cand):
                BASE_INPUT = cand
                break

if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not locate rsna-miccai-brain-tumor-radiogenomic-classification input directory."
    )

SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)



## === cell 1


def _safe_read_csv(path, sort_ids=False):
    if path is None or (isinstance(path, str) and not os.path.exists(path)):
        return None
    df = pd.read_csv(path)
    if "BraTS21ID" in df.columns:
        df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
    if sort_ids and "BraTS21ID" in df.columns:
        df = df.sort_values(by="BraTS21ID").reset_index(drop=True)
    return df


sub056 = _safe_read_csv("../input/testsubmissions/submission (33).csv")
sub084 = _safe_read_csv("../input/testsubmissions/submission (34).csv")
sub083 = _safe_read_csv("../input/testsubmissions/submission (35).csv")
sub077 = _safe_read_csv("../input/testsubmissions/submission (36).csv", sort_ids=True)
sub067mobnet = _safe_read_csv("../input/testsubmissions/submission067mob.csv")
sub067lstm = _safe_read_csv("../input/testsubmissions/submissionlstm067.csv")
sub074effnet74 = _safe_read_csv("../input/testsubmissions/submission067mob.csv")
sub0633dcnn = _safe_read_csv("../input/testsubmissions/submissionlstm067.csv")

subs = {
    "sub056": sub056,
    "sub084": sub084,
    "sub083": sub083,
    "sub077": sub077,
    "sub067mobnet": sub067mobnet,
    "sub067lstm": sub067lstm,
    "sub074effnet74": sub074effnet74,
    "sub0633dcnn": sub0633dcnn,
}
missing = [k for k, v in subs.items() if v is None]



## === cell 2
if len(missing) == 0:
    def _reindex_to_sample(df):
        df2 = df.copy()
        df2["BraTS21ID"] = df2["BraTS21ID"].astype(str).str.zfill(5)
        df2 = df2.set_index("BraTS21ID").reindex(sample_sub["BraTS21ID"])
        return df2.reset_index()

    sub056_a = _reindex_to_sample(sub056)
    sub084_a = _reindex_to_sample(sub084)
    sub083_a = _reindex_to_sample(sub083)
    sub077_a = _reindex_to_sample(sub077)
    sub067lstm_a = _reindex_to_sample(sub067lstm)
    sub067mobnet_a = _reindex_to_sample(sub067mobnet)
    sub074effnet74_a = _reindex_to_sample(sub074effnet74)
    sub0633dcnn_a = _reindex_to_sample(sub0633dcnn)

    fsubmission = sub084_a.copy()
    fsubmission["MGMT_value"] = (
        sub056_a["MGMT_value"].values * 0.05
        + sub084_a["MGMT_value"].values * 0.40
        + sub083_a["MGMT_value"].values * 0.15
        + sub077_a["MGMT_value"].values * 0.15
        + sub067lstm_a["MGMT_value"].values * 0.05
        + sub067mobnet_a["MGMT_value"].values * 0.05
        + sub074effnet74_a["MGMT_value"].values * 0.10
        + sub0633dcnn_a["MGMT_value"].values * 0.05
    )
    fsubmission["BraTS21ID"] = fsubmission["BraTS21ID"].astype(str).str.zfill(5)
else:
    fsubmission = sample_sub.copy()
    fsubmission["MGMT_value"] = 0.5

fsubmission.head()



## === cell 3
submissionDF01 = fsubmission.set_index("BraTS21ID")
scoreDict01 = submissionDF01["MGMT_value"].to_dict()

listOfStudyPaths = glob.glob(os.path.join(TEST_DIR, "*"))
listOfStudies = [os.path.basename(eachPath) for eachPath in listOfStudyPaths]
listOfStudies = [str(x).zfill(5) for x in listOfStudies]

predList = []
for eachStudy in listOfStudies:
    if eachStudy not in scoreDict01 or pd.isna(scoreDict01[eachStudy]):
        predList.append(0.5)
    else:
        predList.append(float(scoreDict01[eachStudy]))

submissionDF = pd.DataFrame({"BraTS21ID": listOfStudies, "MGMT_value": predList})

submissionDF = (
    submissionDF.set_index("BraTS21ID").reindex(sample_sub["BraTS21ID"]).reset_index()
)
submissionDF["MGMT_value"] = submissionDF["MGMT_value"].astype(float).clip(0.0, 1.0)

submission_path = "submission.csv"
submissionDF.to_csv(submission_path, index=False)
submissionDF.head()



## === cell 4
print("BASE_INPUT:", BASE_INPUT)
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print("Missing external blend files:", missing)
print("Submission shape:", submissionDF.shape)
print("Saved:", os.path.abspath("submission.csv"))
print(submissionDF.dtypes)
print(submissionDF.isna().sum())
print(submissionDF.head(10))
