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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.9112965725853808

# 6. Current score

0.58571

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.58571) has done: 'The notebook fails because it tries to read many out-of-environment “../input/model-*” blended submission files that don’t exist here, so nothing downstream is defined and no CSV is written. I keep the same blending-style “produce a submission from existing predictions” core idea, but add a safe fallback: if those external blend files are missing, build a valid submission directly from the provided `sample_submission.csv` (neutral baseline) so a `.csv` is always produced end-to-end. I also make the code robust to different Kaggle directory layouts by probing the known dataset locations you listed. This is score-neutral/low (since current score is not yielded) but guarantees a valid submission file is created.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os



## === cell 1


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _read_submission_or_none(path):
    if path is None or (not os.path.exists(path)):
        return None
    df = pd.read_csv(path)
    if "Id" not in df.columns:
        for c in ["image_id", "ImageId", "id"]:
            if c in df.columns:
                df = df.rename(columns={c: "Id"})
                break
    if "Label" not in df.columns:
        for c in ["label", "pred", "prediction", "prob"]:
            if c in df.columns:
                df = df.rename(columns={c: "Label"})
                break
    if not {"Id", "Label"}.issubset(df.columns):
        raise ValueError(
            f"Submission file {path} does not contain required columns Id, Label. Found: {df.columns.tolist()}"
        )
    return df[["Id", "Label"]].copy()


blend_paths = {
    "sub_10": "../input/model-1-fold0/submission_fold0_epoch35.csv",
    "sub_20": "../input/model-2-fold0/submission_fold0_epoch38.csv",
    "sub_30": "../input/model-3-fold0/submission_fold0_epoch39.csv",
    "sub_11": "../input/model-1-fold1/submission_fold1_epoch_30.csv",
    "sub_21": "../input/model-2-fold1/submission_fold1_epoch_34.csv",
    "sub_31": "../input/model-3-fold1/submission_fold1_epoch_38.csv",
    "sub_12": "../input/model-1-fold2/submission_fold2_epoch_34.csv",
    "sub_22": "../input/model-2-fold2/submission_fold2_epoch_37.csv",
    "sub_32": "../input/model-3-fold2/submission_fold2_epoch_39.csv",
    "sub_13": "../input/model-1-fold3/submission_fold3_epoch_35.csv",
    "sub_23": "../input/model-2-fold3/submission_fold3_epoch_36.csv",
    "sub_33": "../input/model-3-fold3/submission_fold3_epoch_39.csv",
    "sub_14": "../input/model-1-fold4/submission_fold4_epoch_33.csv",
    "sub_24": "../input/model-2-fold4/submission_fold4_epoch_35.csv",
    "sub_34": "../input/model-3-fold4/submission_fold4_epoch_36.csv",
    "sub_b3_13": "../input/b3-fold3-m1/submission_fold3_b3_m1.csv",
    "sub_b3_23": "../input/b3-fold3-m2/submission_fold3_b3_m2.csv",
    "sub_b3_33": "../input/b3-fold3-m3/submission_fold3_b3_m3.csv",
    "sub_b3_10": "../input/b3-fold0-m1/submission_fold0_b3_m1.csv",
    "sub_b3_20": "../input/b3-fold0-m2/submission_fold0_b3_m2.csv",
    "sub_b3_30": "../input/b3-fold0-m3/submission_fold0_b3_m3.csv",
    "sub_oc_1": "../input/oc-m1/submission_openclose_m1.csv",
    "sub_oc_2": "../input/oc-m2/submission_openclose_m2.csv",
    "sub_oc_3": "../input/oc-m3/submission_openclose_m3.csv",
}

subs = {}
missing = []
for k, p in blend_paths.items():
    df = _read_submission_or_none(p)
    if df is None:
        missing.append(p)
    subs[k] = df

sample_path = _first_existing(
    [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/alaska2-image-steganalysis/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/alaska2-image-steganalysis/sample_submission.csv",
        "../input/sample_submission.csv",
        "../input/alaska2-image-steganalysis/sample_submission.csv",
    ]
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in the expected Kaggle paths."
    )

sample_sub = pd.read_csv(sample_path)
if not {"Id", "Label"}.issubset(sample_sub.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns. Found: {sample_sub.columns.tolist()}"
    )



## === cell 2

available_subs = [df for df in subs.values() if df is not None]

if len(available_subs) == len(subs):
    for k in list(subs.keys()):
        subs[k] = subs[k].sort_values(by="Id").reset_index(drop=True)

    base = subs["sub_10"][["Id"]].copy()
    base = base.sort_values("Id").reset_index(drop=True)

    merged = base
    for k, df in subs.items():
        merged = merged.merge(
            df.rename(columns={"Label": k}), on="Id", how="left", validate="one_to_one"
        )

    for k in subs.keys():
        if merged[k].isna().any():
            merged[k] = merged[k].fillna(0.0)

    w0 = 1 / 24
    merged["Label"] = 0.0
    for k in subs.keys():
        merged["Label"] += w0 * merged[k].astype(float)

    final_sub = merged[["Id", "Label"]].copy()
else:
    final_sub = sample_sub[["Id"]].copy()
    final_sub["Label"] = 0.0



## === cell 3
final_sub["Id"] = final_sub["Id"].astype(str)
final_sub["Label"] = (
    pd.to_numeric(final_sub["Label"], errors="coerce").fillna(0.0).astype(float)
)

final_sub = sample_sub[["Id"]].merge(
    final_sub, on="Id", how="left", validate="one_to_one"
)
final_sub["Label"] = final_sub["Label"].fillna(0.0)

out_path = "submission_Jul20_blend_b3_b2_oc.csv"
final_sub.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(final_sub.head())
print(f"Rows: {len(final_sub)}, Columns: {final_sub.columns.tolist()}")
