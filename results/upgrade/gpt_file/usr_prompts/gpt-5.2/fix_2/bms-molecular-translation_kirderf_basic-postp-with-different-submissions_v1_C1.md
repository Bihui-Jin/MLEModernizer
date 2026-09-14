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
Provided with images of chemicals, predict the corresponding International Chemical Identifier (InChI) text string of the image.

## Metric
Mean [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) between the InChi strings you submit and the ground truth InChi values.

## Submission Format
For each `image_id` in the test set, you must predict the InChi string of the molecule in the corresponding image. The file should contain a header and have the following format:

```
image_id,InChI
00000d2a601c,InChI=1S/H2O/h1H2
00001f7fc849,InChI=1S/H2O/h1H2
000037687605,InChI=1S/H2O/h1H2
etc.
```

## Dataset
- **train/** - the training images, arranged in a 3-level folder structure by `image_id`
- **test/** - the test images, arranged in the same folder structure as `train/`
- **train_labels.csv** - ground truth InChi labels for the training images
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
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        input/
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        working/
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
```

-> data/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/bms-molecular-translation/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/bms-molecular-translation/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> data/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> input/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> (stopped after 10 files for performance)

# 5. Target score

4.662682011302231

# 6. Current score

109.59778

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 109.59778) has done: 'Your notebook fails because it references external Kaggle dataset submissions that are not present in your environment, so `pd.read_csv()` raises `FileNotFoundError` and all later variables (like `sub0`, `subfinal`) never get created. I make the ensemble robust by (1) filtering `subs` down to files that actually exist, (2) falling back to the provided `sample_submission.csv` if none exist so a valid `.csv` is always produced, and (3) fixing pandas chained-indexing issues and the broken vote-summing logic so it runs deterministically end-to-end. This keeps the same intended “pick the submission with more matching strings vs others” core logic, but makes it safe and executable in this Kaggle filesystem. Since no valid submission was previously produced (“Not yielded”), the primary goal is to generate a correctly formatted `submission.csv`; score improvement isn’t possible without the missing external prediction files, so the fallback output the sample submission content.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
subs = [
    "../input/inchi-efficientnetb7-25-epochs-inference/submission.csv",
    "../input/efficientnet-multi-layer-lstm-inference/submission.csv",
    "../input/batched-beam-search-inference-changed-parameters/submission.csv",
    "../input/batched-beam-search-inference/submission.csv",
    "../input/subbeem/submission.789.csv",
]

existing_subs = [p for p in subs if os.path.exists(p)]

SAMPLE_CANDIDATES = [
    "/kaggle/input/bms-molecular-translation/sample_submission.csv",
    "/kaggle/data/bms-molecular-translation/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_path = next((p for p in SAMPLE_CANDIDATES if os.path.exists(p)), None)



## === cell 2
sub_dfs = []
if len(existing_subs) > 0:
    for p in existing_subs:
        df = pd.read_csv(p)
        if "image_id" not in df.columns or "InChI" not in df.columns:
            raise ValueError(
                f"Submission at {p} must contain columns ['image_id','InChI'], got {list(df.columns)}"
            )
        df = df.sort_values(by="image_id", ascending=True).reset_index(drop=True)
        sub_dfs.append(df)
else:
    if sample_path is None:
        raise FileNotFoundError(
            "No existing submission files found in ../input/* and no sample_submission.csv found in expected locations."
        )
    df = pd.read_csv(sample_path)
    df = df.sort_values(by="image_id", ascending=True).reset_index(drop=True)
    sub_dfs.append(df)

for ii, df in enumerate(sub_dfs):
    vars()[f"sub{ii}"] = df



## === cell 3
sums = []
subfinal = sub_dfs[0].copy()

for ii in range(len(sub_dfs)):
    subfinal[f"InChI{ii}"] = sub_dfs[ii]["InChI"].values
    subfinal[f"InChI{ii}sum"] = 0
    sums.append(f"InChI{ii}")



## === cell 4
for ii in range(len(sub_dfs)):
    others = [col for col in sums if col != f"InChI{ii}"]
    score_col = f"InChI{ii}sum"
    cand_col = f"InChI{ii}"

    for other_col in others:
        subfinal.loc[subfinal[cand_col] == subfinal[other_col], score_col] += 1



## === cell 5
score_cols = [f"InChI{i}sum" for i in range(len(sub_dfs))]
inchi_cols = [f"InChI{i}" for i in range(len(sub_dfs))]

best_idx = (
    subfinal[score_cols].to_numpy().argmax(axis=1)
)  # ties go to the first (consistent with argmax)
best_inchi = subfinal[inchi_cols].to_numpy()[np.arange(len(subfinal)), best_idx]
subfinal["InChI"] = best_inchi

subfinal["image_id"] = subfinal["image_id"].astype(str)
subfinal["InChI"] = subfinal["InChI"].astype(str)



## === cell 6
subfinal[["image_id", "InChI"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subfinal[["image_id", "InChI"]].shape)
print(subfinal.head())
