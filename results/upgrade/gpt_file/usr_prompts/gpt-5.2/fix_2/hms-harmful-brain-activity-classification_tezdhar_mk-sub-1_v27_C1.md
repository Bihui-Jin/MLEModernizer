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
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

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
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.3111068021557354

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import glob
import pandas as pd
import numpy as np



## === cell 1
sys.path.append("/kaggle/input/hms-mk-codes/")



## === cell 2
import subprocess


def _run(cmd: str):
    print(cmd)
    r = subprocess.run(
        cmd,
        shell=True,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(r.stdout)


_run(
    "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
)
_run(
    "pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
)
_run(
    "pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
)
_run(
    "pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/3495119200.py in <cell line: 0>()
     16 
     17 
---> 18 _run(
     19     "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
     20 )

/tmp/ipykernel_11/3495119200.py in _run(cmd)
      5 def _run(cmd: str):
      6     print(cmd)
----> 7     r = subprocess.run(
      8         cmd,
      9         shell=True,

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command 'pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall' returned non-zero exit status 1.

## === cell 3
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 4
_run(
    f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/510443872.py in <cell line: 0>()
      1 # Convert parquet -> npy using provided repository script (core logic preserved).
----> 2 _run(
      3     f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
      4 )
      5 

/tmp/ipykernel_11/3495119200.py in _run(cmd)
      5 def _run(cmd: str):
      6     print(cmd)
----> 7     r = subprocess.run(
      8         cmd,
      9         shell=True,

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command 'cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir=/kaggle/input/hms-harmful-brain-activity-classification --out_dir=/kaggle/working' returned non-zero exit status 2.

## === cell 5
print("Listing /kaggle/input/hms-mk-data (if exists):")
if os.path.exists("/kaggle/input/hms-mk-data"):
    print("\n".join(os.listdir("/kaggle/input/hms-mk-data")[:50]))
else:
    print("Path not found: /kaggle/input/hms-mk-data")



## === cell 6
cmds = [
    ("fold0_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold0_v0.csv"),
    ("fold1_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold1_v0.csv"),
    ("fold2_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold2_v0.csv"),
    ("fold3_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold3_v0.csv"),
    ("fold4_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold4_v0.csv"),
    ("fold0_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold0_v2.csv"),
    ("fold1_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold1_v2.csv"),
    ("fold2_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold2_v2.csv"),
    ("fold3_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold3_v2.csv"),
    ("fold4_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold4_v2.csv"),
]

for ckpt, experiment, out_name in cmds:
    _run(
        "cd /kaggle/input/hms-mk-codes && "
        f"python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path=/kaggle/input/hms-mk-data/{ckpt} hydra=test +model.test_output_dir={OUT_PATH} "
        f"experiment={experiment} +model.net.pretrained=False"
    )
    src_csv = "/kaggle/working/submission.csv"
    dst_csv = f"/kaggle/working/{out_name}"
    if not os.path.exists(src_csv):
        raise FileNotFoundError(
            f"Expected {src_csv} to be produced by inference but it was not found."
        )
    os.replace(src_csv, dst_csv)
    print(f"Saved {dst_csv}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/919463409.py in <cell line: 0>()
     15 
     16 for ckpt, experiment, out_name in cmds:
---> 17     _run(
     18         "cd /kaggle/input/hms-mk-codes && "
     19         f"python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "

/tmp/ipykernel_11/3495119200.py in _run(cmd)
      5 def _run(cmd: str):
      6     print(cmd)
----> 7     r = subprocess.run(
      8         cmd,
      9         shell=True,

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command 'cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=/kaggle/input/hms-harmful-brain-activity-classification data.test_eegs_dir=/kaggle/working ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt hydra=test +model.test_output_dir=/kaggle/working experiment=conv1d_tfm2d_pseudo +model.net.pretrained=False' returned non-zero exit status 2.

## === cell 7
if "/kaggle/input/hms-mk-codes" not in sys.path:
    sys.path.append("/kaggle/input/hms-mk-codes")

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print(f"Warning: could not import TARGET_COLS from src.settings due to: {repr(e)}")
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

TARGET_COLS = list(TARGET_COLS)
print("TARGET_COLS:", TARGET_COLS)


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2"),
    weights=(0.3, 0.7),
    base_submission_path=f"{DATA_PATH}/sample_submission.csv",
):
    """
    Minimal but robust ensembling:
    - Reads each fold/version submission
    - Aligns by eeg_id (prevents row-order mismatch)
    - Weighted sum then renormalize to sum=1 (required by competition)
    """
    if len(versions) != len(weights):
        raise ValueError(
            f"versions and weights must have same length, got {len(versions)} vs {len(weights)}"
        )

    base = pd.read_csv(base_submission_path)
    if "eeg_id" not in base.columns:
        raise ValueError("sample_submission must contain eeg_id")
    base_ids = base[["eeg_id"]].copy()

    acc = None
    total_w = 0.0

    for fold in folds:
        for version, w in zip(versions, weights):
            path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing prediction file: {path}")

            df = pd.read_csv(path)
            missing = ({"eeg_id"} | set(TARGET_COLS)) - set(df.columns)
            if missing:
                raise ValueError(f"{path} missing columns: {missing}")

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = base_ids.merge(df, on="eeg_id", how="left", validate="one_to_one")

            if df[TARGET_COLS].isna().any().any():
                bad = (
                    df.loc[df[TARGET_COLS].isna().any(axis=1), "eeg_id"]
                    .head(10)
                    .tolist()
                )
                raise ValueError(
                    f"{path}: NaNs after merging by eeg_id. Example missing eeg_id(s): {bad}"
                )

            pred = df[TARGET_COLS].to_numpy(dtype=np.float64)
            if acc is None:
                acc = np.zeros_like(pred)
            acc += pred * float(w)
            total_w += float(w)

    acc /= max(total_w, 1e-12)

    acc = np.clip(acc, 1e-15, 1.0)
    acc = acc / acc.sum(axis=1, keepdims=True)

    out = base_ids.copy()
    out[TARGET_COLS] = acc
    return out




## === cell 8
sol = merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0", "v2"), weights=(0.3, 0.7))

sample = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
sol = sample[["eeg_id"]].merge(sol, on="eeg_id", how="left", validate="one_to_one")
sol = sol[["eeg_id"] + TARGET_COLS]

p = sol[TARGET_COLS].to_numpy(dtype=np.float64)
p = np.clip(p, 1e-15, 1.0)
p = p / p.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = p

out_path = "/kaggle/working/submission.csv"
sol.to_csv(out_path, index=False)
print(f"Wrote: {out_path} with shape {sol.shape}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1836663593.py in <cell line: 0>()
----> 1 sol = merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0", "v2"), weights=(0.3, 0.7))
      2 
      3 # Final schema check + enforce column order exactly as sample_submission.
      4 sample = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
      5 sol = sample[["eeg_id"]].merge(sol, on="eeg_id", how="left", validate="one_to_one")

/tmp/ipykernel_11/2173443393.py in merge_preds(folds, versions, weights, base_submission_path)
     50             path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
     51             if not os.path.exists(path):
---> 52                 raise FileNotFoundError(f"Missing prediction file: {path}")
     53 
     54             df = pd.read_csv(path)

FileNotFoundError: Missing prediction file: /kaggle/working/submission_fold0_v0.csv

## === cell 9
sol.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3448818667.py in <cell line: 0>()
----> 1 sol.head()

NameError: name 'sol' is not defined
