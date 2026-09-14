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

0.3408716306310721

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

MK_CODES_PATH = "/kaggle/input/hms-mk-codes"
if MK_CODES_PATH not in sys.path:
    sys.path.append(MK_CODES_PATH)



## === cell 1
import subprocess


def _run(cmd: str):
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/1803309813.py in <cell line: 0>()
      9 
     10 
---> 11 _run(
     12     "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
     13 )

/tmp/ipykernel_11/1803309813.py in _run(cmd)
      6 def _run(cmd: str):
      7     print(cmd)
----> 8     subprocess.run(cmd, shell=True, check=True)
      9 
     10 

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command 'pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall' returned non-zero exit status 1.

## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 3
import subprocess

subprocess.run(
    f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
    shell=True,
    check=True,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/848344419.py in <cell line: 0>()
      2 import subprocess
      3 
----> 4 subprocess.run(
      5     f"cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
      6     shell=True,

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command 'cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir=/kaggle/input/hms-harmful-brain-activity-classification --out_dir=/kaggle/working' returned non-zero exit status 2.

## === cell 4
import subprocess

subprocess.run("ls /kaggle/input/hms-mk-data", shell=True, check=True)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/1045810687.py in <cell line: 0>()
      1 import subprocess
      2 
----> 3 subprocess.run("ls /kaggle/input/hms-mk-data", shell=True, check=True)
      4 

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command 'ls /kaggle/input/hms-mk-data' returned non-zero exit status 2.

## === cell 5
import subprocess


def run_test_and_move(fold: int, version: str, ckpt: str, experiment: str):
    cmd = (
        f"cd /kaggle/input/hms-mk-codes && "
        f"python -m test "
        f"paths.data_dir={DATA_PATH} "
        f"data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path={ckpt} "
        f"hydra=test "
        f"+model.test_output_dir={OUT_PATH} "
        f"experiment={experiment} "
        f"+model.net.pretrained=False"
    )
    subprocess.run(cmd, shell=True, check=True)
    src_file = "/kaggle/working/submission.csv"
    dst_file = f"/kaggle/working/submission_fold{fold}_{version}.csv"
    if not os.path.exists(src_file):
        raise FileNotFoundError(
            f"Expected {src_file} to be created by inference, but it does not exist."
        )
    subprocess.run(f"mv {src_file} {dst_file}", shell=True, check=True)


for fold in range(5):
    run_test_and_move(
        fold=fold,
        version="v0",
        ckpt=f"/kaggle/input/hms-mk-data/fold{fold}_pseudo_log.ckpt",
        experiment="conv1d_pseudo",
    )

for fold in range(5):
    run_test_and_move(
        fold=fold,
        version="v1",
        ckpt=f"/kaggle/input/hms-mk-data/fold{fold}_resv2.ckpt",
        experiment="conv1d_resv2",
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/2775895911.py in <cell line: 0>()
     26 
     27 for fold in range(5):
---> 28     run_test_and_move(
     29         fold=fold,
     30         version="v0",

/tmp/ipykernel_11/2775895911.py in run_test_and_move(fold, version, ckpt, experiment)
     15         f"+model.net.pretrained=False"
     16     )
---> 17     subprocess.run(cmd, shell=True, check=True)
     18     src_file = "/kaggle/working/submission.csv"
     19     dst_file = f"/kaggle/working/submission_fold{fold}_{version}.csv"

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command 'cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir=/kaggle/input/hms-harmful-brain-activity-classification data.test_eegs_dir=/kaggle/working ckpt_path=/kaggle/input/hms-mk-data/fold0_pseudo_log.ckpt hydra=test +model.test_output_dir=/kaggle/working experiment=conv1d_pseudo +model.net.pretrained=False' returned non-zero exit status 2.

## === cell 6
import pandas as pd
import numpy as np

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception:
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]


def merge_preds(folds=(0, 1, 2), versions=("v0",), workdir="/kaggle/working"):
    preds_list = []
    base_df = None

    for fold in folds:
        for version in versions:
            path = os.path.join(workdir, f"submission_fold{fold}_{version}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing prediction file: {path}")
            df = pd.read_csv(path)

            missing = [
                c for c in (["eeg_id"] + list(TARGET_COLS)) if c not in df.columns
            ]
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")

            if base_df is None:
                base_df = df[["eeg_id"] + list(TARGET_COLS)].copy()
            else:
                if not np.array_equal(base_df["eeg_id"].values, df["eeg_id"].values):
                    df = (
                        df.set_index("eeg_id")
                        .loc[base_df["eeg_id"].values]
                        .reset_index()
                    )

            preds_list.append(df[TARGET_COLS].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(preds_list, axis=0), axis=0)

    eps = 1e-12
    preds = np.clip(preds, eps, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = base_df[["eeg_id"]].copy()
    out[TARGET_COLS] = preds
    return out




## === cell 7
sol = merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0", "v1"))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4031600108.py in <cell line: 0>()
----> 1 sol = merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0", "v1"))
      2 

/tmp/ipykernel_11/267602020.py in merge_preds(folds, versions, workdir)
     25             path = os.path.join(workdir, f"submission_fold{fold}_{version}.csv")
     26             if not os.path.exists(path):
---> 27                 raise FileNotFoundError(f"Missing prediction file: {path}")
     28             df = pd.read_csv(path)
     29 

FileNotFoundError: Missing prediction file: /kaggle/working/submission_fold0_v0.csv

## === cell 8
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)
print(f"Wrote: {sub_path} with shape={sol.shape}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/827340277.py in <cell line: 0>()
      1 # Write final submission
      2 sub_path = "/kaggle/working/submission.csv"
----> 3 sol.to_csv(sub_path, index=False)
      4 print(f"Wrote: {sub_path} with shape={sol.shape}")
      5 

NameError: name 'sol' is not defined

## === cell 9
sol.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3448818667.py in <cell line: 0>()
----> 1 sol.head()

NameError: name 'sol' is not defined
