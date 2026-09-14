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

0.3390572951476089

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
from pathlib import Path

CODE_PATH = "/kaggle/input/hms-mk-codes"
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)

print("Python:", sys.version)
print("CODE_PATH exists:", os.path.exists(CODE_PATH))
print("sys.path contains CODE_PATH:", CODE_PATH in sys.path)



## === cell 1
import subprocess, shlex


def _run(cmd):
    print("RUN:", cmd)
    subprocess.check_call(shlex.split(cmd))


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
/tmp/ipykernel_11/2406955956.py in <cell line: 0>()
      8 
      9 
---> 10 _run(
     11     "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
     12 )

/tmp/ipykernel_11/2406955956.py in _run(cmd)
      5 def _run(cmd):
      6     print("RUN:", cmd)
----> 7     subprocess.check_call(shlex.split(cmd))
      8 
      9 

/usr/lib/python3.11/subprocess.py in check_call(*popenargs, **kwargs)
    411         if cmd is None:
    412             cmd = popenargs[0]
--> 413         raise CalledProcessError(retcode, cmd)
    414     return 0
    415 

CalledProcessError: Command '['pip', 'install', '/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl', '--no-index', '--no-deps', '--force-reinstall']' returned non-zero exit status 1.

## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)



## === cell 3
import subprocess

subprocess.check_call(
    [
        "python",
        "-m",
        "src.convert_parquet_to_npy",
        f"--data_dir={DATA_PATH}",
        f"--out_dir={OUT_PATH}",
    ],
    cwd=CODE_PATH,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1938758840.py in <cell line: 0>()
      2 
      3 # Convert parquet->npy as required by the external code (leave core logic unchanged)
----> 4 subprocess.check_call(
      5     [
      6         "python",

/usr/lib/python3.11/subprocess.py in check_call(*popenargs, **kwargs)
    406     check_call(["ls", "-l"])
    407     """
--> 408     retcode = call(*popenargs, **kwargs)
    409     if retcode:
    410         cmd = kwargs.get("args")

/usr/lib/python3.11/subprocess.py in call(timeout, *popenargs, **kwargs)
    387     retcode = call(["ls", "-l"])
    388     """
--> 389     with Popen(*popenargs, **kwargs) as p:
    390         try:
    391             return p.wait(timeout=timeout)

/usr/lib/python3.11/subprocess.py in __init__(self, args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell, cwd, env, universal_newlines, startupinfo, creationflags, restore_signals, start_new_session, pass_fds, user, group, extra_groups, encoding, errors, text, umask, pipesize, process_group)
   1024                             encoding=encoding, errors=errors)
   1025 
-> 1026             self._execute_child(args, executable, preexec_fn, close_fds,
   1027                                 pass_fds, cwd, env,
   1028                                 startupinfo, creationflags, shell,

/usr/lib/python3.11/subprocess.py in _execute_child(self, args, executable, preexec_fn, close_fds, pass_fds, cwd, env, startupinfo, creationflags, shell, p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite, restore_signals, gid, gids, uid, umask, start_new_session, process_group)
   1953                         err_msg = os.strerror(errno_num)
   1954                     if err_filename is not None:
-> 1955                         raise child_exception_type(errno_num, err_msg, err_filename)
   1956                     else:
   1957                         raise child_exception_type(errno_num, err_msg)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms-mk-codes'

## === cell 4
import os

print("Listing /kaggle/input/hms-mk-data (if present):")
print(
    os.listdir("/kaggle/input/hms-mk-data")[:50]
    if os.path.exists("/kaggle/input/hms-mk-data")
    else "NOT FOUND"
)



## === cell 5
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 6
import subprocess
from pathlib import Path

fold_ckpts = [
    "/kaggle/input/hms-mk-data/fold0_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold1_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold2_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold3_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold4_pseudo_log.ckpt",
]

for fold, ckpt_path in enumerate(fold_ckpts):
    cmd = [
        "python",
        "-m",
        "test",
        f"paths.data_dir={DATA_PATH}",
        f"data.test_eegs_dir={OUT_PATH}",
        f"ckpt_path={ckpt_path}",
        "hydra=test",
        f"+model.test_output_dir={OUT_PATH}",
        "+model.net.pretrained=False",
    ]
    subprocess.check_call(cmd, cwd=CODE_PATH)
    src_sub = Path("/kaggle/working/submission.csv")
    dst_sub = Path(f"/kaggle/working/submission_fold{fold}.csv")
    if not src_sub.exists():
        raise FileNotFoundError(
            f"Expected {src_sub} to be created by fold {fold} inference."
        )
    src_sub.replace(dst_sub)

print(
    "Created fold submissions:",
    [p.name for p in Path("/kaggle/working").glob("submission_fold*.csv")],
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2237253560.py in <cell line: 0>()
     23         "+model.net.pretrained=False",
     24     ]
---> 25     subprocess.check_call(cmd, cwd=CODE_PATH)
     26     src_sub = Path("/kaggle/working/submission.csv")
     27     dst_sub = Path(f"/kaggle/working/submission_fold{fold}.csv")

/usr/lib/python3.11/subprocess.py in check_call(*popenargs, **kwargs)
    406     check_call(["ls", "-l"])
    407     """
--> 408     retcode = call(*popenargs, **kwargs)
    409     if retcode:
    410         cmd = kwargs.get("args")

/usr/lib/python3.11/subprocess.py in call(timeout, *popenargs, **kwargs)
    387     retcode = call(["ls", "-l"])
    388     """
--> 389     with Popen(*popenargs, **kwargs) as p:
    390         try:
    391             return p.wait(timeout=timeout)

/usr/lib/python3.11/subprocess.py in __init__(self, args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell, cwd, env, universal_newlines, startupinfo, creationflags, restore_signals, start_new_session, pass_fds, user, group, extra_groups, encoding, errors, text, umask, pipesize, process_group)
   1024                             encoding=encoding, errors=errors)
   1025 
-> 1026             self._execute_child(args, executable, preexec_fn, close_fds,
   1027                                 pass_fds, cwd, env,
   1028                                 startupinfo, creationflags, shell,

/usr/lib/python3.11/subprocess.py in _execute_child(self, args, executable, preexec_fn, close_fds, pass_fds, cwd, env, startupinfo, creationflags, shell, p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite, restore_signals, gid, gids, uid, umask, start_new_session, process_group)
   1953                         err_msg = os.strerror(errno_num)
   1954                     if err_filename is not None:
-> 1955                         raise child_exception_type(errno_num, err_msg, err_filename)
   1956                     else:
   1957                         raise child_exception_type(errno_num, err_msg)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms-mk-codes'

## === cell 7
import numpy as np
import pandas as pd

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print("WARNING: Could not import TARGET_COLS from src.settings due to:", repr(e))
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

TARGET_COLS = list(TARGET_COLS)
ID_COL = "eeg_id"


def merge_preds(folds=(0, 1, 2, 3, 4)):
    preds_list = []
    sol_ref = None

    for fold in folds:
        fp = f"/kaggle/working/submission_fold{fold}.csv"
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing fold prediction file: {fp}")

        sol = pd.read_csv(fp)
        missing = [c for c in ([ID_COL] + TARGET_COLS) if c not in sol.columns]
        if missing:
            raise ValueError(
                f"{fp} missing columns: {missing}. Found: {list(sol.columns)}"
            )

        sol = sol[[ID_COL] + TARGET_COLS].copy()
        sol[TARGET_COLS] = sol[TARGET_COLS].astype("float64")

        if sol_ref is None:
            sol_ref = sol[[ID_COL]].copy()
        else:
            if not sol_ref[ID_COL].equals(sol[ID_COL]):
                sol = (
                    sol.set_index(ID_COL).reindex(sol_ref[ID_COL].values).reset_index()
                )
                if sol[TARGET_COLS].isna().any().any():
                    raise ValueError(
                        f"After reindexing, NaNs appeared in fold {fold} predictions (ID mismatch)."
                    )

        preds_list.append(sol[TARGET_COLS].values)

    preds = np.mean(np.stack(preds_list, axis=0), axis=0)

    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = sol_ref.copy()
    out[TARGET_COLS] = preds
    return out




## === cell 8
sol = merge_preds(folds=(0, 1, 2, 3, 4))
sol.to_csv("/kaggle/working/submission.csv", index=False)

row_sums = sol[TARGET_COLS].sum(axis=1).values
print("submission shape:", sol.shape)
print("min/max row sum:", float(row_sums.min()), float(row_sums.max()))
print(sol.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1018736848.py in <cell line: 0>()
----> 1 sol = merge_preds(folds=(0, 1, 2, 3, 4))
      2 sol.to_csv("/kaggle/working/submission.csv", index=False)
      3 
      4 # Sanity checks for Kaggle submission validity
      5 row_sums = sol[TARGET_COLS].sum(axis=1).values

/tmp/ipykernel_11/3976249523.py in merge_preds(folds)
     27         fp = f"/kaggle/working/submission_fold{fold}.csv"
     28         if not os.path.exists(fp):
---> 29             raise FileNotFoundError(f"Missing fold prediction file: {fp}")
     30 
     31         sol = pd.read_csv(fp)

FileNotFoundError: Missing fold prediction file: /kaggle/working/submission_fold0.csv

## === cell 9
sol

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/397228740.py in <cell line: 0>()
      1 # Display final submission dataframe
----> 2 sol

NameError: name 'sol' is not defined
