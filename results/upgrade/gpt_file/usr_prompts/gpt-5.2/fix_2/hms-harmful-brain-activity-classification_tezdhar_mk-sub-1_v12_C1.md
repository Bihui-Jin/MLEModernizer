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

0.3385842184983305

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

CODE_ROOT = "/kaggle/input/hms-mk-codes"
sys.path.append(CODE_ROOT)
sys.path.append(os.path.join(CODE_ROOT, "src"))

print("Python:", sys.version)
print("Added to sys.path:", CODE_ROOT, "and", os.path.join(CODE_ROOT, "src"))



## === cell 1
import pathlib
import subprocess

req_dir = pathlib.Path("/kaggle/input/requirements-mk")
wheels = [
    (
        "antlr4_python3_runtime-4.9.2-py3-none-any.whl",
        ["--no-index", "--no-deps", "--force-reinstall"],
    ),
    ("omegaconf-2.3.0-py3-none-any.whl", ["--no-index", "--no-deps"]),
    ("hydra_core-1.3.2-py3-none-any.whl", ["--no-index", "--no-deps"]),
    ("lightning-2.2.1-py3-none-any.whl", ["--no-deps", "--no-index"]),
]

for whl, extra in wheels:
    p = req_dir / whl
    if p.exists():
        cmd = [sys.executable, "-m", "pip", "install", str(p)] + extra
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print(f"Wheel not found, skipping install: {p}")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 3
import subprocess

cmd = [
    sys.executable,
    "-m",
    "src.convert_parquet_to_npy",
    f"--data_dir={DATA_PATH}",
    f"--out_dir={OUT_PATH}",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd, cwd="/kaggle/input/hms-mk-codes")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3157411047.py in <cell line: 0>()
     10 ]
     11 print("Running:", " ".join(cmd))
---> 12 subprocess.check_call(cmd, cwd="/kaggle/input/hms-mk-codes")
     13 

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

print("Listing /kaggle/working (OUT_PATH):")
print(sorted(os.listdir(OUT_PATH))[:50])



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
import pathlib
import shutil

fold_ckpts = [
    "/kaggle/input/hms-mk-data/fold0_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold1_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold2_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold3_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold4_pseudo_log.ckpt",
]

for i, ckpt in enumerate(fold_ckpts):
    cmd = [
        sys.executable,
        "-m",
        "test",
        f"paths.data_dir={DATA_PATH}",
        f"data.test_eegs_dir={OUT_PATH}",
        f"ckpt_path={ckpt}",
        "hydra=test",
        f"+model.test_output_dir={OUT_PATH}",
        "+model.net.pretrained=False",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd="/kaggle/input/hms-mk-codes")

    src_sub = pathlib.Path("/kaggle/working/submission.csv")
    dst_sub = pathlib.Path(f"/kaggle/working/submission_fold{i}.csv")
    if not src_sub.exists():
        raise FileNotFoundError(
            f"Expected {src_sub} to be created by fold {i} inference, but it was not found."
        )
    shutil.move(str(src_sub), str(dst_sub))
    print("Wrote:", dst_sub)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/366684676.py in <cell line: 0>()
     25     ]
     26     print("Running:", " ".join(cmd))
---> 27     subprocess.check_call(cmd, cwd="/kaggle/input/hms-mk-codes")
     28 
     29     src_sub = pathlib.Path("/kaggle/working/submission.csv")

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
import pandas as pd
import numpy as np
import pathlib


def merge_preds(folds=(0, 1, 2, 3, 4), data_path=DATA_PATH, workdir="/kaggle/working"):
    sample = pd.read_csv(f"{data_path}/sample_submission.csv")
    target_cols = [c for c in sample.columns if c != "eeg_id"]
    if len(target_cols) != 6:
        raise ValueError(
            f"Expected 6 target columns, got {len(target_cols)}: {target_cols}"
        )

    preds_list = []
    base_df = None

    for fold in folds:
        fpath = pathlib.Path(workdir) / f"submission_fold{fold}.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing fold submission file: {fpath}")
        df = pd.read_csv(fpath)

        if "eeg_id" not in df.columns:
            raise ValueError(f"{fpath} missing 'eeg_id' column.")
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{fpath} missing target columns: {missing}")

        if base_df is None:
            base_df = df[["eeg_id"] + target_cols].copy()
        else:
            if not np.array_equal(base_df["eeg_id"].values, df["eeg_id"].values):
                df = df.set_index("eeg_id").loc[base_df["eeg_id"].values].reset_index()

        preds_list.append(df[target_cols].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(preds_list, axis=0), axis=0)

    preds = np.clip(preds, 1e-15, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = base_df.copy()
    out[target_cols] = preds
    return out


sol = merge_preds()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2138677924.py in <cell line: 0>()
     51 
     52 
---> 53 sol = merge_preds()
     54 

/tmp/ipykernel_11/2138677924.py in merge_preds(folds, data_path, workdir)
     20         fpath = pathlib.Path(workdir) / f"submission_fold{fold}.csv"
     21         if not fpath.exists():
---> 22             raise FileNotFoundError(f"Missing fold submission file: {fpath}")
     23         df = pd.read_csv(fpath)
     24 

FileNotFoundError: Missing fold submission file: /kaggle/working/submission_fold0.csv

## === cell 8
out_path = "/kaggle/working/submission.csv"
sol.to_csv(out_path, index=False)

target_cols = [c for c in sol.columns if c != "eeg_id"]
row_sums = sol[target_cols].sum(axis=1).values
print("Saved:", out_path)
print("Rows:", len(sol), "Cols:", list(sol.columns))
print("Row-sum min/max:", float(row_sums.min()), float(row_sums.max()))
print(sol.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/157971599.py in <cell line: 0>()
      1 # Write final submission with required name and validate row-wise sums.
      2 out_path = "/kaggle/working/submission.csv"
----> 3 sol.to_csv(out_path, index=False)
      4 
      5 # Validation checks

NameError: name 'sol' is not defined

## === cell 9
sol

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1793929092.py in <cell line: 0>()
----> 1 sol

NameError: name 'sol' is not defined
