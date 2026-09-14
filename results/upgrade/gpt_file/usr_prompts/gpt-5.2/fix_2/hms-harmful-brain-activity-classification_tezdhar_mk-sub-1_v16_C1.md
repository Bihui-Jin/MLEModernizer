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

0.3626664036379633

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
import numpy as np
import pandas as pd

CODE_PATH = "/kaggle/input/hms-mk-codes"
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)



## === cell 1
import subprocess


def _pip_install(path, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", path, "--no-index", "--no-deps"]
    if extra_args:
        cmd += extra_args
    subprocess.run(cmd, check=True)


_pip_install(
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    extra_args=["--force-reinstall"],
)
_pip_install("/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl")
_pip_install("/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
        "--no-deps",
        "--no-index",
    ],
    check=True,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/3522604645.py in <cell line: 0>()
     11 
     12 
---> 13 _pip_install(
     14     "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
     15     extra_args=["--force-reinstall"],

/tmp/ipykernel_11/3522604645.py in _pip_install(path, extra_args)
      8     if extra_args:
      9         cmd += extra_args
---> 10     subprocess.run(cmd, check=True)
     11 
     12 

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command '['/usr/bin/python3', '-m', 'pip', 'install', '/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl', '--no-index', '--no-deps', '--force-reinstall']' returned non-zero exit status 1.

## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)



## === cell 3
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.convert_parquet_to_npy",
        "--data_dir",
        DATA_PATH,
        "--out_dir",
        OUT_PATH,
    ],
    cwd=CODE_PATH,
    check=True,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2740729235.py in <cell line: 0>()
      1 # Convert parquet EEG to npy using provided script (kept same core behavior as original cell)
----> 2 subprocess.run(
      3     [
      4         sys.executable,
      5         "-m",

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    546         kwargs['stderr'] = PIPE
    547 
--> 548     with Popen(*popenargs, **kwargs) as process:
    549         try:
    550             stdout, stderr = process.communicate(input, timeout=timeout)

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
print("Listing /kaggle/input/hms-mk-data (first 50 entries):")
mk_data_path = "/kaggle/input/hms-mk-data"
if os.path.isdir(mk_data_path):
    entries = sorted(os.listdir(mk_data_path))
    print("\n".join(entries[:50]))
else:
    print("Directory not found:", mk_data_path)




## === cell 5
def run_test_and_move(ckpt_path, experiment, out_csv):
    cmd = [
        sys.executable,
        "-m",
        "test",
        f"paths.data_dir={DATA_PATH}",
        f"data.test_eegs_dir={OUT_PATH}",
        f"ckpt_path={ckpt_path}",
        "hydra=test",
        f"+model.test_output_dir={OUT_PATH}",
        f"experiment={experiment}",
        "+model.net.pretrained=False",
    ]
    subprocess.run(cmd, cwd=CODE_PATH, check=True)
    src_csv = os.path.join(OUT_PATH, "submission.csv")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Expected inference output not found: {src_csv}")
    os.replace(src_csv, os.path.join(OUT_PATH, out_csv))


for fold in range(5):
    run_test_and_move(
        ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_pseudo_log.ckpt",
        experiment="conv1d_pseudo",
        out_csv=f"submission_fold{fold}_v0.csv",
    )

for fold in range(5):
    run_test_and_move(
        ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_resv2.ckpt",
        experiment="conv1d_resv2",
        out_csv=f"submission_fold{fold}_v1.csv",
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4023705166.py in <cell line: 0>()
     23 # v0: conv1d_pseudo
     24 for fold in range(5):
---> 25     run_test_and_move(
     26         ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_pseudo_log.ckpt",
     27         experiment="conv1d_pseudo",

/tmp/ipykernel_11/4023705166.py in run_test_and_move(ckpt_path, experiment, out_csv)
     14         "+model.net.pretrained=False",
     15     ]
---> 16     subprocess.run(cmd, cwd=CODE_PATH, check=True)
     17     src_csv = os.path.join(OUT_PATH, "submission.csv")
     18     if not os.path.exists(src_csv):

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    546         kwargs['stderr'] = PIPE
    547 
--> 548     with Popen(*popenargs, **kwargs) as process:
    549         try:
    550             stdout, stderr = process.communicate(input, timeout=timeout)

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

## === cell 6
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


def merge_preds(folds=(0, 1, 2), versions=("v0",), work_dir=OUT_PATH):
    """
    Fixes logic bug: original code overwrote `sol` inside the inner loop and only used last file.
    This averages predictions across all (fold, version) files provided.
    """
    all_preds = []
    base_sol = None

    for fold in folds:
        for version in versions:
            fn = os.path.join(work_dir, f"submission_fold{fold}_{version}.csv")
            if not os.path.exists(fn):
                raise FileNotFoundError(f"Missing prediction file: {fn}")
            sol = pd.read_csv(fn)
            if base_sol is None:
                base_sol = sol[["eeg_id"]].copy()
            else:
                if not np.array_equal(base_sol["eeg_id"].values, sol["eeg_id"].values):
                    sol = (
                        sol.set_index("eeg_id")
                        .loc[base_sol["eeg_id"].values]
                        .reset_index()
                    )
            all_preds.append(sol[TARGET_COLS].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(all_preds, axis=0), axis=0)

    preds = np.clip(preds, 1e-15, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = base_sol.copy()
    out[TARGET_COLS] = preds
    return out




## === cell 7
sol = merge_preds(folds=[0, 1, 2, 3, 4], versions=["v0", "v1"])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2184905050.py in <cell line: 0>()
----> 1 sol = merge_preds(folds=[0, 1, 2, 3, 4], versions=["v0", "v1"])
      2 

/tmp/ipykernel_11/2552104364.py in merge_preds(folds, versions, work_dir)
     25             fn = os.path.join(work_dir, f"submission_fold{fold}_{version}.csv")
     26             if not os.path.exists(fn):
---> 27                 raise FileNotFoundError(f"Missing prediction file: {fn}")
     28             sol = pd.read_csv(fn)
     29             if base_sol is None:

FileNotFoundError: Missing prediction file: /kaggle/working/submission_fold0_v0.csv

## === cell 8
sub_path = "/kaggle/working/submission.csv"
sol = sol[["eeg_id"] + TARGET_COLS].copy()

vals = sol[TARGET_COLS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-15, None)
vals = vals / vals.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = vals

sol.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sol.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2851772283.py in <cell line: 0>()
      1 # Ensure correct column order and write submission.csv
      2 sub_path = "/kaggle/working/submission.csv"
----> 3 sol = sol[["eeg_id"] + TARGET_COLS].copy()
      4 
      5 # Final strict normalization check

NameError: name 'sol' is not defined

## === cell 9
sol.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3448818667.py in <cell line: 0>()
----> 1 sol.head()

NameError: name 'sol' is not defined
