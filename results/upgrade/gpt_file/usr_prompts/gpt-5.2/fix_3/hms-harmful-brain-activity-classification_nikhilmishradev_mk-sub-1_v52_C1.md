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

0.3070588873339696

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
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd



## === cell 1
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
OUT_PATH2 = "/kaggle/working/v2"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
Path(OUT_PATH2).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None):
    """Run a shell command and raise if it fails (so we fail fast)."""
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)




## === cell 2
sys.path.append("/kaggle/input/hms-mk-codes/")



## === cell 3
run_cmd(
    "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
)
run_cmd(
    "pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
)
run_cmd(
    "pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
)
run_cmd(
    "pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/977030038.py in <cell line: 0>()
      1 # NOTE: In a Kaggle notebook these were `!pip install ...`.
      2 # Here we run them via subprocess for script-compatibility.
----> 3 run_cmd(
      4     "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
      5 )

/tmp/ipykernel_11/2570128624.py in run_cmd(cmd, cwd)
     11     """Run a shell command and raise if it fails (so we fail fast)."""
     12     print("Running:", cmd)
---> 13     subprocess.run(cmd, shell=True, cwd=cwd, check=True)
     14 
     15 

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command 'pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall' returned non-zero exit status 1.

## === cell 4
run_cmd(
    f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
    cwd="/kaggle/input/hms-mk-codes",
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2357605841.py in <cell line: 0>()
      1 # Convert parquet to npy using codebase v1
----> 2 run_cmd(
      3     f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
      4     cwd="/kaggle/input/hms-mk-codes",
      5 )

/tmp/ipykernel_11/2570128624.py in run_cmd(cmd, cwd)
     11     """Run a shell command and raise if it fails (so we fail fast)."""
     12     print("Running:", cmd)
---> 13     subprocess.run(cmd, shell=True, cwd=cwd, check=True)
     14 
     15 

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

## === cell 5
mk_data_dir = Path("/kaggle/input/hms-mk-data")
print("Exists /kaggle/input/hms-mk-data:", mk_data_dir.exists())
if mk_data_dir.exists():
    print("Sample files:", sorted([p.name for p in mk_data_dir.glob("*")])[:10])



## === cell 6
for fold in [0, 1, 2, 3, 4]:
    run_cmd(
        "python -m test "
        f"paths.data_dir={DATA_PATH} "
        f"data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path=/kaggle/input/hms-mk-data/fold{fold}_levit_pseudo.ckpt "
        "hydra=test "
        f"+model.test_output_dir={OUT_PATH} "
        "experiment=conv1d_tfm2d_pseudo "
        "+model.net.pretrained=False",
        cwd="/kaggle/input/hms-mk-codes",
    )
    src = Path("/kaggle/working/submission.csv")
    dst = Path(f"/kaggle/working/submission_fold{fold}_v0.csv")
    if src.exists():
        src.replace(dst)
    else:
        raise FileNotFoundError(f"Expected {src} to be created by inference.")

for fold in [0, 1, 2, 3, 4]:
    run_cmd(
        "python -m test "
        f"paths.data_dir={DATA_PATH} "
        f"data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path=/kaggle/input/hms-mk-data/fold{fold}_effb3_sim_pseudo.ckpt "
        "hydra=test "
        f"+model.test_output_dir={OUT_PATH} "
        "experiment=conv1d_effv2_pseudo "
        "+model.net.pretrained=False",
        cwd="/kaggle/input/hms-mk-codes",
    )
    src = Path("/kaggle/working/submission.csv")
    dst = Path(f"/kaggle/working/submission_fold{fold}_v3.csv")
    if src.exists():
        src.replace(dst)
    else:
        raise FileNotFoundError(f"Expected {src} to be created by inference.")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/22719758.py in <cell line: 0>()
      1 # Run test inference for v0 and v3 models (fold 0-4), saving fold CSVs
      2 for fold in [0, 1, 2, 3, 4]:
----> 3     run_cmd(
      4         "python -m test "
      5         f"paths.data_dir={DATA_PATH} "

/tmp/ipykernel_11/2570128624.py in run_cmd(cmd, cwd)
     11     """Run a shell command and raise if it fails (so we fail fast)."""
     12     print("Running:", cmd)
---> 13     subprocess.run(cmd, shell=True, cwd=cwd, check=True)
     14 
     15 

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

## === cell 7
try:
    sys.path.remove("/kaggle/input/hms-mk-codes/")
except ValueError:
    pass
sys.path.append("/kaggle/input/hms-mk-codesv2")



## === cell 8
run_cmd(
    f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH2}",
    cwd="/kaggle/input/hms-mk-codesv2",
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2811552115.py in <cell line: 0>()
      1 # Convert parquet to npy using codebase v2
----> 2 run_cmd(
      3     f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH2}",
      4     cwd="/kaggle/input/hms-mk-codesv2",
      5 )

/tmp/ipykernel_11/2570128624.py in run_cmd(cmd, cwd)
     11     """Run a shell command and raise if it fails (so we fail fast)."""
     12     print("Running:", cmd)
---> 13     subprocess.run(cmd, shell=True, cwd=cwd, check=True)
     14 
     15 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms-mk-codesv2'

## === cell 9
for fold in [0, 1, 2, 3, 4]:
    run_cmd(
        "python -m test "
        f"paths.data_dir={DATA_PATH} "
        "data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG "
        f"data.test_dataset.eeg_dir={OUT_PATH2}/test_eegs "
        f"ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold{fold}.ckpt "
        "hydra=test "
        f"+model.test_output_dir={OUT_PATH} "
        "experiment=clean_tfm_corr_pseudo "
        "data.num_workers=2 "
        "+model.net.pretrained=False",
        cwd="/kaggle/input/hms-mk-codesv2",
    )
    src = Path("/kaggle/working/submission.csv")
    dst = Path(f"/kaggle/working/submission_fold{fold}_v4.csv")
    if src.exists():
        src.replace(dst)
    else:
        raise FileNotFoundError(f"Expected {src} to be created by inference.")

for fold in [0, 1, 2, 3, 4]:
    run_cmd(
        "python -m test "
        f"paths.data_dir={DATA_PATH} "
        "data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG "
        f"data.test_dataset.eeg_dir={OUT_PATH2}/test_eegs "
        f"ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold{fold}.ckpt "
        "hydra=test "
        f"+model.test_output_dir={OUT_PATH} "
        "experiment=clean_effb1_corr_pseudo "
        "data.num_workers=2 "
        "+model.net.pretrained=False",
        cwd="/kaggle/input/hms-mk-codesv2",
    )
    src = Path("/kaggle/working/submission.csv")
    dst = Path(f"/kaggle/working/submission_fold{fold}_v5.csv")
    if src.exists():
        src.replace(dst)
    else:
        raise FileNotFoundError(f"Expected {src} to be created by inference.")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/140185788.py in <cell line: 0>()
      1 # Run test inference for v4 and v5 models (fold 0-4), saving fold CSVs
      2 for fold in [0, 1, 2, 3, 4]:
----> 3     run_cmd(
      4         "python -m test "
      5         f"paths.data_dir={DATA_PATH} "

/tmp/ipykernel_11/2570128624.py in run_cmd(cmd, cwd)
     11     """Run a shell command and raise if it fails (so we fail fast)."""
     12     print("Running:", cmd)
---> 13     subprocess.run(cmd, shell=True, cwd=cwd, check=True)
     14 
     15 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms-mk-codesv2'

## === cell 10
p1 = Path("/kaggle/working/submission_fold1_v4.csv")
p2 = Path("/kaggle/working/submission_fold1_v5.csv")
if p1.exists():
    print("Head:", p1)
    print(pd.read_csv(p1, nrows=5).head())
if p2.exists():
    print("Head:", p2)
    print(pd.read_csv(p2, nrows=5).head())



## === cell 11
DEFAULT_TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _get_target_cols():
    ss_path = f"{DATA_PATH}/sample_submission.csv"
    if os.path.exists(ss_path):
        ss = pd.read_csv(ss_path, nrows=1)
        cols = [c for c in ss.columns if c != "eeg_id"]
        if len(cols) == 6:
            return cols
    return DEFAULT_TARGET_COLS


TARGET_COLS = _get_target_cols()


def merge_preds(folds, versions, weights):
    """
    Robust merge: align on eeg_id; skip any missing per-fold files (prevents FileNotFoundError).
    Then clip + renormalize to ensure valid probabilities for KL divergence.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    base = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")[["eeg_id"]].copy()
    pred_sum = np.zeros((len(base), len(TARGET_COLS)), dtype=np.float64)
    used = 0

    for fold in folds:
        for w, version in zip(weights, versions):
            path = Path(f"/kaggle/working/submission_fold{fold}_{version}.csv")
            if not path.exists():
                print("WARNING: missing prediction file, skipping:", str(path))
                continue

            sol = pd.read_csv(path)
            need_cols = ["eeg_id"] + TARGET_COLS
            missing_cols = [c for c in need_cols if c not in sol.columns]
            if missing_cols:
                raise ValueError(f"{path} is missing columns: {missing_cols}")

            aligned = base.merge(
                sol[need_cols], on="eeg_id", how="left", validate="one_to_one"
            )
            vals = aligned[TARGET_COLS].to_numpy(dtype=np.float64)

            if np.isnan(vals).any():
                vals = np.nan_to_num(vals, nan=1.0 / len(TARGET_COLS))

            pred_sum += w * vals
            used += 1

    if used == 0:
        raise RuntimeError("No prediction CSVs were found; cannot ensemble.")

    pred_sum = np.clip(pred_sum, 1e-12, None)
    pred_sum = pred_sum / pred_sum.sum(axis=1, keepdims=True)

    out = base.copy()
    out[TARGET_COLS] = pred_sum
    return out




## === cell 12
sol = merge_preds(
    folds=[0, 1, 2, 3, 4],
    versions=["v0", "v3", "v4", "v5"],
    weights=[0.25, 0.25, 0.25, 0.25],
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3887200726.py in <cell line: 0>()
      1 # Keep original ensemble intent: equal weighting across (v0, v3, v4, v5) and folds.
----> 2 sol = merge_preds(
      3     folds=[0, 1, 2, 3, 4],
      4     versions=["v0", "v3", "v4", "v5"],
      5     weights=[0.25, 0.25, 0.25, 0.25],

/tmp/ipykernel_11/923516892.py in merge_preds(folds, versions, weights)
     60 
     61     if used == 0:
---> 62         raise RuntimeError("No prediction CSVs were found; cannot ensemble.")
     63 
     64     pred_sum = np.clip(pred_sum, 1e-12, None)

RuntimeError: No prediction CSVs were found; cannot ensemble.

## === cell 13
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape=", sol.shape)

assert list(sol.columns) == ["eeg_id"] + TARGET_COLS, "Submission columns mismatch."
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))
assert np.all(np.isfinite(row_sums)), "Non-finite row sums."
assert np.max(np.abs(row_sums - 1.0)) < 1e-6, "Row probabilities do not sum to 1."
assert sol["eeg_id"].nunique() == len(sol), "Duplicate eeg_id in submission."
print(sol.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4255905911.py in <cell line: 0>()
      1 # Write final submission (must be /kaggle/working/submission.csv), and validate format.
      2 sub_path = "/kaggle/working/submission.csv"
----> 3 sol.to_csv(sub_path, index=False)
      4 print("Wrote:", sub_path, "shape=", sol.shape)
      5 

NameError: name 'sol' is not defined
