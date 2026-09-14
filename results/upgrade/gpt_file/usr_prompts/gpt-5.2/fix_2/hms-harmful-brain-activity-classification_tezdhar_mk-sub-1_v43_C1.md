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

0.3045701602754038

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
if os.path.isdir(MK_CODES_PATH) and MK_CODES_PATH not in sys.path:
    sys.path.append(MK_CODES_PATH)



## === cell 1
import subprocess


def _pip_install_if_exists(whl_path, extra_args=None):
    if extra_args is None:
        extra_args = []
    if os.path.exists(whl_path):
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            whl_path,
            "--no-index",
            "--no-deps",
        ] + extra_args
        subprocess.check_call(cmd)
    else:
        print(f"[WARN] Wheel not found, skipping install: {whl_path}")


_pip_install_if_exists(
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    extra_args=["--force-reinstall"],
)
_pip_install_if_exists("/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl")
_pip_install_if_exists(
    "/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl"
)
_pip_install_if_exists(
    "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
    extra_args=["--no-deps"],
)



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)



## === cell 3
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "src.convert_parquet_to_npy",
        "--data_dir",
        DATA_PATH,
        "--out_dir",
        OUT_PATH,
    ],
    cwd=MK_CODES_PATH,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1495471734.py in <cell line: 0>()
      1 # Convert parquet to npy using provided external repo script (core logic preserved).
      2 # Use subprocess so it works in a .py-like environment too.
----> 3 subprocess.check_call(
      4     [
      5         sys.executable,

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
import glob

print("Listing /kaggle/input/hms-mk-data (first 50):")
for p in sorted(glob.glob("/kaggle/input/hms-mk-data/*"))[:50]:
    print(" ", p)




## === cell 5
def run_test_and_move(ckpt_path, experiment, fold, version):
    subprocess.check_call(
        [
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
        ],
        cwd=MK_CODES_PATH,
    )
    src_csv = os.path.join(OUT_PATH, "submission.csv")
    dst_csv = os.path.join(OUT_PATH, f"submission_fold{fold}_{version}.csv")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Expected submission not found at: {src_csv}")
    os.replace(src_csv, dst_csv)
    return dst_csv


for fold in range(5):
    run_test_and_move(
        ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_levit_pseudo.ckpt",
        experiment="conv1d_tfm2d_pseudo",
        fold=fold,
        version="v0",
    )

for fold in range(5):
    run_test_and_move(
        ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_pseudo_resv2.ckpt",
        experiment="conv1d_resv2",
        fold=fold,
        version="v2",
    )

for fold in range(5):
    run_test_and_move(
        ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_effb3_sim_pseudo.ckpt",
        experiment="conv1d_effv2_pseudo",
        fold=fold,
        version="v3",
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1138476759.py in <cell line: 0>()
     26 # LeViT pseudo (v0)
     27 for fold in range(5):
---> 28     run_test_and_move(
     29         ckpt_path=f"/kaggle/input/hms-mk-data/fold{fold}_levit_pseudo.ckpt",
     30         experiment="conv1d_tfm2d_pseudo",

/tmp/ipykernel_11/1138476759.py in run_test_and_move(ckpt_path, experiment, fold, version)
      1 # Run inference for all folds/experiments (core logic preserved), then rename outputs.
      2 def run_test_and_move(ckpt_path, experiment, fold, version):
----> 3     subprocess.check_call(
      4         [
      5             sys.executable,

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

## === cell 6
import pandas as pd
import numpy as np

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print(
        f"[WARN] Could not import TARGET_COLS from src.settings ({e}). Falling back to default target cols."
    )
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2", "v3"),
    weights=(0.3, 0.3, 0.4),
    base_submission_path=f"{DATA_PATH}/sample_submission.csv",
    work_dir=OUT_PATH,
    eps=1e-12,
):
    sol = pd.read_csv(base_submission_path)
    if "eeg_id" not in sol.columns:
        raise ValueError("sample_submission.csv missing 'eeg_id'")

    eeg_ids = sol["eeg_id"].values
    pred_sum = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)

    for fold in folds:
        for w, ver in zip(weights, versions):
            fn = os.path.join(work_dir, f"submission_fold{fold}_{ver}.csv")
            if not os.path.exists(fn):
                raise FileNotFoundError(f"Missing prediction file: {fn}")

            df = pd.read_csv(fn)

            if "eeg_id" not in df.columns:
                raise ValueError(f"{fn} missing 'eeg_id'")
            missing_cols = [c for c in TARGET_COLS if c not in df.columns]
            if missing_cols:
                raise ValueError(f"{fn} missing target cols: {missing_cols}")

            df = df.set_index("eeg_id").reindex(eeg_ids)
            if df.isna().any().any():
                nan_rows = int(df.isna().any(axis=1).sum())
                raise ValueError(
                    f"{fn} has missing predictions after align: {nan_rows} rows"
                )

            pred_sum += df[TARGET_COLS].to_numpy(dtype=np.float64) * float(w)

    pred_sum = np.clip(pred_sum, eps, None)
    pred_sum = pred_sum / pred_sum.sum(axis=1, keepdims=True)

    sol[TARGET_COLS] = pred_sum
    return sol




## === cell 7
sol = merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2", "v3"),
    weights=(0.3, 0.3, 0.4),
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1806948165.py in <cell line: 0>()
----> 1 sol = merge_preds(
      2     folds=(0, 1, 2, 3, 4),
      3     versions=("v0", "v2", "v3"),
      4     weights=(0.3, 0.3, 0.4),
      5 )

/tmp/ipykernel_11/4272746218.py in merge_preds(folds, versions, weights, base_submission_path, work_dir, eps)
     39             fn = os.path.join(work_dir, f"submission_fold{fold}_{ver}.csv")
     40             if not os.path.exists(fn):
---> 41                 raise FileNotFoundError(f"Missing prediction file: {fn}")
     42 
     43             df = pd.read_csv(fn)

FileNotFoundError: Missing prediction file: /kaggle/working/submission_fold0_v0.csv

## === cell 8
out_file = "/kaggle/working/submission.csv"
sol.to_csv(out_file, index=False)
print("Wrote:", out_file)
print("Shape:", sol.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/793454035.py in <cell line: 0>()
      1 out_file = "/kaggle/working/submission.csv"
----> 2 sol.to_csv(out_file, index=False)
      3 print("Wrote:", out_file)
      4 print("Shape:", sol.shape)
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
