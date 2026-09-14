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

0.78029

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.75645) has done: 'I remove the hard dependency on the missing external code repo (`/kaggle/input/hms-mk-codes`) and replace the broken subprocess-based pipeline with a lightweight, fully in-notebook baseline that reads the provided `train.csv`/`test.csv` and produces a valid probabilistic submission. This fixes the immediate runtime errors (missing directory, missing fold files, undefined `sol`) and guarantees a `submission.csv` with the correct columns and row-wise probabilities summing to 1. To move the KL score toward a reasonable target without changing any unavailable model logic, I use a stable, competition-appropriate fallback: patient-level smoothed label priors learned from train (and global prior for unseen patients), with proper normalization and clipping for numerical safety. This should yield a materially better score than uniform predictions while staying fast and robust within the 600s limit.'
- What this solution (achieved 0.91682) has done: 'Your current baseline is already valid but too weak (KL 0.756 > target 0.339, lower is better), so we make the smallest “still-the-same-idea” upgrade: compute smoothed class priors at the EEG-recording level (by `eeg_id`) instead of the patient level, then fall back to patient-level smoothing, then global prior. This preserves the same core logic (label-prior smoothing from train metadata only) while using a much more directly relevant key for test (`eeg_id` is unique per row and matches the submission index), which should materially reduce KL. We also add a tiny Dirichlet-style additive smoothing on the vote counts before normalization to avoid pathological zeros and improve calibration without changing semantics. The output remains a proper probability distribution per row and writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.78029) has done: 'Your current priors are averaged at the row level (mean of per-row normalized votes), which can overweight small-vote rows and miscalibrate probabilities for KL. I keep the same “metadata-only smoothed priors” core logic, but compute priors by summing vote counts within each group (eeg_id/patient_id) and then normalizing once, which is a minimal semantic fix that usually reduces KL. I also replace the slow per-key loop in `make_smoothed_prior` with a vectorized formulation (same math) to improve stability and runtime, and add a final alignment step to ensure submission order exactly matches `sample_submission.csv`. Output remains a valid probability distribution per row and still writes `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys

CODE_ROOT = "/kaggle/input/hms-mk-codes"
if os.path.isdir(CODE_ROOT):
    sys.path.append(CODE_ROOT)
    sys.path.append(os.path.join(CODE_ROOT, "src"))
    print("Added to sys.path:", CODE_ROOT, "and", os.path.join(CODE_ROOT, "src"))
else:
    print(
        "NOTE: External code repo not found at", CODE_ROOT, "- using fallback pipeline."
    )

print("Python:", sys.version)



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

if req_dir.exists():
    for whl, extra in wheels:
        p = req_dir / whl
        if p.exists():
            cmd = [sys.executable, "-m", "pip", "install", str(p)] + extra
            print("Running:", " ".join(cmd))
            subprocess.check_call(cmd)
        else:
            print(f"Wheel not found, skipping install: {p}")
else:
    print("NOTE: requirements-mk not found; skipping wheel installs.")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

train_csv = os.path.join(DATA_PATH, "train.csv")
test_csv = os.path.join(DATA_PATH, "test.csv")
sample_csv = os.path.join(DATA_PATH, "sample_submission.csv")

for p in [train_csv, test_csv, sample_csv]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

print("DATA_PATH:", DATA_PATH)
print("OUT_PATH:", OUT_PATH)



## === cell 3
import os

print("Listing /kaggle/working (OUT_PATH):")
print(sorted(os.listdir(OUT_PATH))[:50])



## === cell 4
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 5
import numpy as np
import pandas as pd

train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
sample = pd.read_csv(sample_csv)

target_cols = [c for c in sample.columns if c != "eeg_id"]
if target_cols != [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]:
    if len(target_cols) != 6:
        raise ValueError(
            f"Expected 6 target columns, got {len(target_cols)}: {target_cols}"
        )

eps_count = 0.5  # additive smoothing on counts to avoid zeros and improve KL stability

train_counts = train[["eeg_id", "patient_id"] + target_cols].copy()
for c in target_cols:
    train_counts[c] = train_counts[c].astype(np.float64)

global_counts = (
    train_counts[target_cols].sum(axis=0).to_numpy(dtype=np.float64) + eps_count
)
global_mean = global_counts / global_counts.sum()
global_mean = np.clip(global_mean, 1e-15, 1.0)
global_mean = global_mean / global_mean.sum()


def make_smoothed_prior_from_counts(group_key: str, alpha: float) -> pd.DataFrame:
    grp_counts = train_counts.groupby(group_key)[target_cols].sum().astype(np.float64)
    grp_total = grp_counts.sum(axis=1).to_numpy(dtype=np.float64)

    grp_counts_sm = grp_counts.to_numpy(dtype=np.float64) + eps_count

    grp_prob = grp_counts_sm / grp_counts_sm.sum(axis=1, keepdims=True)

    n = grp_total.reshape(-1, 1)
    post = (n * grp_prob + alpha * global_mean.reshape(1, -1)) / (n + alpha)

    post = np.clip(post, 1e-15, 1.0)
    post = post / post.sum(axis=1, keepdims=True)

    out = pd.DataFrame(post, index=grp_counts.index, columns=target_cols)
    return out


alpha_eeg = 10.0
alpha_patient = 20.0

eeg_smoothed = make_smoothed_prior_from_counts("eeg_id", alpha=alpha_eeg)
patient_smoothed = make_smoothed_prior_from_counts("patient_id", alpha=alpha_patient)

preds = np.zeros((len(test), len(target_cols)), dtype=np.float64)
test_eeg_ids = test["eeg_id"].to_numpy()
test_pids = test["patient_id"].to_numpy()

eeg_index = eeg_smoothed.index
patient_index = patient_smoothed.index

for i, (eid, pid) in enumerate(zip(test_eeg_ids, test_pids)):
    if eid in eeg_index:
        preds[i] = eeg_smoothed.loc[eid, target_cols].to_numpy(dtype=np.float64)
    elif pid in patient_index:
        preds[i] = patient_smoothed.loc[pid, target_cols].to_numpy(dtype=np.float64)
    else:
        preds[i] = global_mean

preds = np.clip(preds, 1e-15, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

submission_fold0 = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for j, c in enumerate(target_cols):
    submission_fold0[c] = preds[:, j]

fold0_path = os.path.join(OUT_PATH, "submission_fold0.csv")
submission_fold0.to_csv(fold0_path, index=False)
print("Wrote fallback fold submission:", fold0_path, "rows:", len(submission_fold0))



## === cell 6
import pandas as pd
import numpy as np
import pathlib


def merge_preds(folds=(0,), data_path=DATA_PATH, workdir="/kaggle/working"):
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

    out = sample[["eeg_id"]].merge(out, on="eeg_id", how="left")
    out[target_cols] = out[target_cols].to_numpy(dtype=np.float64)
    out[target_cols] = np.clip(out[target_cols].to_numpy(dtype=np.float64), 1e-15, 1.0)
    out[target_cols] = out[target_cols].to_numpy(dtype=np.float64) / out[
        target_cols
    ].sum(axis=1, keepdims=True)
    return out


sol = merge_preds()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2387190839.py in <cell line: 0>()
     52 
     53 
---> 54 sol = merge_preds()
     55 

/tmp/ipykernel_11/2387190839.py in merge_preds(folds, data_path, workdir)
     48     out[target_cols] = out[target_cols].to_numpy(dtype=np.float64) / out[
     49         target_cols
---> 50     ].sum(axis=1, keepdims=True)
     51     return out
     52 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in sum(self, axis, skipna, numeric_only, min_count, **kwargs)
  11668         **kwargs,
  11669     ):
> 11670         result = super().sum(axis, skipna, numeric_only, min_count, **kwargs)
  11671         return result.__finalize__(self, method="sum")
  11672 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in sum(self, axis, skipna, numeric_only, min_count, **kwargs)
  12504         **kwargs,
  12505     ):
> 12506         return self._min_count_stat_function(
  12507             "sum", nanops.nansum, axis, skipna, numeric_only, min_count, **kwargs
  12508         )

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _min_count_stat_function(self, name, func, axis, skipna, numeric_only, min_count, **kwargs)
  12469     ):
  12470         assert name in ["sum", "prod"], name
> 12471         nv.validate_func(name, (), kwargs)
  12472 
  12473         validate_bool_kwarg(skipna, "skipna", none_allowed=False)

/usr/local/lib/python3.11/dist-packages/pandas/compat/numpy/function.py in validate_func(fname, args, kwargs)
    416 
    417     validation_func = _validation_funcs[fname]
--> 418     return validation_func(args, kwargs)

/usr/local/lib/python3.11/dist-packages/pandas/compat/numpy/function.py in __call__(self, args, kwargs, fname, max_fname_arg_count, method)
     86             validate_kwargs(fname, kwargs, self.defaults)
     87         elif method == "both":
---> 88             validate_args_and_kwargs(
     89                 fname, args, kwargs, max_fname_arg_count, self.defaults
     90             )

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in validate_args_and_kwargs(fname, args, kwargs, max_fname_arg_count, compat_args)
    221 
    222     kwargs.update(args_dict)
--> 223     validate_kwargs(fname, kwargs, compat_args)
    224 
    225 

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in validate_kwargs(fname, kwargs, compat_args)
    163     kwds = kwargs.copy()
    164     _check_for_invalid_keys(fname, kwargs, compat_args)
--> 165     _check_for_default_values(fname, kwds, compat_args)
    166 
    167 

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in _check_for_default_values(fname, arg_val_dict, compat_args)
     79 
     80         if not match:
---> 81             raise ValueError(
     82                 f"the '{key}' parameter is not supported in "
     83                 f"the pandas implementation of {fname}()"

ValueError: the 'keepdims' parameter is not supported in the pandas implementation of sum()

## === cell 7
out_path = "/kaggle/working/submission.csv"
sol.to_csv(out_path, index=False)

target_cols = [c for c in sol.columns if c != "eeg_id"]
row_sums = sol[target_cols].sum(axis=1).values
print("Saved:", out_path)
print("Rows:", len(sol), "Cols:", list(sol.columns))
print("Row-sum min/max:", float(row_sums.min()), float(row_sums.max()))
print(sol.head())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/543042759.py in <cell line: 0>()
      1 out_path = "/kaggle/working/submission.csv"
----> 2 sol.to_csv(out_path, index=False)
      3 
      4 target_cols = [c for c in sol.columns if c != "eeg_id"]
      5 row_sums = sol[target_cols].sum(axis=1).values

NameError: name 'sol' is not defined

## === cell 8
sol

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1793929092.py in <cell line: 0>()
----> 1 sol

NameError: name 'sol' is not defined
