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

0.323113851522087

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.39779) has done: 'Your run fails because the fold prediction CSVs are never created: the earlier `python -m test ...` commands depend on external Kaggle inputs (`/kaggle/input/hms-mk-codes` and `/kaggle/input/hms-mk-data`) that are not present in this environment, so the subsequent merge can’t find `/kaggle/working/submission_fold*_v3.csv`. To make the notebook run end-to-end and always produce a valid `submission.csv`, I add guards that detect missing external assets and skip those steps instead of crashing. When predictions aren’t available, the script falls back to a safe baseline using the class prior from `train.csv` (normalized vote counts), ensuring rows sum to 1 and the submission format is correct. The merge function is also made robust to either produce an averaged ensemble (when files exist) or the baseline (when they don’t), which should yield a reasonable (not necessarily optimal) KL score rather than “Not yielded”.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
from pathlib import Path


def _in_ipython():
    try:
        get_ipython  # noqa: F401
        return True
    except Exception:
        return False


MK_CODES_PATH = "/kaggle/input/hms-mk-codes/"
if os.path.isdir(MK_CODES_PATH):
    sys.path.append(MK_CODES_PATH)



## === cell 1
if _in_ipython():
    wheels_dir = "/kaggle/input/requirements-mk"
    if os.path.isdir(wheels_dir):
        get_ipython().run_line_magic(
            "pip",
            f"install {wheels_dir}/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall",
        )
        get_ipython().run_line_magic(
            "pip",
            f"install {wheels_dir}/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps",
        )
        get_ipython().run_line_magic(
            "pip",
            f"install {wheels_dir}/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps",
        )
        get_ipython().run_line_magic(
            "pip",
            f"install {wheels_dir}/lightning-2.2.1-py3-none-any.whl --no-deps --no-index",
        )
    else:
        print(f"[WARN] {wheels_dir} not found; skipping custom wheel installs.")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)



## === cell 3
if _in_ipython() and os.path.isdir(MK_CODES_PATH):
    get_ipython().system(
        f"cd {MK_CODES_PATH} && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
    )
else:
    print("[WARN] hms-mk-codes not available; skipping parquet->npy conversion.")



## === cell 4
if _in_ipython():
    if os.path.exists("/kaggle/input/hms-mk-data"):
        get_ipython().system("ls /kaggle/input/hms-mk-data")
    else:
        print(
            "[WARN] /kaggle/input/hms-mk-data not found; cannot list model checkpoints."
        )




## === cell 5
def _run_fold_infer(fold: int, version: str = "v3") -> bool:
    if not _in_ipython():
        return False
    if not os.path.isdir(MK_CODES_PATH):
        return False
    ckpt = f"/kaggle/input/hms-mk-data/fold{fold}_effb3_sim_pseudo.ckpt"
    if not os.path.exists(ckpt):
        return False

    cmd = (
        f"cd {MK_CODES_PATH} && python -m test "
        f"paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path={ckpt} hydra=test +model.test_output_dir={OUT_PATH} "
        f"experiment=conv1d_effb1_v3_pseudo +model.net.pretrained=False"
    )
    get_ipython().system(cmd)
    src = f"{OUT_PATH}/submission.csv"
    dst = f"{OUT_PATH}/submission_fold{fold}_{version}.csv"
    if os.path.exists(src):
        get_ipython().system(f"mv {src} {dst}")
        return True
    return False


any_fold = False
for f in [0, 1, 2, 3, 4]:
    ok = _run_fold_infer(f, version="v3")
    any_fold = any_fold or ok

if not any_fold:
    print(
        "[WARN] No fold prediction files were generated (missing repo/checkpoints). Will use baseline submission."
    )



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


def _baseline_from_train_priors(data_path: str) -> np.ndarray:
    """
    Score-friendly fallback (no test leakage): use global class prior from train votes.
    Produces a valid probability vector summing to 1.
    """
    train_path = os.path.join(data_path, "train.csv")
    train = pd.read_csv(train_path, usecols=TARGET_COLS)
    votes = train[TARGET_COLS].to_numpy(dtype=np.float64)
    row_sums = votes.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = votes / row_sums
    prior = probs.mean(axis=0)
    prior = np.clip(prior, 1e-12, 1.0)
    prior = prior / prior.sum()
    return prior


def merge_preds(folds=(0, 1, 2), versions=("v0",), weights=(1.0,)):
    """
    Merge per-fold submissions by weighted averaging and renormalize to sum to 1.
    Bugfix: if expected fold files are missing, fall back to a valid baseline submission
    rather than raising (so we always yield submission.csv end-to-end).
    """
    sample_path = os.path.join(DATA_PATH, "sample_submission.csv")
    sol = pd.read_csv(sample_path)

    preds_sum = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)
    used = 0

    for fold in folds:
        for weight, version in zip(weights, versions):
            path = f"{OUT_PATH}/submission_fold{fold}_{version}.csv"
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)

            if "eeg_id" in df.columns and not df["eeg_id"].equals(sol["eeg_id"]):
                df = df.set_index("eeg_id").loc[sol["eeg_id"]].reset_index()

            preds_sum += df[TARGET_COLS].to_numpy(dtype=np.float64) * float(weight)
            used += 1

    if used == 0:
        prior = _baseline_from_train_priors(DATA_PATH)
        preds = np.tile(prior, (len(sol), 1))
        sol[TARGET_COLS] = preds
        return sol

    preds = preds_sum / used

    preds = np.clip(preds, 1e-12, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    sol[TARGET_COLS] = preds
    return sol




## === cell 7
sol = merge_preds(folds=[0, 1, 2, 3, 4], versions=["v3"], weights=[1.0])



## === cell 8
out_file = "/kaggle/working/submission.csv"
sol.to_csv(out_file, index=False)

row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
print("Saved:", out_file)
print("Submission shape:", sol.shape)
print("Row-sum min/max:", float(row_sums.min()), float(row_sums.max()))
print(sol.head())
