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

0.3391724675284024

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The runtime failure is because the fold inference commands never produced `/kaggle/working/submission.csv`, so the subsequent `mv` commands didn’t create `submission_fold*.csv` files and the ensemble step can’t find them. I make the pipeline robust by (1) skipping IPython-only `pip` magics when not available, (2) checking each fold run’s return code and whether `submission.csv` was created before moving it, and (3) allowing `merge_preds` to ensemble only the folds that actually exist (or fall back to the sample submission probabilities if none exist) so a valid `.csv` is always produced. This preserves the core modeling/inference logic (still calls the same external `python -m test` with the same checkpoints), but prevents missing-file crashes and guarantees the final submission sums to 1. The output be written to `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.33917), so we should improve predictive quality without changing the underlying model/test invocation. The most likely silent issue hurting score is row misalignment when ensembling: each fold’s `submission_fold*.csv` may be in a different order than `sample_submission.csv`, and your current merge can reorder rows and introduce NaNs, corrupting the averaged probabilities. I change `merge_preds` to strictly align predictions to the sample’s `eeg_id` order via an index-based reindex (no sorting), verify no duplicate/missing ids per fold, and only then average and renormalize. This preserves the core fold inference pipeline and ensemble semantics, but fixes alignment so probabilities correspond to the correct `eeg_id`s, which should substantially reduce KL.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), so we should fix the most likely remaining “silent” issue that can destroy KL while still producing a valid-looking CSV: averaging raw fold probabilities instead of averaging in log-space (geometric mean), which is usually better calibrated for KL and ensembling. I keep the same fold inference pipeline and the same checkpoints, but change only the ensembling step to compute a log-mean of probabilities with a tiny epsilon for numerical safety, then renormalize to sum to 1. I also add strict clipping/renormalization per fold before ensembling to prevent any tiny negatives/zeros from breaking log-space averaging. This is a minimal change focused on improving predictive quality without altering the core model/test calls.'
- What this solution (achieved 1.40995) has done: 'The current score is far worse than the target (lower-is-better), so the most likely remaining high-impact issue is that the external `python -m test` command is not actually running different folds: it never passes a `fold=` override, so you can end up producing identical predictions five times (or the same default fold each time), making the “ensemble” ineffective. I make a minimal change to `run_fold` to pass `fold={fold}` into Hydra so each checkpoint is evaluated on its intended fold configuration, while keeping the same model/test entrypoint and checkpoints. I also add a lightweight sanity check that warns if fold outputs look identical (helps catch silent failures without changing semantics), and keep your aligned log-mean ensembling and strict normalization.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
from pathlib import Path

sys.path.append("/kaggle/input/hms-mk-codes/")




## === cell 1
def _safe_pip_install(cmd: str) -> int:
    """
    Run pip install command safely. In Kaggle, this may or may not be needed
    depending on the environment; failures shouldn't crash the whole run.
    """
    try:
        return subprocess.run(cmd, shell=True, check=False).returncode
    except Exception as e:
        print(f"[WARN] pip install command failed to execute: {e}")
        return 1


wheels = [
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    "/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl",
    "/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl",
    "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
]
if all(os.path.exists(w) for w in wheels):
    _safe_pip_install(
        "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl "
        "--no-index --no-deps --force-reinstall"
    )
    _safe_pip_install(
        "pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
    )
    _safe_pip_install(
        "pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
    )
    _safe_pip_install(
        "pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
    )
else:
    print(
        "[INFO] requirements-mk wheels not found in /kaggle/input; skipping local wheel installs."
    )



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)



## === cell 3
cmd = (
    f"cd /kaggle/input/hms-mk-codes && "
    f'python -m src.convert_parquet_to_npy --data_dir="{DATA_PATH}" --out_dir="{OUT_PATH}"'
)
ret = subprocess.run(cmd, shell=True, check=False)
if ret.returncode != 0:
    print(
        f"[WARN] convert_parquet_to_npy returned non-zero exit code: {ret.returncode}"
    )
else:
    print("[INFO] convert_parquet_to_npy completed.")



## === cell 4
subprocess.run("ls -la /kaggle/input | head -n 200", shell=True, check=False)



## === cell 5
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]




## === cell 6
def run_fold(
    fold: int, ckpt_path: str, data_path: str = DATA_PATH, out_path: str = OUT_PATH
) -> str | None:
    fold_out = f"/kaggle/working/submission_fold{fold}.csv"
    base_sub = "/kaggle/working/submission.csv"

    if os.path.exists(base_sub):
        try:
            os.remove(base_sub)
        except Exception as e:
            print(f"[WARN] Could not remove stale {base_sub}: {e}")

    cmd = (
        f"cd /kaggle/input/hms-mk-codes && "
        f"python -m test "
        f'paths.data_dir="{data_path}" '
        f'data.test_eegs_dir="{out_path}" '
        f'ckpt_path="{ckpt_path}" '
        f"hydra=test "
        f'+model.test_output_dir="{out_path}" '
        f"experiment=conv1d_pseudo "
        f"+model.net.pretrained=False "
        f"fold={fold}"
    )
    print(f"[INFO] Running fold {fold}: {cmd}")
    ret = subprocess.run(cmd, shell=True, check=False)
    if ret.returncode != 0:
        print(
            f"[WARN] Fold {fold} test command failed with exit code {ret.returncode}. Skipping this fold."
        )
        return None

    if not os.path.exists(base_sub):
        print(
            f"[WARN] Fold {fold} completed but {base_sub} not found. Skipping this fold."
        )
        return None

    try:
        os.replace(base_sub, fold_out)
        print(f"[INFO] Wrote {fold_out}")
        return fold_out
    except Exception as e:
        print(f"[WARN] Could not move {base_sub} -> {fold_out}: {e}")
        return None


fold_ckpts = {
    0: "/kaggle/input/hms-mk-data/fold0_pseudo_log.ckpt",
    1: "/kaggle/input/hms-mk-data/fold1_pseudo_log.ckpt",
    2: "/kaggle/input/hms-mk-data/fold2_pseudo_log.ckpt",
    3: "/kaggle/input/hms-mk-data/fold3_pseudo_log.ckpt",
    4: "/kaggle/input/hms-mk-data/fold4_pseudo_log.ckpt",
}
produced = {}
for f, ckpt in fold_ckpts.items():
    produced[f] = run_fold(f, ckpt)

print(
    "[INFO] Produced fold files:", {k: v for k, v in produced.items() if v is not None}
)



## === cell 7
import pandas as pd
import numpy as np

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
    sample_sub_path="/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
):
    """
    Keep same ensemble intent, but robustly align by eeg_id and use log-space mean.
    """
    sample = pd.read_csv(sample_sub_path)
    if "eeg_id" not in sample.columns:
        raise ValueError("sample_submission.csv is missing 'eeg_id' column.")
    for c in TARGET_COLS:
        if c not in sample.columns:
            raise ValueError(
                f"sample_submission.csv is missing required target column: {c}"
            )

    sample_ids = sample["eeg_id"].to_numpy()
    sample_index = pd.Index(sample_ids, name="eeg_id")

    eps = 1e-12

    log_accum = None
    used_folds = 0
    used_fold_ids = []
    fold_signatures = {}

    for fold in folds:
        fold_path = f"/kaggle/working/submission_fold{fold}.csv"
        if not os.path.exists(fold_path):
            print(f"[WARN] Missing fold prediction file: {fold_path} (skipping)")
            continue

        sol = pd.read_csv(fold_path)
        if "eeg_id" not in sol.columns:
            print(f"[WARN] {fold_path} is missing 'eeg_id' column (skipping)")
            continue

        missing_cols = [c for c in TARGET_COLS if c not in sol.columns]
        if missing_cols:
            print(f"[WARN] {fold_path} missing columns {missing_cols} (skipping)")
            continue

        if sol["eeg_id"].duplicated().any():
            print(f"[WARN] {fold_path} has duplicate eeg_id rows (skipping)")
            continue

        sol_idx = sol.set_index("eeg_id")[TARGET_COLS]
        sol_idx = sol_idx.reindex(sample_index)

        if sol_idx.isna().any().any():
            n_missing = int(sol_idx.isna().any(axis=1).sum())
            print(
                f"[WARN] {fold_path} missing predictions for {n_missing} eeg_id(s) after alignment (skipping)"
            )
            continue

        fold_preds = sol_idx.to_numpy(dtype=np.float64)

        fold_preds = np.nan_to_num(fold_preds, nan=0.0, posinf=0.0, neginf=0.0)
        fold_preds = np.clip(fold_preds, 0.0, 1.0)
        rs = fold_preds.sum(axis=1, keepdims=True)
        zero_mask = rs[:, 0] == 0
        if np.any(zero_mask):
            fold_preds[zero_mask] = 1.0 / len(TARGET_COLS)
            rs = fold_preds.sum(axis=1, keepdims=True)
        fold_preds = fold_preds / rs

        fold_preds = np.clip(fold_preds, eps, 1.0)
        fold_preds = fold_preds / fold_preds.sum(axis=1, keepdims=True)

        fold_signatures[fold] = float(np.round(fold_preds[:50].mean(), 12))

        fold_log = np.log(fold_preds)

        if log_accum is None:
            log_accum = fold_log
        else:
            log_accum += fold_log

        used_folds += 1
        used_fold_ids.append(fold)

    if len(set(fold_signatures.values())) <= 1 and len(fold_signatures) > 1:
        print(
            "[WARN] Fold predictions appear nearly identical by quick signature; "
            "this can hurt ensemble quality. (Not failing; just warning.)"
        )

    if used_folds == 0:
        print(
            "[WARN] No fold predictions loaded; falling back to sample_submission probabilities."
        )
        preds = sample[TARGET_COLS].to_numpy(dtype=np.float64)
    else:
        print(f"[INFO] Ensembling folds (aligned, log-mean): {used_fold_ids}")
        log_mean = log_accum / used_folds
        preds = np.exp(log_mean)

    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.clip(preds, 0.0, None)
    row_sums = preds.sum(axis=1, keepdims=True)
    zero_mask = row_sums[:, 0] == 0
    if np.any(zero_mask):
        preds[zero_mask] = 1.0 / len(TARGET_COLS)
        row_sums = preds.sum(axis=1, keepdims=True)
    preds = preds / row_sums

    out = sample.copy()
    out[TARGET_COLS] = preds
    return out




## === cell 8
sol = merge_preds(folds=(0, 1, 2, 3, 4))



## === cell 9
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)

assert (
    sol.columns.tolist() == ["eeg_id"] + TARGET_COLS
), "Submission columns do not match required format."
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
if not np.all(np.isfinite(row_sums)):
    raise ValueError("Non-finite row sums detected in submission.")
max_abs_deviation = float(np.max(np.abs(row_sums - 1.0)))
print(
    f"Wrote submission: {sub_path}  shape={sol.shape}  max|sum-1|={max_abs_deviation:.3e}"
)
print(sol.head())



## === cell 10
sol
