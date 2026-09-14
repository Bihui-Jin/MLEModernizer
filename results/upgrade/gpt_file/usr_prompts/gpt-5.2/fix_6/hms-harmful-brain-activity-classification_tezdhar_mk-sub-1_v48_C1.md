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

0.3346129051398209

# 6. Current score

1.68479

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the failing external `pip install`/Hydra code paths and instead generate a stable, valid baseline submission directly from `train.csv` by computing the global mean label distribution (vote counts normalized to probabilities). This fixes the runtime errors that prevent any `.csv` from being written, while keeping evaluation semantics correct (probabilities per class summing to 1) and typically achieves a reasonable KL score baseline. I also keep the existing merge/ensemble code but make it gracefully fall back to the baseline if no fold prediction files exist. Finally, I add strict sanity checks to guarantee the submission matches `sample_submission.csv` column order and constraints.'
- What this solution (achieved 1.68479) has done: 'Your current submission is a pure global-prior baseline, which is stable but far from the target KL score; the smallest legitimate improvement without changing core modeling is to make the prior *patient-aware* (still just aggregating label votes) because test has `patient_id` and label distribution is strongly patient-dependent. I compute per-patient vote distributions from `train.csv`, fall back to the global prior for unseen patients, and then align predictions to `test.csv`’s `eeg_id` order (instead of relying on `sample_submission` ordering) to avoid any subtle row misalignment. This keeps the same overall “mean distribution baseline” logic while usually moving KL substantially downward. The merge/ensemble fallback behavior is preserved; it now fall back to the improved patient-aware baseline.'
- What this solution (achieved 1.68479) has done: 'Your current patient-aware prior is still too coarse because it ignores strong per-recording effects in `spectrogram_id`; the smallest improvement that keeps the same “aggregate vote distribution” core logic is to use a hierarchical prior: spectrogram-specific if available, else patient-specific, else global. This stays within your existing baseline semantics (no EEG/spectrogram feature extraction or model training), but typically reduces KL a lot because test provides `spectrogram_id` and label distribution differs by recording context. I also fix a subtle alignment risk by mapping priors via DataFrame joins (instead of reindexing a patient_prior table by a raw NumPy array), while keeping the same output format and probability normalization checks. The merge/ensemble fallback remains unchanged, and the final `submission.csv` is still written to `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, pathlib, time, shlex, subprocess
import numpy as np
import pandas as pd

np.random.seed(0)

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
OUT_PATH2 = "/kaggle/working/v2"

SAMPLE_SUB_PATH = f"{DATA_PATH}/sample_submission.csv"
TRAIN_CSV_PATH = f"{DATA_PATH}/train.csv"
TEST_CSV_PATH = f"{DATA_PATH}/test.csv"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(OUT_PATH2, exist_ok=True)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]


def _run(cmd):
    """Keep helper for optional external commands, but do not hard-fail the notebook."""
    print(f"\n[RUN] {cmd}")
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if p.stdout:
        print(p.stdout)
    if p.returncode != 0:
        if p.stderr:
            print(p.stderr)
        raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
    return p


print("Loaded sample_submission with columns:", sample_sub.columns.tolist())
print("Target columns:", TARGET_COLS)
print("Loaded test.csv with columns:", test_df.columns.tolist(), "rows:", len(test_df))



## === cell 1
missing_inputs = []
for p in [
    "/kaggle/input/hms-mk-codes",
    "/kaggle/input/hms-mk-codesv2",
    "/kaggle/input/requirements-mk",
    "/kaggle/input/hms-mk-data",
]:
    if not os.path.exists(p):
        missing_inputs.append(p)

if missing_inputs:
    print("External model assets not available; will use baseline submission.")
    for p in missing_inputs:
        print("  missing:", p)
else:
    print(
        "External assets appear available (unexpected). Baseline will still be produced as fallback."
    )



## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)

for c in TARGET_COLS:
    if c not in train_df.columns:
        raise ValueError(f"train.csv missing required target column: {c}")
if "patient_id" not in train_df.columns:
    raise ValueError("train.csv missing patient_id (needed for patient-aware prior)")
if "patient_id" not in test_df.columns:
    raise ValueError("test.csv missing patient_id (needed for patient-aware prior)")
if "spectrogram_id" not in train_df.columns:
    raise ValueError(
        "train.csv missing spectrogram_id (needed for spectrogram-aware prior)"
    )
if "spectrogram_id" not in test_df.columns:
    raise ValueError(
        "test.csv missing spectrogram_id (needed for spectrogram-aware prior)"
    )

global_vote_sums = train_df[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)
if not np.isfinite(global_vote_sums).all() or global_vote_sums.sum() <= 0:
    raise RuntimeError("Invalid global vote sums computed from train.csv")
global_prior = global_vote_sums / global_vote_sums.sum()
global_prior = np.clip(global_prior, 1e-15, 1.0)
global_prior = global_prior / global_prior.sum()

patient_vote_sums = (
    train_df.groupby("patient_id", observed=True)[TARGET_COLS].sum().astype(np.float64)
)
patient_priors = patient_vote_sums.div(patient_vote_sums.sum(axis=1), axis=0)
patient_priors = patient_priors.replace([np.inf, -np.inf], np.nan).fillna(
    1.0 / len(TARGET_COLS)
)
patient_priors = patient_priors.clip(1e-15, 1.0)
patient_priors = patient_priors.div(patient_priors.sum(axis=1), axis=0)

spec_vote_sums = (
    train_df.groupby("spectrogram_id", observed=True)[TARGET_COLS]
    .sum()
    .astype(np.float64)
)
spec_priors = spec_vote_sums.div(spec_vote_sums.sum(axis=1), axis=0)
spec_priors = spec_priors.replace([np.inf, -np.inf], np.nan).fillna(
    1.0 / len(TARGET_COLS)
)
spec_priors = spec_priors.clip(1e-15, 1.0)
spec_priors = spec_priors.div(spec_priors.sum(axis=1), axis=0)

print("Computed global prior (train mean distribution):")
for c, v in zip(TARGET_COLS, global_prior):
    print(f"  {c}: {v:.6f}")
print("Sum:", global_prior.sum())
print("Computed patient-specific priors for patients:", len(patient_priors))
print("Computed spectrogram-specific priors for spectrograms:", len(spec_priors))



## === cell 3
baseline = test_df[["eeg_id", "patient_id", "spectrogram_id"]].copy()

spec_map = spec_priors.reset_index().rename(
    columns={"spectrogram_id": "spectrogram_id"}
)
baseline = baseline.merge(
    spec_map, on="spectrogram_id", how="left", suffixes=("", "_spec")
)

pat_map = patient_priors.reset_index().rename(columns={"patient_id": "patient_id"})
baseline = baseline.merge(pat_map, on="patient_id", how="left", suffixes=("", "_pat"))

spec_vals = baseline[TARGET_COLS].to_numpy(dtype=np.float64)
pat_vals = baseline[[f"{c}_pat" for c in TARGET_COLS]].to_numpy(dtype=np.float64)

use_pat = np.isnan(spec_vals).any(axis=1)
final_vals = spec_vals.copy()
if use_pat.any():
    final_vals[use_pat] = pat_vals[use_pat]

use_global = np.isnan(final_vals).any(axis=1)
if use_global.any():
    final_vals[use_global] = global_prior

final_vals = np.nan_to_num(
    final_vals,
    nan=1.0 / len(TARGET_COLS),
    posinf=1.0 / len(TARGET_COLS),
    neginf=1.0 / len(TARGET_COLS),
)
final_vals = np.clip(final_vals, 1e-15, 1.0)
final_vals = final_vals / final_vals.sum(axis=1, keepdims=True)

baseline = pd.DataFrame({"eeg_id": baseline["eeg_id"].values})
baseline[TARGET_COLS] = final_vals

baseline = (
    baseline.set_index("eeg_id").reindex(sample_sub["eeg_id"].values).reset_index()
)
if baseline[TARGET_COLS].isna().any().any():
    baseline[TARGET_COLS] = baseline[TARGET_COLS].fillna(global_prior)

probs = baseline[TARGET_COLS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
baseline[TARGET_COLS] = probs

baseline_fp = os.path.join(OUT_PATH, "submission_baseline.csv")
baseline = baseline[["eeg_id"] + TARGET_COLS]
baseline.to_csv(baseline_fp, index=False)
print(f"Wrote baseline submission to: {baseline_fp}")
print(baseline.head())




## === cell 4
def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v4", "v5"),
    weights=(0.5, 0.5),
    base_dir="/kaggle/working",
    allow_missing=True,
    fallback_df=None,
):
    if len(versions) != len(weights):
        raise ValueError("versions and weights must have the same length")

    sol = sample_sub.copy()
    eeg_order = sol["eeg_id"].values
    pred_sum = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)
    weight_sum = 0.0

    missing = []
    used = 0
    for fold in folds:
        for version, w in zip(versions, weights):
            fp = os.path.join(base_dir, f"submission_fold{fold}_{version}.csv")
            if not os.path.exists(fp):
                missing.append(fp)
                if allow_missing:
                    continue
                raise FileNotFoundError(f"Missing prediction file: {fp}")

            df = pd.read_csv(fp)
            if "eeg_id" not in df.columns:
                raise ValueError(f"{fp} missing eeg_id column")
            for c in TARGET_COLS:
                if c not in df.columns:
                    raise ValueError(f"{fp} missing required column: {c}")

            df = df.set_index("eeg_id").reindex(eeg_order)
            if df.isna().any().any():
                df[TARGET_COLS] = df[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

            arr = df[TARGET_COLS].to_numpy(dtype=np.float64)
            pred_sum += arr * float(w)
            weight_sum += float(w)
            used += 1

    if weight_sum == 0.0:
        if fallback_df is None:
            raise RuntimeError(
                "No prediction files were merged (all missing) and no fallback was provided. "
                f"First 3 missing examples: {missing[:3]}"
            )
        print(
            "merge_preds: no fold prediction files found; using fallback predictions (baseline)."
        )
        sol = fallback_df.copy()
        sol = sol[sample_sub.columns.tolist()]
        return sol

    pred_sum /= weight_sum

    pred_sum = np.nan_to_num(
        pred_sum,
        nan=1.0 / len(TARGET_COLS),
        posinf=1.0 / len(TARGET_COLS),
        neginf=1.0 / len(TARGET_COLS),
    )
    pred_sum = np.clip(pred_sum, 1e-15, 1.0)
    pred_sum /= pred_sum.sum(axis=1, keepdims=True)

    sol[TARGET_COLS] = pred_sum
    print(f"merge_preds: merged {used} files; missing {len(missing)} (allowed).")
    if missing:
        print("Example missing:", missing[0])
    return sol




## === cell 5
sol = merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v4", "v5"),
    weights=(0.5, 0.5),
    base_dir=OUT_PATH,
    allow_missing=True,
    fallback_df=baseline,
)



## === cell 6
for c in ["eeg_id"] + TARGET_COLS:
    if c not in sol.columns:
        raise ValueError(f"Submission missing required column: {c}")

sol = sol[sample_sub.columns.tolist()].copy()

probs = sol[TARGET_COLS].to_numpy(dtype=np.float64)
probs = np.nan_to_num(
    probs,
    nan=1.0 / len(TARGET_COLS),
    posinf=1.0 / len(TARGET_COLS),
    neginf=1.0 / len(TARGET_COLS),
)
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = probs

out_fp = os.path.join(OUT_PATH, "submission.csv")
sol.to_csv(out_fp, index=False)
print(f"Wrote final submission to: {out_fp}")
print(sol.head())



## === cell 7
assert sol.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission"
assert list(sol.columns) == list(
    sample_sub.columns
), "Column order mismatch vs sample_submission"
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums)), "Non-finite probabilities detected"
assert (
    np.max(np.abs(row_sums - 1.0)) < 1e-6
), "Probabilities do not sum to 1 within tolerance"
print("Submission checks passed.")
