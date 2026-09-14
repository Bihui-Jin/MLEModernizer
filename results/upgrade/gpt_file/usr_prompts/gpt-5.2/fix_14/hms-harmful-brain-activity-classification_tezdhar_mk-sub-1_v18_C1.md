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

0.3626664036379633

# 6. Current score

1.01228

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I remove the failing dependency installs and external code calls because the referenced Kaggle datasets (`/kaggle/input/hms-mk-codes` and `/kaggle/input/hms-mk-data`) are not available in this environment, which is the root cause of the runtime errors. To still produce a valid end-to-end submission, I replace the missing model inference with a simple, stable baseline that predicts the mean class distribution from the training labels, properly normalized and clipped for KL safety. This keeps evaluation semantics correct (probability outputs summing to 1 with the required columns) and yield a nontrivial score instead of failing to generate a CSV. The final code writes `/kaggle/working/submission.csv` with the exact required format.'
- What this solution (achieved 0.7306) has done: 'The crash comes from mapping with a non-unique `eeg_id` index in `test.csv`; `Series.map` requires a uniquely-valued index. I fix this by de-duplicating `test` on `eeg_id` (keeping the first) before building the `eeg_id -> patient_id` mapping, which restores deterministic behavior and keeps the same baseline logic. I also add a small safety guard to ensure we always output all required target columns (and in the correct order) even if merges introduce unexpected columns. Finally, the script writes `/kaggle/working/submission.csv` and validates row-sum-to-1 and strict positivity to avoid submission rejection.'
- What this solution (achieved 0.7507) has done: 'Your current baseline already uses patient-level label priors; the simplest way to move the KL score down toward the target is to make those priors more informative while keeping the same overall logic (a smoothed mean distribution). I replace the plain per-patient mean with a smoothed estimate that shrinks each patient toward the global mean based on how many training rows that patient has (empirical Bayes–style), which typically improves generalization. I also add a tiny “power” sharpening step (followed by renormalization) to reduce over-uniform predictions, which often helps KL in this competition without changing the prediction semantics (still valid probabilities summing to 1). Everything else (I/O, columns, normalization, and output path) stays the same and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.79909) has done: 'Your current score (0.7507, lower-is-better) is still far from the target (0.3627), so we should improve generalization without changing the core “patient prior + global fallback” logic. The most effective minimal tweak for KL here is to compute patient priors from *soft counts* (vote totals) rather than averaging per-row normalized distributions, because rows have varying numbers of annotators and should be weighted accordingly. I keep the same shrinkage-to-global structure, but replace the patient mean with a Dirichlet-smoothed normalized sum of votes (and compute the global mean the same way), which typically reduces KL substantially while preserving identical semantics (probabilities, same columns, same mapping). I also remove the “power sharpening” since it can easily hurt calibration under KL; this is a minimal post-processing change aimed at improving the score toward the target.'
- What this solution (achieved 0.83947) has done: 'Your current baseline is entirely patient-prior driven; to move KL down toward the target with minimal risk, we make the priors slightly more informative without changing the approach. Concretely: (1) compute patient/global priors at the *eeg_id* level (aggregating overlapping subsamples) so repeated windows don’t distort counts, and (2) replace the vote-total-based shrinkage weight with an *effective sample size* based on the number of distinct eeg recordings for that patient (more stable). We keep the same “Dirichlet-smoothed vote proportions + shrink-to-global + global fallback” semantics and still output a valid probability submission summing to 1. This should generally improve calibration/generalization (lower KL) while staying well within the existing logic.'
- What this solution (achieved 0.81847) has done: 'Your current approach is a patient-level prior with shrinkage to a global prior; the main score drag is that the current shrinkage weight ignores how many *votes* support each patient distribution (it only uses number of EEGs), which can underweight strong patient evidence and overtrust weak evidence. I keep the same “Dirichlet-smoothed vote proportions + shrink-to-global + global fallback” logic, but compute the shrinkage weight from an effective sample size based on each patient’s total vote counts (at eeg-aggregated level), and tune the shrinkage scale to be a bit less aggressive. This is a minimal, directly-relevant change that typically improves KL calibration/generalization while preserving the same prediction semantics and output format. Everything else (I/O, aggregation at eeg_id level, Dirichlet smoothing, and submission validity checks) stays the same.'
- What this solution (achieved 0.81847) has done: 'Your current patient-prior baseline is sound, but it likely underfits because it cannot distinguish different EEGs within the same patient and your shrinkage uses a fixed vote-scale. To move the KL score down toward the target with minimal semantic change, I keep the same “Dirichlet-smoothed vote proportions + shrink-to-global + global fallback” core logic and add an EEG-level prior (when an `eeg_id` appears in train) that shrinks toward the patient prior, which in turn shrinks toward global. This adds only a small hierarchical layer and should materially improve predictions for test EEGs whose `eeg_id` exists in training without changing any modeling approach. I also derive shrinkage weights directly from total vote counts at each level (EEG and patient), keeping the rest of the pipeline and submission validity checks unchanged.'
- What this solution (achieved 0.85202) has done: 'Your current baseline is underperforming the target (0.81847 vs 0.36267, lower-is-better), so we should improve calibration with the smallest possible semantic change. I keep the same hierarchical “Dirichlet-smoothed vote proportions + shrinkage” logic, but (1) increase the Dirichlet pseudo-count to reduce overconfident spikes from small vote totals (helps KL), and (2) make the shrinkage weights use a slightly stronger pull to priors by increasing the taus modestly (also helps KL stability). I also add an optional final blend with the global mean (very small epsilon) to prevent extreme probabilities that are heavily penalized by KL, while keeping outputs valid and normalized. All I/O paths, aggregation level, and submission schema remain unchanged.'
- What this solution (achieved 0.84998) has done: 'Your current score (0.852, lower-is-better) is still far from the target (0.3627), so we should improve the KL by making the hierarchical priors better calibrated without changing the core “Dirichlet-smoothed vote proportions + shrinkage (EEG→patient→global) + global fallback” logic. The most likely issue is that the EEG-level prior is being merged incorrectly due to an unnecessary/incorrect rename of the index column, which can silently drop most EEG-level matches and force weaker patient/global predictions. I fix the EEG merge key to use the real `eeg_id` column, and I also compute the final prediction in a clearer hierarchical way (EEG if available else patient else global), keeping the same semantics and probability constraints. Finally, I remove the extra global blending step (which can worsen KL by over-smoothing), while keeping strict positivity via Dirichlet smoothing and clipping.'
- What this solution (achieved 1.00824) has done: 'Your baseline is already hierarchical (EEG→patient→global), so the most direct way to reduce KL toward the target is to make the priors more representative of the *label distribution* seen by the metric: normalize each training row into probabilities, then average at the eeg_id level (so overlapping windows don’t overweight) and then at patient/global levels. This preserves the same core logic (hierarchical shrinkage with Dirichlet-style smoothing and fallbacks) but fixes the main mismatch in your current code: using raw vote-count totals biases patients/EEGs with more raters rather than better reflecting per-sample class probabilities. I also keep your merge/prediction flow intact and only adjust the computations feeding `global_mean`, `patient_mean`, and `eeg_mean`. Submission writing/validation stays unchanged and still produce `/kaggle/working/submission.csv`.'
- What this solution (achieved 1.01228) has done: 'Your current score is far above the target (lower-is-better), so we should make the hierarchical priors more faithful to the evaluation target while keeping the same EEG→patient→global fallback structure. The smallest impactful change is to compute eeg-level and patient/global priors using vote-weighted averaging of per-row probabilities (so samples with more annotator votes contribute proportionally, matching the “observed target” better), instead of a plain mean over rows/eegs. We keep the same smoothing, the same merge/prediction logic, and the same submission formatting/validity checks. This should reduce KL (improve score) toward your target without changing the overall approach.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
os.makedirs(OUT_PATH, exist_ok=True)

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

np.random.seed(0)



## === cell 1
train_path = f"{DATA_PATH}/train.csv"
test_path = f"{DATA_PATH}/test.csv"
sample_path = f"{DATA_PATH}/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

assert set(["eeg_id", "patient_id"]).issubset(test.columns)
assert all(c in train.columns for c in (["patient_id", "eeg_id"] + TARGET_COLS))
assert all(c in sample.columns for c in ["eeg_id"] + TARGET_COLS)

print("Loaded:")
print(" train:", train.shape, " test:", test.shape, " sample:", sample.shape)



## === cell 2
votes = train[TARGET_COLS].to_numpy(dtype=np.float64)
votes = np.where(np.isfinite(votes), votes, 0.0)
votes = np.clip(votes, 0.0, None)

K = len(TARGET_COLS)

alpha = 0.10

row_tot = votes.sum(axis=1, keepdims=True)
row_tot = np.where(row_tot <= 0.0, 1.0, row_tot)
row_probs = votes / row_tot
row_probs = (row_probs + alpha) / (row_probs.sum(axis=1, keepdims=True) + alpha * K)

train_probs_df = pd.DataFrame(row_probs, columns=TARGET_COLS)
train_probs_df["patient_id"] = train["patient_id"].values
train_probs_df["eeg_id"] = train["eeg_id"].values

row_w = votes.sum(axis=1).astype(np.float64)
row_w = np.where(np.isfinite(row_w) & (row_w > 0), row_w, 1.0)
train_probs_df["row_w"] = row_w


def _weighted_mean_df(g: pd.DataFrame) -> pd.Series:
    w = g["row_w"].to_numpy(dtype=np.float64)
    wsum = w.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        w = np.ones(len(g), dtype=np.float64)
        wsum = float(len(g))
    out = {}
    for c in TARGET_COLS:
        x = g[c].to_numpy(dtype=np.float64)
        out[c] = float(np.sum(x * w) / wsum)
    out["patient_id"] = g["patient_id"].iloc[0]
    out["eeg_w"] = float(
        wsum
    )  # effective weight at eeg level for later patient/global aggregation
    return pd.Series(out)


eeg_level = (
    train_probs_df.groupby("eeg_id", as_index=False)
    .apply(_weighted_mean_df)
    .reset_index(drop=True)
)

eeg_w = eeg_level["eeg_w"].to_numpy(dtype=np.float64)
eeg_wsum = eeg_w.sum()
if not np.isfinite(eeg_wsum) or eeg_wsum <= 0:
    eeg_w = np.ones(len(eeg_level), dtype=np.float64)
    eeg_wsum = float(len(eeg_level))

global_mean = (eeg_level[TARGET_COLS].to_numpy(dtype=np.float64) * eeg_w[:, None]).sum(
    axis=0
) / eeg_wsum
global_mean = np.clip(global_mean, 1e-15, 1.0)
global_mean = global_mean / global_mean.sum()

patient_tbl = eeg_level[["patient_id", "eeg_w"] + TARGET_COLS].copy()


def _patient_weighted_mean(g: pd.DataFrame) -> pd.Series:
    w = g["eeg_w"].to_numpy(dtype=np.float64)
    wsum = w.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        w = np.ones(len(g), dtype=np.float64)
        wsum = float(len(g))
    out = {}
    for c in TARGET_COLS:
        x = g[c].to_numpy(dtype=np.float64)
        out[c] = float(np.sum(x * w) / wsum)
    out["patient_w"] = float(wsum)
    out["patient_n_eegs"] = float(len(g))
    return pd.Series(out)


patient_agg = patient_tbl.groupby("patient_id").apply(_patient_weighted_mean)
patient_probs = patient_agg[TARGET_COLS]
patient_n_eegs = patient_agg["patient_n_eegs"].astype(np.float64)
patient_wsum = patient_agg["patient_w"].astype(np.float64)

tau_patient = 6.0
w_patient = (patient_n_eegs / (patient_n_eegs + tau_patient)).to_numpy(dtype=np.float64)

patient_mean = patient_probs.to_numpy(dtype=np.float64) * w_patient[
    :, None
] + global_mean[None, :] * (1.0 - w_patient[:, None])
patient_mean = np.clip(patient_mean, 1e-15, 1.0)
patient_mean = patient_mean / patient_mean.sum(axis=1, keepdims=True)
patient_mean = pd.DataFrame(
    patient_mean, index=patient_probs.index, columns=TARGET_COLS
)

eeg_probs = eeg_level.set_index("eeg_id")[TARGET_COLS].astype(np.float64)
eeg_patient = eeg_level.set_index("eeg_id")["patient_id"]

w_eeg = 0.90

eeg_pid = eeg_patient.values
patient_prior_for_eeg = patient_mean.reindex(eeg_pid).to_numpy(dtype=np.float64)
missing_pat_mask = ~np.isfinite(patient_prior_for_eeg).all(axis=1)
if missing_pat_mask.any():
    patient_prior_for_eeg[missing_pat_mask, :] = global_mean[None, :]

eeg_mean = eeg_probs.to_numpy(dtype=np.float64) * w_eeg + patient_prior_for_eeg * (
    1.0 - w_eeg
)
eeg_mean = np.clip(eeg_mean, 1e-15, 1.0)
eeg_mean = eeg_mean / eeg_mean.sum(axis=1, keepdims=True)
eeg_mean = pd.DataFrame(eeg_mean, index=eeg_probs.index, columns=TARGET_COLS)

print("Global mean class probabilities from train (eeg-aggregated, vote-weighted):")
for c, p in zip(TARGET_COLS, global_mean):
    print(f"  {c}: {p:.6f}")
print("Sum:", global_mean.sum())
print(
    "Unique train patients:",
    train["patient_id"].nunique(),
    " | Unique test patients:",
    test["patient_id"].nunique(),
    " | Unique train eeg_ids:",
    train["eeg_id"].nunique(),
)



## === cell 3
test_unique = test.drop_duplicates(subset=["eeg_id"], keep="first").copy()

sol = sample[["eeg_id"]].copy()

test_pid = test_unique.set_index("eeg_id")["patient_id"]

missing = sol.loc[~sol["eeg_id"].isin(test_pid.index), "eeg_id"]
if len(missing) > 0:
    raise ValueError(
        f"Found eeg_ids in sample_submission missing from test.csv: {missing.iloc[:5].tolist()}"
    )

base = pd.DataFrame({"eeg_id": sol["eeg_id"].values})
base["patient_id"] = base["eeg_id"].map(test_pid)

base = base.merge(
    eeg_mean.reset_index(),  # columns: ['eeg_id'] + TARGET_COLS
    on="eeg_id",
    how="left",
    suffixes=("", "_eeg"),
)

base = base.merge(
    patient_mean.reset_index(),  # columns: ['patient_id'] + TARGET_COLS
    on="patient_id",
    how="left",
    suffixes=("", "_patient"),
)

K = len(TARGET_COLS)
pred = np.empty((len(base), K), dtype=np.float64)
for i, c in enumerate(TARGET_COLS):
    eeg_col = base[c]
    pat_col = base[f"{c}_patient"]
    vals = eeg_col.where(eeg_col.notna(), pat_col)
    vals = vals.fillna(global_mean[i]).to_numpy(dtype=np.float64)
    pred[:, i] = vals

pred = np.clip(pred, 1e-15, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

for i, c in enumerate(TARGET_COLS):
    sol[c] = pred[:, i]

sol = sol[["eeg_id"] + TARGET_COLS]



## === cell 4
out_path = f"{OUT_PATH}/submission.csv"
sol.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path}")
print(sol.head())



## === cell 5
assert sol.shape[1] == 1 + len(TARGET_COLS)
assert (
    sol.columns.tolist() == ["eeg_id"] + TARGET_COLS
), "Submission columns/order mismatch."
assert sol["eeg_id"].isna().sum() == 0
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums))
assert (
    np.max(np.abs(row_sums - 1.0)) < 1e-9
), f"Row sums not 1. Max deviation={np.max(np.abs(row_sums - 1.0))}"
assert (sol[TARGET_COLS].to_numpy() > 0).all()
print(
    "Submission looks valid: probabilities sum to 1 for all rows and are strictly positive."
)
