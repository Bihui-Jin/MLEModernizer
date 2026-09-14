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

0.3408716306310721

# 6. Current score

0.89406

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the failing dependency installs and external “mk-codes/mk-data” calls, since those inputs aren’t available in your provided environment and currently stop execution before any submission is written. To keep the pipeline valid end-to-end, I replace the broken inference stage with a minimal, deterministic baseline that uses the training vote distributions (properly normalized) to produce valid per-class probabilities for every test `eeg_id`. I also harden the submission creation to enforce required columns, row alignment to `sample_submission.csv`, numeric type, clipping, and row-wise probability normalization so the file always passes Kaggle’s format checks. This yield a valid `submission.csv` and a reasonable KL baseline score, moving you toward the target rather than failing to submit.'
- What this solution (achieved 0.83266) has done: 'Your current baseline predicts the same global class prior for every test row, which is stable but too weak (KL ≈ 1.42) versus your target (≈ 0.34). To move toward the target with minimal semantic changes, I keep the “prior-based” idea but condition it on `patient_id` (a strong metadata signal available in both train/test) by using a smoothed patient-specific vote distribution when possible, otherwise falling back to the global prior. I also add simple Laplace/Dirichlet smoothing and blend patient-prior with global prior to avoid overfitting sparse patients while improving calibration. The output is still a valid probability distribution per row and is aligned to `sample_submission.csv` to guarantee a valid submission.'
- What this solution (achieved 0.83266) has done: 'Your current metadata-only baseline is too weak versus the target (lower-is-better), so we should gently increase signal while keeping the same “smoothed prior” core logic. The smallest meaningful improvement is to condition priors not just on `patient_id` but also on `eeg_id` when that EEG appears in training (train has many rows per `eeg_id`, and test `eeg_id`s may overlap), falling back to patient-prior, then global prior. To avoid overfitting and keep calibration stable for KL, we keep the same Dirichlet/Laplace smoothing + blending structure, just add one extra hierarchy level (`eeg_id` → `patient_id` → global) with conservative blending weights. Submission creation remains aligned to `sample_submission.csv` and enforces strict probability normalization.'
- What this solution (achieved 0.83266) has done: 'Your current approach is a hierarchical, smoothed prior (eeg_id → patient_id → global), but it likely underuses strong metadata signals that exist in both train and test. To move the KL score down toward the target with minimal semantic change, I add one more conservative hierarchy level based on `spectrogram_id` (often highly informative and present in both train/test), placed above `eeg_id`. I keep the exact same smoothing+blending semantics and just extend the lookup chain (`spectrogram_id → eeg_id → patient_id → global`) with cautious weights so calibration remains stable. I also make the mapping robust in case `sample_submission` ordering differs from `test.csv` by building lookup dicts and defaulting safely.'
- What this solution (achieved 0.84251) has done: 'Your current hierarchical smoothed-prior is valid but likely under-confident because it falls back too often and it ignores extra structure inside train where each `spectrogram_id`/`eeg_id` can have multiple label rows with varying label quality. To move the KL down toward the target with minimal semantic change, I keep the exact same inference idea (hierarchical priors + Dirichlet smoothing + blending) but (1) normalize each train row’s votes to a probability distribution before aggregation (so recordings with more raters don’t dominate purely by count), and (2) blend hierarchically rather than “pick first available”, so when a key exists we still softly incorporate lower levels/global for better calibration. I also compute blends based on evidence strength (total votes for that group) to avoid overconfident priors for sparse groups while keeping the core logic intact. The submission creation remains aligned to `sample_submission.csv` and strictly enforces per-row probability normalization.'
- What this solution (achieved 0.86524) has done: 'Your current hierarchical-prior is still too weak versus the target (0.84251 vs 0.34087, lower is better), so the smallest safe way to improve is to use a stronger, still-metadata-only signal that exists in both train and test: the `expert_consensus` label distribution conditioned on `spectrogram_id`/`eeg_id`/`patient_id`. I keep the exact same “hierarchical smoothed prior + evidence-strength blending + fallback to global” core logic, but I augment each group’s aggregated distribution by blending the per-row normalized votes with a one-hot from `expert_consensus` (with a small weight), which typically tightens predictions without changing the overall approach. I also fix the current “strength” definition to match the aggregated units (number of rows contributing to a group) since you are averaging per-row probabilities; this makes the evidence-based blending weight behave as intended and improves calibration for KL. The submission creation and strict row-wise probability normalization stay unchanged so the output remains a valid Kaggle submission.'
- What this solution (achieved 0.86524) has done: 'Your current KL (0.865) is still far above the target (0.341), so we should safely increase predictive signal while keeping the same “hierarchical smoothed prior + evidence-strength blending” core logic. The smallest high-impact addition is to insert one more hierarchy level keyed by `(patient_id, spectrogram_id)` (and optionally `(patient_id, eeg_id)`) because these combinations often capture session-specific patterns better than either key alone, without changing the modeling paradigm. I keep the same per-row vote normalization + small expert-consensus blending, and reuse the same Dirichlet smoothing and strength-based blending—just extending the lookup chain. Submission alignment and strict probability normalization remain unchanged to guarantee a valid CSV.'
- What this solution (achieved 0.86524) has done: 'Your current KL (0.865, lower-is-better) is still far from the target (0.341), so we need a small but meaningful metadata-only signal boost while preserving the same hierarchical smoothed-prior approach. The safest improvement is to aggregate training labels at the `label_id` level first (since `train.csv` contains many overlapping subsamples and repeated label sets), then build the same group priors from these de-duplicated units; this reduces noise and double-counting without changing inference semantics. I keep your expert-consensus blending, but apply it per `label_id` (and recompute evidence strength as number of labels) so the evidence-based blending weights behave more consistently for KL. Submission creation, alignment to `sample_submission.csv`, and strict probability normalization remain unchanged.'
- What this solution (achieved 0.85714) has done: 'Your current score (0.86524, lower-is-better) is still far above the target (0.34087), so we should increase predictive signal while keeping the exact same “hierarchical smoothed priors + evidence-strength blending” core approach. The smallest high-impact fix is to stop smoothing the *summed probabilities* with a Dirichlet alpha as if they were counts; instead, treat each group as a mean distribution with an *effective sample size* (=strength) and apply Dirichlet shrinkage toward the global prior using that strength, which is the correct conjugate form and usually improves KL calibration. I keep your same hierarchy, same features, and same loops, but replace `counts + alpha` with a principled shrinkage `p = (n*mean + tau*global)/(n+tau)` using your existing strength scales as taus. This is a minimal semantic correction (still a smoothed hierarchical prior) that should move the KL downward toward the target without changing I/O or submission format.'
- What this solution (achieved 0.84588) has done: 'Your current metadata-only hierarchical prior is still much worse than the target (0.857 vs 0.341, lower is better), so we need a small, safe bump in signal without changing the overall approach. The most likely issue is that your group aggregations are mixing keys inconsistently because `label_meta` takes the mode of `spectrogram_id/eeg_id/patient_id`, which can silently mis-assign labels when duplicates/overlaps exist; switching to a deterministic “first row per label_id” preserves the de-duplication idea but avoids mode-instability and typically improves calibration for KL. Next, your shrinkage step currently uses a fixed tau per hierarchy and a separate blend weight; we keep the same shrinkage+blend core logic but make tau proportional to the existing strength_scale (not +alpha) and slightly reduce over-shrinkage so informative groups (spectrogram/ps/pe) can influence predictions more. Finally, we keep strict submission alignment and probability normalization exactly as before to ensure the CSV always passes format checks.'
- What this solution (achieved 0.84588) has done: 'Your current score (0.84588, lower-is-better) is still far above the target (0.34087), so we need a modest signal increase while keeping the exact same “hierarchical smoothed priors + evidence-strength blending” core logic. The smallest high-impact fix is to correct the training aggregation mismatch: you compute `label_prob` as a mean per `label_id` but then later sum these means without accounting for how many original rows contributed to each `label_id`, which underweights labels that appear multiple times and hurts calibration. I keep the same hierarchy and blending, but (1) carry a per-`label_id` weight (`label_n`) and use **weighted sums/strengths** for all group priors (spectrogram/eeg/patient/ps/pe), so “strength” reflects the true amount of evidence. I also make the shrinkage step consistent with that weighted evidence by using `n_eff = n + alpha_floor` where `n` is now the weighted strength, without changing the model semantics or submission format.'
- What this solution (achieved 0.84471) has done: 'We keep your exact “hierarchical smoothed priors + evidence-strength blending” approach, but fix one mismatch that can blunt signal: you currently average per-`label_id` and then weight by `label_n`, which can over-emphasize repeated overlapping windows rather than the *actual rater evidence* for that label. We instead carry a per-`label_id` effective evidence weight equal to the total votes for that label (sum of raw vote counts across rows in the label), and use that as the group “strength” and weighting for aggregated priors; this is still the same model semantics, just a better-calibrated notion of evidence for KL. Additionally, we compute `global_prior` from these de-duplicated, vote-weighted `label_id` units (instead of raw rows), which reduces overlap bias and usually improves calibration. All I/O paths, columns, and the submission normalization/format checks remain unchanged.'
- What this solution (achieved 0.84471) has done: 'Your current hierarchical prior is structurally fine but it likely underperforms because the group “prob_sum” vectors you build are not actually probability sums: you’re summing `label_prob * label_votes` (which scales like votes), but later you divide by `strength` (also votes), yielding a weighted mean that can be overly dominated by high-vote labels and can interact badly with the shrinkage/blend schedule for KL. With minimal change to core logic, I switch the group aggregates to store (a) the **weighted mean label probability** per group and (b) the corresponding **total evidence strength**, so `_shrunk_prior` uses a stable mean distribution directly (no extra division by `n` inside). I also fix the shrinkage formula to use `mean` as-is and only weight it by `n_eff` vs `tau`, keeping your same hierarchy, smoothing, and blending semantics but making the math consistent. Submission alignment/format checks remain identical so it still writes a valid `submission.csv`.'
- What this solution (achieved 0.89406) has done: 'We keep your exact hierarchical smoothed-prior approach, but adjust only the blend schedule to better match KL: right now the model can become too “peaky” when a key has high strength, which is usually punished by KL on this task. Specifically, we (1) apply a small temperature smoothing to the final probabilities (closer to global prior, less overconfidence) and (2) slightly reduce the maximum effective per-level blend weights while leaving the hierarchy, shrinkage math, and all aggregations unchanged. This is minimal code change (pure post-processing + conservative weight tweak) and should move the score downward toward your target (lower is better) without risking invalid submissions. The CSV writing and strict row-wise normalization remain unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_PATH, "sample_submission.csv")

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

for p in [TRAIN_CSV, TEST_CSV, SAMPLE_SUB_CSV]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

print("Data files found.")
print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_CSV:", TEST_CSV)
print("SAMPLE_SUB_CSV:", SAMPLE_SUB_CSV)

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

missing_train = [
    c
    for c in (
        ["patient_id", "eeg_id", "spectrogram_id", "expert_consensus", "label_id"]
        + TARGET_COLS
    )
    if c not in train.columns
]
missing_test = [
    c for c in ["eeg_id", "patient_id", "spectrogram_id"] if c not in test.columns
]
missing_sample = [c for c in (["eeg_id"] + TARGET_COLS) if c not in sample_sub.columns]
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")
if missing_sample:
    raise ValueError(f"sample_submission.csv missing columns: {missing_sample}")

train_key = train[
    [
        "label_id",
        "spectrogram_id",
        "eeg_id",
        "patient_id",
        "expert_consensus",
    ]
    + TARGET_COLS
].copy()

train_key["label_id"] = train_key["label_id"].astype(np.int64)
train_key["spectrogram_id"] = train_key["spectrogram_id"].astype(np.int64)
train_key["eeg_id"] = train_key["eeg_id"].astype(np.int64)
train_key["patient_id"] = train_key["patient_id"].astype(np.int64)

votes = train_key[TARGET_COLS].astype(np.float64).to_numpy()
row_sum = votes.sum(axis=1, keepdims=True)
row_sum = np.where(row_sum > 0, row_sum, 1.0)
row_probs = votes / row_sum

cons_map = {
    "Seizure": "seizure_vote",
    "LPD": "lpd_vote",
    "GPD": "gpd_vote",
    "LRDA": "lrda_vote",
    "GRDA": "grda_vote",
    "Other": "other_vote",
}
cons_idx = {k: i for i, k in enumerate(TARGET_COLS)}
expert_idx = (
    train_key["expert_consensus"]
    .map(cons_map)
    .map(cons_idx)
    .fillna(-1)
    .astype(np.int64)
    .to_numpy()
)

expert_onehot = np.zeros_like(row_probs)
valid = expert_idx >= 0
expert_onehot[np.arange(len(expert_idx))[valid], expert_idx[valid]] = 1.0

EXPERT_WEIGHT = 0.15
row_probs_enhanced = (1.0 - EXPERT_WEIGHT) * row_probs + EXPERT_WEIGHT * expert_onehot

train_key_prob = train_key[
    ["label_id", "spectrogram_id", "eeg_id", "patient_id", "expert_consensus"]
].copy()
train_key_prob[TARGET_COLS] = row_probs_enhanced

train_key_prob_sorted = train_key_prob.sort_values(
    ["label_id"], kind="mergesort"
).reset_index(drop=True)

train_key_votes_sorted = train_key[["label_id"] + TARGET_COLS].copy()
train_key_votes_sorted = train_key_votes_sorted.sort_values(
    ["label_id"], kind="mergesort"
).reset_index(drop=True)

label_vote_total = (
    train_key_votes_sorted.groupby("label_id", sort=False)[TARGET_COLS]
    .sum()
    .sum(axis=1)
    .astype(np.float64)
    .rename("label_votes")
)

label_prob = (
    train_key_prob_sorted.groupby("label_id", sort=False)[TARGET_COLS]
    .mean()
    .astype(np.float64)
)

label_meta = train_key_prob_sorted.drop_duplicates("label_id", keep="first")[
    ["label_id", "spectrogram_id", "eeg_id", "patient_id"]
].set_index("label_id")

train_label = (
    label_meta.join(label_prob, how="inner")
    .join(label_vote_total, how="inner")
    .reset_index(drop=False)
)

train_label["label_votes"] = train_label["label_votes"].clip(lower=1.0)

global_prior = (
    train_label[TARGET_COLS].to_numpy(dtype=np.float64)
    * train_label["label_votes"].to_numpy(dtype=np.float64)[:, None]
).sum(axis=0)
global_prior = np.clip(global_prior, 1e-12, None)
global_prior = global_prior / global_prior.sum()

print(
    "Computed global class prior (label-vote-weighted):",
    dict(zip(TARGET_COLS, global_prior.round(6))),
)
print(
    "Train rows:", len(train), "Test rows:", len(test), "Sample rows:", len(sample_sub)
)

for c in TARGET_COLS:
    train_label[f"{c}__w"] = train_label[c] * train_label["label_votes"]


def _group_weighted_mean(df: pd.DataFrame, by):
    wsum = (
        df.groupby(by, sort=False)[[f"{c}__w" for c in TARGET_COLS]]
        .sum()
        .rename(columns={f"{c}__w": c for c in TARGET_COLS})
        .astype(np.float64)
    )
    strength = df.groupby(by, sort=False)["label_votes"].sum().astype(np.float64)
    mean = (
        wsum.div(strength.replace(0.0, np.nan), axis=0).fillna(0.0).astype(np.float64)
    )
    return mean, strength


spect_prob_means, spect_strength = _group_weighted_mean(train_label, "spectrogram_id")
eeg_prob_means, eeg_strength = _group_weighted_mean(train_label, "eeg_id")
patient_prob_means, patient_strength = _group_weighted_mean(train_label, "patient_id")
ps_prob_means, ps_strength = _group_weighted_mean(
    train_label, ["patient_id", "spectrogram_id"]
)
pe_prob_means, pe_strength = _group_weighted_mean(train_label, ["patient_id", "eeg_id"])

test_patients = set(test["patient_id"].astype(np.int64).unique().tolist())
train_patients = set(patient_prob_means.index.astype(np.int64).tolist())
seen_p = len(test_patients & train_patients)
unseen_p = len(test_patients - train_patients)

test_eegs = set(test["eeg_id"].astype(np.int64).unique().tolist())
train_eegs = set(eeg_prob_means.index.astype(np.int64).tolist())
seen_e = len(test_eegs & train_eegs)
unseen_e = len(test_eegs - train_eegs)

test_specs = set(test["spectrogram_id"].astype(np.int64).unique().tolist())
train_specs = set(spect_prob_means.index.astype(np.int64).tolist())
seen_s = len(test_specs & train_specs)
unseen_s = len(test_specs - train_specs)

print(
    f"Unique test patients:     {len(test_patients)} | seen in train: {seen_p} | unseen: {unseen_p}"
)
print(
    f"Unique test eeg_id:       {len(test_eegs)} | seen in train: {seen_e} | unseen: {unseen_e}"
)
print(
    f"Unique test spectrograms: {len(test_specs)} | seen in train: {seen_s} | unseen: {unseen_s}"
)




## === cell 1
def make_submission_hierarchical_prior(
    sample_submission: pd.DataFrame,
    test_df: pd.DataFrame,
    global_prior: np.ndarray,
    spect_prob_means: pd.DataFrame,
    eeg_prob_means: pd.DataFrame,
    patient_prob_means: pd.DataFrame,
    ps_prob_means: pd.DataFrame,
    pe_prob_means: pd.DataFrame,
    spect_strength: pd.Series,
    eeg_strength: pd.Series,
    patient_strength: pd.Series,
    ps_strength: pd.Series,
    pe_strength: pd.Series,
    target_cols: list[str],
    alpha_ps: float = 1.0,
    alpha_pe: float = 1.0,
    alpha_spect: float = 1.0,
    alpha_eeg: float = 1.0,
    alpha_patient: float = 2.0,
    base_blend_ps: float = 0.72,
    base_blend_pe: float = 0.68,
    base_blend_spect: float = 0.60,
    base_blend_eeg: float = 0.55,
    base_blend_patient: float = 0.70,
    strength_scale_ps: float = 6.0,
    strength_scale_pe: float = 6.0,
    strength_scale_spect: float = 12.0,
    strength_scale_eeg: float = 10.0,
    strength_scale_patient: float = 8.0,
    final_temperature: float = 1.12,
    max_level_blend: float = 0.90,
) -> pd.DataFrame:
    """
    Create submission aligned to sample_submission order.

    Soft hierarchical blend: global -> patient -> (patient,eeg) -> eeg -> (patient,spectrogram) -> spectrogram.
    """
    sample_eeg_ids = sample_submission["eeg_id"].astype(np.int64).to_numpy()

    test_map = test_df[["eeg_id", "patient_id", "spectrogram_id"]].copy()
    test_map["eeg_id"] = test_map["eeg_id"].astype(np.int64)
    test_map["patient_id"] = test_map["patient_id"].astype(np.int64)
    test_map["spectrogram_id"] = test_map["spectrogram_id"].astype(np.int64)

    pid_by_eeg = dict(
        zip(test_map["eeg_id"].to_numpy(), test_map["patient_id"].to_numpy())
    )
    sid_by_eeg = dict(
        zip(test_map["eeg_id"].to_numpy(), test_map["spectrogram_id"].to_numpy())
    )

    pids = np.array([pid_by_eeg.get(eid, -1) for eid in sample_eeg_ids], dtype=np.int64)
    sids = np.array([sid_by_eeg.get(eid, -1) for eid in sample_eeg_ids], dtype=np.int64)

    K = len(target_cols)
    eps = 1e-12
    probs = np.empty((len(sample_eeg_ids), K), dtype=np.float64)

    spect_index = spect_prob_means.index.to_numpy(dtype=np.int64)
    spect_to_row = {sid: i for i, sid in enumerate(spect_index)}
    eeg_index = eeg_prob_means.index.to_numpy(dtype=np.int64)
    eeg_to_row = {eid: i for i, eid in enumerate(eeg_index)}
    patient_index = patient_prob_means.index.to_numpy(dtype=np.int64)
    patient_to_row = {pid: i for i, pid in enumerate(patient_index)}

    ps_to_row = {k: i for i, k in enumerate(ps_prob_means.index.to_list())}
    pe_to_row = {k: i for i, k in enumerate(pe_prob_means.index.to_list())}

    spect_strength_map = spect_strength.to_dict()
    eeg_strength_map = eeg_strength.to_dict()
    patient_strength_map = patient_strength.to_dict()
    ps_strength_map = ps_strength.to_dict()
    pe_strength_map = pe_strength.to_dict()

    tau_patient = float(strength_scale_patient)
    tau_pe = float(strength_scale_pe)
    tau_eeg = float(strength_scale_eeg)
    tau_ps = float(strength_scale_ps)
    tau_spect = float(strength_scale_spect)

    base_blend_ps = min(base_blend_ps * 0.92, 1.0)
    base_blend_pe = min(base_blend_pe * 0.92, 1.0)
    base_blend_spect = min(base_blend_spect * 0.92, 1.0)
    base_blend_eeg = min(base_blend_eeg * 0.92, 1.0)
    base_blend_patient = min(base_blend_patient * 0.92, 1.0)

    for i, (sid, eid, pid) in enumerate(zip(sids, sample_eeg_ids, pids)):
        p = global_prior.copy()

        def _shrunk_prior(
            mean_vec: np.ndarray, n_strength: float, tau: float, alpha_floor: float
        ) -> np.ndarray:
            n = float(n_strength)
            if n <= 0:
                return global_prior
            n_eff = n + float(alpha_floor)
            mean = np.clip(mean_vec, eps, None)
            mean = mean / mean.sum()
            prior = (n_eff * mean + tau * global_prior) / (n_eff + tau)
            prior = np.clip(prior, eps, None)
            prior = prior / prior.sum()
            return prior

        pid_row = patient_to_row.get(pid, None)
        if pid_row is not None:
            mean_vec = patient_prob_means.iloc[pid_row].to_numpy(dtype=np.float64)
            strength = float(patient_strength_map.get(int(pid), 0.0))
            patient_prior = _shrunk_prior(
                mean_vec, strength, tau_patient, alpha_patient
            )
            w = base_blend_patient * (strength / (strength + strength_scale_patient))
            w = min(float(w), float(max_level_blend))
            p = (1.0 - w) * p + w * patient_prior

        pe_key = (int(pid), int(eid))
        pe_row = pe_to_row.get(pe_key, None)
        if pe_row is not None:
            mean_vec = pe_prob_means.iloc[pe_row].to_numpy(dtype=np.float64)
            strength = float(pe_strength_map.get(pe_key, 0.0))
            pe_prior = _shrunk_prior(mean_vec, strength, tau_pe, alpha_pe)
            w = base_blend_pe * (strength / (strength + strength_scale_pe))
            w = min(float(w), float(max_level_blend))
            p = (1.0 - w) * p + w * pe_prior

        eeg_row = eeg_to_row.get(eid, None)
        if eeg_row is not None:
            mean_vec = eeg_prob_means.iloc[eeg_row].to_numpy(dtype=np.float64)
            strength = float(eeg_strength_map.get(int(eid), 0.0))
            eeg_prior = _shrunk_prior(mean_vec, strength, tau_eeg, alpha_eeg)
            w = base_blend_eeg * (strength / (strength + strength_scale_eeg))
            w = min(float(w), float(max_level_blend))
            p = (1.0 - w) * p + w * eeg_prior

        ps_key = (int(pid), int(sid))
        ps_row = ps_to_row.get(ps_key, None)
        if ps_row is not None:
            mean_vec = ps_prob_means.iloc[ps_row].to_numpy(dtype=np.float64)
            strength = float(ps_strength_map.get(ps_key, 0.0))
            ps_prior = _shrunk_prior(mean_vec, strength, tau_ps, alpha_ps)
            w = base_blend_ps * (strength / (strength + strength_scale_ps))
            w = min(float(w), float(max_level_blend))
            p = (1.0 - w) * p + w * ps_prior

        spect_row = spect_to_row.get(sid, None)
        if spect_row is not None:
            mean_vec = spect_prob_means.iloc[spect_row].to_numpy(dtype=np.float64)
            strength = float(spect_strength_map.get(int(sid), 0.0))
            spect_prior = _shrunk_prior(mean_vec, strength, tau_spect, alpha_spect)
            w = base_blend_spect * (strength / (strength + strength_scale_spect))
            w = min(float(w), float(max_level_blend))
            p = (1.0 - w) * p + w * spect_prior

        probs[i] = p

    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum(axis=1, keepdims=True)

    t = float(final_temperature)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("final_temperature must be finite and > 0")
    if abs(t - 1.0) > 1e-12:
        logp = np.log(np.clip(probs, eps, 1.0))
        logp = logp / t
        logp = logp - logp.max(axis=1, keepdims=True)
        probs = np.exp(logp)
        probs = np.clip(probs, eps, None)
        probs = probs / probs.sum(axis=1, keepdims=True)

    out = sample_submission[["eeg_id"]].copy()
    out[target_cols] = probs
    return out


sol = make_submission_hierarchical_prior(
    sample_submission=sample_sub,
    test_df=test,
    global_prior=global_prior,
    spect_prob_means=spect_prob_means,
    eeg_prob_means=eeg_prob_means,
    patient_prob_means=patient_prob_means,
    ps_prob_means=ps_prob_means,
    pe_prob_means=pe_prob_means,
    spect_strength=spect_strength,
    eeg_strength=eeg_strength,
    patient_strength=patient_strength,
    ps_strength=ps_strength,
    pe_strength=pe_strength,
    target_cols=TARGET_COLS,
    alpha_ps=1.0,
    alpha_pe=1.0,
    alpha_spect=1.0,
    alpha_eeg=1.0,
    alpha_patient=2.0,
    base_blend_ps=0.72,
    base_blend_pe=0.68,
    base_blend_spect=0.60,
    base_blend_eeg=0.55,
    base_blend_patient=0.70,
    strength_scale_ps=6.0,
    strength_scale_pe=6.0,
    strength_scale_spect=12.0,
    strength_scale_eeg=10.0,
    strength_scale_patient=8.0,
    final_temperature=1.12,
    max_level_blend=0.90,
)

assert list(sol.columns) == ["eeg_id"] + TARGET_COLS
assert sol[TARGET_COLS].isna().sum().sum() == 0
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-9):
    raise ValueError(f"Row sums not 1.0; min={row_sums.min()}, max={row_sums.max()}")

print("Submission dataframe ready:", sol.shape)
print(sol.head())



## === cell 2
sub_path = os.path.join(OUT_PATH, "submission.csv")
sol.to_csv(sub_path, index=False)
print(f"Wrote: {sub_path} with shape={sol.shape}")
print(sol.describe(include="all"))
