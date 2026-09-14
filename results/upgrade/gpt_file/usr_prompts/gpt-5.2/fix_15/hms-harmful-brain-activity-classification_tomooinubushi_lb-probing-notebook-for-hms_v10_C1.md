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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.989084

# 6. Current score

1.48862

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I make the smallest changes needed to (1) ensure the notebook runs in Kaggle (your `display()` calls can error outside notebooks), (2) guarantee the submission probabilities are always valid (sum to 1, non-negative, no NaNs) even if your hypothesis logic falls through, and (3) keep your current “constant-mean prior” core logic unchanged so the score is stable and should improve from “Not yielded” to a valid baseline. Concretely, I compute `mean_vote_ratio` directly from `train.csv` vote columns (a legitimate prior) when your hypothesis-based preset ends up NaN, then renormalize the row to sum to one. This avoids invalid submissions and is expected to reduce KL versus a uniform guess while preserving your overall approach.'
- What this solution (achieved 1.41937) has done: 'Your current submission is a constant prior over classes, so the smallest legitimate way to move KL down toward the target is to use a *better prior* without changing the overall “no-model” core logic. I keep the same structure (compute a single probability vector and assign it to every test row), but compute that vector more robustly by (1) aggregating votes at the `eeg_id` level (matching the submission granularity) and (2) applying light Bayesian/Laplace smoothing so no class gets overconfident and KL is typically reduced. I also remove the expensive directory listing/parquet scanning (not used for predictions) to keep runtime safely under 600s. The output remains a valid `submission.csv` with per-row probabilities summing to 1.'
- What this solution (achieved 1.3976) has done: 'Your current approach predicts the same class-probability vector for every test row; the smallest legitimate way to reduce KL (lower-is-better) toward 0.989084 is to make that single vector closer to the label distribution at the submission granularity (`eeg_id`). I keep the constant-prior core logic unchanged, but compute the prior from **per-`eeg_id` normalized vote distributions** (instead of globally summing raw votes across EEGs), which better matches how each EEG contributes to the objective and typically improves calibration. I also switch to a very light Jeffreys-style smoothing (`alpha=0.5`) to avoid tiny probabilities without making the prior too uniform, then re-normalize to guarantee valid rows. Everything else (no model, no feature extraction) remains the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.43163) has done: 'Your current submission is a single constant prior for all test EEGs; to move KL down toward the target (lower-is-better), the smallest legitimate improvement is to make that prior better match the *test-set* class mix without changing the “no model / constant vector” core logic. We can do that by computing the prior at the correct unit (`eeg_id`) as you already do, but additionally reweight each training EEG by how similar its `patient_id` appears in the test set (a light domain-adaptation step using metadata only). This keeps identical semantics (still one probability vector for every row) while typically improving calibration vs. a pure global prior. We keep smoothing and strict renormalization so the submission is always valid.'
- What this solution (achieved 1.39412) has done: 'Your current approach is still a single constant probability vector for every test EEG, so the only safe way to reduce KL (lower-is-better) without changing core logic is to make that one vector better calibrated. I keep the same “constant prior for all rows” structure, but compute the prior from training **vote proportions** (votes normalized per row) and then average them with **test-patient reweighting**, which better matches how the competition target is a distribution per labeled segment. I also fix a small bug where the smoothing used an incorrect “sample size” proxy (`n_eegs` rather than a weight-based effective count), which can over/under-smooth and hurt KL; the correction keeps the same smoothing idea but makes it consistent with your weighting. Finally, I keep the strict renormalization so the submission is always valid and sums to 1.'
- What this solution (achieved 1.43081) has done: 'Your current solution is a constant probability vector for every test `eeg_id`, so the only safe way to lower KL toward 0.989084 without changing core logic is to make that single vector a better-calibrated prior. I keep the “one vector for all rows” approach, but compute the prior at the submission granularity by first aggregating training vote-probabilities to `eeg_id` (so overlapping segments don’t overweight some EEGs), then averaging those EEG-level distributions with the same test-patient reweighting you already use. I keep your same Jeffreys smoothing idea, but apply it using the effective number of EEGs after weighting (consistent with this aggregation), then strictly renormalize to guarantee a valid submission.'
- What this solution (achieved 1.46383) has done: 'Your current submission uses one constant class-probability vector, so the smallest safe way to reduce KL toward the target is to make that single vector better calibrated without changing the overall “constant prior for all rows” logic. The main issue is that the current patient reweighting can be noisy and can overfit the test patient mix, which has been hurting score (your score regressed). I keep the same EEG-level aggregation and Jeffreys smoothing, but replace the high-variance per-patient ratio weights with a low-variance convex blend between the unweighted global prior and the patient-reweighted prior (shrinkage), which typically improves KL stability. I also compute the unweighted prior from the exact same EEG-level distributions to keep semantics consistent and then apply the same strict renormalization to guarantee a valid submission.'
- What this solution (achieved 1.48692) has done: 'Your current score (1.46383, lower-is-better) is worse than the target (0.989084), and the recent regression suggests the test-patient reweighting is adding noise rather than helping. To move the KL down with minimal change while preserving the “single constant probability vector for all rows” core logic, I disable the patient reweighting by setting the shrinkage weight to 0 (i.e., use the robust EEG-level global prior only). I keep your EEG-level aggregation and Jeffreys-style smoothing exactly as-is (same logic, just removing the noisy blend term), and keep the strict renormalization so the submission is always valid. This should move you back toward your previously better constant-prior scores (~1.39) without changing the fundamental approach.'
- What this solution (achieved 1.48862) has done: 'To move your KL score down toward the target while preserving the “single constant probability vector for all rows” core logic, I adjust only how that constant prior is computed: instead of averaging per-row vote proportions (which overweights EEGs with many overlapping segments), compute the prior from **vote counts aggregated at `eeg_id`** and then normalize to a distribution per EEG before averaging across EEGs. This keeps the same constant-vector submission semantics but typically improves calibration at the submission granularity and should recover/beat your earlier ~1.39 scores (reducing the gap to 0.989084). I keep your Jeffreys smoothing (`alpha=0.5`) and strict renormalization to guarantee valid probabilities summing to 1. No model/training/feature extraction is introduced, and runtime stays well under 600s.'
- What this solution (achieved 1.41922) has done: 'Your current solution always predicts a single constant probability vector, so the only safe way to move the KL score down (lower-is-better) without changing the core logic is to make that one vector closer to the true marginal distribution at the *submission unit* (`eeg_id`). I keep the “one vector for all rows” approach, but compute the prior as the **average of per-`eeg_id` vote distributions weighted by each EEG’s total number of votes**, which better reflects label reliability and typically reduces KL versus an unweighted average. I keep your Jeffreys smoothing and strict renormalization exactly as before to avoid invalid rows, and I leave patient reweighting disabled (since it previously regressed). This is a minimal, targeted change confined to how the constant prior is estimated.'
- What this solution (achieved 1.41937) has done: 'Your current pipeline is a constant-prior submission, so the only safe way to reduce KL (lower-is-better) without changing the core approach is to make that single probability vector closer to the evaluation unit’s marginal target distribution. I keep your exact “one vector for all rows” logic, but change the prior estimation from a vote-weighted mean of per-EEG distributions to a **Dirichlet-multinomial posterior mean computed from aggregated vote counts** across EEGs (still using only train labels, no test leakage). This is a minimal, targeted calibration change that typically improves KL for distributional targets because it matches the generative assumption behind vote counts, while retaining your Jeffreys-style smoothing (alpha=0.5) in a mathematically consistent way. I also keep the strict renormalization/positivity guardrails so the submission always validates.'
- What this solution (achieved 1.3976) has done: 'Your current solution is a single constant probability vector, so the only minimal way to reduce KL toward the target is to make that one vector better match the per-EEG label distribution seen in train (same submission granularity) without introducing any modeling. I keep the constant-prior core logic, but compute the prior from EEG-level normalized vote distributions (rather than raw global count totals), and then apply the same small Jeffreys smoothing and strict renormalization so the submission is always valid. This typically improves calibration for KL because it reduces overweighting EEGs/rows with more total votes. No feature extraction, model training, or loop changes are introduced, and it still writes `submission.csv` in `/kaggle/working/`.'
- What this solution (achieved 1.41937) has done: 'I keep your “single constant probability vector for every test EEG” core logic unchanged, but make that vector closer to the training marginal at the **submission unit** by using the Dirichlet-multinomial posterior mean from **aggregated vote counts across EEGs** (instead of averaging per-EEG normalized distributions). This is a minimal calibration change that usually lowers KL for this competition because the targets are vote-count–derived distributions and the posterior mean matches that generative structure. I also remove the accidental “add alpha to probabilities” step (which is not the intended Jeffreys smoothing) and instead apply Jeffreys smoothing correctly as a prior on counts. Submission formatting, strict renormalization, and file paths remain the same.'
- What this solution (achieved 1.48862) has done: 'Your current approach is a single constant probability vector, and the score (1.41937, lower-is-better) is still far above the target (0.989084), so the smallest safe improvement is to make that constant vector closer to the true marginal target distribution while keeping the “no model” core logic unchanged. The main calibration issue is that you first aggregate vote counts to `eeg_id` but then you immediately discard that and compute the posterior from global totals, which can overweight EEGs with many overlapping segments in train. I keep the same Dirichlet-multinomial/Jeffreys smoothing idea, but compute it from **EEG-level normalized vote distributions** (submission granularity) and then convert that to a Dirichlet posterior mean with an effective sample size, which usually reduces KL for this competition while preserving the constant-prior semantics. I also keep strict non-negativity and renormalization so the submission always validates.'

# 9. Code solution

## === cell 0
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set(style="whitegrid")
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
all_df = pd.concat([train, test]).reset_index(drop=True)

try:
    display(train.head())
    display(test.head())
    display(all_df.head())
except NameError:
    print(train.head())
    print(test.head())
    print(all_df.head())



## === cell 1
print(len(train.eeg_id.unique()))
print(len(train.spectrogram_id.unique()))
print(len(train.patient_id.unique()))
print(len(train.eeg_id.unique()) / len(train.patient_id.unique()))
print(len(train.spectrogram_id.unique()) / len(train.patient_id.unique()))



## === cell 2
train_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
test_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
test_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

train_spectrogram_files = []
test_spectrogram_files = []
train_eeg_files = []
test_eeg_files = []

print(
    "Skipped listing parquet directories (not needed for this constant-prior submission)."
)




## === cell 3
def get_files_info(files, file_dir):
    nan_ratio = []
    shapes = []
    for file in files:
        data = np.array(pd.read_parquet(f"{file_dir}{file}"))
        nan_ratio.append(np.isnan(data).sum() / len(data.flatten()))
        shapes.append(data.shape)
    nan_ratio = np.array(nan_ratio)
    shapes = np.array(shapes)
    return nan_ratio, shapes




## === cell 4
hypothesis0 = False
hypothesis1 = False

if len(train.spectrogram_id.unique()) / len(train.patient_id.unique()) < 5:
    hypothesis0 = True

if len(all_df.eeg_id.unique()) == (
    len(train.eeg_id.unique()) + len(test.eeg_id.unique())
):
    if len(all_df.patient_id.unique()) == (
        len(train.patient_id.unique()) + len(test.patient_id.unique())
    ):
        if len(all_df.spectrogram_id.unique()) == (
            len(train.spectrogram_id.unique()) + len(test.spectrogram_id.unique())
        ):
            hypothesis1 = True

print(f"hypothesis0: {hypothesis0}")
print(f"hypothesis1: {hypothesis1}")



## === cell 5
hypotheses = []
hypotheses.append(len(test.eeg_id.unique()) == len(test))
hypotheses.append(len(test.spectrogram_id.unique()) == len(test))
hypotheses.append(len(test.patient_id.unique()) != len(test))
hypotheses.append(True)  # file count checks skipped (see cell 3)
hypotheses.append(True)  # file count checks skipped (see cell 3)
hypotheses.append(
    len(train.spectrogram_id.unique()) / len(train.patient_id.unique()) < 6
)
hypotheses.append(
    len(train.spectrogram_id.unique()) / len(train.patient_id.unique()) > 4
)

print(f"hypotheses: {hypotheses}")
hypotheses = all(hypotheses)
print(f"hyposetheses: {hypotheses}")



## === cell 6
sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

targets = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_votes = train[["eeg_id", "patient_id"] + targets].copy()
for c in targets:
    train_votes[c] = pd.to_numeric(train_votes[c], errors="coerce").fillna(0.0)

eeg_counts = train_votes.groupby("eeg_id", as_index=False)[targets].sum()

counts_mat = eeg_counts[targets].to_numpy(dtype=float)
counts_mat = np.nan_to_num(counts_mat, nan=0.0, posinf=0.0, neginf=0.0)
counts_mat = np.clip(counts_mat, 0.0, None)

row_sums = counts_mat.sum(axis=1, keepdims=True)
valid = row_sums[:, 0] > 0

K = len(targets)
alpha = 0.5  # Jeffreys prior per class (kept)

p_eeg = np.full((counts_mat.shape[0], K), 1.0 / K, dtype=float)
p_eeg[valid] = counts_mat[valid] / row_sums[valid]

p_bar = p_eeg.mean(axis=0)
p_bar = np.nan_to_num(p_bar, nan=1.0 / K, posinf=1.0 / K, neginf=1.0 / K)
p_bar = np.clip(p_bar, 1e-15, None)
p_bar = p_bar / p_bar.sum()

n_eff = float(p_eeg.shape[0])
posterior_mean = (n_eff * p_bar + alpha) / (n_eff + alpha * K)

posterior_mean = np.nan_to_num(
    posterior_mean, nan=1.0 / K, posinf=1.0 / K, neginf=1.0 / K
)
posterior_mean = np.clip(posterior_mean, 1e-15, None)
posterior_mean = posterior_mean / posterior_mean.sum()

mean_vote_ratio = dict(zip(targets, posterior_mean.tolist()))

for target in targets:
    sub[target] = float(mean_vote_ratio[target])

vals = sub[targets].to_numpy(dtype=float)
vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
vals = np.clip(vals, 1e-15, None)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[targets] = vals

try:
    display(sub.head())
except NameError:
    print(sub.head())



## === cell 7
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", sub.shape)
print(
    "Row-sum check (min/max):",
    sub[targets].sum(axis=1).min(),
    sub[targets].sum(axis=1).max(),
)
