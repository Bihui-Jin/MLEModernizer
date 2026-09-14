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

No external packages required in the script and installed.

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

0.4124392191517156

# 6. Current score

1.25414

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'Your notebook didn’t yield a score because it relies on external Kaggle Datasets (`/kaggle/input/hms-mk-codes`, `/kaggle/input/requirements-mk`, `/kaggle/input/hms-mk-data`) that are not present in your provided environment, so it can’t run end-to-end to generate `submission.csv`. To make it run and produce a valid submission within the given constraints, I’m replacing the missing external pipeline with a minimal, deterministic baseline that reads only the provided competition files and outputs a properly-normalized probability submission. This generate a valid `submission.csv` (so you can finally obtain a Kaggle score) and should land in a reasonable KL range by using the train-set mean class distribution (a standard safe baseline for KL). The core “training approach” in your original code cannot be preserved because the required code/checkpoint assets are unavailable; this is the smallest viable change to unblock scoring.'
- What this solution (achieved 1.01346) has done: 'I fix the merge failure by removing the incorrect `validate="one_to_one"` constraint and instead safely mapping `patient_id` from `test.csv` to the submission `eeg_id` order (test has repeated `eeg_id` in some environments). I also add a guard to ensure the mapping is deterministic by dropping duplicates on `eeg_id` (keeping the first), and I keep the rest of your probability-prior logic unchanged to preserve evaluation semantics. Finally, I ensure `out_file` is always defined by writing the submission after successful construction, so the last cell can read/print it without crashing.'
- What this solution (achieved 0.97422) has done: 'Your current solution is a patient-conditional prior with a global fallback; to move the KL score down toward the target, the smallest safe improvement is to reduce overconfidence by smoothing every prediction toward the global class distribution (a standard calibration step for KL). This preserves your core logic (same priors, same mapping) but adds one mixing hyperparameter `alpha` to shrink patient means toward the global mean, which typically improves KL on this competition. I also compute the global mean as a **vote-weighted** mean across all training rows (instead of averaging per-row normalized distributions), which better matches the label-generation process and usually reduces KL. Finally, I keep strict normalization and submission formatting unchanged.'
- What this solution (achieved 1.00185) has done: 'To move KL down toward the target with minimal risk, I keep your patient-conditional prior + global fallback exactly as-is, and only improve the *patient mean estimation* to better match the label-generation process. Specifically, instead of taking a simple mean of per-row normalized distributions (which overweights rows with few annotators), I compute a **vote-weighted patient distribution** by summing raw votes per patient and normalizing—this typically reduces KL while preserving your core “prior” logic. I keep your existing global vote-weighted mean, smoothing toward global with the same `alpha`, and strict normalization to ensure a valid submission. No model/training loops are introduced; runtime remains fast and within limits.'
- What this solution (achieved 1.19019) has done: 'We keep your patient-conditional prior + global fallback exactly the same, but tune the single smoothing hyperparameter `alpha` in a deterministic, minimal way using a fast patient-wise cross-validation on the training data itself. This directly targets the KL metric: for unseen patients in validation we use the global prior, and for seen patients we use the patient prior (then apply the same smoothing), matching test-time behavior. This adds no new model, no feature extraction, and stays within runtime by evaluating only a small grid of `alpha` values with vectorized aggregation. Finally, we write `submission.csv` exactly as before with strict normalization.'
- What this solution (achieved 1.32901) has done: 'Your current score is worse than the target (lower is better), so we should cautiously improve KL with the smallest change that keeps your patient-prior core logic intact. The biggest likely issue is that the CV used to pick `alpha` is patient-wise, but its fold assignment is based on factorized patient codes modulo K, which can create imbalanced folds and a suboptimal `alpha`. I switch to a deterministic, balanced fold assignment over unique patients (round-robin after sorting by patient total vote mass) and slightly expand the `alpha` grid (still tiny and fast) so we can pick a better smoothing strength without changing the model. Everything else (vote-weighted patient priors, global fallback, smoothing, strict normalization, submission format/path) remains the same.'
- What this solution (achieved 1.32901) has done: 'Your current score (1.32901, lower-is-better) is much worse than the target (0.4124), so we should improve KL with the smallest possible change while keeping your “patient prior + global fallback + smoothing” core logic intact. The biggest remaining weakness is that patient priors are estimated from *all* training rows, including many near-duplicates from overlapping segments, which can distort patient distributions; we instead compute vote-weighted priors on a deduplicated label unit (`label_id`) to better match the ground-truth generation and reduce noise. We keep your balanced patient-wise CV selection of `alpha`, but run it on the same deduplicated labels so it selects a smoothing strength consistent with the revised priors. All I/O paths, submission schema, and normalization remain unchanged, and runtime stays fast (CSV-only).'
- What this solution (achieved 1.38094) has done: 'We keep your exact “patient prior + global fallback + alpha smoothing” logic, and only adjust how alpha is selected so it better matches the leaderboard objective (KL on the unknown test distribution). The smallest relevant change is to tune alpha using a more realistic CV simulation of test-time behavior by grouping at the `eeg_id` level (since train has many overlapping windows per `eeg_id`, while test has one row per `eeg_id`), while still enforcing patient-wise folds to avoid leakage. Concretely: aggregate labels to one distribution per `eeg_id` (vote-summed, then normalized), then run the same balanced patient-fold CV on those `eeg_id` units to pick alpha; the final training priors and submission generation remain unchanged. This should reduce noise/overlap distortion during alpha selection and move KL down toward your target without changing the prediction formula.'
- What this solution (achieved 1.25414) has done: 'Your current KL (1.38094, lower-is-better) is far worse than the target (0.4124), so we should improve it with the smallest change that keeps your “patient prior + global fallback + alpha smoothing” logic intact. The most likely issue is distribution shift between train and test: test contains many patients not (or barely) represented in train, so a single global prior is too crude. I add one extra backoff level: a **patient-conditional prior blended with a recording-conditional prior (by `eeg_id`)** built from train (vote-summed), then keep the same alpha smoothing toward the global prior; this is still the same prior-based approach, just with a slightly better base distribution when an `eeg_id` has historical labels. I also tune **both** blending weights (alpha smoothing to global, and beta mixing eeg-level vs patient-level) using the same patient-wise CV framework on `eeg_id`-aggregated labels, keeping everything deterministic and fast (CSV-only).'
- What this solution (achieved 1.25414) has done: 'Your current KL is far above the target (lower is better), so the smallest likely win without changing the core “prior + backoff + smoothing” logic is to fix two CV/estimation mismatches that can select bad (alpha, beta). First, in CV you build an `oof_eeg_prob_df` from the already-aggregated `cv_agg` rows, but you then group again by `eeg_id`, which is redundant and can distort the intended “eeg-level prior”; we instead use the true `eeg_id`-level distributions directly for OOF. Second, we should prevent leakage for the eeg-level prior by only allowing an eeg-level prior if that `eeg_id` exists in the OOF training portion (otherwise fall back), which better matches test-time behavior and should reduce KL. Everything else (vote-weighted priors, blending formula, normalization, output schema/path) remains unchanged.'
- What this solution (achieved 1.25414) has done: 'We keep your exact “patient prior + eeg prior + global smoothing” prediction formula, and focus only on making the CV tuning match test-time behavior better so it can pick a less harmful (alpha, beta). The main minimal fix is to prevent leakage and misuse in the eeg-level prior during CV: we should only use an eeg-level prior for a validation eeg_id if that eeg_id exists in the training portion of that fold (otherwise fall back), and we should build that prior from OOF training rows (vote-summed per eeg_id) rather than incorrectly reusing y_true from the validation units. This keeps everything CSV-only and deterministic, but should move KL down toward your target by selecting more sensible blending weights. Submission writing/normalization stays unchanged and still produces `submission.csv`.'
- What this solution (achieved 1.25414) has done: 'We keep your exact “patient prior + eeg prior + global smoothing” submission formula unchanged, and only fix the CV tuning so it no longer (a) accidentally builds the EEG prior from validation targets and (b) uses `cv_agg` in a way that makes the EEG prior effectively unavailable in CV (because each `eeg_id` appears once after aggregation). Concretely, during each fold we build the EEG-level vote-summed prior from the *fold’s training rows in the original `train_label` table*, so the `(alpha, beta)` selection better matches the real test-time behavior where an EEG prior exists only if that `eeg_id` was seen in training. This is a minimal, metric-aligned change that should reduce KL (move toward your 0.412 target) without changing architecture/training/feature extraction (still CSV-only, deterministic, fast). Submission writing, normalization, and paths remain exactly the same.'
- What this solution (achieved 1.25414) has done: 'We keep your exact prior/backoff prediction formula, but fix a key mismatch: your `eeg_pred` currently uses `eeg_mean_df` built from **deduplicated `label_id` rows**, while the CV that selects `(alpha, beta)` evaluates using `eeg_id`-aggregated targets from **all train rows**. The smallest score-improving change is to make the EEG-level prior estimation consistent with the CV/unit-of-evaluation by building `eeg_mean_df` from the same `train.groupby("eeg_id").sum()` vote aggregation (vote-weighted), while keeping patient/global priors as-is. This should reduce the CV-to-test mismatch and move KL down toward the target without changing the core “patient prior + eeg prior + global smoothing” logic. Submission formatting, normalization, and paths remain unchanged and it still writes `submission.csv`.'
- What this solution (achieved 1.25414) has done: 'Your current score is far worse than the target (KL lower-is-better), so we should make a small change that legitimately improves calibration without altering your core “prior/backoff + smoothing” logic. The biggest minimal win is to fix a mismatch: you tune `(alpha, beta)` on one set of priors (fold-specific patient/eeg priors), but you generate the final submission using full-data priors with a different “availability” behavior, which can make the tuned weights harmful. I keep your exact prediction formula, but during CV I precompute fold-specific patient and eeg priors using the same aggregation logic as final inference (vote-summed then normalized) and use fast reindex lookups instead of rebuilding DataFrames per fold. This makes CV match inference semantics more closely (less mismatch/leakage), which should move KL downward toward the target while staying CSV-only and deterministic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
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

train_csv = os.path.join(DATA_PATH, "train.csv")
test_csv = os.path.join(DATA_PATH, "test.csv")
sample_sub_csv = os.path.join(DATA_PATH, "sample_submission.csv")

assert os.path.exists(train_csv), f"Missing: {train_csv}"
assert os.path.exists(test_csv), f"Missing: {test_csv}"
assert os.path.exists(sample_sub_csv), f"Missing: {sample_sub_csv}"



## === cell 2
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
sub = pd.read_csv(sample_sub_csv)

assert "eeg_id" in test.columns
assert "eeg_id" in sub.columns
assert (
    "patient_id" in train.columns and "patient_id" in test.columns
), "patient_id required for conditional prior"
for c in TARGET_COLS:
    assert c in train.columns, f"train missing target col {c}"
    assert c in sub.columns, f"sample_submission missing target col {c}"

sub = sub[["eeg_id"] + TARGET_COLS].copy()



## === cell 3
eps = 1e-6

if "label_id" in train.columns:
    train_label = train.groupby("label_id", sort=False, as_index=False).agg(
        patient_id=("patient_id", "first"),
        eeg_id=("eeg_id", "first"),
        **{c: (c, "sum") for c in TARGET_COLS},
    )
else:
    train_label = train[["patient_id", "eeg_id"] + TARGET_COLS].copy()

label_votes = train_label[TARGET_COLS].to_numpy(dtype=np.float64)

global_votes = label_votes.sum(axis=0)
global_mean = global_votes / max(global_votes.sum(), 1.0)
global_mean = np.clip(global_mean, eps, 1.0)
global_mean = global_mean / global_mean.sum()

patient_votes_df = train_label.groupby("patient_id", sort=False)[TARGET_COLS].sum()
patient_votes = patient_votes_df.to_numpy(dtype=np.float64)
patient_sums = patient_votes.sum(axis=1, keepdims=True)
patient_sums[patient_sums == 0.0] = 1.0
patient_probs = patient_votes / patient_sums
patient_probs = np.clip(patient_probs, eps, 1.0)
patient_probs = patient_probs / patient_probs.sum(axis=1, keepdims=True)
patient_mean_df = patient_votes_df.copy()
patient_mean_df.loc[:, TARGET_COLS] = patient_probs

eeg_votes_df = train.groupby("eeg_id", sort=False)[TARGET_COLS].sum()
eeg_votes = eeg_votes_df.to_numpy(dtype=np.float64)
eeg_sums = eeg_votes.sum(axis=1, keepdims=True)
eeg_sums[eeg_sums == 0.0] = 1.0
eeg_probs = eeg_votes / eeg_sums
eeg_probs = np.clip(eeg_probs, eps, 1.0)
eeg_probs = eeg_probs / eeg_probs.sum(axis=1, keepdims=True)
eeg_mean_df = eeg_votes_df.copy()
eeg_mean_df.loc[:, TARGET_COLS] = eeg_probs

global_mean




## === cell 4
def kl_divergence(p_true, p_pred, eps=1e-15):
    p_true = np.clip(p_true, eps, 1.0)
    p_pred = np.clip(p_pred, eps, 1.0)
    p_true = p_true / p_true.sum(axis=1, keepdims=True)
    p_pred = p_pred / p_pred.sum(axis=1, keepdims=True)
    return np.mean(np.sum(p_true * (np.log(p_true) - np.log(p_pred)), axis=1))


cv_agg = train.groupby("eeg_id", sort=False, as_index=False).agg(
    patient_id=("patient_id", "first"),
    **{c: (c, "sum") for c in TARGET_COLS},
)

y_votes = cv_agg[TARGET_COLS].to_numpy(dtype=np.float64)
y_sums = y_votes.sum(axis=1, keepdims=True)
y_sums[y_sums == 0.0] = 1.0
y_true = y_votes / y_sums
y_true = np.clip(y_true, eps, 1.0)
y_true = y_true / y_true.sum(axis=1, keepdims=True)

patient_ids = cv_agg["patient_id"].to_numpy()
pid_codes, pid_uniques = pd.factorize(patient_ids, sort=False)
n_pat = len(pid_uniques)

n_folds = 5

pat_vote_mass = np.zeros(n_pat, dtype=np.float64)
np.add.at(pat_vote_mass, pid_codes, y_votes.sum(axis=1))
pat_order = np.argsort(-pat_vote_mass, kind="mergesort")
pat_to_fold = np.empty(n_pat, dtype=np.int32)
for i, p in enumerate(pat_order):
    pat_to_fold[p] = i % n_folds
fold_id = pat_to_fold[pid_codes]

fold_votes = np.zeros((n_folds, len(TARGET_COLS)), dtype=np.float64)
for f in range(n_folds):
    fold_votes[f] = y_votes[fold_id == f].sum(axis=0)
total_votes = y_votes.sum(axis=0)

alpha_grid = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], dtype=np.float64)
beta_grid = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)

best_alpha = 0.35
best_beta = 0.5
best_kl = np.inf

all_eeg_ids = cv_agg["eeg_id"].to_numpy()

fold_global_mean = []
fold_patient_prior = []
fold_seen_patient = []
fold_eeg_prior = []

for f in range(n_folds):
    in_fold = fold_id == f
    oof = ~in_fold

    oof_global_votes = total_votes - fold_votes[f]
    oof_global_mean = oof_global_votes / max(oof_global_votes.sum(), 1.0)
    oof_global_mean = np.clip(oof_global_mean, eps, 1.0)
    oof_global_mean = oof_global_mean / oof_global_mean.sum()
    fold_global_mean.append(oof_global_mean)

    pat_oof_votes = np.zeros((n_pat, len(TARGET_COLS)), dtype=np.float64)
    np.add.at(pat_oof_votes, pid_codes[oof], y_votes[oof])
    pat_oof_sums = pat_oof_votes.sum(axis=1, keepdims=True)
    seen_pat = pat_oof_sums[:, 0] > 0.0
    pat_oof_sums[pat_oof_sums == 0.0] = 1.0
    pat_oof_probs = pat_oof_votes / pat_oof_sums
    pat_oof_probs = np.clip(pat_oof_probs, eps, 1.0)
    pat_oof_probs = pat_oof_probs / pat_oof_probs.sum(axis=1, keepdims=True)
    fold_patient_prior.append(pat_oof_probs)
    fold_seen_patient.append(seen_pat)

    oof_eeg_ids = set(all_eeg_ids[oof].tolist())
    train_oof = train[train["eeg_id"].isin(oof_eeg_ids)]
    if len(train_oof) > 0:
        oof_eeg_votes_df = train_oof.groupby("eeg_id", sort=False)[TARGET_COLS].sum()
        oof_eeg_votes = oof_eeg_votes_df.to_numpy(dtype=np.float64)
        oof_eeg_sums = oof_eeg_votes.sum(axis=1, keepdims=True)
        oof_eeg_sums[oof_eeg_sums == 0.0] = 1.0
        oof_eeg_probs = oof_eeg_votes / oof_eeg_sums
        oof_eeg_probs = np.clip(oof_eeg_probs, eps, 1.0)
        oof_eeg_probs = oof_eeg_probs / oof_eeg_probs.sum(axis=1, keepdims=True)
        fold_eeg_prior.append(
            pd.DataFrame(
                oof_eeg_probs, columns=TARGET_COLS, index=oof_eeg_votes_df.index
            )
        )
    else:
        fold_eeg_prior.append(None)

for alpha in alpha_grid:
    for beta in beta_grid:
        fold_kls = []
        for f in range(n_folds):
            in_fold = fold_id == f

            oof_global_mean = fold_global_mean[f]
            pat_oof_probs = fold_patient_prior[f]
            seen_pat = fold_seen_patient[f]
            eeg_prior_df = fold_eeg_prior[f]

            fold_eeg_ids = all_eeg_ids[in_fold]
            fold_patient_codes = pid_codes[in_fold]

            base_pat = pat_oof_probs[fold_patient_codes]
            unseen_pat_rows = ~seen_pat[fold_patient_codes]
            if np.any(unseen_pat_rows):
                base_pat = base_pat.copy()
                base_pat[unseen_pat_rows] = oof_global_mean.reshape(1, -1)

            if eeg_prior_df is not None:
                base_eeg = eeg_prior_df.reindex(fold_eeg_ids).to_numpy(dtype=np.float64)
                unseen_eeg_rows = np.isnan(base_eeg).any(axis=1)
                if np.any(unseen_eeg_rows):
                    base_eeg = base_eeg.copy()
                    base_eeg[unseen_eeg_rows] = oof_global_mean.reshape(1, -1)
            else:
                base_eeg = np.repeat(
                    oof_global_mean.reshape(1, -1),
                    repeats=fold_eeg_ids.shape[0],
                    axis=0,
                )

            base = beta * base_eeg + (1.0 - beta) * base_pat
            base = np.clip(base, eps, 1.0)
            base = base / base.sum(axis=1, keepdims=True)

            pred_fold = alpha * base + (1.0 - alpha) * oof_global_mean.reshape(1, -1)
            pred_fold = np.clip(pred_fold, eps, 1.0)
            pred_fold = pred_fold / pred_fold.sum(axis=1, keepdims=True)

            fold_kls.append(kl_divergence(y_true[in_fold], pred_fold, eps=1e-15))

        mean_kl = float(np.mean(fold_kls))
        if mean_kl < best_kl:
            best_kl = mean_kl
            best_alpha = float(alpha)
            best_beta = float(beta)

print(
    f"Selected alpha={best_alpha:.2f}, beta={best_beta:.2f} via patient-wise CV on eeg_id-aggregated labels (mean KL={best_kl:.6f})"
)



## === cell 5
test_patients = test[["eeg_id", "patient_id"]].copy()
test_patients = test_patients.drop_duplicates(subset=["eeg_id"], keep="first")

merge_df = pd.DataFrame({"eeg_id": sub["eeg_id"].values}).merge(
    test_patients, on="eeg_id", how="left"
)
assert (
    merge_df["patient_id"].isna().sum() == 0
), "All submission eeg_id must exist in test.csv"

patient_pred = patient_mean_df.reindex(merge_df["patient_id"].values)[
    TARGET_COLS
].to_numpy(dtype=np.float64)
eeg_pred = eeg_mean_df.reindex(merge_df["eeg_id"].values)[TARGET_COLS].to_numpy(
    dtype=np.float64
)

patient_unseen = np.isnan(patient_pred).any(axis=1)
eeg_unseen = np.isnan(eeg_pred).any(axis=1)
if np.any(patient_unseen):
    patient_pred = patient_pred.copy()
    patient_pred[patient_unseen] = global_mean.reshape(1, -1)
if np.any(eeg_unseen):
    eeg_pred = eeg_pred.copy()
    eeg_pred[eeg_unseen] = global_mean.reshape(1, -1)

beta = best_beta
base = beta * eeg_pred + (1.0 - beta) * patient_pred
base = np.clip(base, eps, 1.0)
base = base / base.sum(axis=1, keepdims=True)

alpha = best_alpha
pred = alpha * base + (1.0 - alpha) * global_mean.reshape(1, -1)

pred = np.clip(pred, eps, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

sub.loc[:, TARGET_COLS] = pred

row_sum_check = sub[TARGET_COLS].sum(axis=1).to_numpy()
max_dev = np.max(np.abs(row_sum_check - 1.0))
assert max_dev < 1e-10, f"Row probabilities do not sum to 1 (max deviation {max_dev})"

out_file = os.path.join(OUT_PATH, "submission.csv")
sub.to_csv(out_file, index=False)

out_file, sub.shape



## === cell 6
print(pd.read_csv(out_file).head())
print("Saved:", out_file)
