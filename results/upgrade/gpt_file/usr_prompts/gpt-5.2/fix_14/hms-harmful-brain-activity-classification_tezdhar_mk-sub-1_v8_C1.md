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

0.3839901784373968

# 6. Current score

0.75919

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39771) has done: 'I remove the hard dependency on the missing `/kaggle/input/hms-mk-codes` package by replacing the external conversion + inference subprocess calls with a small, local baseline that reads the provided `train.csv` and outputs class probabilities per `eeg_id`. This fixes the runtime errors (missing directory, missing ckpt/scripts) and guarantees a correctly formatted `submission.csv` is created. To keep the logic minimal and stable, the prediction be the smoothed per-class mean vote distribution from training, which is a valid probability vector and a reasonable KL baseline. Finally, I keep (and slightly harden) your existing submission validation/normalization so the CSV always passes Kaggle’s row-sum constraint.'
- What this solution (achieved 0.77226) has done: 'Your current submission uses a single global prior (mean class distribution) for every test row, which is a very weak KL baseline and explains the large gap to the target. To move the score down toward 0.384 with minimal logic change, I keep the same “vote-normalize then average” approach but compute the prior per `patient_id` (and fall back to the global prior if a patient is unseen), which usually improves calibration in this competition. I also average at the `eeg_id` level inside each patient to reduce overweighting patients with many overlapping training crops, without changing the overall method. The rest of your submission validation/normalization stays intact to guarantee a valid CSV.'
- What this solution (achieved 1.65469) has done: 'Your current approach is a patient-specific prior with a small shrink toward the global prior; the score gap suggests it’s still too coarse. To move KL down toward the 0.384 target with minimal logic change, I keep the exact same “vote-normalize → average” baseline, but compute priors at a slightly more specific level: `(patient_id, expert_consensus)` (fallback to patient-only, then global). This remains a pure prior model (no EEG/spectrogram feature extraction, no training loop), but usually improves calibration because the test set’s label distribution varies meaningfully by patient and consensus pattern. I also replace the slow Python loop with a vectorized merge/fill, which doesn’t change semantics but is safer and faster within the 600s limit. Submission validation/normalization stays the same so the CSV always passes row-sum constraints.'
- What this solution (achieved 0.77562) has done: 'Your current priors are being selected using a test-time `expert_consensus="Other"`, which almost never matches the training consensus keys, so you’re effectively falling back to the (weaker) patient-only/global priors and losing performance. I keep the exact same “vote-normalize → average → prior with fallbacks” logic, but change the most specific key from `(patient_id, expert_consensus)` to `(patient_id, spectrogram_id)` because `spectrogram_id` exists in both train and test and is a more relevant grouping for this dataset. I also keep the same shrink+alpha smoothing and the same fallback chain, just swapping in the spectrogram-level table first; this should reduce KL (lower is better) toward your 0.384 target with minimal risk. Submission formatting/normalization remains unchanged to guarantee a valid CSV.'
- What this solution (achieved 0.77562) has done: 'Your current approach is a hierarchical prior (patient+spectrogram → patient → global); to move KL down toward the 0.384 target with minimal change, we should make the most-specific prior less noisy and more directly tied to the shared label distribution in train/test. I keep the exact same “vote-normalize → average priors → fallback chain” core logic, but switch the most-specific table from `(patient_id, spectrogram_id)` to `spectrogram_id`-only, since `spectrogram_id` is present in both splits and aggregates more data per key. I also add a tiny count-based shrink so spectrograms with few unique `eeg_id`s rely slightly more on the patient/global priors, improving calibration without changing semantics. Submission validation/normalization remains unchanged to guarantee a valid CSV.'
- What this solution (achieved 0.77562) has done: 'Your current baseline is already valid but too coarse, so to move the KL score down toward the 0.384 target (lower is better) with minimal change, I keep the same “vote-normalize → average priors → hierarchical fallback” core logic and only adjust how the most-specific prior is formed. Specifically, I build a `spectrogram_id` prior that is lightly blended with the `patient_id` prior (instead of shrinking spectrogram directly to global), because patient information is available at test time and usually improves calibration without changing the modeling approach. I also make the shrink factor depend on how many unique `eeg_id`s each spectrogram has in training, so sparse spectrograms rely more on patient/global priors (stability). Submission formatting/normalization remain intact to guarantee a valid CSV.'
- What this solution (achieved 0.77562) has done: 'Your current hierarchical prior is likely failing to improve because the spectrogram-level blend is inadvertently using the wrong columns: after the merge, `spec_val` is taken from the already-most-specific column (or NaNs) instead of the spectrogram-prior column, so the intended count-based shrink toward patient prior is mostly ineffective. I make a minimal fix so the blend uses the spectrogram prior (`*_s`) and only applies when that prior exists, while keeping your core “vote-normalize → average priors → fallback chain” logic unchanged. I also compute the shrink vector once (instead of inside the loop) to keep semantics the same but avoid repeated conversions and potential subtle dtype issues. Submission validation/normalization remains intact to guarantee a valid CSV.'
- What this solution (achieved 0.77562) has done: 'We keep your exact hierarchical “vote-normalize → average priors → fallback chain” approach, but make the most-specific prior less sparse by switching from `spec_patient_df` (spectrogram+patient) to a stronger `eeg_id`-level prior, since `eeg_id` exists in both train and test and directly matches the submission granularity. This is a minimal change (still just averaged label distributions; no EEG/spectrogram feature extraction, no model training) and should reduce KL substantially from the current 0.77562 toward your 0.38399 target (lower is better). We retain your existing spectrogram→patient→global fallback logic (with the same smoothing/shrink ideas) for any `eeg_id` unseen in training. Submission normalization/validation remains unchanged to guarantee a valid CSV with rows summing to 1.'
- What this solution (achieved 0.77562) has done: 'Your score is far above the target (0.77562 vs 0.38399; lower is better), so we need a real improvement but still within your “prior-only” core logic. The biggest low-risk gain is to remove noise from using per-crop vote distributions by first aggregating votes at the `label_id` level (the consolidated label unit), then averaging per `eeg_id`—this keeps the exact same “vote-normalize → group-mean priors → hierarchical fallback” approach but reduces duplicated/overlapping-window bias. I also make the `eeg_id` prior shrink depend on how many unique `label_id`s an `eeg_id` has (more evidence → less shrink), which improves calibration without changing semantics. Finally, I keep your existing prediction merge/fallback and the submission validation/renormalization to guarantee a valid CSV.'
- What this solution (achieved 0.77562) has done: 'We keep your exact “vote-normalize → label-level aggregate → eeg/spec/patient/global priors with shrink + fallback” core logic, but fix a subtle mismatch in the spectrogram shrink: the current `n_eeg` is computed globally per `spectrogram_id`, while the prior you ultimately want to use in test should better reflect how much evidence exists for that *patient’s* spectrogram context. With a minimal change, we compute a `(patient_id, spectrogram_id)` evidence count from training and use it to drive the shrink that blends spectrogram and patient priors at test time, which should reduce KL (lower is better) from 0.77562 toward your 0.38399 target without changing modeling semantics. We keep the existing eeg_id prior as the first choice (no change), and only improve the second-stage spec→patient blend when eeg_id is missing/unseen. Submission formatting/normalization stays intact to guarantee a valid CSV.'
- What this solution (achieved 0.77562) has done: 'Your current score (0.77562, lower is better) is far above the target (0.38399), so we need a meaningful improvement while keeping your “prior-only, vote-normalize → group-mean priors → hierarchical fallback” core logic intact. The biggest low-risk fix is that the model is currently using an `eeg_id` prior built from training crops, but `test.eeg_id` values are essentially unseen, so you almost always fall back to the weaker spec/patient/global priors; we instead build the most-specific prior at the test-matching key `spectrogram_id`. We keep your existing label-level de-duplication and shrinkage idea, but strengthen the spec→patient blend by (a) computing spec evidence using unique `label_id` (not `eeg_id`) and (b) always filling missing predictions from the blended spec/patient prior before falling back to patient/global. These are minimal, semantics-preserving changes that should reduce KL (improve score) without introducing new models or feature extraction, and the submission normalization/validation remains unchanged to guarantee a valid CSV.'
- What this solution (achieved 0.75919) has done: 'Your current hierarchical prior still falls back often and is likely miscalibrated because it uses raw (soft) mean vote distributions as probabilities without any temperature control; KL is very sensitive to overconfident wrong probabilities. I keep your exact prior-building and fallback chain, but add one minimal post-processing step: apply a mild “probability tempering” (power transform) to all predicted rows and re-normalize, which typically improves KL by reducing overconfidence. To avoid overshooting the target, I make this tempering gentle and deterministic, and keep all existing smoothing/normalization checks unchanged. This change is isolated to the final prediction matrix right before writing `submission.csv`, so core logic and semantics remain the same while nudging the score downward (better) toward 0.384.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(0)

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)

SAMPLE_SUB_PATH = os.path.join(DATA_PATH, "sample_submission.csv")
TEST_PATH = os.path.join(DATA_PATH, "test.csv")
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")

SUB_PATH = os.path.join(OUT_PATH, "submission.csv")

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["eeg_id", "patient_id", "spectrogram_id", "label_id"] + TARGET_COLS,
)

votes = train[TARGET_COLS].to_numpy(dtype=np.float64)
row_sums = votes.sum(axis=1, keepdims=True)

zero_mask = row_sums.squeeze() == 0
if np.any(zero_mask):
    votes[zero_mask, :] = 1.0
    row_sums = votes.sum(axis=1, keepdims=True)

probs = votes / row_sums
probs_df = pd.DataFrame(probs, columns=TARGET_COLS)
probs_df["eeg_id"] = train["eeg_id"].values
probs_df["patient_id"] = train["patient_id"].values
probs_df["spectrogram_id"] = train["spectrogram_id"].values
probs_df["label_id"] = train["label_id"].values

label_level = probs_df.groupby(
    ["patient_id", "spectrogram_id", "eeg_id", "label_id"], as_index=False
)[TARGET_COLS].mean()

eeg_level = label_level.groupby(
    ["patient_id", "spectrogram_id", "eeg_id"], as_index=False
)[TARGET_COLS].mean()

alpha = 1e-3

global_prior = eeg_level[TARGET_COLS].mean(axis=0).to_numpy(dtype=np.float64)
global_prior = (global_prior + alpha) / (global_prior.sum() + alpha * len(TARGET_COLS))

patient_prior_df = eeg_level.groupby("patient_id", as_index=False)[TARGET_COLS].mean()

spec_prior_df = label_level.groupby("spectrogram_id", as_index=False)[
    TARGET_COLS
].mean()

eeg_prior_df = eeg_level.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()

spec_counts = label_level.groupby("spectrogram_id", as_index=False).agg(
    n_label=("label_id", "nunique")
)
spec_prior_df = spec_prior_df.merge(spec_counts, on="spectrogram_id", how="left")
spec_prior_df["n_label"] = spec_prior_df["n_label"].fillna(0).astype(np.float64)

spec_patient_counts = label_level.groupby(
    ["patient_id", "spectrogram_id"], as_index=False
).agg(n_label_ps=("label_id", "nunique"))
spec_patient_counts["n_label_ps"] = (
    spec_patient_counts["n_label_ps"].fillna(0).astype(np.float64)
)

eeg_counts = label_level.groupby("eeg_id", as_index=False).agg(
    n_label=("label_id", "nunique")
)
eeg_prior_df = eeg_prior_df.merge(eeg_counts, on="eeg_id", how="left")
eeg_prior_df["n_label"] = eeg_prior_df["n_label"].fillna(0).astype(np.float64)

shrink_p = 0.05
p_mat = patient_prior_df[TARGET_COLS].to_numpy(dtype=np.float64)
p_mat = (1.0 - shrink_p) * p_mat + shrink_p * global_prior[None, :]
p_mat = (p_mat + alpha) / (p_mat.sum(axis=1, keepdims=True) + alpha * len(TARGET_COLS))
patient_prior_df[TARGET_COLS] = p_mat

s_mat = spec_prior_df[TARGET_COLS].to_numpy(dtype=np.float64)
s_mat = (s_mat + alpha) / (s_mat.sum(axis=1, keepdims=True) + alpha * len(TARGET_COLS))
spec_prior_df[TARGET_COLS] = s_mat

eeg_prior_df = eeg_prior_df.merge(
    eeg_level[["eeg_id", "patient_id"]].drop_duplicates("eeg_id"),
    on="eeg_id",
    how="left",
)
eeg_prior_df = eeg_prior_df.merge(
    patient_prior_df[["patient_id"] + TARGET_COLS],
    on="patient_id",
    how="left",
    suffixes=("", "_p"),
)

base_shrink_eeg = 0.08
k_eeg = 3.0
shrink_eeg_vec = eeg_prior_df["n_label"].to_numpy(dtype=np.float64)
shrink_eeg_vec = (base_shrink_eeg * (k_eeg / (shrink_eeg_vec + k_eeg))).astype(
    np.float64
)

e_mat = eeg_prior_df[TARGET_COLS].to_numpy(dtype=np.float64)
e_mat = (e_mat + alpha) / (e_mat.sum(axis=1, keepdims=True) + alpha * len(TARGET_COLS))

p_for_eeg = eeg_prior_df[[f"{c}_p" for c in TARGET_COLS]].to_numpy(dtype=np.float64)
p_for_eeg = np.where(np.isfinite(p_for_eeg), p_for_eeg, global_prior[None, :])

e_mat = (1.0 - shrink_eeg_vec[:, None]) * e_mat + shrink_eeg_vec[:, None] * p_for_eeg
e_mat = (e_mat + alpha) / (e_mat.sum(axis=1, keepdims=True) + alpha * len(TARGET_COLS))
eeg_prior_df[TARGET_COLS] = e_mat

test = pd.read_csv(TEST_PATH, usecols=["eeg_id", "patient_id", "spectrogram_id"])

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
pred_df = test[["eeg_id", "patient_id", "spectrogram_id"]].copy()

pred_df = pred_df.merge(
    eeg_prior_df[["eeg_id"] + TARGET_COLS],
    on=["eeg_id"],
    how="left",
)

pred_df = pred_df.merge(
    spec_prior_df[["spectrogram_id"] + TARGET_COLS + ["n_label"]],
    on=["spectrogram_id"],
    how="left",
    suffixes=("", "_s"),
)

pred_df = pred_df.merge(
    patient_prior_df,
    on=["patient_id"],
    how="left",
    suffixes=("", "_p"),
)

pred_df = pred_df.merge(
    spec_patient_counts,
    on=["patient_id", "spectrogram_id"],
    how="left",
)

evidence = (
    pred_df["n_label_ps"]
    .fillna(pred_df["n_label"])
    .fillna(0)
    .astype(np.float64)
    .to_numpy()
)

base_shrink_sp = 0.10
k = 10.0
shrink_vec = (base_shrink_sp * (k / (evidence + k))).astype(np.float64)

for c in TARGET_COLS:
    spec_prior_col = f"{c}_s"
    pat_prior_col = f"{c}_p"

    spec_val = pred_df[spec_prior_col].to_numpy(dtype=np.float64)
    pat_val = pred_df[pat_prior_col].to_numpy(dtype=np.float64)

    pat_val = np.where(
        np.isfinite(pat_val), pat_val, global_prior[TARGET_COLS.index(c)]
    )

    spec_is_finite = np.isfinite(spec_val)

    blended = np.full_like(spec_val, np.nan, dtype=np.float64)
    blended[spec_is_finite] = (1.0 - shrink_vec[spec_is_finite]) * spec_val[
        spec_is_finite
    ] + shrink_vec[spec_is_finite] * pat_val[spec_is_finite]

    fill_mask = pred_df[c].isna() & np.isfinite(blended)
    pred_df.loc[fill_mask, c] = blended[fill_mask.to_numpy()]

for c in TARGET_COLS:
    pred_df[c] = pred_df[c].fillna(pred_df[f"{c}_p"])
for j, c in enumerate(TARGET_COLS):
    pred_df[c] = pred_df[c].fillna(global_prior[j])

sub[TARGET_COLS] = pred_df[TARGET_COLS].to_numpy(dtype=np.float64)

TEMPER_POWER = 0.85
vals = sub[TARGET_COLS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-12, None)
vals = np.power(vals, TEMPER_POWER)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGET_COLS] = vals

sub.to_csv(SUB_PATH, index=False)
print(
    "Wrote baseline submission (eeg_id prior -> blended spec+patient -> patient -> global) with tempering:",
    SUB_PATH,
)
print(sub.head())



## === cell 1
if not os.path.exists(SUB_PATH):
    raise FileNotFoundError(f"Expected submission not found at: {SUB_PATH}")

sub = pd.read_csv(SUB_PATH)
sample = pd.read_csv(SAMPLE_SUB_PATH)

if "eeg_id" not in sub.columns:
    raise ValueError(
        f"`submission.csv` missing `eeg_id` column. Found columns: {list(sub.columns)}"
    )

missing = [c for c in TARGET_COLS if c not in sub.columns]
if missing:
    raise ValueError(
        f"`submission.csv` missing required columns: {missing}. Found columns: {list(sub.columns)}"
    )

sub = sub.set_index("eeg_id")
sample_ids = sample["eeg_id"].values
sub = sub.reindex(sample_ids)

if sub.isnull().any().any():
    sub[TARGET_COLS] = sub[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

for c in TARGET_COLS:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

sub[TARGET_COLS] = sub[TARGET_COLS].replace([np.inf, -np.inf], np.nan)
sub[TARGET_COLS] = sub[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

vals = sub[TARGET_COLS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 0.0, None)
row_sums = vals.sum(axis=1, keepdims=True)
zero_mask = row_sums.squeeze() == 0.0
if np.any(zero_mask):
    vals[zero_mask, :] = 1.0 / len(TARGET_COLS)
    row_sums = vals.sum(axis=1, keepdims=True)
vals = vals / row_sums

fixed = pd.DataFrame(vals, columns=TARGET_COLS)
fixed.insert(0, "eeg_id", sample_ids)

final_sums = fixed[TARGET_COLS].sum(axis=1).values
if not np.all(np.isfinite(final_sums)) or np.max(np.abs(final_sums - 1.0)) > 1e-6:
    raise ValueError(
        "Row sums are not 1 after fixing; refusing to write invalid submission."
    )

fixed.to_csv(SUB_PATH, index=False)

print("Wrote valid submission:", SUB_PATH)
print(fixed.head())



## === cell 2
with open(SUB_PATH, "r") as f:
    for _ in range(6):
        print(f.readline().rstrip("\n"))
