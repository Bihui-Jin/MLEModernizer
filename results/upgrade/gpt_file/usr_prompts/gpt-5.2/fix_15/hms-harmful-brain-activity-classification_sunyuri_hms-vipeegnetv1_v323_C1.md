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

3.13

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

0.2871303650374164

# 6. Current score

1.41858

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash caused by an incompatibility between TensorFlow and the newer `protobuf` API (the `'MessageFactory' object has no attribute 'GetPrototype'` error) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. I also make the Kaggle path/model discovery more robust so the notebook doesn’t fail when no `models*` dataset is attached, and fall back to a valid uniform-probability submission in that case (score be poor but it “yield” a valid submission). Finally, I fix the test batch slicing bug that incorrectly indexes `test` rows (it uses `len(preds_all)` inside `max(...)`), ensuring predictions align 1:1 with `test.csv` rows and the submission sums to 1 per row.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf implementation is forced *before any TensorFlow/protobuf-related imports*, and by making the Kaggle path detection reliable under Python 3.13. I also fix the main reason your score is extremely poor: in inference you currently don’t load any preprocessed EEGs (and in Kaggle you also disabled training), so the notebook often falls back to uniform predictions; I enable on-the-fly EEG preprocessing for test-time when weights are present, while keeping the model and generator logic unchanged. Additionally, I make the training-data loading robust (so it won’t crash when `train.csv` hasn’t been generated yet) without changing the training approach. Finally, I keep the submission strictly valid (row alignment, probability normalization, correct columns, `.csv` suffix).'
- What this solution (achieved 0.96732) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it is not compatible with this Kaggle Python 3.13 environment), while keeping the pipeline end-to-end and producing a valid `submission.csv`. Since your current score (1.40995, lower-is-better) is far from the target (0.2871), I replace the uniform fallback with a stronger but still simple/fast baseline: class priors estimated from `train.csv` vote distributions, then (optionally) mildly conditioned on `patient_id` priors (computed from train), and finally normalized to sum to 1. This is score-improving, does not use any forbidden leakage, and guarantees valid probabilities for every test row. The output file name, columns, and row alignment match `sample_submission.csv`.'
- What this solution (achieved 1.10642) has done: 'Your current baseline uses only global class priors plus a weak patient prior blend, so it can’t get close to the target KL because it ignores the strongest available signal at test-time: the provided `spectrogram_id`. With minimal changes and no model/training, I add a spectrogram-conditioned prior (computed from train vote distributions grouped by `spectrogram_id`) and blend it with the global prior (and keep the existing patient blend as a smaller backoff). This preserves your core “prior-based probability” logic, just adds a better conditioning variable already present in both train and test. I also keep strict probability normalization/clipping to guarantee a valid submission.'
- What this solution (achieved 1.24905) has done: 'Your current solution is a prior-only blender; to move the KL score down toward the 0.287 target (lower is better), the smallest meaningful improvement is to make the conditioning priors closer to the evaluation target distribution without changing the overall “prior blending” logic. I (1) compute spectrogram/patient priors as **vote-sum-weighted** means (so rows with more annotator agreement contribute proportionally), (2) add a tiny **Dirichlet-style smoothing** toward the global prior for rare spectrogram/patient IDs to prevent noisy overconfident priors, and (3) slightly retune the blend weights to rely a bit more on spectrogram (stronger signal) while keeping the same inference structure and strict probability normalization for a valid submission.'
- What this solution (achieved 1.34198) has done: 'Your current score (1.24905, lower-is-better) is still far from the target (0.2871), so we should improve the prior-only blender without changing its core “global + group-conditioned priors” logic. The main low-risk gain is to condition on variables that generalize better to test: `eeg_id` is present in both train/test and in train it aggregates many overlapping labeled windows per recording, so an `eeg_id`-conditioned prior is typically stronger than `spectrogram_id`/`patient_id`. I add an `eeg_id` weighted/smoothed prior exactly like your existing group priors, and retune blend weights to rely primarily on `eeg_id`, then `spectrogram_id`, then `patient_id`, keeping the same Dirichlet-style backoff and strict probability normalization. All changes are minimal and keep the pipeline fast and submission-valid.'
- What this solution (achieved 1.34635) has done: 'Your current prior blender is still far from the target KL (1.34198 vs 0.2871, lower-is-better), so the smallest safe improvement is to make the group-conditioned priors less noisy and more aligned to how the test labels are aggregated (per eeg_id). I keep the same “global + (eeg/spec/patient) smoothed priors + linear blending” core logic, but (1) aggregate train labels to **eeg_id-level** first (weighted by vote totals) before computing any priors to reduce overlap-induced noise, (2) compute group priors using a faster/cleaner weighted-sum aggregation (same math), and (3) slightly increase smoothing (tau) to avoid overconfident wrong priors that hurt KL. This should improve score while keeping runtime low and ensuring the submission stays strictly valid (row alignment, clipping, sum-to-1).'
- What this solution (achieved 1.37971) has done: 'Your current prior-blending approach is sound but still too noisy/overconfident for KL, so I make the smallest changes that reduce that noise while keeping the same “global + (eeg/spec/patient) smoothed priors + linear blending” core logic. Specifically, I (1) strengthen the reliability weighting by using per-`eeg_id` vote totals as the effective sample size for smoothing (not the already-aggregated weights), and (2) increase the Dirichlet backoff (tau) slightly and rebalance alphas a bit toward the more stable `eeg_id` prior and away from weaker priors. This keeps identical semantics (still just priors computed from train metadata) but should move the public KL down from 1.346 toward your 0.287 target without changing architecture/training (none is used). The script still run end-to-end and write a valid `submission.csv` with rows aligned to `sample_submission.csv` and probabilities summing to 1.'
- What this solution (achieved 1.38348) has done: 'Your current score (1.37971, lower-is-better) is far above the target (0.28713), so we need a small but meaningful improvement without changing the overall “global prior + smoothed group priors + linear blending” core logic. The biggest low-risk issue in the current code is that it blends multiple priors sequentially with fixed alphas, which unintentionally over-anchors to the global prior and underuses the most informative available key (`eeg_id`) when present; I change this to a single convex mixture of (global, eeg, spectrogram, patient) priors so the intended weights are actually respected. I also make the Dirichlet backoff use a smaller tau for `eeg_id` and slightly larger taus for noisier groups, plus compute the group reliability `m` as the effective vote mass already aggregated at eeg-level (same semantics, but better-calibrated), which usually helps KL by avoiding overconfident wrong distributions. These are minimal changes that preserve your priors-only approach and still guarantee perfectly valid probabilities and a correct `submission.csv`.'
- What this solution (achieved 1.39721) has done: 'Your current prior-blender is likely being hurt by (1) weak utilization of *test-time duplicates* (the same `eeg_id` appears multiple times in `test.csv`) and (2) slightly misaligned smoothing strength for KL (too much reliance on noisy group priors for rare IDs, too little shrinkage for uncertain cases). I keep the exact same “global + smoothed (eeg/spec/patient) priors + convex mixture” core logic, but (a) compute predictions once per unique `eeg_id` and broadcast back to all rows to reduce variance and improve calibration, and (b) modestly retune the taus/alphas to be a bit more conservative on `spectrogram_id`/`patient_id` while leaning on the more stable `eeg_id` prior. This is a minimal change (no new model/training, same features, same semantics), but it typically improves KL by reducing inconsistent probabilities across duplicate IDs. The script still runs end-to-end and writes a strictly valid `submission.csv` with correct columns and row sums of 1.'
- What this solution (achieved 1.40906) has done: 'The row explosion comes from merging `sample_submission` with `pred_df` on `eeg_id` when `pred_df` contains duplicate `eeg_id` rows (because test has multiple rows per `eeg_id`), producing a many-to-many merge. I fix this by generating exactly one prediction row per unique `eeg_id` (still using your existing `predict_for_row` and your same priors/blend logic) and then merging 1:1 into `sample_submission`. I also add a small safety assertion to guarantee `pred_df.eeg_id` is unique and keep the same probability clipping/normalization so the submission is valid and stable. This is a pure correctness fix (it should also improve score vs the broken submission because predictions align properly and no duplication occurs).'
- What this solution (achieved 1.41859) has done: 'Your current score (1.40906, lower-is-better) is far worse than the target (0.28713), and the main issue is that the solution is still “priors-only” and cannot approach the target without using the actual EEG/spectrogram signals. Since large architectural/training changes are disallowed, the smallest legitimate improvement we can make within your current logic is to make the priors themselves better calibrated for KL by (1) using **vote-count Dirichlet smoothing in probability space** (instead of smoothing already-normalized probabilities), and (2) adding a **record-level (eeg_id) + spectrogram_id + patient_id hierarchical backoff** computed from the original vote counts with consistent pseudo-counts. This keeps the same overall approach (global + group-conditioned priors + convex mixture; no model, no training) but typically reduces KL because it avoids overconfident wrong distributions for rare IDs and aligns the smoothing with how labels are generated (counts). The submission writing, alignment to `sample_submission.csv`, and sum-to-1 constraints are kept strict and unchanged.'
- What this solution (achieved 1.41858) has done: 'Your current prior blender is mathematically sound, but it likely over-weights `eeg_id` priors that are *not available* for many test eeg_ids (causing frequent fallback to the global prior and weak differentiation), and it also underuses the strongest *shared* key between train and test: `spectrogram_id`. I keep the exact same “global + smoothed group priors + convex mixture” core logic, but (1) compute group priors using a faster, fully weighted count-sum aggregation (fixing a subtle inefficiency/fragility in the current grouping helper), and (2) retune the mixture weights and taus to lean more on `spectrogram_id` (more likely to overlap between train/test) while being more conservative on `patient_id`, which is noisier for KL. These are minimal changes that preserve evaluation semantics (still just priors from train metadata; no model/training) but should move KL down from ~1.42 toward your 0.287 target. The submission writing, alignment to `sample_submission.csv`, and strict sum-to-1 normalization remain unchanged.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


NEEDTRAIN = (
    True  # original flag; on Kaggle we will not train and will only create submission
)
LOAD_MODELS_FROM = "modelsxxxxxxx"  # kept for compatibility; unused in this baseline

cwd_parts = os.getcwd().split(os.sep)
if len(cwd_parts) > 1 and cwd_parts[1] == "home":
    PLATFORM = "local"
elif len(cwd_parts) > 1 and cwd_parts[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = False
else:
    PLATFORM = "local"

DATATYPE = ["eeg"]  # preserved but unused in this baseline
print("DATATYPE:", DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SEED = 2024
np.random.seed(SEED)

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
sample_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")

df = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

TARGETS = sample_sub.columns.tolist()[1:]
print("Train shape:", df.shape)
print("Test shape:", test.shape)
print("Targets:", TARGETS)



## === cell 1
votes = df[TARGETS].astype(np.float32).values
vote_sums = votes.sum(axis=1).astype(np.float32)
vote_sums[vote_sums == 0] = 1.0  # safeguard

tmp = df[["eeg_id", "spectrogram_id", "patient_id"]].copy()
for k, t in enumerate(TARGETS):
    tmp[t] = votes[:, k].astype(np.float32)
tmp["_w"] = vote_sums.astype(np.float32)

grp_eeg = tmp.groupby("eeg_id", sort=False)
eeg_counts = grp_eeg[TARGETS].sum().astype(np.float32)  # counts per class
eeg_w = grp_eeg["_w"].sum().astype(np.float32)  # total votes (effective sample size)
eeg_meta = grp_eeg[["spectrogram_id", "patient_id"]].first()

df2 = eeg_counts.join(eeg_meta, how="left").reset_index()
print("Aggregated eeg-level train shape:", df2.shape)

global_counts = eeg_counts.sum(axis=0).values.astype(np.float32)
global_prior = np.clip(global_counts, 1e-8, None)
global_prior = global_prior / global_prior.sum()

use_patient = "patient_id" in df2.columns and "patient_id" in test.columns
use_spectrogram = "spectrogram_id" in df2.columns and "spectrogram_id" in test.columns
use_eeg = "eeg_id" in df2.columns and "eeg_id" in test.columns


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32, copy=False), 1e-8, 1.0)
    s = float(p.sum())
    if s <= 0:
        return (np.ones_like(p, dtype=np.float32) / len(p)).astype(np.float32)
    return p / s


def group_counts_and_mass(
    ids: pd.Series, counts_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    g = counts_df.copy()
    g["_gid"] = ids.astype("int64", copy=False).values
    g_counts = g.groupby("_gid", sort=False)[TARGETS].sum().astype(np.float32)
    g_mass = g_counts.sum(axis=1).astype(np.float32)
    g_counts.index.name = None
    return g_counts, g_mass


patient_counts, patient_mass = (None, None)
if use_patient:
    patient_counts, patient_mass = group_counts_and_mass(
        df2["patient_id"], df2[TARGETS]
    )

spectrogram_counts, spectrogram_mass = (None, None)
if use_spectrogram:
    spectrogram_counts, spectrogram_mass = group_counts_and_mass(
        df2["spectrogram_id"], df2[TARGETS]
    )

eeg_prior_counts = df2.set_index("eeg_id")[TARGETS].astype(np.float32)
eeg_mass = eeg_prior_counts.sum(axis=1).astype(np.float32)

print("Global prior:", dict(zip(TARGETS, global_prior.round(6))))
print(
    "use_patient:",
    use_patient,
    "| use_spectrogram:",
    use_spectrogram,
    "| use_eeg:",
    use_eeg,
)



## === cell 2
alpha_spec = 0.24 if use_spectrogram else 0.0
alpha_eeg = 0.62 if use_eeg else 0.0
alpha_pat = 0.02 if use_patient else 0.0

alpha_eeg = float(np.clip(alpha_eeg, 0.0, 1.0))
alpha_spec = float(np.clip(alpha_spec, 0.0, 1.0))
alpha_pat = float(np.clip(alpha_pat, 0.0, 1.0))

total_alpha = alpha_eeg + alpha_spec + alpha_pat
if total_alpha > 0.93:
    scale = 0.93 / total_alpha
    alpha_eeg *= scale
    alpha_spec *= scale
    alpha_pat *= scale
alpha_global = 1.0 - (alpha_eeg + alpha_spec + alpha_pat)

eeg_tau = 220.0 if use_eeg else 0.0
spec_tau = 260.0 if use_spectrogram else 0.0
pat_tau = 900.0 if use_patient else 0.0

test_pids = test["patient_id"].values if use_patient else None
test_sids = test["spectrogram_id"].values if use_spectrogram else None
test_eids = test["eeg_id"].values if use_eeg else None


def smoothed_prob_from_counts(counts_vec: np.ndarray, tau: float) -> np.ndarray:
    c = np.clip(counts_vec.astype(np.float32, copy=False), 0.0, None)
    post = c + float(tau) * global_prior.astype(np.float32, copy=False)
    return _norm(post)


def predict_for_row(i: int) -> np.ndarray:
    p_g = global_prior

    p_e = global_prior
    p_s = global_prior
    p_p = global_prior

    if use_eeg:
        eid = int(test_eids[i])
        if eid in eeg_prior_counts.index:
            c = eeg_prior_counts.loc[eid].values.astype(np.float32, copy=False)
            p_e = smoothed_prob_from_counts(c, eeg_tau)

    if use_spectrogram:
        sid = int(test_sids[i])
        if sid in spectrogram_counts.index:
            c = spectrogram_counts.loc[sid].values.astype(np.float32, copy=False)
            p_s = smoothed_prob_from_counts(c, spec_tau)

    if use_patient:
        pid = int(test_pids[i])
        if pid in patient_counts.index:
            c = patient_counts.loc[pid].values.astype(np.float32, copy=False)
            p_p = smoothed_prob_from_counts(c, pat_tau)

    p = alpha_global * p_g + alpha_eeg * p_e + alpha_spec * p_s + alpha_pat * p_p
    return _norm(p)


if use_eeg:
    first_idx_by_eeg = (
        test.reset_index()[["index", "eeg_id"]]
        .drop_duplicates(subset=["eeg_id"], keep="first")
        .set_index("eeg_id")["index"]
        .astype(int)
    )
    unique_eeg_ids = first_idx_by_eeg.index.astype("int64").values

    preds_unique = np.zeros((len(unique_eeg_ids), len(TARGETS)), dtype=np.float32)
    for j, eid in enumerate(unique_eeg_ids):
        i0 = int(first_idx_by_eeg.loc[eid])
        preds_unique[j] = predict_for_row(i0)

    preds_unique = np.clip(preds_unique, 1e-8, 1.0)
    preds_unique = preds_unique / preds_unique.sum(axis=1, keepdims=True)

    pred_df = pd.DataFrame(preds_unique, columns=TARGETS)
    pred_df.insert(0, "eeg_id", unique_eeg_ids)

    if pred_df["eeg_id"].duplicated().any():
        raise RuntimeError("pred_df has duplicate eeg_id values; merge would explode.")
else:
    preds = np.zeros((len(test), len(TARGETS)), dtype=np.float32)
    for i in range(len(test)):
        preds[i] = predict_for_row(i)

    preds = np.clip(preds, 1e-8, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    pred_df = pd.DataFrame(preds, columns=TARGETS)
    pred_df.insert(0, "eeg_id", test["eeg_id"].astype("int64").values)

    if pred_df["eeg_id"].duplicated().any():
        pred_df = pred_df.groupby("eeg_id", sort=False, as_index=False)[TARGETS].mean()
        arr = np.clip(pred_df[TARGETS].values.astype(np.float32), 1e-8, 1.0)
        pred_df[TARGETS] = arr / arr.sum(axis=1, keepdims=True)

sub = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")
if sub[TARGETS].isna().any().any():
    miss = sub[TARGETS].isna().any(axis=1).values
    sub.loc[miss, TARGETS] = global_prior[None, :]

sub[TARGETS] = sub[TARGETS].astype(np.float32)
sub[TARGETS] = np.clip(sub[TARGETS].values, 1e-8, 1.0)
sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].values.sum(axis=1, keepdims=True)

sub = sub[["eeg_id"] + TARGETS]

if sub.shape[0] != sample_sub.shape[0]:
    raise RuntimeError(
        f"Row count mismatch: sub={sub.shape[0]} vs sample_submission={sample_sub.shape[0]}"
    )
if sub.columns.tolist() != sample_sub.columns.tolist():
    raise RuntimeError(
        f"Column mismatch: {sub.columns.tolist()} vs {sample_sub.columns.tolist()}"
    )

row_sums = sub[TARGETS].sum(axis=1).values
print(
    "Blend weights:",
    {
        "global": alpha_global,
        "eeg": alpha_eeg,
        "spec": alpha_spec,
        "patient": alpha_pat,
    },
)
print("Taus:", {"eeg_tau": eeg_tau, "spec_tau": spec_tau, "pat_tau": pat_tau})
print("Row sums (min/max):", float(row_sums.min()), float(row_sums.max()))
print(sub.head())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
