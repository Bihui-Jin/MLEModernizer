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

0.2854737642477286

# 6. Current score

0.77501

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.01346) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory has no attribute GetPrototype`) by removing TensorFlow/Keras usage and replacing it with a lightweight, deterministic fallback that still produces a valid probability submission. Since your current score was “Not yielded”, the priority is to run end-to-end and write a correctly formatted `submission.csv` whose rows sum to 1. The fallback uses patient-level class priors from `train.csv` (and global priors for unseen patients), which is score-better than a uniform guess but keeps logic simple and stable within the 600s timeout. I also ensure column order matches `sample_submission.csv` exactly and add numerical safety (clipping + renormalization).'
- What this solution (achieved 0.77501) has done: 'Your current baseline (patient-level mean of per-row vote probabilities) is leaving performance on the table because the training metadata contains multiple rows per patient and per EEG with varying annotator counts, and KL is better matched by aggregating *counts* (Dirichlet-multinomial intuition) rather than averaging already-normalized distributions. I keep the same overall approach (priors-only, no EEG/spectrogram reading) but (1) compute patient/global priors by summing raw vote counts with a small symmetric pseudocount for stability, (2) back off unseen patients to global, and (3) for seen `eeg_id` in train, use an `eeg_id`-level prior (still a prior lookup, same semantics) which should move the score down toward your target with minimal risk. I also keep strict normalization/clipping so every row sums to 1 and submission stays valid. These changes should improve KL materially while preserving the lightweight deterministic logic and runtime.'
- What this solution (achieved 0.77501) has done: 'You’re already using a solid “priors-only” approach; the biggest remaining gap to the target (0.775 -> 0.285, lower is better) is that the mapping is too coarse and doesn’t use the strongest metadata signal available without touching EEG/spectrograms: the `spectrogram_id`. I keep the same core logic (lookup priors; no model/training/feature extraction) but add a higher-specificity prior: `spectrogram_id`-level counts → probabilities, and use a strict backoff order `spectrogram_id -> eeg_id -> patient_id -> global`. I also make the `in` checks O(1) by using dict `.get()` (same semantics, faster) and keep the same smoothing + clipping/renormalization so the submission remains valid under KL. These minimal changes should reduce KL materially by better matching test records to consolidated train labels when IDs overlap.'
- What this solution (achieved 1.01345) has done: 'You’re far above the target KL (0.775 → 0.285, lower is better), so the most direct minimal improvement is to make the priors more faithful to the competition’s label-generation process: normalize each training row’s votes by its total annotators, then aggregate those *probability* vectors (not raw counts) for each key. This keeps the same priors-only/backoff core logic and the same submission semantics, but fixes a key mismatch (train rows have varying total votes; summing raw votes over-weights high-annotator rows under KL). I also keep your smoothing/clipping/renormalization, but apply smoothing in “probability space” with a very small epsilon so we don’t blunt informative priors. The backoff order stays `spectrogram_id -> eeg_id -> patient_id -> global`, and the script still runs fast and writes a valid `submission.csv`.'
- What this solution (achieved 0.77501) has done: 'Your current KL (1.01345; lower is better) is far above the target (0.28547), so we should improve score, but with minimal changes and the same “priors-only/backoff” logic. The main issue is the current code averages per-row normalized vote distributions equally, which can underuse stronger-signal rows with more annotators; for KL against “observed target” proportions, using vote-count-weighted aggregation is typically closer to how the targets are generated. I keep the exact same backoff order (`spectrogram_id -> eeg_id -> patient_id -> global`) and smoothing/clipping/renorm, but change the group priors to be computed from summed counts (Dirichlet-style) with a small pseudocount, and also compute the global prior the same way for consistency. This is a small, deterministic change that should materially lower KL toward your target while preserving runtime and submission validity.'
- What this solution (achieved 0.77501) has done: 'Your current approach is already “priors-only with backoff”; the smallest change likely to move KL down toward the target is to make the priors closer to the competition target generation by aggregating at the *label_id* level first (to avoid overcounting overlapping/duplicated windows), then summing those label-level counts into spectrogram/eeg/patient/global priors. This preserves the same backoff order and the same smoothing/clipping/renormalization semantics, but reduces noise from repeated segments that can distort priors and hurt KL. I also keep your alpha/clip settings and ensure the submission still strictly sums to 1 per row. No EEG/spectrogram files are loaded, so runtime stays well under the limit.'

# 9. Code solution

## === cell 0
"""
Lightweight deterministic baseline for HMS - Harmful Brain Activity Classification.

Score-matching intent (lower is better; current 0.77501 -> target ~0.285):
- Keep the same "priors-only" core logic (no EEG/spectrogram loading, no ML training loop).
- Minimal but impactful adjustment: de-duplicate / consolidate training supervision by aggregating
  votes at the `label_id` level first (each label set corresponds to one set of rater votes),
  then build spectrogram/eeg/patient/global priors from those consolidated counts.
  This directly targets KL by reducing overcounting from overlapping windows that share labels.
- Preserve the same backoff order:
    spectrogram_id -> eeg_id -> patient_id -> global
- Keep clipping + renormalization so every row sums to 1 (submission validity).
"""

import os
import numpy as np
import pandas as pd

SEED = 2024
np.random.seed(SEED)

LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
sample_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

TARGETS = [c for c in sample.columns if c != "eeg_id"]
assert len(TARGETS) == 6, f"Expected 6 target columns, got {len(TARGETS)}: {TARGETS}"

train_votes = train[TARGETS].astype(np.float64).values


def _normalize_probs(arr, eps=1e-6):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(arr, dtype=np.float64) / len(arr)
    return arr / s


def _smooth_counts_to_probs(counts_vec, alpha=0.5, clip_eps=1e-6):
    """
    Votes treated as counts: sum counts per key, add symmetric pseudocount alpha, normalize.
    """
    v = np.asarray(counts_vec, dtype=np.float64)
    v = np.where(np.isfinite(v), v, 0.0)
    v = np.clip(v, 0.0, None)
    v = v + float(alpha)
    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(v, dtype=np.float64) / len(v)
    p = v / s
    p = np.clip(p, clip_eps, 1.0)
    p = p / p.sum()
    return p




## === cell 1
alpha = 0.5
clip_eps = 1e-6

cols = ["label_id", "patient_id", "eeg_id", "spectrogram_id"] + TARGETS
df = train[cols].copy()

for c in TARGETS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float64)

label_agg = df.groupby("label_id", as_index=False).agg(
    patient_id=("patient_id", "first"),
    eeg_id=("eeg_id", "first"),
    spectrogram_id=("spectrogram_id", "first"),
    **{c: (c, "sum") for c in TARGETS},
)

global_counts = label_agg[TARGETS].sum(axis=0).values
global_prior_vec = _smooth_counts_to_probs(
    global_counts, alpha=alpha, clip_eps=clip_eps
)

patient_sum = label_agg.groupby("patient_id")[TARGETS].sum()
patient_prior_map = {
    pid: _smooth_counts_to_probs(
        patient_sum.loc[pid, TARGETS].values, alpha=alpha, clip_eps=clip_eps
    )
    for pid in patient_sum.index.values
}

eeg_sum = label_agg.groupby("eeg_id")[TARGETS].sum()
eeg_prior_map = {
    eid: _smooth_counts_to_probs(
        eeg_sum.loc[eid, TARGETS].values, alpha=alpha, clip_eps=clip_eps
    )
    for eid in eeg_sum.index.values
}

spec_sum = label_agg.groupby("spectrogram_id")[TARGETS].sum()
spec_prior_map = {
    sid: _smooth_counts_to_probs(
        spec_sum.loc[sid, TARGETS].values, alpha=alpha, clip_eps=clip_eps
    )
    for sid in spec_sum.index.values
}

preds = np.zeros((len(test), len(TARGETS)), dtype=np.float64)

test_spec_ids = test["spectrogram_id"].values
test_eeg_ids = test["eeg_id"].values
test_patient_ids = test["patient_id"].values

for i in range(len(test)):
    sid = test_spec_ids[i]
    eid = test_eeg_ids[i]
    pid = test_patient_ids[i]

    p = spec_prior_map.get(sid)
    if p is None:
        p = eeg_prior_map.get(eid)
        if p is None:
            p = patient_prior_map.get(pid, global_prior_vec)

    preds[i] = p

preds = np.clip(preds, clip_eps, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)




## === cell 2
sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for j, col in enumerate(TARGETS):
    sub[col] = preds[:, j].astype(np.float32)

sub = sub[["eeg_id"] + TARGETS]

row_sum_check = sub[TARGETS].sum(axis=1).values
if not np.all(np.isfinite(row_sum_check)):
    raise ValueError("Non-finite row sums in submission.")
max_dev = float(np.max(np.abs(row_sum_check - 1.0)))
print("Submission shape:", sub.shape)
print("Max deviation from sum=1:", max_dev)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(sub.head())
