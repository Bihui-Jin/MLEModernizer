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

1.34635

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

row_probs = votes / vote_sums[:, None]

tmp = df[["eeg_id", "spectrogram_id", "patient_id"]].copy()
for k, t in enumerate(TARGETS):
    tmp[t] = row_probs[:, k].astype(np.float32)
tmp["_w"] = vote_sums.astype(np.float32)

grp_eeg = tmp.groupby("eeg_id", sort=False)
w_eeg = grp_eeg["_w"].sum().astype(np.float32)

wsums = {}
for t in TARGETS:
    wsums[t] = (
        (tmp[t] * tmp["_w"]).groupby(tmp["eeg_id"], sort=False).sum().astype(np.float32)
    )
eeg_level = pd.DataFrame(wsums)
eeg_level = eeg_level.div(w_eeg, axis=0)

eeg_meta = grp_eeg[["spectrogram_id", "patient_id"]].first()

df2 = (
    eeg_level.join(eeg_meta, how="left")
    .reset_index()
    .rename(columns={"index": "eeg_id"})
)
print("Aggregated eeg-level train shape:", df2.shape)

w = w_eeg.reindex(df2["eeg_id"].values).values.astype(np.float32)
w_sum = float(w.sum()) if float(w.sum()) > 0 else 1.0

row_probs2 = df2[TARGETS].astype(np.float32).values
global_prior = (row_probs2 * w[:, None]).sum(axis=0) / w_sum
global_prior = np.clip(global_prior, 1e-8, 1.0)
global_prior = global_prior / global_prior.sum()

use_patient = "patient_id" in df2.columns and "patient_id" in test.columns
use_spectrogram = "spectrogram_id" in df2.columns and "spectrogram_id" in test.columns
use_eeg = "eeg_id" in df2.columns and "eeg_id" in test.columns


def weighted_group_prior_fast(
    df_ids: pd.Series, row_probs_arr: np.ndarray, weights: np.ndarray, targets
):
    ids = df_ids.astype("int64", copy=False)
    w_ser = pd.Series(weights, index=ids)
    w_by = w_ser.groupby(level=0, sort=False).sum().astype(np.float32)

    num_df = {}
    for k, t in enumerate(targets):
        s = pd.Series(row_probs_arr[:, k] * weights, index=ids)
        num_df[t] = s.groupby(level=0, sort=False).sum().astype(np.float32)

    num_df = pd.DataFrame(num_df)
    prior = num_df.div(w_by, axis=0)
    return prior, w_by


patient_prior, patient_w = (None, None)
if use_patient:
    patient_prior, patient_w = weighted_group_prior_fast(
        df2["patient_id"], row_probs2, w, TARGETS
    )

spectrogram_prior, spectrogram_w = (None, None)
if use_spectrogram:
    spectrogram_prior, spectrogram_w = weighted_group_prior_fast(
        df2["spectrogram_id"], row_probs2, w, TARGETS
    )

eeg_prior, eeg_w = (None, None)
if use_eeg:
    eeg_prior = df2.set_index("eeg_id")[TARGETS].astype(np.float32)
    eeg_w = (
        pd.Series(w, index=df2["eeg_id"].astype("int64", copy=False))
        .groupby(level=0, sort=False)
        .sum()
    )

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
alpha_eeg = 0.74 if use_eeg else 0.0
alpha_spec = 0.18 if use_spectrogram else 0.0
alpha_pat = 0.05 if use_patient else 0.0

alpha_eeg = float(np.clip(alpha_eeg, 0.0, 1.0))
alpha_spec = float(np.clip(alpha_spec, 0.0, 1.0))
alpha_pat = float(np.clip(alpha_pat, 0.0, 1.0))

if alpha_eeg + alpha_spec + alpha_pat > 0.95:
    scale = 0.95 / (alpha_eeg + alpha_spec + alpha_pat)
    alpha_eeg *= scale
    alpha_spec *= scale
    alpha_pat *= scale

eeg_tau = 25.0 if use_eeg else 0.0
spec_tau = 50.0 if use_spectrogram else 0.0
pat_tau = 35.0 if use_patient else 0.0

preds = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

test_pids = test["patient_id"].values if use_patient else None
test_sids = test["spectrogram_id"].values if use_spectrogram else None
test_eids = test["eeg_id"].values if use_eeg else None

for i in range(len(test)):
    p = global_prior.copy()

    if use_eeg:
        eid = int(test_eids[i])
        if eid in eeg_prior.index:
            ep = eeg_prior.loc[eid].values.astype(np.float32)
            ep = np.clip(ep, 1e-8, 1.0)
            ep = ep / ep.sum()

            m = float(eeg_w.loc[eid]) if eid in eeg_w.index else 0.0
            lam = m / (m + eeg_tau) if (m + eeg_tau) > 0 else 0.0
            ep = lam * ep + (1.0 - lam) * global_prior
            ep = np.clip(ep, 1e-8, 1.0)
            ep = ep / ep.sum()

            p = (1.0 - alpha_eeg) * p + alpha_eeg * ep

    if use_spectrogram:
        sid = int(test_sids[i])
        if sid in spectrogram_prior.index:
            sp = spectrogram_prior.loc[sid].values.astype(np.float32)
            sp = np.clip(sp, 1e-8, 1.0)
            sp = sp / sp.sum()

            m = float(spectrogram_w.loc[sid]) if sid in spectrogram_w.index else 0.0
            lam = m / (m + spec_tau) if (m + spec_tau) > 0 else 0.0
            sp = lam * sp + (1.0 - lam) * global_prior
            sp = np.clip(sp, 1e-8, 1.0)
            sp = sp / sp.sum()

            p = (1.0 - alpha_spec) * p + alpha_spec * sp

    if use_patient:
        pid = int(test_pids[i])
        if pid in patient_prior.index:
            pp = patient_prior.loc[pid].values.astype(np.float32)
            pp = np.clip(pp, 1e-8, 1.0)
            pp = pp / pp.sum()

            m = float(patient_w.loc[pid]) if pid in patient_w.index else 0.0
            lam = m / (m + pat_tau) if (m + pat_tau) > 0 else 0.0
            pp = lam * pp + (1.0 - lam) * global_prior
            pp = np.clip(pp, 1e-8, 1.0)
            pp = pp / pp.sum()

            p = (1.0 - alpha_pat) * p + alpha_pat * pp

    preds[i] = p

preds = np.clip(preds, 1e-8, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for j, t in enumerate(TARGETS):
    sub[t] = preds[:, j]

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
print("Row sums (min/max):", float(row_sums.min()), float(row_sums.max()))
print(sub.head())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
