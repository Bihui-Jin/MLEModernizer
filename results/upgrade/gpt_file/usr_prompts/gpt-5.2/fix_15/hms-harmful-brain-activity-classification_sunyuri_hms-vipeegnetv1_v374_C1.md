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

0.3404325331581851

# 6. Current score

1.11406

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the intentional RuntimeError stop so the notebook always completes and writes `submission.csv`. Because TensorFlow import is currently failing due to a protobuf incompatibility (`MessageFactory.GetPrototype`), I make the script robust by catching that failure and deterministically falling back to a valid uniform-probability submission (rows sum to 1). I also adjust the cell numbering to start at 1 (your provided script starts at cell 0), which is required by the execution harness. These changes are minimal, unblock end-to-end execution, and guarantee a valid `.csv` submission even when TF can’t be used.'
- What this solution (achieved 1.39779) has done: 'I fix the runtime failure by avoiding the TensorFlow import entirely (the protobuf `MessageFactory.GetPrototype` crash happens during import and is not needed to write a valid submission). Then I replace the uniform fallback with a score-improving, still-minimal “prior-based” prediction: compute the average class distribution from `train.csv` vote counts (converted to probabilities) and use that same distribution for every test row—this is a legitimate, stable baseline that typically beats uniform on KL for this competition. Finally, I ensure the submission columns match exactly and rows sum to 1, and keep paths unchanged.'
- What this solution (achieved 1.39779) has done: 'Your current approach (constant class-prior for all rows) is a solid safe baseline, but it’s leaving score on the table because KL on this competition strongly benefits from predicting *per-eeg_id* label distributions rather than per-row distributions (train.csv has multiple overlapping clips per eeg_id). The smallest legitimate improvement that preserves your “no TF, no model” core logic is to compute the empirical class prior **grouped by eeg_id** (aggregate votes across all rows belonging to the same eeg_id), then use that as the prediction for any matching eeg_id in test; for any test eeg_id not seen in train, fall back to the global prior. This uses only metadata already allowed, avoids leakage (train labels only), stays deterministic, and usually moves KL much closer to your target. I also keep the strict “rows sum to 1” normalization and column ordering identical to sample_submission to avoid submission-format penalties.'
- What this solution (achieved 1.46771) has done: 'Your current baseline is already deterministic and valid, but it’s still far from the target because it can’t distinguish test eeg_ids that aren’t present in train and it ignores a strong available signal: patient-level label priors. The smallest legitimate improvement (no model, no TF, same “vote-prior” logic) is to add a second fallback: if a test `eeg_id` is unseen in train, use the aggregated prior for that `patient_id`; only if the patient is also unseen do we fall back to the global prior. This preserves evaluation semantics, keeps outputs as proper probabilities summing to 1, and typically reduces KL materially versus global-only fallback. I also keep the submission column order identical to `sample_submission.csv` and add a tiny safety smoothing/renorm to prevent any numerical issues.'
- What this solution (achieved 1.46771) has done: 'Your current score is much worse than the target (lower is better), so we should improve predictive specificity without changing the “no model / prior-based” core logic. The smallest legitimate gain beyond your current hierarchy is to use the already-available `spectrogram_id` as an additional, very strong grouping signal: add an intermediate fallback `eeg_id -> spectrogram_id -> patient_id -> global prior`. This keeps the same vote-aggregation approach, just reuses train metadata more effectively for test rows whose `eeg_id` is unseen but whose `spectrogram_id` (or patient) may be seen in train. I also keep your strict normalization/clipping to ensure every row sums to 1 and the submission columns exactly match `sample_submission.csv`.'
- What this solution (achieved 1.47425) has done: 'Your current score is much worse than the target (lower is better), so we should improve the *same prior-based* approach by making the priors more accurate without changing to a real model. The smallest high-impact fix is to aggregate vote distributions at the **label_id level first** (because multiple train rows can share the same label set due to overlaps), then build eeg/spectrogram/patient priors from these de-duplicated labels to reduce bias. Additionally, we should compute priors using **total vote counts** (not mean of per-row normalized probabilities), which better matches the KL target distribution definition. These are minimal logic-preserving changes (still “aggregate votes → normalize”) and should move the KL down toward your target.'
- What this solution (achieved 1.47425) has done: 'We keep your prior-based “aggregate votes → normalize → hierarchical fallback” core logic intact, but fix two high-impact issues that are likely inflating KL: (1) `label_id` de-dup currently uses `.first()` for metadata, which can mismatch if a `label_id` appears with multiple eeg/spectrogram entries; we instead aggregate priors directly from the original rows using *vote-count sums* without relying on a single metadata row per `label_id`. (2) We make the hierarchy slightly more faithful by building each group’s vote totals after de-duplication using a stable, deterministic “one row per label_id per group” rule (sum votes within `(group_col, label_id)` first, then sum across label_id), preventing duplicated label sets from being counted multiple times inside a group. These changes are minimal, deterministic, preserve evaluation semantics, and should move the score down toward your target without introducing any model/training. Submission formatting and strict probability normalization are preserved.'
- What this solution (achieved 0.83424) has done: 'We keep your current “aggregate votes → normalize → hierarchical fallback” approach, but make two small changes that typically reduce KL without introducing any model: (1) compute group priors from **raw vote sums per (group, label_id)** but also allow a tunable **shrinkage toward the global prior**, which helps avoid overconfident per-group distributions that can hurt KL on unseen/shifted test cases; (2) speed up and stabilize lookup by using pandas `Series.map`/numpy indexing instead of a Python loop, preserving identical semantics. We choose a conservative shrinkage strength (alpha=0.35) to move the score down from ~1.47 toward your 0.34 target without changing the core logic. Submission formatting, column order, and strict row-sum-to-1 normalization are preserved.'
- What this solution (achieved 1.00185) has done: 'We keep your exact “aggregate votes → normalize → hierarchical fallback” baseline and only adjust the calibration that most directly affects KL: the shrinkage strength toward the global prior. Since your current score (0.83424, lower is better) is still far above the target (0.3404), we should reduce overconfident group priors a bit more by increasing `shrink_alpha`, which typically improves KL by preventing tiny probabilities on classes that sometimes appear in test. To stay minimal and stable, we won’t change the hierarchy, de-dup, or any feature/modeling—just tune `shrink_alpha` to a more conservative value (0.65) and keep the same strict normalization and submission formatting.'
- What this solution (achieved 1.13446) has done: 'We keep your exact hierarchical vote-prior baseline and only tune the one parameter that directly controls KL calibration: `shrink_alpha`. Your current score (1.00185, lower is better) is still far above the target (0.3404), so we should reduce overconfident group priors further by increasing shrinkage toward the global prior, which typically lowers KL by avoiding near-zero probabilities on plausible classes. No hierarchy, de-dup logic, vote aggregation, or submission formatting is changed—only the shrinkage strength. The script still runs end-to-end without TensorFlow and writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.93409) has done: 'Your current score is worse than the target (lower is better), so we should improve KL by slightly reducing the over-aggressive shrinkage toward the global prior that likely makes predictions too generic. To keep core logic identical (same hierarchy, same vote aggregation, same de-dup by (group,label_id), same normalization), the only change is tuning `shrink_alpha` downward from 0.80 to a more balanced 0.55, which typically improves KL by keeping useful group-specific signal while still avoiding near-zero probabilities. Everything else—including submission column order and strict row-sum-to-1 normalization—remains unchanged to preserve evaluation semantics and stability. The script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 1.0413) has done: 'We keep your exact hierarchical prior logic (group vote sums with (group,label_id) de-dup, eeg→spectrogram→patient→global fallback, strict normalization) and only adjust the single calibration knob that most directly impacts KL: `shrink_alpha`. Your current score (0.93409, lower is better) is still far above the target (0.34043), which suggests predictions are still too overconfident/too spiky for KL; increasing shrinkage modestly typically lowers KL by preventing tiny probabilities on plausible classes. To stay minimal and stable, we won’t change any aggregation, hierarchy, or output format—only tune `shrink_alpha` upward from 0.55 to 0.70. The script still run end-to-end and write a valid `submission.csv` with correct columns and row sums.'
- What this solution (achieved 1.11406) has done: 'We keep your exact hierarchical vote-prior pipeline (vote-count sums with (group,label_id) de-dup, eeg→spectrogram→patient→global fallback, strict normalization, and submission column order) and only tune the single calibration knob that directly impacts KL: `shrink_alpha`. Your current score (1.0413, lower is better) suggests predictions are still too “spiky/overconfident” for KL, so we increase shrinkage modestly from 0.70 to 0.78 to move probabilities closer to the safer global prior without changing any aggregation or hierarchy. This is the smallest legitimate change expected to reduce KL while preserving core logic and runtime. The script still run end-to-end and write a valid `submission.csv` with rows summing to 1.'

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

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

os.environ["KERAS_BACKEND"] = "tensorflow"

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # local
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # kaggle
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "kaggle"
    NEEDTRAIN = False

datatype = "spe"
print(datatype)

DATATYPE = ["spe", "eeg"]  # kept for compatibility with original code

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))




## === cell 1
def make_hierarchical_prior_submission(
    load_data_from: str, train_df: pd.DataFrame, targets, shrink_alpha: float = 0.78
):
    """
    Preserve the same core logic (vote aggregation -> normalize -> hierarchical fallback),
    and adjust only calibration via shrinkage toward the global prior.

    Change (relevance: improve score toward target): tune shrink_alpha from 0.70 -> 0.78.
    For KL divergence, too-peaky group priors can be heavily penalized when the true target
    distribution retains mass across multiple classes; slightly stronger shrinkage typically
    reduces extreme probabilities while keeping hierarchy signal:
        p_final = (1 - alpha) * p_group + alpha * p_global

    Everything else (de-dup by (group,label_id), hierarchy order, normalization, column order)
    remains unchanged to preserve evaluation semantics.
    """
    test = pd.read_csv(os.path.join(load_data_from, "test.csv"))
    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

    targets = list(targets)
    id_cols = ["eeg_id", "spectrogram_id", "patient_id", "label_id"]
    cols_needed = id_cols + targets
    base = train_df.loc[:, cols_needed].copy()

    label_votes = base.groupby("label_id", sort=False)[targets].sum().astype(np.float64)
    global_votes = label_votes.sum(axis=0).values.astype(np.float64)
    global_votes = np.clip(global_votes, 0.0, None)
    if global_votes.sum() == 0.0:
        global_prior = np.ones(len(targets), dtype=np.float64) / len(targets)
    else:
        global_prior = global_votes / global_votes.sum()
    global_prior = np.clip(global_prior, 1e-12, None)
    global_prior = global_prior / global_prior.sum()

    def build_group_prior_dedup_by_label(df_base: pd.DataFrame, group_col: str):
        """
        Deterministic de-duplication:
        1) Sum votes within (group_col, label_id) so each label set contributes once per group.
        2) Sum across label_id per group -> normalize -> then shrink toward global_prior.
        Returns a DataFrame indexed by group id with columns=targets.
        """
        g1 = (
            df_base.groupby([group_col, "label_id"], sort=False)[targets]
            .sum()
            .astype(np.float64)
        )
        g = g1.groupby(level=0, sort=False).sum().astype(np.float64)

        sums = g.sum(axis=1).values.astype(np.float64)
        sums[sums == 0.0] = 1.0
        probs = (g.values.T / sums).T
        probs = np.clip(probs, 1e-12, None)
        probs = (probs.T / probs.sum(axis=1)).T

        a = float(shrink_alpha)
        if not (0.0 <= a <= 1.0):
            a = 0.78
        probs = (1.0 - a) * probs + a * global_prior[None, :]
        probs = np.clip(probs, 1e-12, None)
        probs = (probs.T / probs.sum(axis=1)).T

        out = pd.DataFrame(probs, index=g.index.astype(np.int64), columns=targets)
        return out

    eeg_priors = build_group_prior_dedup_by_label(base, "eeg_id")
    spe_priors = build_group_prior_dedup_by_label(base, "spectrogram_id")
    pat_priors = build_group_prior_dedup_by_label(base, "patient_id")

    test_eeg = test["eeg_id"].astype(np.int64).values
    test_spe = test["spectrogram_id"].astype(np.int64).values
    test_pat = test["patient_id"].astype(np.int64).values

    p_eeg = eeg_priors.reindex(test_eeg).to_numpy(dtype=np.float64)
    p_spe = spe_priors.reindex(test_spe).to_numpy(dtype=np.float64)
    p_pat = pat_priors.reindex(test_pat).to_numpy(dtype=np.float64)

    p = np.tile(global_prior[None, :], (len(test), 1)).astype(np.float64)

    m_pat = ~np.isnan(p_pat[:, 0])
    p[m_pat] = p_pat[m_pat]

    m_spe = ~np.isnan(p_spe[:, 0])
    p[m_spe] = p_spe[m_spe]

    m_eeg = ~np.isnan(p_eeg[:, 0])
    p[m_eeg] = p_eeg[m_eeg]

    eps = 1e-12
    p = np.clip(p, eps, None)
    p = (p.T / p.sum(axis=1)).T

    for j, c in enumerate(targets):
        sub[c] = p[:, j].astype(np.float64)

    sample = pd.read_csv(os.path.join(load_data_from, "sample_submission.csv"))
    sub = sub[sample.columns.tolist()]

    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("Submission shape:", sub.shape)
    print("Shrink alpha:", float(shrink_alpha))
    print(
        "Global prior (final fallback):",
        {t: float(v) for t, v in zip(targets, global_prior)},
    )
    print("Unique label_id used for global prior:", int(label_votes.shape[0]))
    print(
        "Known eeg_id priors:",
        int(eeg_priors.shape[0]),
        "/",
        int(train_df["eeg_id"].nunique()),
    )
    print(
        "Known spectrogram_id priors:",
        int(spe_priors.shape[0]),
        "/",
        int(train_df["spectrogram_id"].nunique()),
    )
    print(
        "Known patient_id priors:",
        int(pat_priors.shape[0]),
        "/",
        int(train_df["patient_id"].nunique()),
    )
    print(sub.head())
    return sub


_ = make_hierarchical_prior_submission(LOAD_DATA_FROM, df, TARGETS, shrink_alpha=0.78)



## === cell 2
print("Done. submission.csv is ready.")
print(
    "Note: TensorFlow import skipped; submission created via hierarchical vote-prior baseline "
    "with (group,label_id) de-duplication, vote-count-based priors, and tuned shrinkage "
    "toward the global prior (eeg_id -> spectrogram_id -> patient_id -> global prior)."
)
