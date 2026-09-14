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

1.1537

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the failing external `pip install`/Hydra code paths and instead generate a stable, valid baseline submission directly from `train.csv` by computing the global mean label distribution (vote counts normalized to probabilities). This fixes the runtime errors that prevent any `.csv` from being written, while keeping evaluation semantics correct (probabilities per class summing to 1) and typically achieves a reasonable KL score baseline. I also keep the existing merge/ensemble code but make it gracefully fall back to the baseline if no fold prediction files exist. Finally, I add strict sanity checks to guarantee the submission matches `sample_submission.csv` column order and constraints.'
- What this solution (achieved 1.68479) has done: 'Your current submission is a pure global-prior baseline, which is stable but far from the target KL score; the smallest legitimate improvement without changing core modeling is to make the prior *patient-aware* (still just aggregating label votes) because test has `patient_id` and label distribution is strongly patient-dependent. I compute per-patient vote distributions from `train.csv`, fall back to the global prior for unseen patients, and then align predictions to `test.csv`’s `eeg_id` order (instead of relying on `sample_submission` ordering) to avoid any subtle row misalignment. This keeps the same overall “mean distribution baseline” logic while usually moving KL substantially downward. The merge/ensemble fallback behavior is preserved; it now fall back to the improved patient-aware baseline.'
- What this solution (achieved 1.68479) has done: 'Your current patient-aware prior is still too coarse because it ignores strong per-recording effects in `spectrogram_id`; the smallest improvement that keeps the same “aggregate vote distribution” core logic is to use a hierarchical prior: spectrogram-specific if available, else patient-specific, else global. This stays within your existing baseline semantics (no EEG/spectrogram feature extraction or model training), but typically reduces KL a lot because test provides `spectrogram_id` and label distribution differs by recording context. I also fix a subtle alignment risk by mapping priors via DataFrame joins (instead of reindexing a patient_prior table by a raw NumPy array), while keeping the same output format and probability normalization checks. The merge/ensemble fallback remains unchanged, and the final `submission.csv` is still written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.79884) has done: 'Your current hierarchical prior is a good “no-model” baseline, but it’s likely overfitting sparsely-seen spectrogram_ids; the smallest improvement toward the target KL is to smooth each group prior toward the global prior using a simple empirical-Bayes (additive pseudocount) scheme. This keeps the exact same core logic (aggregate vote distributions; spectrogram→patient→global fallback) while reducing extreme probabilities that are heavily penalized by KL. I also fix the spectrogram-merge column collision bug (your current merge overwrites baseline columns and makes the NaN-detection logic ineffective), ensuring spectrogram priors are actually used when present. Finally, I keep all submission alignment/normalization checks identical so the output remains valid.'
- What this solution (achieved 0.79884) has done: 'Your current score (0.79884, lower-is-better) is still far from the target (0.3346), so we should improve it while keeping the same “hierarchical prior from aggregated votes” core logic. The smallest high-impact change is to compute the priors at the *eeg_id* level first (then smooth), since train has many overlapping subsamples per eeg_id and test is keyed by eeg_id; this typically matches the evaluation unit better than spectrogram_id alone. We keep the same spectrogram→patient→global fallback structure, but add an eeg_id-specific smoothed prior at the top of the hierarchy: eeg_id → spectrogram_id → patient_id → global. This is still purely aggregation + empirical-Bayes smoothing (no new model/feature extraction), and we preserve the same submission alignment and normalization guarantees.'
- What this solution (achieved 0.7974) has done: 'Your current score (0.79884, lower-is-better) is still far from the target (0.3346), so we should improve it while keeping the same hierarchical “aggregate vote distribution” core logic. The smallest high-impact change is to avoid over-counting overlapping subsamples by first aggregating votes to one row per `eeg_id` (the evaluation unit) when building all priors; this keeps the same prior/smoothing logic but reduces label noise/leak from duplicated windows. I also make the smoothing strength adaptive to group sample size (more smoothing for sparse groups) while retaining the same additive pseudocount scheme and global prior anchor. All submission alignment/normalization checks and the merge fallback behavior remain unchanged, and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 1.07656) has done: 'Your current score (0.7974, lower-is-better) is still far above the target (0.3346), so we should improve it while keeping the same core “hierarchical aggregated-vote priors + smoothing” approach. The most impactful minimal change is to build priors using the *true label distribution* per training row (votes normalized to probabilities) rather than raw vote counts, because the evaluation KL is defined on probability targets and rows have varying annotator counts; this prevents high-vote rows from dominating the prior. We keep the same hierarchy (eeg_id → spectrogram_id → patient_id → global) and the same adaptive smoothing concept, but we apply smoothing in “effective-sample-size” space (alpha based on number of training rows per group), which is more consistent with probability averaging. All alignment, fallback, and submission normalization checks remain the same, and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 1.15761) has done: 'Your current approach is a hierarchical aggregated-vote prior with adaptive smoothing, but the biggest remaining issue for KL is that it uses hard fallbacks (eeg→spec→patient→global), which creates discontinuities and overconfident group distributions for sparse/mismatched groups. I keep the same priors and smoothing you already compute, but change the *combination step* to a soft mixture of available priors (weighted by each group’s effective sample size), which typically reduces KL by avoiding extreme probabilities and better calibrating uncertainty. This is a minimal change: no new features/models, no new training loop, and the submission format/normalization checks remain identical. I also store the per-group effective weights (n and alpha) so the mixture uses the same “adaptive smoothing strength” semantics you already defined.'
- What this solution (achieved 1.17478) has done: 'Your current score (1.1576, lower-is-better) is much worse than the target (0.3346), so we should improve while keeping the same “hierarchical aggregated-probability priors + smoothing + soft mixture” core logic. The smallest high-impact adjustment is to re-tune the mixture weights so they rely more on the best-matching identifiers (eeg_id, then spectrogram_id) and less on the global prior, while also making the global weight adaptive (stronger only when little group evidence exists). I keep your existing prior construction and adaptive smoothing intact, and only change the final mixture weighting step plus add a tiny Dirichlet-style floor that reduces KL penalties from overconfident near-zeros. The script still run end-to-end and write `/kaggle/working/submission.csv` with correct columns and row-wise probability sums of 1.'
- What this solution (achieved 1.1537) has done: 'We keep your exact “hierarchical priors + adaptive smoothing + soft mixture” pipeline, but fix the key issue driving the bad KL: the mixture weights are currently dominated by large global evidence terms, causing the final predictions to collapse toward a near-constant distribution. I change only the weighting semantics so that (1) each group’s *actual sample size n* drives its influence (not n+alpha), and (2) the global weight is truly small when strong group evidence exists. Finally, I reduce the uniform Dirichlet floor slightly (still preventing near-zeros) so it doesn’t wash out informative priors.'

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
for needed in ["patient_id", "spectrogram_id", "eeg_id"]:
    if needed not in train_df.columns:
        raise ValueError(f"train.csv missing {needed}")
for needed in ["patient_id", "spectrogram_id", "eeg_id"]:
    if needed not in test_df.columns:
        raise ValueError(f"test.csv missing {needed}")

votes = train_df[TARGET_COLS].astype(np.float64)
row_totals = votes.sum(axis=1).to_numpy(dtype=np.float64)
row_totals = np.clip(row_totals, 1.0, None)
train_df_probs = train_df[["eeg_id", "patient_id", "spectrogram_id"]].copy()
train_df_probs[TARGET_COLS] = votes.to_numpy(dtype=np.float64) / row_totals[:, None]

train_eeg_prob = (
    train_df_probs.groupby("eeg_id", observed=True)
    .agg(
        {
            "patient_id": "first",
            "spectrogram_id": "first",
            **{c: "mean" for c in TARGET_COLS},
        }
    )
    .reset_index()
)

global_prior = train_eeg_prob[TARGET_COLS].mean(axis=0).to_numpy(dtype=np.float64)
if not np.isfinite(global_prior).all() or global_prior.sum() <= 0:
    raise RuntimeError("Invalid global prior computed from train.csv")
global_prior = np.clip(global_prior, 1e-15, 1.0)
global_prior = global_prior / global_prior.sum()

ALPHA_BASE = 80.0
ALPHA_POWER = 0.5


def _smoothed_group_priors_adaptive_prob(
    df, group_col, target_cols, global_prior_vec, alpha_base, alpha_power
):
    grp = df.groupby(group_col, observed=True)[target_cols]
    mean_probs = grp.mean().astype(np.float64)
    n = grp.size().to_numpy(dtype=np.float64)

    alpha_i = alpha_base / np.power(n + 1.0, alpha_power)
    alpha_i = np.clip(alpha_i, 5.0, alpha_base)

    pri_sm = (mean_probs * n[:, None] + (alpha_i[:, None] * global_prior_vec)) / (
        n + alpha_i
    )[:, None]
    pri_sm = pri_sm.replace([np.inf, -np.inf], np.nan).fillna(1.0 / len(target_cols))
    pri_sm = pri_sm.clip(1e-15, 1.0)
    pri_sm = pri_sm.div(pri_sm.sum(axis=1), axis=0)
    return pri_sm, mean_probs, n, alpha_i


eeg_priors_sm, eeg_mean_probs, eeg_n, eeg_alpha = _smoothed_group_priors_adaptive_prob(
    train_eeg_prob, "eeg_id", TARGET_COLS, global_prior, ALPHA_BASE, ALPHA_POWER
)
patient_priors_sm, patient_mean_probs, patient_n, patient_alpha = (
    _smoothed_group_priors_adaptive_prob(
        train_eeg_prob, "patient_id", TARGET_COLS, global_prior, ALPHA_BASE, ALPHA_POWER
    )
)
spec_priors_sm, spec_mean_probs, spec_n, spec_alpha = (
    _smoothed_group_priors_adaptive_prob(
        train_eeg_prob,
        "spectrogram_id",
        TARGET_COLS,
        global_prior,
        ALPHA_BASE,
        ALPHA_POWER,
    )
)

print("Computed global prior (mean of eeg_id-aggregated label probabilities):")
for c, v in zip(TARGET_COLS, global_prior):
    print(f"  {c}: {v:.6f}")
print("Sum:", global_prior.sum())
print("Computed eeg-specific priors for eeg_ids:", len(eeg_priors_sm))
print("Computed patient-specific priors for patients:", len(patient_priors_sm))
print("Computed spectrogram-specific priors for spectrograms:", len(spec_priors_sm))
print("Adaptive smoothing alpha_base:", ALPHA_BASE, "alpha_power:", ALPHA_POWER)
print(
    "Example alpha ranges:",
    "eeg:",
    float(np.min(eeg_alpha)),
    float(np.max(eeg_alpha)),
    "patient:",
    float(np.min(patient_alpha)),
    float(np.max(patient_alpha)),
    "spec:",
    float(np.min(spec_alpha)),
    float(np.max(spec_alpha)),
)



## === cell 3
baseline = test_df[["eeg_id", "patient_id", "spectrogram_id"]].copy()

eeg_map = eeg_priors_sm.reset_index().rename(
    columns={c: f"{c}_eeg" for c in TARGET_COLS}
)
spec_map = spec_priors_sm.reset_index().rename(
    columns={c: f"{c}_spec" for c in TARGET_COLS}
)
pat_map = patient_priors_sm.reset_index().rename(
    columns={c: f"{c}_pat" for c in TARGET_COLS}
)

eeg_w = pd.DataFrame({"eeg_id": eeg_priors_sm.index, "w_eeg": eeg_n.astype(np.float64)})
spec_w = pd.DataFrame(
    {"spectrogram_id": spec_priors_sm.index, "w_spec": spec_n.astype(np.float64)}
)
pat_w = pd.DataFrame(
    {"patient_id": patient_priors_sm.index, "w_pat": patient_n.astype(np.float64)}
)

baseline = baseline.merge(eeg_map, on="eeg_id", how="left")
baseline = baseline.merge(eeg_w, on="eeg_id", how="left")

baseline = baseline.merge(spec_map, on="spectrogram_id", how="left")
baseline = baseline.merge(spec_w, on="spectrogram_id", how="left")

baseline = baseline.merge(pat_map, on="patient_id", how="left")
baseline = baseline.merge(pat_w, on="patient_id", how="left")

eeg_vals = baseline[[f"{c}_eeg" for c in TARGET_COLS]].to_numpy(dtype=np.float64)
spec_vals = baseline[[f"{c}_spec" for c in TARGET_COLS]].to_numpy(dtype=np.float64)
pat_vals = baseline[[f"{c}_pat" for c in TARGET_COLS]].to_numpy(dtype=np.float64)

w_eeg = baseline["w_eeg"].to_numpy(dtype=np.float64)
w_spec = baseline["w_spec"].to_numpy(dtype=np.float64)
w_pat = baseline["w_pat"].to_numpy(dtype=np.float64)

has_eeg = np.isfinite(eeg_vals).all(axis=1)
has_spec = np.isfinite(spec_vals).all(axis=1)
has_pat = np.isfinite(pat_vals).all(axis=1)

w_eeg = np.where(has_eeg & np.isfinite(w_eeg), w_eeg, 0.0)
w_spec = np.where(has_spec & np.isfinite(w_spec), w_spec, 0.0)
w_pat = np.where(has_pat & np.isfinite(w_pat), w_pat, 0.0)

eeg_vals = np.nan_to_num(eeg_vals, nan=0.0, posinf=0.0, neginf=0.0)
spec_vals = np.nan_to_num(spec_vals, nan=0.0, posinf=0.0, neginf=0.0)
pat_vals = np.nan_to_num(pat_vals, nan=0.0, posinf=0.0, neginf=0.0)

MIX_EEG_MULT = 3.0
MIX_SPEC_MULT = 2.0
MIX_PAT_MULT = 1.0

W_GLOBAL_BASE = 0.25
W_GLOBAL_MAX = 25.0
evidence = w_eeg + w_spec + w_pat
w_global = W_GLOBAL_BASE + (W_GLOBAL_MAX - W_GLOBAL_BASE) * (
    1.0 / (1.0 + evidence / 8.0)
)
w_global = w_global.astype(np.float64)

w_eeg_m = w_eeg * MIX_EEG_MULT
w_spec_m = w_spec * MIX_SPEC_MULT
w_pat_m = w_pat * MIX_PAT_MULT

num = (
    eeg_vals * w_eeg_m[:, None]
    + spec_vals * w_spec_m[:, None]
    + pat_vals * w_pat_m[:, None]
    + global_prior[None, :] * w_global[:, None]
)
den = w_eeg_m + w_spec_m + w_pat_m + w_global
den = np.clip(den, 1e-12, None)

final_vals = num / den[:, None]
final_vals = np.nan_to_num(
    final_vals,
    nan=1.0 / len(TARGET_COLS),
    posinf=1.0 / len(TARGET_COLS),
    neginf=1.0 / len(TARGET_COLS),
)

EPS_DIRICHLET = 5e-4
final_vals = (1.0 - EPS_DIRICHLET) * final_vals + EPS_DIRICHLET * (
    1.0 / len(TARGET_COLS)
)

final_vals = np.clip(final_vals, 1e-15, 1.0)
final_vals = final_vals / final_vals.sum(axis=1, keepdims=True)

baseline_out = pd.DataFrame({"eeg_id": baseline["eeg_id"].values})
baseline_out[TARGET_COLS] = final_vals

baseline_out = (
    baseline_out.set_index("eeg_id").reindex(sample_sub["eeg_id"].values).reset_index()
)
if baseline_out[TARGET_COLS].isna().any().any():
    baseline_out[TARGET_COLS] = baseline_out[TARGET_COLS].fillna(global_prior)

probs = baseline_out[TARGET_COLS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
baseline_out[TARGET_COLS] = probs

baseline_fp = os.path.join(OUT_PATH, "submission_baseline.csv")
baseline_out = baseline_out[["eeg_id"] + TARGET_COLS]
baseline_out.to_csv(baseline_fp, index=False)
print(f"Wrote baseline submission to: {baseline_fp}")
print(baseline_out.head())




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
    fallback_df=baseline_out,
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
