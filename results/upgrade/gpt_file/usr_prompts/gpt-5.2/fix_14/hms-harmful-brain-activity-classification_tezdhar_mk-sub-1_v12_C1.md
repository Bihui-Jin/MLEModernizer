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

0.3385842184983305

# 6. Current score

0.74795

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75645) has done: 'I remove the hard dependency on the missing external code repo (`/kaggle/input/hms-mk-codes`) and replace the broken subprocess-based pipeline with a lightweight, fully in-notebook baseline that reads the provided `train.csv`/`test.csv` and produces a valid probabilistic submission. This fixes the immediate runtime errors (missing directory, missing fold files, undefined `sol`) and guarantees a `submission.csv` with the correct columns and row-wise probabilities summing to 1. To move the KL score toward a reasonable target without changing any unavailable model logic, I use a stable, competition-appropriate fallback: patient-level smoothed label priors learned from train (and global prior for unseen patients), with proper normalization and clipping for numerical safety. This should yield a materially better score than uniform predictions while staying fast and robust within the 600s limit.'
- What this solution (achieved 0.91682) has done: 'Your current baseline is already valid but too weak (KL 0.756 > target 0.339, lower is better), so we make the smallest “still-the-same-idea” upgrade: compute smoothed class priors at the EEG-recording level (by `eeg_id`) instead of the patient level, then fall back to patient-level smoothing, then global prior. This preserves the same core logic (label-prior smoothing from train metadata only) while using a much more directly relevant key for test (`eeg_id` is unique per row and matches the submission index), which should materially reduce KL. We also add a tiny Dirichlet-style additive smoothing on the vote counts before normalization to avoid pathological zeros and improve calibration without changing semantics. The output remains a proper probability distribution per row and writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.78029) has done: 'Your current priors are averaged at the row level (mean of per-row normalized votes), which can overweight small-vote rows and miscalibrate probabilities for KL. I keep the same “metadata-only smoothed priors” core logic, but compute priors by summing vote counts within each group (eeg_id/patient_id) and then normalizing once, which is a minimal semantic fix that usually reduces KL. I also replace the slow per-key loop in `make_smoothed_prior` with a vectorized formulation (same math) to improve stability and runtime, and add a final alignment step to ensure submission order exactly matches `sample_submission.csv`. Output remains a valid probability distribution per row and still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.78029) has done: 'I fix the runtime error in `merge_preds` caused by using `keepdims` with pandas `.sum()` by switching to a NumPy-based row normalization that preserves the same semantics. I also make the fold file creation and merge robust to any row-order differences by always aligning to `sample_submission.csv`’s `eeg_id` order and using fast membership dictionaries (no score-changing logic). Finally, I ensure `sol` is always created so the final `/kaggle/working/submission.csv` is written with correct columns and row-wise probabilities summing to 1.'
- What this solution (achieved 0.78029) has done: 'I fix the `merge_preds` length mismatch by ensuring each fold file is aggregated to one row per `eeg_id` before aligning to `sample_submission.csv` (your fold file currently has many duplicate `eeg_id`s from `test.csv`, causing the exploded size). I keep the same core logic (smoothed priors from train metadata), but also make the fold creation itself write exactly the sample submission’s `eeg_id` order to prevent future alignment issues. Finally, I re-enable end-to-end execution by guaranteeing `sol` is created and `/kaggle/working/submission.csv` is written with correct columns and row-wise probabilities summing to 1. These are correctness/stability fixes and should also improve KL versus the broken/duplicated merge.'
- What this solution (achieved 0.78029) has done: 'Your current submission is valid but the score is far worse than the target (0.780 vs 0.339, lower is better), so we make the smallest metadata-only upgrade that better matches the KL metric without changing the core “smoothed priors from train.csv” approach. Specifically, instead of predicting at the `eeg_id` level (which almost never appears in train for test), we predict at the `spectrogram_id` level (which aligns better to the labeling unit in train/test), then fall back to `patient_id`, then global. We also keep the same Dirichlet-style smoothing but tune the mixing weights slightly toward the more informative key (`spectrogram_id`) to reduce KL. All I/O paths and the final `submission.csv` format remain unchanged and probabilities are still normalized per row.'
- What this solution (achieved 0.76716) has done: 'Your current approach is a metadata-only smoothed prior, but it likely underperforms because it doesn’t use the most predictive signal available in `train.csv`: the `expert_consensus` categorical label. To move the KL score down toward the target while preserving the same “prior from train metadata → lookup by keys → fallback” core logic, I compute a smoothed class-probability table by `expert_consensus`, and at test-time I infer each test row’s consensus distribution via the patient’s mix of consensus labels in train (then still fall back to spectrogram/patient/global as before). This keeps everything fast, avoids any modeling/feature extraction changes, and improves calibration because it uses a strong label proxy derived only from training metadata. I also keep the existing Dirichlet-style smoothing and strict row-wise normalization so the submission remains valid.'
- What this solution (achieved 0.76716) has done: 'Your current metadata-prior approach is valid but underpowered because it ignores the strongest easy signal available in `train.csv`: the matched `spectrogram_id` labels distribution itself. I keep the same “smoothed priors from grouped vote counts + fallback + normalization” core logic, but (1) compute the spectrogram prior using *all train rows for that spectrogram* (already done) and then (2) add a small, safe blend with the patient prior (and patient-consensus prior) to reduce overconfidence when a spectrogram has sparse/biased labels. This typically lowers KL (better calibration) without changing the modeling paradigm. I also remove slow dict lookups in the prediction loop by precomputing aligned arrays for test rows (same math, less room for key-mismatch mistakes) and keep strict clipping/renormalization so every row sums to 1.'
- What this solution (achieved 0.77303) has done: 'Your current score is far worse than the target (0.767 vs 0.339, lower is better), so the smallest safe improvement is to better calibrate your existing metadata-prior blend without changing the underlying “grouped vote-count priors + fallback” logic. I keep the same priors (spectrogram/patient/patient-from-consensus/global) but replace the hard weights with a data-dependent shrinkage based on how much vote evidence a spectrogram has in train, so sparse spectrograms rely more on patient/global and dense ones rely more on spectrogram. I also apply a tiny final mix with the global prior to reduce overconfident probabilities (helpful for KL), while preserving normalization and submission format. All paths stay the same and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.7509) has done: 'Your current metadata-prior blend is likely miscalibrated for KL because it uses vote *counts* (which vary by row) without first normalizing to per-row probabilities; this can overweight rows with more annotators and push distributions to be too sharp. I keep the same core logic (spectrogram prior + patient/patient-from-consensus fallback + global shrinkage), but change the training aggregation to sum **per-row normalized vote probabilities** instead of raw vote counts, which tends to improve calibration and reduce KL. I also make the spectrogram “strength” used for shrinkage consistent with that same aggregation (effective sample size = number of rows per spectrogram), so the data-dependent weighting behaves as intended. All I/O paths remain unchanged, runtime stays fast, and the script still writes `/kaggle/working/submission.csv` with rows summing to 1.'
- What this solution (achieved 0.74795) has done: 'Your current score (0.7509, lower-is-better) is far from the target (0.3386), so we should improve calibration while keeping the same “metadata-only smoothed priors + fallback + normalization” core logic. The minimal, high-impact fix is to stop using “rows per spectrogram” as the confidence/strength signal (which is only loosely related to label certainty) and instead use the *effective sample size from normalized vote distributions* (ESS) per spectrogram, then use that ESS in the same shrinkage formula. This preserves your architecture and semantics (still a smoothed prior lookup blended with fallback), but makes the weight on the spectrogram prior better aligned to how confident the aggregated distribution really is, which typically reduces KL. I also slightly reduce the unconditional global mixing (w_global) to avoid washing out informative priors; it remains a small safety term for KL.'
- What this solution (achieved 0.74795) has done: 'You’re currently far above the target (0.74795 vs 0.3386, lower is better), so we should improve KL by making the existing metadata-prior blend better calibrated without changing the overall approach. The smallest high-impact change is to stop using the summed per-row ESS as “strength” (it can overstate confidence) and instead use an **effective concentration** derived from the aggregated distribution’s entropy (via Dirichlet ESS ≈ 1/∑p²), which better reflects how sharp/uncertain each spectrogram’s label mix is. We then use that concentration both to (a) adapt the spectrogram-vs-regularizer mixing weight and (b) very lightly temper overconfident spectrogram priors toward the global prior when concentration is high—both are calibration tweaks that typically reduce KL. All paths, columns, and the core “grouped priors + fallback + normalize” logic remain the same, and the script still writes `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys

CODE_ROOT = "/kaggle/input/hms-mk-codes"
if os.path.isdir(CODE_ROOT):
    sys.path.append(CODE_ROOT)
    sys.path.append(os.path.join(CODE_ROOT, "src"))
    print("Added to sys.path:", CODE_ROOT, "and", os.path.join(CODE_ROOT, "src"))
else:
    print(
        "NOTE: External code repo not found at", CODE_ROOT, "- using fallback pipeline."
    )

print("Python:", sys.version)



## === cell 1
import pathlib
import subprocess

req_dir = pathlib.Path("/kaggle/input/requirements-mk")
wheels = [
    (
        "antlr4_python3_runtime-4.9.2-py3-none-any.whl",
        ["--no-index", "--no-deps", "--force-reinstall"],
    ),
    ("omegaconf-2.3.0-py3-none-any.whl", ["--no-index", "--no-deps"]),
    ("hydra_core-1.3.2-py3-none-any.whl", ["--no-index", "--no-deps"]),
    ("lightning-2.2.1-py3-none-any.whl", ["--no-deps", "--no-index"]),
]

if req_dir.exists():
    for whl, extra in wheels:
        p = req_dir / whl
        if p.exists():
            cmd = [sys.executable, "-m", "pip", "install", str(p)] + extra
            print("Running:", " ".join(cmd))
            subprocess.check_call(cmd)
        else:
            print(f"Wheel not found, skipping install: {p}")
else:
    print("NOTE: requirements-mk not found; skipping wheel installs.")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

train_csv = os.path.join(DATA_PATH, "train.csv")
test_csv = os.path.join(DATA_PATH, "test.csv")
sample_csv = os.path.join(DATA_PATH, "sample_submission.csv")

for p in [train_csv, test_csv, sample_csv]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

print("DATA_PATH:", DATA_PATH)
print("OUT_PATH:", OUT_PATH)



## === cell 3
import os

print("Listing /kaggle/working (OUT_PATH):")
print(sorted(os.listdir(OUT_PATH))[:50])



## === cell 4
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 5
import numpy as np
import pandas as pd

train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
sample = pd.read_csv(sample_csv)

target_cols = [c for c in sample.columns if c != "eeg_id"]
if target_cols != [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]:
    if len(target_cols) != 6:
        raise ValueError(
            f"Expected 6 target columns, got {len(target_cols)}: {target_cols}"
        )

eps_row = 1e-12
train_counts = train[
    ["spectrogram_id", "eeg_id", "patient_id", "expert_consensus"] + target_cols
].copy()
for c in target_cols:
    train_counts[c] = train_counts[c].astype(np.float64)

row_sum = train_counts[target_cols].sum(axis=1).astype(np.float64)
row_sum = row_sum.mask(row_sum <= 0.0, np.nan)
row_probs = train_counts[target_cols].div(row_sum, axis=0)
row_probs = row_probs.fillna(1.0 / len(target_cols)).astype(np.float64)
row_probs = np.clip(row_probs.to_numpy(dtype=np.float64), 1e-15, 1.0)
row_probs = row_probs / row_probs.sum(axis=1, keepdims=True)
train_counts[target_cols] = row_probs

eps_count = 0.5

global_counts = (
    train_counts[target_cols].sum(axis=0).to_numpy(dtype=np.float64) + eps_count
)
global_mean = global_counts / global_counts.sum()
global_mean = np.clip(global_mean, 1e-15, 1.0)
global_mean = global_mean / global_mean.sum()


def make_smoothed_prior_from_counts(group_key: str, alpha: float) -> pd.DataFrame:
    grp_counts = train_counts.groupby(group_key)[target_cols].sum().astype(np.float64)
    grp_total = grp_counts.sum(axis=1).to_numpy(dtype=np.float64)

    grp_counts_sm = grp_counts.to_numpy(dtype=np.float64) + eps_count
    grp_prob = grp_counts_sm / grp_counts_sm.sum(axis=1, keepdims=True)

    n = grp_total.reshape(-1, 1)
    post = (n * grp_prob + alpha * global_mean.reshape(1, -1)) / (n + alpha)

    post = np.clip(post, 1e-15, 1.0)
    post = post / post.sum(axis=1, keepdims=True)

    return pd.DataFrame(post, index=grp_counts.index, columns=target_cols)


alpha_spec = 5.0
alpha_patient = 20.0
alpha_consensus = 5.0

spec_smoothed = make_smoothed_prior_from_counts("spectrogram_id", alpha=alpha_spec)
patient_smoothed = make_smoothed_prior_from_counts("patient_id", alpha=alpha_patient)
cons_smoothed = make_smoothed_prior_from_counts(
    "expert_consensus", alpha=alpha_consensus
)

spec_prior = {
    k: spec_smoothed.loc[k, target_cols].to_numpy(dtype=np.float64)
    for k in spec_smoothed.index
}
patient_prior = {
    k: patient_smoothed.loc[k, target_cols].to_numpy(dtype=np.float64)
    for k in patient_smoothed.index
}
cons_prior = {
    k: cons_smoothed.loc[k, target_cols].to_numpy(dtype=np.float64)
    for k in cons_smoothed.index
}

cons_counts_by_patient = (
    train_counts.groupby(["patient_id", "expert_consensus"], dropna=False)
    .size()
    .unstack(fill_value=0)
)
cons_cols = list(cons_counts_by_patient.columns)

eps_cons = 0.25
cons_mat = cons_counts_by_patient.to_numpy(dtype=np.float64) + eps_cons
cons_prob_mat = cons_mat / cons_mat.sum(axis=1, keepdims=True)

cons2prob_mat = np.zeros((len(cons_cols), len(target_cols)), dtype=np.float64)
for i, lab in enumerate(cons_cols):
    v = cons_prior.get(lab, None)
    if v is None:
        v = global_mean
    cons2prob_mat[i] = v

patient_from_cons = cons_prob_mat @ cons2prob_mat
patient_from_cons = np.clip(patient_from_cons, 1e-15, 1.0)
patient_from_cons = patient_from_cons / patient_from_cons.sum(axis=1, keepdims=True)

patient_cons_prior = {
    pid: patient_from_cons[i]
    for i, pid in enumerate(cons_counts_by_patient.index.to_numpy())
}

spec_conc = {}
for sid, v in spec_prior.items():
    v = np.clip(np.asarray(v, dtype=np.float64), 1e-15, 1.0)
    v = v / v.sum()
    ess = 1.0 / max(float(np.sum(v * v)), 1e-15)  # in [1, K]
    K = float(len(target_cols))
    conc = K / ess
    spec_conc[sid] = conc

test_eeg_ids = sample["eeg_id"].to_numpy()
test_unique = test.drop_duplicates("eeg_id").set_index("eeg_id")

aligned = test_unique.reindex(test_eeg_ids)
if aligned.isna().any().any():
    raise ValueError("Some sample eeg_id not found in test.csv after alignment.")

test_pids = aligned["patient_id"].to_numpy()
test_sids = aligned["spectrogram_id"].to_numpy()

tau = 10.0  # was 40.0; with new conc definition, smaller tau gives useful spectrogram weight

w_global = 0.01

gamma = 0.15  # tempering strength, small to keep changes minimal

preds = np.zeros((len(test_eeg_ids), len(target_cols)), dtype=np.float64)
for i in range(len(test_eeg_ids)):
    sid = test_sids[i]
    pid = test_pids[i]

    v_spec = spec_prior.get(sid, None)
    v_patc = patient_cons_prior.get(pid, None) if pid is not None else None
    v_pat = patient_prior.get(pid, None) if pid is not None else None

    v_reg = (
        v_patc if v_patc is not None else (v_pat if v_pat is not None else global_mean)
    )

    if v_spec is not None:
        n_eff = float(spec_conc.get(sid, 0.0))
        w_spec = n_eff / (n_eff + tau)

        v_spec_adj = (1.0 - gamma * w_spec) * v_spec + (gamma * w_spec) * global_mean
        v = w_spec * v_spec_adj + (1.0 - w_spec) * v_reg
    else:
        v = v_reg

    v = (1.0 - w_global) * v + w_global * global_mean
    preds[i] = v

preds = np.clip(preds, 1e-15, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

submission_fold0 = pd.DataFrame({"eeg_id": test_eeg_ids})
for j, c in enumerate(target_cols):
    submission_fold0[c] = preds[:, j]

fold0_path = os.path.join(OUT_PATH, "submission_fold0.csv")
submission_fold0.to_csv(fold0_path, index=False)
print("Wrote fallback fold submission:", fold0_path, "rows:", len(submission_fold0))



## === cell 6
import numpy as np
import pandas as pd
import pathlib


def merge_preds(folds=(0,), data_path=DATA_PATH, workdir="/kaggle/working"):
    sample = pd.read_csv(f"{data_path}/sample_submission.csv")
    target_cols = [c for c in sample.columns if c != "eeg_id"]
    if len(target_cols) != 6:
        raise ValueError(
            f"Expected 6 target columns, got {len(target_cols)}: {target_cols}"
        )

    preds_list = []
    base_eeg_ids = None

    for fold in folds:
        fpath = pathlib.Path(workdir) / f"submission_fold{fold}.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing fold submission file: {fpath}")
        df = pd.read_csv(fpath)

        if "eeg_id" not in df.columns:
            raise ValueError(f"{fpath} missing 'eeg_id' column.")
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{fpath} missing target columns: {missing}")

        if df["eeg_id"].duplicated().any():
            df = df.groupby("eeg_id", as_index=False)[target_cols].mean()

        df = sample[["eeg_id"]].merge(
            df[["eeg_id"] + target_cols], on="eeg_id", how="left"
        )
        if df[target_cols].isna().any().any():
            raise ValueError(
                f"{fpath} produced NaNs after alignment; missing eeg_id predictions."
            )

        if base_eeg_ids is None:
            base_eeg_ids = df["eeg_id"].to_numpy()
        else:
            if not np.array_equal(base_eeg_ids, df["eeg_id"].to_numpy()):
                raise ValueError(
                    "Internal alignment error: eeg_id order mismatch after merge."
                )

        preds_list.append(df[target_cols].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(preds_list, axis=0), axis=0)
    preds = np.clip(preds, 1e-15, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = sample[["eeg_id"]].copy()
    out[target_cols] = preds

    arr = out[target_cols].to_numpy(dtype=np.float64)
    arr = np.clip(arr, 1e-15, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    out[target_cols] = arr
    return out


sol = merge_preds()



## === cell 7
out_path = "/kaggle/working/submission.csv"
sol.to_csv(out_path, index=False)

target_cols = [c for c in sol.columns if c != "eeg_id"]
row_sums = sol[target_cols].sum(axis=1).values
print("Saved:", out_path)
print("Rows:", len(sol), "Cols:", list(sol.columns))
print("Row-sum min/max:", float(row_sums.min()), float(row_sums.max()))
print(sol.head())



## === cell 8
sol
