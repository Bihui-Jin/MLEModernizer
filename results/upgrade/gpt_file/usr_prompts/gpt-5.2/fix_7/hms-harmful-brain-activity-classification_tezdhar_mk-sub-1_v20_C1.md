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

0.84251

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the failing dependency installs and external “mk-codes/mk-data” calls, since those inputs aren’t available in your provided environment and currently stop execution before any submission is written. To keep the pipeline valid end-to-end, I replace the broken inference stage with a minimal, deterministic baseline that uses the training vote distributions (properly normalized) to produce valid per-class probabilities for every test `eeg_id`. I also harden the submission creation to enforce required columns, row alignment to `sample_submission.csv`, numeric type, clipping, and row-wise probability normalization so the file always passes Kaggle’s format checks. This yield a valid `submission.csv` and a reasonable KL baseline score, moving you toward the target rather than failing to submit.'
- What this solution (achieved 0.83266) has done: 'Your current baseline predicts the same global class prior for every test row, which is stable but too weak (KL ≈ 1.42) versus your target (≈ 0.34). To move toward the target with minimal semantic changes, I keep the “prior-based” idea but condition it on `patient_id` (a strong metadata signal available in both train/test) by using a smoothed patient-specific vote distribution when possible, otherwise falling back to the global prior. I also add simple Laplace/Dirichlet smoothing and blend patient-prior with global prior to avoid overfitting sparse patients while improving calibration. The output is still a valid probability distribution per row and is aligned to `sample_submission.csv` to guarantee a valid submission.'
- What this solution (achieved 0.83266) has done: 'Your current metadata-only baseline is too weak versus the target (lower-is-better), so we should gently increase signal while keeping the same “smoothed prior” core logic. The smallest meaningful improvement is to condition priors not just on `patient_id` but also on `eeg_id` when that EEG appears in training (train has many rows per `eeg_id`, and test `eeg_id`s may overlap), falling back to patient-prior, then global prior. To avoid overfitting and keep calibration stable for KL, we keep the same Dirichlet/Laplace smoothing + blending structure, just add one extra hierarchy level (`eeg_id` → `patient_id` → global) with conservative blending weights. Submission creation remains aligned to `sample_submission.csv` and enforces strict probability normalization.'
- What this solution (achieved 0.83266) has done: 'Your current approach is a hierarchical, smoothed prior (eeg_id → patient_id → global), but it likely underuses strong metadata signals that exist in both train and test. To move the KL score down toward the target with minimal semantic change, I add one more conservative hierarchy level based on `spectrogram_id` (often highly informative and present in both train/test), placed above `eeg_id`. I keep the exact same smoothing+blending semantics and just extend the lookup chain (`spectrogram_id → eeg_id → patient_id → global`) with cautious weights so calibration remains stable. I also make the mapping robust in case `sample_submission` ordering differs from `test.csv` by building lookup dicts and defaulting safely.'
- What this solution (achieved 0.84251) has done: 'Your current hierarchical smoothed-prior is valid but likely under-confident because it falls back too often and it ignores extra structure inside train where each `spectrogram_id`/`eeg_id` can have multiple label rows with varying label quality. To move the KL down toward the target with minimal semantic change, I keep the exact same inference idea (hierarchical priors + Dirichlet smoothing + blending) but (1) normalize each train row’s votes to a probability distribution before aggregation (so recordings with more raters don’t dominate purely by count), and (2) blend hierarchically rather than “pick first available”, so when a key exists we still softly incorporate lower levels/global for better calibration. I also compute blends based on evidence strength (total votes for that group) to avoid overconfident priors for sparse groups while keeping the core logic intact. The submission creation remains aligned to `sample_submission.csv` and strictly enforces per-row probability normalization.'

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



## === cell 1
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

missing_train = [
    c
    for c in (["patient_id", "eeg_id", "spectrogram_id"] + TARGET_COLS)
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

votes = train[TARGET_COLS].astype(np.float64).to_numpy()
row_sum = votes.sum(axis=1, keepdims=True)
row_sum = np.where(row_sum > 0, row_sum, 1.0)
row_probs = votes / row_sum
global_prior = row_probs.mean(axis=0)
global_prior = global_prior / global_prior.sum()

print("Computed global class prior:", dict(zip(TARGET_COLS, global_prior.round(6))))
print(
    "Train rows:", len(train), "Test rows:", len(test), "Sample rows:", len(sample_sub)
)

train_key = train[["spectrogram_id", "eeg_id", "patient_id"] + TARGET_COLS].copy()
train_key["spectrogram_id"] = train_key["spectrogram_id"].astype(np.int64)
train_key["eeg_id"] = train_key["eeg_id"].astype(np.int64)
train_key["patient_id"] = train_key["patient_id"].astype(np.int64)

train_key["_row_vote_sum"] = train_key[TARGET_COLS].sum(axis=1).astype(np.float64)
safe_row_sum = train_key["_row_vote_sum"].to_numpy()
safe_row_sum = np.where(safe_row_sum > 0, safe_row_sum, 1.0)
train_key_prob = train_key.copy()
train_key_prob[TARGET_COLS] = (
    train_key_prob[TARGET_COLS].astype(np.float64).to_numpy() / safe_row_sum[:, None]
)

spect_prob_sums = (
    train_key_prob.groupby("spectrogram_id", sort=False)[TARGET_COLS]
    .sum()
    .astype(np.float64)
)
eeg_prob_sums = (
    train_key_prob.groupby("eeg_id", sort=False)[TARGET_COLS].sum().astype(np.float64)
)
patient_prob_sums = (
    train_key_prob.groupby("patient_id", sort=False)[TARGET_COLS]
    .sum()
    .astype(np.float64)
)

spect_strength = (
    train_key.groupby("spectrogram_id", sort=False)["_row_vote_sum"]
    .sum()
    .astype(np.float64)
)
eeg_strength = (
    train_key.groupby("eeg_id", sort=False)["_row_vote_sum"].sum().astype(np.float64)
)
patient_strength = (
    train_key.groupby("patient_id", sort=False)["_row_vote_sum"]
    .sum()
    .astype(np.float64)
)

test_patients = set(test["patient_id"].astype(np.int64).unique().tolist())
train_patients = set(patient_prob_sums.index.astype(np.int64).tolist())
seen_p = len(test_patients & train_patients)
unseen_p = len(test_patients - train_patients)

test_eegs = set(test["eeg_id"].astype(np.int64).unique().tolist())
train_eegs = set(eeg_prob_sums.index.astype(np.int64).tolist())
seen_e = len(test_eegs & train_eegs)
unseen_e = len(test_eegs - train_eegs)

test_specs = set(test["spectrogram_id"].astype(np.int64).unique().tolist())
train_specs = set(spect_prob_sums.index.astype(np.int64).tolist())
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




## === cell 2
def make_submission_hierarchical_prior(
    sample_submission: pd.DataFrame,
    test_df: pd.DataFrame,
    global_prior: np.ndarray,
    spect_prob_sums: pd.DataFrame,
    eeg_prob_sums: pd.DataFrame,
    patient_prob_sums: pd.DataFrame,
    spect_strength: pd.Series,
    eeg_strength: pd.Series,
    patient_strength: pd.Series,
    target_cols: list[str],
    alpha_spect: float = 1.0,
    alpha_eeg: float = 1.0,
    alpha_patient: float = 2.0,
    base_blend_spect: float = 0.60,
    base_blend_eeg: float = 0.55,
    base_blend_patient: float = 0.70,
    strength_scale_spect: float = 12.0,
    strength_scale_eeg: float = 10.0,
    strength_scale_patient: float = 8.0,
) -> pd.DataFrame:
    """
    Create submission aligned to sample_submission order.

    Change (score-relevant, core logic preserved): instead of hard "first-match wins",
    do a soft hierarchical blend: global -> patient -> eeg -> spectrogram (when available).
    Blend weight is adapted by evidence strength to avoid overconfidence for sparse groups.
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

    spect_index = spect_prob_sums.index.to_numpy(dtype=np.int64)
    spect_to_row = {sid: i for i, sid in enumerate(spect_index)}
    eeg_index = eeg_prob_sums.index.to_numpy(dtype=np.int64)
    eeg_to_row = {eid: i for i, eid in enumerate(eeg_index)}
    patient_index = patient_prob_sums.index.to_numpy(dtype=np.int64)
    patient_to_row = {pid: i for i, pid in enumerate(patient_index)}

    spect_strength_map = spect_strength.to_dict()
    eeg_strength_map = eeg_strength.to_dict()
    patient_strength_map = patient_strength.to_dict()

    for i, (sid, eid, pid) in enumerate(zip(sids, sample_eeg_ids, pids)):
        p = global_prior.copy()

        pid_row = patient_to_row.get(pid, None)
        if pid_row is not None:
            counts = patient_prob_sums.iloc[pid_row].to_numpy(dtype=np.float64)
            smoothed = counts + alpha_patient
            patient_prior = smoothed / smoothed.sum()
            strength = float(patient_strength_map.get(int(pid), 0.0))
            w = base_blend_patient * (strength / (strength + strength_scale_patient))
            p = (1.0 - w) * p + w * patient_prior

        eeg_row = eeg_to_row.get(eid, None)
        if eeg_row is not None:
            counts = eeg_prob_sums.iloc[eeg_row].to_numpy(dtype=np.float64)
            smoothed = counts + alpha_eeg
            eeg_prior = smoothed / smoothed.sum()
            strength = float(eeg_strength_map.get(int(eid), 0.0))
            w = base_blend_eeg * (strength / (strength + strength_scale_eeg))
            p = (1.0 - w) * p + w * eeg_prior

        spect_row = spect_to_row.get(sid, None)
        if spect_row is not None:
            counts = spect_prob_sums.iloc[spect_row].to_numpy(dtype=np.float64)
            smoothed = counts + alpha_spect
            spect_prior = smoothed / smoothed.sum()
            strength = float(spect_strength_map.get(int(sid), 0.0))
            w = base_blend_spect * (strength / (strength + strength_scale_spect))
            p = (1.0 - w) * p + w * spect_prior

        probs[i] = p

    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum(axis=1, keepdims=True)

    out = sample_submission[["eeg_id"]].copy()
    out[target_cols] = probs
    return out


sol = make_submission_hierarchical_prior(
    sample_submission=sample_sub,
    test_df=test,
    global_prior=global_prior,
    spect_prob_sums=spect_prob_sums,
    eeg_prob_sums=eeg_prob_sums,
    patient_prob_sums=patient_prob_sums,
    spect_strength=spect_strength,
    eeg_strength=eeg_strength,
    patient_strength=patient_strength,
    target_cols=TARGET_COLS,
    alpha_spect=1.0,
    alpha_eeg=1.0,
    alpha_patient=2.0,
    base_blend_spect=0.60,
    base_blend_eeg=0.55,
    base_blend_patient=0.70,
    strength_scale_spect=12.0,
    strength_scale_eeg=10.0,
    strength_scale_patient=8.0,
)

assert list(sol.columns) == ["eeg_id"] + TARGET_COLS
assert sol[TARGET_COLS].isna().sum().sum() == 0
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-9):
    raise ValueError(f"Row sums not 1.0; min={row_sums.min()}, max={row_sums.max()}")

print("Submission dataframe ready:", sol.shape)
print(sol.head())



## === cell 3
sub_path = os.path.join(OUT_PATH, "submission.csv")
sol.to_csv(sub_path, index=False)
print(f"Wrote: {sub_path} with shape={sol.shape}")
print(sol.describe(include="all"))
