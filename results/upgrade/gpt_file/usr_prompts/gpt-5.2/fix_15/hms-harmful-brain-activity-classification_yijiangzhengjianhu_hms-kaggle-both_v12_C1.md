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

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.4145331432764152

# 6. Current score

0.78609

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'You’re crashing because no model checkpoints are found, so `preds` stays empty and `np.stack(preds)` fails. I make the checkpoint discovery robust to the actual Kaggle input layout and add a safe fallback: if no `.pt` models are available, we generate a valid baseline submission using the training-set mean class distribution (score won’t be great but it yield a valid CSV). I also fix test-time offset handling by creating an `offset` column in `test` (expected by `DataGenerator` in non-test modes, and harmless here), and ensure the inference generator always produces tensors in `float32` with the right shape. All changes are minimal and aimed at getting an end-to-end run and a valid `submission.csv`.'
- What this solution (achieved 0.77167) has done: 'The run currently fails because `sample_submission.csv` contains duplicate `eeg_id`, so asserting uniqueness is incorrect and blocks submission creation. I remove that brittle assertion and instead align predictions to the sample submission by merging on `eeg_id` and then aggregating back to exactly the sample submission’s row count/order (so duplicates are handled correctly). I also add a final safety normalization to guarantee each row sums to 1 (required by Kaggle) and keep the rest of your modeling/calibration logic unchanged.'
- What this solution (achieved 0.78609) has done: 'Your current solution is a pure metadata-prior model, so the main safe lever to reduce KL is calibration: avoid overconfident predictions and reduce near-zero probabilities. I keep your exact aggregation/smoothing logic, but (1) tune the tiny `eps_mix` used during CV (it was fixed at `1e-6`, which is often too small for KL), (2) add a very small per-row “temperature” sharpening/softening (power transform) as an extra calibration parameter, and (3) include both in the same CV grid search so we only change inference based on what actually improves CV-wKL. These are minimal post-processing changes that preserve semantics (still a convex/monotone transform and normalization) and should move your public score down toward the 0.414 target without changing the modeling approach. The script still writes a valid `submission.csv` aligned to `sample_submission.csv` and normalized to sum to 1.'
- What this solution (achieved 0.78609) has done: 'Your current score (0.78609, lower-is-better) is still far from the target (0.41453), so we should improve, but with minimal, metric-aligned changes. The simplest lever that preserves your core “metadata prior + smoothing tables” logic is to fix a mismatch between how you build per-row predictions and how the competition evaluates per-eeg_id: instead of averaging probabilities across rows, we should aggregate underlying vote counts to an eeg-level probability (Dirichlet-consistent) using the same smoothing/calibration tables. I keep your exact table-building and calibration (beta/eps_mix/temp) and your CV grid, but change both CV scoring aggregation and test prediction aggregation to be count-weighted, which better matches KL weighting by number of votes and typically reduces wKL without changing the modeling approach. Finally, I keep the existing submission alignment to `sample_submission.csv` and the safety normalization.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
CLASSES = TARGETS

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"
sample_path = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_sub.shape)

assert set(["eeg_id", "spectrogram_id", "patient_id"]).issubset(test.columns)
assert set(["patient_id", "spectrogram_id"]).issubset(train.columns)
assert set(CLASSES).issubset(train.columns)
assert set(["eeg_id"] + CLASSES).issubset(sample_sub.columns)



## === cell 1
train_votes = train[CLASSES].astype(np.float64)
train_group_cols = ["eeg_id", "patient_id", "spectrogram_id"]
train_agg = (
    pd.concat([train[train_group_cols], train_votes], axis=1)
    .groupby(train_group_cols, as_index=False)[CLASSES]
    .sum()
)

votes = train_agg[CLASSES].values
den = np.clip(votes.sum(axis=1, keepdims=True), 1e-12, None)
probs = votes / den
train_probs_df = pd.DataFrame(probs, columns=CLASSES)

train2 = pd.concat(
    [train_agg[["eeg_id", "patient_id", "spectrogram_id"]].copy(), train_probs_df],
    axis=1,
)

train2_counts = pd.concat(
    [
        train_agg[["eeg_id", "patient_id", "spectrogram_id"]].copy(),
        train_agg[CLASSES].copy(),
    ],
    axis=1,
)

train_eeg_counts = train2_counts.groupby("eeg_id", as_index=False)[CLASSES].sum()
total_counts = train_eeg_counts[CLASSES].sum(axis=0).values.astype(np.float64)
global_prior = np.clip(total_counts, 1e-12, None)
global_prior = global_prior / global_prior.sum()

print("Global prior:", dict(zip(CLASSES, global_prior.round(4))))
print("Aggregated train2 shape:", train2.shape)
print("Aggregated train2_counts shape:", train2_counts.shape)
print("Aggregated train_eeg_counts shape:", train_eeg_counts.shape)




## === cell 2
def kl_divergence(p_true, p_pred, eps=1e-12):
    p_true = np.clip(p_true, eps, 1.0)
    p_true = p_true / p_true.sum(axis=1, keepdims=True)
    p_pred = np.clip(p_pred, eps, 1.0)
    p_pred = p_pred / p_pred.sum(axis=1, keepdims=True)
    return np.sum(p_true * (np.log(p_true) - np.log(p_pred)), axis=1)


def build_smooth_tables_from_counts(
    train_counts_df, global_prior_vec, alpha_patient, alpha_spec
):
    """
    Compute patient/spec tables by aggregating vote COUNTS and applying Dirichlet-style smoothing,
    then convert to probabilities.
    """
    patient_counts = train_counts_df.groupby("patient_id")[CLASSES].sum()
    spec_counts = train_counts_df.groupby("spectrogram_id")[CLASSES].sum()

    patient_total = patient_counts.sum(axis=1).astype(np.float64)
    spec_total = spec_counts.sum(axis=1).astype(np.float64)

    patient_smooth_counts = patient_counts.add(alpha_patient * global_prior_vec, axis=1)
    spec_smooth_counts = spec_counts.add(alpha_spec * global_prior_vec, axis=1)

    patient_smooth = patient_smooth_counts.div(
        np.clip(patient_smooth_counts.sum(axis=1), 1e-12, None), axis=0
    )
    spec_smooth = spec_smooth_counts.div(
        np.clip(spec_smooth_counts.sum(axis=1), 1e-12, None), axis=0
    )

    w_patient = np.clip(patient_total / (patient_total + alpha_patient), 0.0, 0.9)
    w_spec = np.clip(spec_total / (spec_total + alpha_spec), 0.0, 0.9)
    return patient_smooth, spec_smooth, w_patient, w_spec


def predict_from_tables(
    df_meta,
    patient_smooth,
    spec_smooth,
    w_patient,
    w_spec,
    global_prior_vec,
    beta=0.05,
    eps_mix=1e-6,
    temp=1.0,
):
    """
    Add a tiny epsilon-mix before normalization to avoid near-zero probs,
    which can be disproportionately penalized under KL.
    """
    p_idx = df_meta["patient_id"].values
    s_idx = df_meta["spectrogram_id"].values

    p_df = patient_smooth.reindex(p_idx)
    s_df = spec_smooth.reindex(s_idx)

    p_w = w_patient.reindex(p_idx).to_numpy(dtype=np.float64)
    s_w = w_spec.reindex(s_idx).to_numpy(dtype=np.float64)

    p_w = np.nan_to_num(p_w, nan=0.0)
    s_w = np.nan_to_num(s_w, nan=0.0)

    p_probs = p_df.to_numpy(dtype=np.float64)
    s_probs = s_df.to_numpy(dtype=np.float64)

    p_missing = np.isnan(p_probs).any(axis=1)
    s_missing = np.isnan(s_probs).any(axis=1)
    if p_missing.any():
        p_probs[p_missing] = global_prior_vec
    if s_missing.any():
        s_probs[s_missing] = global_prior_vec

    w_sum = np.clip(p_w + s_w, 0.0, 1.0)
    w_prior = 1.0 - w_sum
    base = (
        (p_probs * p_w[:, None])
        + (s_probs * s_w[:, None])
        + (global_prior_vec[None, :] * w_prior[:, None])
    )

    beta = float(np.clip(beta, 0.0, 0.30))
    pred = (1.0 - beta) * base + beta * global_prior_vec[None, :]

    eps_mix = float(np.clip(eps_mix, 0.0, 5e-2))
    if eps_mix > 0:
        pred = (1.0 - eps_mix) * pred + eps_mix * (np.ones_like(pred) / pred.shape[1])

    pred = np.clip(pred, 1e-12, 1.0)

    temp = float(np.clip(temp, 0.5, 2.0))
    if temp != 1.0:
        pred = np.power(pred, 1.0 / temp)

    pred = pred / pred.sum(axis=1, keepdims=True)
    return pred


def aggregate_probs_to_eeg_dirichlet(probs_row, counts_row, eps=1e-12):
    """
    Change (score improvement): Instead of averaging per-row probabilities, aggregate to eeg-level
    via a count-weighted Dirichlet-consistent aggregation:
      pseudo_counts = sum_i (vote_total_i * p_i), then normalize.
    This better matches the competition's KL weighting by vote counts and usually reduces wKL.
    """
    w = np.clip(
        counts_row.sum(axis=1).astype(np.float64), eps, None
    )  # vote totals per row
    pseudo = probs_row.astype(np.float64) * w[:, None]
    agg = pseudo.sum(axis=0, keepdims=True)
    agg = np.clip(agg, eps, None)
    agg = agg / agg.sum(axis=1, keepdims=True)
    return agg.reshape(-1)


patients = train2["patient_id"].values
uniq_pat = pd.unique(patients)
rng = np.random.default_rng(0)
rng.shuffle(uniq_pat)
n_folds = 3
fold_ids = np.array_split(uniq_pat, n_folds)

alpha_patient_grid = [5.0, 10.0, 20.0, 40.0]
alpha_spec_grid = [5.0, 10.0, 20.0, 40.0]
beta_grid = [0.00, 0.02, 0.05, 0.08, 0.12]

eps_mix_grid = [1e-6, 1e-4, 5e-4, 1e-3, 2e-3]
temp_grid = [0.9, 1.0, 1.1, 1.2]

best = None
best_score = np.inf

for ap in alpha_patient_grid:
    for aspec in alpha_spec_grid:
        for beta in beta_grid:
            for eps_mix in eps_mix_grid:
                for temp in temp_grid:
                    fold_scores = []
                    for f in range(n_folds):
                        val_pat = set(fold_ids[f].tolist())
                        is_val = train2["patient_id"].isin(val_pat).values

                        va_df = train2.loc[is_val].copy()
                        tr_counts_df = train2_counts.loc[~is_val].copy()
                        va_counts_df = train2_counts.loc[is_val].copy()

                        tr_eeg_counts = tr_counts_df.groupby("eeg_id", as_index=False)[
                            CLASSES
                        ].sum()
                        tr_total_counts = (
                            tr_eeg_counts[CLASSES].sum(axis=0).values.astype(np.float64)
                        )
                        gp = np.clip(tr_total_counts, 1e-12, None)
                        gp = gp / gp.sum()

                        patient_smooth, spec_smooth, w_patient, w_spec = (
                            build_smooth_tables_from_counts(tr_counts_df, gp, ap, aspec)
                        )

                        va_pred_row = predict_from_tables(
                            va_df[["patient_id", "spectrogram_id"]],
                            patient_smooth,
                            spec_smooth,
                            w_patient,
                            w_spec,
                            gp,
                            beta=beta,
                            eps_mix=eps_mix,
                            temp=temp,
                        )

                        va_pred_row_df = pd.DataFrame(va_pred_row, columns=CLASSES)
                        va_pred_row_df["eeg_id"] = va_df["eeg_id"].values
                        va_counts_row = va_counts_df.copy()
                        va_counts_row["row_weight"] = (
                            va_counts_row[CLASSES].sum(axis=1).astype(np.float64)
                        )

                        merged_row = va_pred_row_df.merge(
                            va_counts_row[["eeg_id", "row_weight"]],
                            on="eeg_id",
                            how="left",
                        )
                        for c in CLASSES:
                            merged_row[c] = merged_row[c].astype(
                                np.float64
                            ) * merged_row["row_weight"].astype(np.float64)

                        va_pred_eeg = merged_row.groupby("eeg_id", as_index=False)[
                            CLASSES
                        ].sum()
                        pred_den = np.clip(
                            va_pred_eeg[CLASSES]
                            .sum(axis=1)
                            .to_numpy(dtype=np.float64)[:, None],
                            1e-12,
                            None,
                        )
                        va_pred_eeg[CLASSES] = (
                            va_pred_eeg[CLASSES].to_numpy(dtype=np.float64) / pred_den
                        )

                        va_eeg_counts = va_counts_df.groupby("eeg_id", as_index=False)[
                            CLASSES
                        ].sum()

                        merged = va_eeg_counts.merge(
                            va_pred_eeg,
                            on="eeg_id",
                            how="inner",
                            suffixes=("_true", "_pred"),
                        )

                        y_true_counts = merged[[c + "_true" for c in CLASSES]].to_numpy(
                            dtype=np.float64
                        )
                        y_pred = merged[[c + "_pred" for c in CLASSES]].to_numpy(
                            dtype=np.float64
                        )

                        y_true = y_true_counts / np.clip(
                            y_true_counts.sum(axis=1, keepdims=True), 1e-12, None
                        )
                        weights = np.clip(
                            y_true_counts.sum(axis=1).astype(np.float64), 1e-12, None
                        )

                        kls = kl_divergence(y_true, y_pred)
                        fold_scores.append(
                            float(np.sum(weights * kls) / np.sum(weights))
                        )

                    cv_score = float(np.mean(fold_scores))
                    print(
                        f"ap={ap:>4}, as={aspec:>4}, beta={beta:>4.2f}, eps_mix={eps_mix:>7g}, temp={temp:>3.1f} -> CV_wKL={cv_score:.6f}"
                    )
                    if cv_score < best_score:
                        best_score = cv_score
                        best = (ap, aspec, beta, eps_mix, temp)

alpha_patient, alpha_spec, beta, eps_mix, temp = best
print("Selected params:", best, "with CV_wKL:", best_score)

patient_smooth, spec_smooth, w_patient, w_spec = build_smooth_tables_from_counts(
    train2_counts, global_prior, alpha_patient, alpha_spec
)
print("Unique patients in train:", patient_smooth.shape[0])
print("Unique spectrograms in train:", spec_smooth.shape[0])



## === cell 3
test_pred_row = predict_from_tables(
    test[["patient_id", "spectrogram_id"]],
    patient_smooth,
    spec_smooth,
    w_patient,
    w_spec,
    global_prior,
    beta=beta,
    eps_mix=eps_mix,
    temp=temp,
)

print("Pred stats min/max:", float(test_pred_row.min()), float(test_pred_row.max()))
print(
    "Row sum min/max:",
    float(test_pred_row.sum(axis=1).min()),
    float(test_pred_row.sum(axis=1).max()),
)



## === cell 4
pred_df = pd.DataFrame(test_pred_row, columns=CLASSES)
pred_df["eeg_id"] = test["eeg_id"].values

pred_df = pred_df.groupby("eeg_id", as_index=False)[CLASSES].mean()

sub = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

miss = sub[CLASSES].isna().any(axis=1)
if miss.any():
    sub.loc[miss, CLASSES] = global_prior

arr = sub[CLASSES].to_numpy(dtype=np.float64)
arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
arr = np.clip(arr, 1e-12, 1.0)
arr = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)
sub[CLASSES] = arr.astype(np.float32)

row_sums = sub[CLASSES].sum(axis=1).values
print("Submission shape:", sub.shape)
print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))
assert len(sub) == len(sample_sub), (len(sub), len(sample_sub))
assert list(sub.columns) == ["eeg_id"] + CLASSES

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(sub.head())
