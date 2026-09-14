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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

1.0598384737455746

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I removed the non‑code text, avoided loading missing pretrained weights, and replaced the heavy model training with a simple baseline that uses the overall class vote distribution from the training set to generate probabilities for every test row. This guarantees the script runs end‑to‑end, creates a valid `submission.csv` with the correct column order, and produces probabilities that sum to 1, moving the KL‑divergence score toward the target without altering the core competition logic.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑distribution baseline with a simple per‑eeg_id probability lookup: for each eeg_id that appears in the training set I compute its class vote distribution and use it for the matching test rows, falling back to the overall class distribution when the eeg_id is unseen. This keeps the core logic unchanged while adding a modest amount of information, which should lower the KL‑divergence (the target metric) and move the score closer to the desired value.'
- What this solution (achieved 1.41937) has done: 'I add a simple smoothing step that blends each eeg_id‑specific probability distribution with the overall class distribution, using the total number of votes for that eeg_id as a weight. This keeps the core “per eeg_id lookup” logic while making predictions less noisy for rarely‑seen ids, which should lower the KL‑divergence and move the score closer to the target. The rest of the pipeline (merge, fallback to overall, row‑sum check, and CSV output) remains unchanged.'
- What this solution (achieved 1.41937) has done: 'I keep the basic per‑eeg probability logic but add a lightweight per‑patient fallback and reduce the smoothing factor for the per‑eeg blend (k = 5). This gives more weight to the specific eeg_id distribution while still backing off to a patient‑level distribution and finally to the overall class distribution, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

base_path = "/kaggle/input/hms-harmful-brain-activity-classification"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(base_path, "test.csv")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def compute_distributions(df):
    total_votes = df[vote_cols].sum()
    overall_probs = (total_votes / total_votes.sum()).values.astype(float)
    overall_series = pd.Series(overall_probs, index=vote_cols)

    agg_votes_eeg = df.groupby("eeg_id")[vote_cols].sum()
    agg_sums_eeg = agg_votes_eeg.sum(axis=1)
    per_eeg_raw = agg_votes_eeg.div(agg_sums_eeg, axis=0)

    k_eeg = 0.5  # reduced smoothing → trust eeg‑specific votes more
    alpha_eeg = agg_sums_eeg / (agg_sums_eeg + k_eeg)
    alpha_eeg = alpha_eeg.reindex(per_eeg_raw.index)

    per_eeg_blended = per_eeg_raw.multiply(alpha_eeg, axis=0) + overall_series.multiply(
        1 - alpha_eeg, axis=0
    )
    per_eeg_blended = per_eeg_blended.div(per_eeg_blended.sum(axis=1), axis=0)
    per_eeg_probs = per_eeg_blended.reset_index()  # eeg_id + vote_cols

    agg_votes_pat = df.groupby("patient_id")[vote_cols].sum()
    agg_sums_pat = agg_votes_pat.sum(axis=1)
    per_pat_raw = agg_votes_pat.div(agg_sums_pat, axis=0)

    k_pat = 1.0  # reduced smoothing for patient level
    alpha_pat = agg_sums_pat / (agg_sums_pat + k_pat)
    alpha_pat = alpha_pat.reindex(per_pat_raw.index)

    per_pat_blended = per_pat_raw.multiply(alpha_pat, axis=0) + overall_series.multiply(
        1 - alpha_pat, axis=0
    )
    per_pat_blended = per_pat_blended.div(per_pat_blended.sum(axis=1), axis=0)
    per_patient_probs = per_pat_blended.reset_index()  # patient_id + vote_cols

    return overall_series, per_eeg_probs, per_patient_probs


rng = np.random.default_rng(42)
shuffle_idx = rng.permutation(len(train_df))
split_pt = int(0.8 * len(train_df))
train_idx = shuffle_idx[:split_pt]
val_idx = shuffle_idx[split_pt:]

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

overall_series, per_eeg_probs, per_patient_probs = compute_distributions(train_split)




## === cell 1
def kl_divergence(true_counts, pred_probs):
    true_sum = true_counts.sum()
    if true_sum == 0:
        return 0.0
    p_true = true_counts / true_sum
    eps = 1e-12
    p_pred = np.clip(pred_probs, eps, 1.0)
    return np.sum(p_true * np.log(p_true / p_pred))


def evaluate_weights(w_eeg, w_pat, w_over):
    val = val_split[["eeg_id", "patient_id"] + vote_cols].copy()
    val = val.merge(per_eeg_probs, on="eeg_id", how="left", suffixes=("", "_eeg"))
    val = val.merge(
        per_patient_probs, on="patient_id", how="left", suffixes=("", "_pat")
    )

    for col in vote_cols:
        val[col].fillna(overall_series[col], inplace=True)
        pat_col = f"{col}_pat"
        val[pat_col].fillna(overall_series[col], inplace=True)

    eeg_arr = val[vote_cols].values
    pat_arr = val[[f"{c}_pat" for c in vote_cols]].values
    overall_arr = overall_series.values

    combined = w_eeg * eeg_arr + w_pat * pat_arr + w_over * overall_arr
    combined = combined / combined.sum(axis=1, keepdims=True)

    kl_vals = []
    for i, row in enumerate(val_split[vote_cols].values):
        kl_vals.append(kl_divergence(row, combined[i]))
    return np.mean(kl_vals)


best_score = np.inf
best_weights = (0.7, 0.2, 0.1)  # fallback

for w_eeg in np.arange(0.6, 0.96, 0.05):
    for w_pat in np.arange(0.0, 0.31, 0.05):
        w_over = 1.0 - w_eeg - w_pat
        if w_over < 0:
            continue
        score = evaluate_weights(w_eeg, w_pat, w_over)
        if score < best_score:
            best_score = score
            best_weights = (w_eeg, w_pat, w_over)

best_w_eeg, best_w_pat, best_w_over = best_weights
print(
    f"Chosen weights → eeg: {best_w_eeg:.2f}, patient: {best_w_pat:.2f}, overall: {best_w_over:.2f}"
)
print(f"Validation KL ≈ {best_score:.6f}")

overall_series, per_eeg_probs, per_patient_probs = compute_distributions(train_df)




## === cell 2
submission = test_df[["eeg_id", "patient_id"]].copy()

submission = submission.merge(per_eeg_probs, on="eeg_id", how="left")
submission = submission.merge(
    per_patient_probs, on="patient_id", how="left", suffixes=("", "_pat")
)

for col in vote_cols:
    pat_col = f"{col}_pat"
    submission[pat_col].fillna(overall_series[col], inplace=True)

eeg_array = np.nan_to_num(submission[vote_cols].values, nan=0.0)
pat_array = np.nan_to_num(submission[[f"{c}_pat" for c in vote_cols]].values, nan=0.0)
overall_array = overall_series.values  # shape (6,)

combined = best_w_eeg * eeg_array + best_w_pat * pat_array + best_w_over * overall_array
submission[vote_cols] = combined

row_sums = submission[vote_cols].sum(axis=1)
submission[vote_cols] = submission[vote_cols].div(row_sums, axis=0)

assert np.allclose(
    submission[vote_cols].sum(axis=1), 1.0, atol=1e-6
), "Row probabilities do not sum to 1."



## === cell 3
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with shape {submission.shape}")
