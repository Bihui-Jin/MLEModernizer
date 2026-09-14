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

0.2843587538511006

# 6. Current score

1.12578

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the TensorFlow‑dependent sections with a simple, pure‑Python baseline that predicts the overall class distribution observed in the training data. This avoids the import error, ensures the script runs to completion, and produces a valid `submission.csv` whose probabilities sum to 1 for each row, moving the score toward the target without needing heavy modelling.'
- What this solution (achieved 1.41937) has done: 'I compute per‑eeg_id vote distributions from the training data and use them for any matching test rows, falling back to the overall class probabilities when an eeg_id is unseen. This small personalization should reduce the KL‑divergence (moving the score closer to the target) while keeping the original baseline logic intact and still writing a valid‐format CSV.'
- What this solution (achieved 1.41937) has done: 'I add a tiny Laplace smoothing constant when computing both the global class probabilities and the per‑eeg id probabilities. This prevents any class having a zero probability for a given eeg_id, which otherwise yields very large KL‑divergence penalties. The change is confined to cell 1, keeps the overall baseline logic unchanged, and retains the final CSV generation unchanged, so the script still runs end‑to‑end and produces a valid submission.'
- What this solution (achieved 0.87464) has done: 'I fix the file‑path errors by dynamically locating the required CSV files instead of using a hard‑coded relative directory. This ensures the script can find train.csv and test.csv in the Kaggle environment, allowing the remainder of the code (global, per‑eeg and per‑patient probability calculations) to run unchanged and produce a valid submission.csv with rows that sum to 1. No core‑logic changes are introduced, preserving the original modeling approach while enabling execution and a proper submission file.'
- What this solution (achieved 0.809) has done: 'I increase the Laplace smoothing constant to 0.1 so the per‑eeg and per‑patient probabilities are less extreme, and I blend the per‑eeg and per‑patient predictions (70 % per‑eeg, 30 % per‑patient) instead of falling back sequentially. This modest regularisation should reduce over‑confident errors and lower the KL‑divergence, moving the score closer to the target while keeping the original baseline logic unchanged.'
- What this solution (achieved 0.90753) has done: 'I increase Laplace smoothing (α = 0.3) to make per‑eeg and per‑patient distributions less extreme and replace the sequential fallback with a weighted blend of per‑eeg, per‑patient and the global class probabilities (weights 0.4, 0.3, 0.3). This modest regularisation should lower the KL‑divergence, moving the score closer to the target while preserving the original baseline logic and still producing a valid `submission.csv`.'
- What this solution (achieved 1.16366) has done: 'The changes lower the Laplace smoothing to keep class distributions sharper, compute how many votes each eeg_id and patient_id contribute, and use these counts to give a data‑driven blend between the per‑eeg, per‑patient and global probabilities (instead of the fixed 0.4/0.3/0.3 weights). This modest regularisation is expected to reduce the KL‑divergence and move the score closer to the target while preserving the original baseline logic and CSV output.'
- What this solution (achieved 1.165) has done: 'I slightly increase the Laplace smoothing constant (to make per‑eeg and per‑patient distributions less extreme) and raise the base global weight so the overall blend leans more on the safe global class probabilities. These modest tweaks keep the original blending logic intact while encouraging less‑confident, more calibrated predictions, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.36333) has done: 'I increase the Laplace smoothing constant to make class distributions less extreme and raise the global‑weight while shrinking the per‑eeg and per‑patient contributions. This keeps the original blending logic but makes the predictions closer to the safe global baseline, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.9888) has done: 'I lower the Laplace smoothing (ALPHA = 0.1) so per‑eeg and per‑patient distributions stay sharper, and I increase their influence in the final blend while reducing the global fallback weight. The scaling constants and max‑weights are adjusted (EEG_SCALE = 10, PATIENT_SCALE = 10, MAX_EEG_WEIGHT = 0.4, MAX_PAT_WEIGHT = 0.4, BASE_GLOBAL_WEIGHT = 0.2) so that rows with enough historical votes rely more on their own eeg_id or patient_id probabilities, which should cut the KL‑divergence and move the score closer to the target. All other logic and the CSV output remain unchanged.'
- What this solution (achieved 1.12578) has done: 'I increase Laplace smoothing and shift the blending weights toward the safe global baseline, which should dampen noisy per‑eeg / per‑patient predictions and lower the KL‑divergence (moving the score closer to the target). The core logic and overall workflow remain unchanged, only the constants that control smoothing and blending are adjusted.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np

train_candidates = glob.glob(os.path.join("**", "train.csv"), recursive=True)
if not train_candidates:
    raise FileNotFoundError("train.csv not found in any subdirectory.")
train_path = train_candidates[0]

df_train = pd.read_csv(train_path)

TARGETS = df_train.columns[-6:]
print("Target columns:", list(TARGETS))

ALPHA = 1.0

vote_sums = df_train[TARGETS].sum().astype(float) + ALPHA
total_votes = vote_sums.sum()
global_probs = vote_sums / total_votes
print("Global class probabilities (fallback) with smoothing:")
print(global_probs)

eeg_group = df_train.groupby("eeg_id")[list(TARGETS)].sum()
eeg_counts = eeg_group.sum(axis=1)  # total votes per eeg_id (no smoothing)
eeg_group_smoothed = eeg_group + ALPHA
eeg_totals = eeg_group_smoothed.sum(axis=1).replace(0, np.nan)
per_eeg_probs = eeg_group_smoothed.div(eeg_totals, axis=0).fillna(global_probs)

patient_group = df_train.groupby("patient_id")[list(TARGETS)].sum()
patient_counts = patient_group.sum(axis=1)  # total votes per patient_id (no smoothing)
patient_group_smoothed = patient_group + ALPHA
patient_totals = patient_group_smoothed.sum(axis=1).replace(0, np.nan)
per_patient_probs = patient_group_smoothed.div(patient_totals, axis=0).fillna(
    global_probs
)

eeg_counts_dict = eeg_counts.to_dict()
patient_counts_dict = patient_counts.to_dict()




## === cell 1
test_candidates = glob.glob(os.path.join("**", "test.csv"), recursive=True)
if not test_candidates:
    raise FileNotFoundError("test.csv not found in any subdirectory.")
test_path = test_candidates[0]

df_test = pd.read_csv(test_path)

submission = pd.DataFrame()
submission["eeg_id"] = df_test["eeg_id"]

BASE_GLOBAL_WEIGHT = 0.55  # higher safe fallback weight
EEG_SCALE = 20.0  # larger denominator ⇒ smaller per‑eeg weight
PATIENT_SCALE = 20.0  # larger denominator ⇒ smaller per‑patient weight
MAX_EEG_WEIGHT = 0.25  # reduced max contribution from per‑eeg
MAX_PAT_WEIGHT = 0.25  # reduced max contribution from per‑patient

for col in TARGETS:
    eeg_vals = df_test["eeg_id"].map(per_eeg_probs[col])
    patient_vals = df_test["patient_id"].map(per_patient_probs[col])

    final_vals = np.full(len(df_test), np.nan)

    for idx, row in df_test.iterrows():
        eid = row["eeg_id"]
        pid = row["patient_id"]

        cnt_eeg = eeg_counts_dict.get(eid, 0.0)
        cnt_pat = patient_counts_dict.get(pid, 0.0)

        w_eeg = min((cnt_eeg / (cnt_eeg + EEG_SCALE)) * MAX_EEG_WEIGHT, MAX_EEG_WEIGHT)
        w_pat = min(
            (cnt_pat / (cnt_pat + PATIENT_SCALE)) * MAX_PAT_WEIGHT, MAX_PAT_WEIGHT
        )

        total = w_eeg + w_pat + BASE_GLOBAL_WEIGHT
        if total > 1.0:
            scale = 1.0 / total
            w_eeg *= scale
            w_pat *= scale
            w_glob = 0.0
        else:
            w_glob = 1.0 - (w_eeg + w_pat)

        prob_eeg = eeg_vals.iloc[idx] if pd.notna(eeg_vals.iloc[idx]) else 0.0
        prob_pat = patient_vals.iloc[idx] if pd.notna(patient_vals.iloc[idx]) else 0.0
        prob_glob = global_probs[col]

        final_vals[idx] = w_glob * prob_glob + w_eeg * prob_eeg + w_pat * prob_pat

    submission[col] = final_vals

row_sums = submission[TARGETS].sum(axis=1)
tolerance = 1e-6
if not np.allclose(row_sums, 1.0, atol=tolerance):
    correction = 1.0 / row_sums
    submission[TARGETS] = submission[TARGETS].multiply(correction, axis=0)

assert np.allclose(
    submission[TARGETS].sum(axis=1), 1.0, atol=1e-6
), "Probabilities do not sum to 1"

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submission.shape}")
