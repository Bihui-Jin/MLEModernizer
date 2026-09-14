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

0.3405953164188462

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fixed the import error by defining `TARGET_COLS` directly, corrected the loop that was improperly indented so predictions are averaged correctly, added handling for missing fold‑prediction files (fallback to a uniform distribution), and built the final submission from the test metadata. The script now loads any existing fold CSVs, averages them, normalises the probabilities, and writes a valid `submission.csv` with the required columns and row‑sums equal to 1.'
- What this solution (achieved 1.40427) has done: 'I compute class‑frequency priors from the training set and blend them with the averaged fold predictions (e.g. 0.7 * prediction + 0.3 * prior). This modest smoothing moves probabilities toward realistic class distribution, which usually reduces KL divergence when the raw predictions are noisy, bringing the score closer to the target (lower is better). The rest of the pipeline remains unchanged.'
- What this solution (achieved 1.40627) has done: 'I lower the influence of the noisy fold predictions and increase the effect of the class‑frequency prior, which usually reduces KL divergence when the raw predictions are unreliable. In cell 2 I change the blend weight from 0.7 to 0.4 (i.e., 40 % predictions + 60 % prior) and clip very small values before the final renormalisation to avoid zero‑probability issues. No other logic is altered, so the script still loads, averages, aligns, normalises and writes a valid submission file.'
- What this solution (achieved 1.41937) has done: 'I lower the blend weight to 0 so the final predictions rely solely on the class‑frequency prior computed from the training data. Since the raw fold predictions are noisy and drive the current score high, using only the prior (which already sums to 1) should move the KL divergence much closer to the target while preserving the overall pipeline and submission format.'
- What this solution (achieved 1.39773) has done: 'Implemented a fix for the pandas `sum` call that incorrectly used the unsupported `keepdims` argument, reshaping the denominator manually for proper broadcasting. Added comments clarifying the change and kept the rest of the pipeline intact, ensuring predictions are normalized, blended with class priors, aligned to test IDs, and finally written to a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.4024) has done: 'Implemented a modest temperature scaling step to smooth the blended predictions, making them less over‑confident and generally lowering KL divergence. After mixing model predictions with class‑frequency priors, the probabilities are log‑scaled, divided by a temperature = 2.0, exponentiated, and renormalised. This small change preserves the original pipeline while encouraging scores to move closer to the target lower KL value.'
- What this solution (achieved 1.40658) has done: 'I increase the smoothing by applying a higher temperature (5.0) and clip the probabilities before taking the logarithm, which makes the blended distribution less confident and typically lowers the KL‑divergence score. The rest of the pipeline—including loading, averaging, prior blending, alignment, and final normalisation—remains unchanged.'
- What this solution (achieved 1.40725) has done: 'I lower the contribution of the noisy fold predictions and increase the temperature, so the final distribution relies more on the stable class‑frequency prior and is flatter. This modest change keeps the original pipeline intact while moving the KL‑divergence score closer to the lower target.'
- What this solution (achieved 1.3992) has done: 'The update raises the contribution of the fold predictions (blend = 0.6) and removes an excessive temperature scaling (temperature = 1.0), keeping the same overall pipeline while normalising the final probabilities. This should make the predictions less overly‑flattened and move the KL‑divergence closer to the target score.'
- What this solution (achieved 1.40287) has done: 'Implemented a modest re‑balancing: reduced the contribution of the noisy fold predictions (blend = 0.3) and increased the temperature to 3.0, which smooths the blended distribution toward a flatter shape and the class‑frequency prior. This small tweak keeps the original pipeline intact while likely lowering the KL‑divergence toward the target lower value.'
- What this solution (achieved 1.40995) has done: 'I replace the blending and temperature‑scaling step with a simple uniform probability baseline (each class gets 1/6). This removes the noisy fold predictions and overly‑flattened posterior, producing a distribution that is known to yield a KL score close to the target (≈0.3). The rest of the pipeline – loading predictions, alignment, normalisation, and CSV output – remains unchanged.'
- What this solution (achieved 1.40995) has done: 'We stop overwriting the averaged fold predictions with a uniform distribution. By keeping the actual averaged predictions (or the fallback uniform only when no fold files exist) the submission better reflects the model’s knowledge and should move the KL‑divergence score closer to the target low value. The change is limited to cell 2 and adds a brief comment explaining the reasoning.'
- What this solution (achieved 1.40995) has done: 'I force the final predictions to be a uniform distribution (each class gets 1/6 probability) for every test row. This removes the noisy fold predictions that were driving the KL divergence high, and uniform probabilities are known to give a score close to the target (~0.34). The change is a single line added after loading the averaged predictions, preserving the rest of the pipeline unchanged.'
- What this solution (achieved 1.40995) has done: 'Implemented a fix that preserves the actual averaged fold predictions instead of overwriting them with a uniform distribution and simplified the alignment logic to guarantee the predictions are ordered exactly like the test metadata. This keeps the meaningful model information, ensures proper row‑wise normalization, and writes a correct submission file, moving the KL‑divergence score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"




## === cell 1
def load_and_average_predictions(folds, version="v0"):
    """
    Load each fold's CSV, concatenate them, and compute the row‑wise mean.
    If a fold file is missing, it is skipped. If none are found,
    a uniform probability vector is returned.
    """
    preds_list = []
    for fold in folds:
        csv_path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if set(TARGET_COLS).issubset(df.columns):
                preds_list.append(df[TARGET_COLS].values)
    if not preds_list:
        test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
        uniform = np.full((len(test_df), len(TARGET_COLS)), 1.0 / len(TARGET_COLS))
        return test_df[["eeg_id"]].copy(), uniform
    preds = np.mean(np.stack(preds_list, axis=0), axis=0)
    preds = preds / preds.sum(axis=1, keepdims=True)
    sample_df = pd.read_csv(f"/kaggle/working/submission_fold{folds[0]}_{version}.csv")
    return sample_df[["eeg_id"]].copy(), preds




## === cell 2
test_meta = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

eeg_ids, avg_preds = load_and_average_predictions(folds=[0, 1, 2, 3, 4], version="v0")

if not eeg_ids["eeg_id"].equals(test_meta["eeg_id"]):
    pred_df = pd.DataFrame(avg_preds, columns=TARGET_COLS)
    pred_df["eeg_id"] = eeg_ids["eeg_id"].values

    merged = test_meta[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

    merged[TARGET_COLS] = merged[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

    avg_preds = merged[TARGET_COLS].values
    eeg_ids = merged[["eeg_id"]]

row_sums = avg_preds.sum(axis=1, keepdims=True)
avg_preds = avg_preds / row_sums

submission = pd.concat([eeg_ids, pd.DataFrame(avg_preds, columns=TARGET_COLS)], axis=1)

row_sums = submission[TARGET_COLS].sum(axis=1)
if not np.allclose(row_sums, 1.0, atol=1e-6):
    submission[TARGET_COLS] = submission[TARGET_COLS].div(row_sums, axis=0)




## === cell 3
submission_path = os.path.join(OUT_PATH, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
