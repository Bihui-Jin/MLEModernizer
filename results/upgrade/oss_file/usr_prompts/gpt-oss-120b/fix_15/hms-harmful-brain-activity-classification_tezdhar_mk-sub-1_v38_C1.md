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

0.3090887487737234

# 6. Current score

0.76863

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix introduces the missing column list, replaces the broken import, and rewrites the prediction‑merging logic to work without the unavailable `src` package. It directly builds a submission by using the overall class vote distribution from the training set, guaranteeing a valid CSV whose probabilities sum to 1.'
- What this solution (achieved 1.57587) has done: 'I replace the single global vote distribution with a patient‑specific distribution: for each test row we look up the average vote distribution of that patient in the training set (falling back to the overall distribution when the patient does not appear). This keeps the same overall workflow but gives more tailored probabilities, which should lower the KL‑divergence and bring the score closer to the target.'
- What this solution (achieved 1.57587) has done: 'I add an EEG‑specific vote distribution: if the test `eeg_id` appears in the training set we use its exact averaged vote vector, otherwise we fall back to the patient‑specific distribution (and finally the overall distribution). This gives a more granular prior and should lower the KL‑divergence, moving the score toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.77964) has done: 'I keep the same hierarchical approach but add a simple count‑based smoothing: when an `eeg_id` or `patient_id` appears only a few times in the training set we blend its specific vote distribution with the overall distribution instead of using it alone. This reduces over‑confident predictions on rare IDs, which typically lowers the KL‑divergence and moves the score closer to the target while preserving the original logic.'
- What this solution (achieved 0.83827) has done: 'I increase the smoothing factor and only apply specific eeg or patient distributions when they have enough supporting rows (≥5). This reduces over‑confident predictions from rare IDs, keeping the hierarchical logic unchanged while moving the KL‑divergence score closer to the target.'
- What this solution (achieved 0.93688) has done: 'Implemented a more aggressive smoothing strategy to bring the KL‑divergence closer to the target.  
* Increased the smoothing constant `k` from 30 to 100 so rare IDs are blended heavily with the overall distribution.  
* Removed the hard `MIN_COUNT` threshold; we now always blend any available `eeg_id` or `patient_id` distribution using the count‑based weight, allowing even minimal information to modestly influence predictions while staying well‑regularized.  
These minimal adjustments keep the original hierarchical logic intact, ensure rows still sum to 1, and are expected to lower the score toward the target.'
- What this solution (achieved 0.83827) has done: 'Implemented a milder smoothing strategy and re‑introduced a minimum‑count guard to avoid over‑confident predictions from very rare IDs.  
- Reduced the smoothing constant `k` from 100 to 30, giving more weight to genuine ID‑specific patterns.  
- Added a `MIN_COUNT = 5` threshold: specific `eeg_id` or `patient_id` distributions are used only when they appear at least five times in the training data; otherwise the overall distribution is used directly.  
These modest adjustments keep the original hierarchical blending logic while expectedly lowering the KL‑divergence toward the target score.'
- What this solution (achieved 1.01461) has done: 'I increase the smoothing constant and allow blending for any available ID count (remove the minimum‑count guard). Using a larger `SMOOTHING_K` means the specific `eeg_id` or `patient_id` distributions have much less weight, keeping the prediction close to the overall distribution while still benefiting from any weak signal. This reduces over‑confident predictions on rare IDs and is expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.83827) has done: 'I lower the smoothing constant and require a minimum count of 5 for using a specific eeg_id or patient_id distribution. This gives more weight to reliable ID‑specific vote vectors while still falling back to the overall distribution for rare IDs, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.83335) has done: 'I lower the minimum‑count guard to allow more patient/eeg specific information and blend both levels sequentially (eeg first, then patient) instead of choosing only one. This adds useful signal while still applying count‑based smoothing, which should pull the KL‑divergence down toward the target.'
- What this solution (achieved 1.01563) has done: 'I increase the smoothing constant and require a higher minimum count so that specific eeg_id or patient_id distributions are blended much less often, keeping predictions closer to the overall (well‑regularized) distribution. This modest change preserves the original hierarchical logic while steering the KL‑divergence lower, moving the score toward the target.'
- What this solution (achieved 0.83827) has done: 'I lower the smoothing constant to give more weight to the specific eeg_id and patient_id distributions (which historically reduced the KL‑divergence) and modify the blending function to renormalise the resulting vector, ensuring each row sums exactly to 1. This keeps the hierarchical logic unchanged while steering the score toward the target.'
- What this solution (achieved 1.41937) has done: 'I raise the `MIN_COUNT` threshold so that the per‑`eeg_id` and per‑`patient_id` specific distributions are never applied (they would require an unrealistically large count). This forces the submission to use only the overall vote distribution for every test row, which is expected to yield a lower KL‑divergence and move the score closer to the target while keeping the original logic unchanged.'
- What this solution (achieved 0.76863) has done: 'I lower the thresholds so that the model actually uses the EEG‑ and patient‑specific vote distributions instead of only the overall average. By setting `MIN_COUNT` to 0 and reducing `SMOOTHING_K` to 1 the blending weight becomes close to 1 for any available specific distribution, letting the predictions reflect the observed training frequencies while still keeping a tiny amount of smoothing to avoid zeroes. This minimal change preserves the existing hierarchical logic and should move the KL‑divergence score down toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

SMOOTHING_K = 1
MIN_COUNT = 0  # no minimum count guard; always consider specific info when present


def _blend_probs(
    specific: pd.Series, overall: pd.Series, count: int, k: int = SMOOTHING_K
) -> pd.Series:
    """
    Blend a specific probability vector with the overall distribution.
    The blending weight α = count / (count + k) gives more weight to better‑represented IDs.
    After blending we renormalise so the probabilities sum to 1.
    """
    alpha = count / (count + k) if count > 0 else 0.0
    blended = alpha * specific + (1 - alpha) * overall
    blended_sum = blended.sum()
    if blended_sum == 0:
        return overall.copy()
    return blended / blended_sum


def build_submission(data_path: str, out_path: str) -> pd.DataFrame:
    """
    Create a submission using hierarchical vote distributions with count‑based smoothing:
    1. Start from the overall distribution.
    2. If the test `eeg_id` appears in training, blend its averaged vote
       distribution with the current probabilities.
    3. If the `patient_id` appears in training, blend the patient‑level
       distribution with the current probabilities.
    This sequential blending incorporates both levels of information while keeping
    predictions well‑regularized.
    """
    train_path = os.path.join(data_path, "train.csv")
    train_df = pd.read_csv(train_path)

    eps = 1e-12

    overall_sums = train_df[TARGET_COLS].sum()
    overall_probs = (overall_sums + eps) / (overall_sums.sum() + eps * len(TARGET_COLS))

    patient_sums = train_df.groupby("patient_id")[TARGET_COLS].sum()
    patient_counts = train_df.groupby("patient_id").size()
    patient_probs = (patient_sums + eps).div(
        patient_sums.sum(axis=1) + eps * len(TARGET_COLS), axis=0
    )

    eeg_sums = train_df.groupby("eeg_id")[TARGET_COLS].sum()
    eeg_counts = train_df.groupby("eeg_id").size()
    eeg_probs = (eeg_sums + eps).div(
        eeg_sums.sum(axis=1) + eps * len(TARGET_COLS), axis=0
    )

    test_path = os.path.join(data_path, "test.csv")
    test_df = pd.read_csv(test_path)

    submission = pd.DataFrame()
    submission["eeg_id"] = test_df["eeg_id"]

    for idx, row in test_df.iterrows():
        eid = row["eeg_id"]
        pid = row["patient_id"]

        probs = overall_probs.copy()

        if eid in eeg_probs.index:
            probs = _blend_probs(eeg_probs.loc[eid], probs, eeg_counts.loc[eid])

        if pid in patient_probs.index:
            probs = _blend_probs(patient_probs.loc[pid], probs, patient_counts.loc[pid])

        for col in TARGET_COLS:
            submission.at[idx, col] = probs[col]

    row_sums = submission[TARGET_COLS].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Row probabilities do not sum to 1."

    out_file = os.path.join(out_path, "submission.csv")
    submission.to_csv(out_file, index=False)
    return submission




## === cell 1
submission_df = build_submission(DATA_PATH, OUT_PATH)
print("Submission saved to:", os.path.join(OUT_PATH, "submission.csv"))
print(submission_df.head())
