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

0.2930461997319395

# 6. Current score

0.78589

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix removes the problematic TensorFlow import and replaces the heavy training/inference pipeline with a lightweight baseline that computes class‑wise average vote proportions from the training set and applies these proportions to every test sample, ensuring a valid `submission.csv` is created without runtime errors.'
- What this solution (achieved 1.68479) has done: 'I keep the original data loading and global class‑probability baseline, but add a lightweight per‑patient adjustment: compute each patient’s vote distribution from the training set and use it for any test rows belonging to that patient, falling back to the overall distribution when the patient is unseen. This small personalization respects the original logic while giving predictions that better reflect the training data, so the KL divergence should move closer to the target (lower ≈ better). The script still writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.81597) has done: 'Implemented a fix for the DataFrame construction error and added a lightweight smoothing step to the probability calculation.  
* Replaced the invalid `tolist()` call by using the DataFrame returned directly from `apply`.  
* Updated `get_probs` to blend specific (eeg/patient) probabilities with the overall class distribution, improving calibration while keeping the original logic intact.'
- What this solution (achieved 0.85517) has done: 'I keep the overall data‑loading and submission logic unchanged but improve the probability blending. The new `get_probs` function now:
1. Adds a tiny epsilon to avoid zero probabilities and renormalises each distribution.
2. When an exact `eeg_id` match exists, it blends EEG‑specific, patient‑specific (if available), and global class probabilities (0.5 + 0.3 + 0.2).  
3. When only a patient match exists it blends patient‑specific with the global distribution (0.6 + 0.4).  
4. Otherwise it falls back to the smoothed global distribution.  
These modest adjustments should calibrate predictions better and move the KL divergence closer to the target while preserving the original workflow.'
- What this solution (achieved 0.78004) has done: 'I adjust the blending weights in `get_probs` so that predictions rely more heavily on the most specific information available (EEG‑level, then patient‑level) and less on the global baseline. By increasing the influence of EEG‑specific and patient‑specific distributions, the model should better reflect the true class proportions observed in the training data, which is expected to lower the KL‑divergence and move the score nearer the target (while keeping the overall workflow unchanged).'
- What this solution (achieved 0.76863) has done: 'The update replaces the fixed blending weights with a data‑driven weighting that scales each distribution (global, patient‑level, EEG‑level) by the amount of training data supporting it. By giving more influence to larger patient or EEG groups, the predictions become better calibrated, which should lower the KL‑divergence and move the score nearer the target while keeping the overall workflow unchanged.'
- What this solution (achieved 1.05318) has done: 'I replace the blending logic with a hierarchy‑first approach: use the exact EEG‑level distribution when it exists, otherwise fall back to the patient‑level distribution, and only use the global prior as a last resort. This keeps the same data sources and overall workflow but gives more specific (and therefore better calibrated) probabilities, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.20657) has done: 'I replace the pure‑fallback hierarchy with a lightweight blending that mixes EEG‑specific, patient‑specific and global class probabilities, weighting each component by how much training data supports it. This adds a small amount of personalization while keeping the original workflow, and the renormalisation ensures rows still sum to 1, which should lower the KL‑divergence toward the target.'
- What this solution (achieved 0.77964) has done: 'I lower the dominant global weight from 1000 to 10 so that patient‑ and EEG‑specific distributions have a much larger influence on each prediction. This small tweak keeps the original blending logic intact while making the model more personalized, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.77606) has done: 'I replace the linear count‑based weights with a gentler log‑based weighting so that EEG‑ and patient‑specific distributions influence each prediction but cannot dominate the global prior. This smoother blending should produce better‑calibrated probabilities and lower the KL‑divergence, moving the score toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.77122) has done: 'We keep the overall workflow unchanged but replace the log‑scaled blending weights with raw count‑based weights (plus 1) so that patient‑ and EEG‑specific distributions have a stronger influence, which should better capture the true class ratios and reduce the KL‑divergence toward the target score. Minor comments are added to explain the adjustment, and the cell index is reset to start at 1.'
- What this solution (achieved 1.03892) has done: 'The change reduces the influence of patient‑specific and EEG‑specific distributions by keeping a strong global prior (weight 10) and using much smaller log‑scaled weights for the personalized components. This smoother blending keeps the original workflow but yields predictions closer to the overall class distribution, which empirically lowers the KL‑divergence and moves the score toward the target.'
- What this solution (achieved 0.78589) has done: 'I lower the global prior weight and replace the log‑scaled patient/eeg weights with simple count‑based weights (plus 1) so the personalized distributions dominate each prediction. This should reduce the KL divergence and move the score closer to the target while keeping the overall workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

LOAD_DATA_FROM = (
    "/kaggle/input/hms-harmful-brain-activity-classification"
    if os.path.isdir("/kaggle/input")
    else "./data/hms-harmful-brain-activity-classification"
)

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
train_df = pd.read_csv(train_path)

TARGETS = train_df.columns[-6:]

class_totals = train_df[TARGETS].sum().astype(float)
class_probs = class_totals / class_totals.sum()

patient_sums = train_df.groupby("patient_id")[TARGETS].sum()
patient_probs = patient_sums.div(patient_sums.sum(axis=1), axis=0)
patient_counts = train_df.groupby("patient_id").size()

eeg_sums = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_probs = eeg_sums.div(eeg_sums.sum(axis=1), axis=0)
eeg_counts = train_df.groupby("eeg_id").size()

test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test_df = pd.read_csv(test_path)


def get_probs(row):
    """
    Blend a much weaker global prior with count‑based patient‑ and EEG‑specific
    distributions.  Using raw counts (+1) gives the personalized information
    greater influence, which is expected to lower the KL divergence (lower is better).
    """
    eid = row["eeg_id"]
    pid = row["patient_id"]
    eps = 1e-6

    global_weight = 0.5
    weighted_sum = class_probs.values * global_weight
    weight_total = global_weight

    if pid in patient_probs.index:
        p_cnt = patient_counts.get(pid, 0)
        p_weight = float(p_cnt) + 1.0  # raw count + 1 to avoid zero
        weighted_sum += patient_probs.loc[pid].values * p_weight
        weight_total += p_weight

    if eid in eeg_probs.index:
        e_cnt = eeg_counts.get(eid, 0)
        e_weight = float(e_cnt) + 1.0  # raw count + 1
        weighted_sum += eeg_probs.loc[eid].values * e_weight
        weight_total += e_weight

    prob = weighted_sum / weight_total
    prob = prob + eps  # avoid exact zeros
    prob = prob / prob.sum()  # enforce sum‑to‑1
    return pd.Series(prob, index=TARGETS)


prob_df = test_df.apply(get_probs, axis=1)

submission = pd.concat([test_df[["eeg_id"]], prob_df], axis=1)

row_sums = submission[TARGETS].sum(axis=1)
submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("Submission preview:")
print(submission.head())
