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

0.4575461801753707

# 6. Current score

1.41423

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.68479) has done: 'I remove the TensorFlow import and related GPU/precision code that triggers a protobuf error, and replace the simple global‑probability baseline with a modest per‑patient averaging heuristic: for each patient we compute the normalized vote distribution from the training set and use it for test rows belonging to that patient, falling back to the overall global distribution otherwise. This fixes the runtime error and should improve the KL‑divergence score while keeping the core logic unchanged.'
- What this solution (achieved 1.05318) has done: 'I improve the heuristic by first trying to use the per‑`eeg_id` vote distribution (which is more specific than per‑patient) and fall back to the patient‑level or global distribution when necessary. I also add a tiny epsilon to avoid zero probabilities that can explode the KL score, then renormalise. The core workflow stays the same, only the probability‑lookup logic is enhanced.'
- What this solution (achieved 1.05318) has done: 'I add a simple weighting scheme that blends the EEG‑specific probability with the patient‑level probability based on how many votes the EEG record has. This reduces the impact of noisy EEG‑level estimates (which caused the high KL score) while still keeping the core logic. A tiny Laplace smoothing is retained to avoid zero probabilities, and the final probabilities are re‑normalised to guarantee they sum to one.'
- What this solution (achieved 1.05318) has done: 'I add a simple Dirichlet‑style smoothing that blends each EEG‑level vote distribution with the overall global distribution before the existing patient‑level blending. This keeps the same overall workflow but gives every class a non‑zero probability, which usually lowers the KL‑divergence and moves the score closer to the target.'
- What this solution (achieved 1.40933) has done: 'The changes strengthen the Bayesian smoothing for patient‑level probabilities, increase the global prior used for EEG‑level smoothing, and replace the linear vote‑weighting with a softer `votes/(votes+MAX_WEIGHT_VOTES)` formula. These adjustments keep the original workflow while making the predictions more regularised, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.39581) has done: 'I reduce the amount of global smoothing and trust the EEG‑specific vote distribution more, because the current heavy smoothing (large ALPHA, BETA and a high MAX_WEIGHT_VOTES) makes predictions too generic, leading to a high KL‑divergence. By lowering ALPHA and BETA and decreasing MAX_WEIGHT_VOTES the model leans more on available EEG‑level information while still keeping a small prior to avoid zeros, which should move the score toward the lower‑than‑target range.'
- What this solution (achieved 1.41423) has done: 'I increase the smoothing parameters so the predictions rely more on the global vote distribution, which reduces over‑fitting to sparse EEG‑ or patient‑specific counts and moves the KL‑divergence closer to the lower target value. The core workflow stays identical; only BETA, ALPHA, and MAX_WEIGHT_VOTES are adjusted.'
- What this solution (achieved 1.37636) has done: 'We reduce the smoothing strength so the model relies more on the EEG‑ and patient‑specific vote counts (which historically lowered the KL‑divergence). Specifically, we set `ALPHA` and `BETA` to a small value (0.1) and make `MAX_WEIGHT_VOTES` smaller (20) so that EEG‑level information gets a larger weight. These minimal constant changes keep the core workflow unchanged while moving the score toward the lower target.'
- What this solution (achieved 1.41423) has done: 'I keep the overall heuristic unchanged but strengthen the smoothing so predictions rely more on the global and patient‑level distributions, which reduces noisy EEG‑specific estimates and should lower the KL‑divergence. Specifically, I raise **ALPHA** and **BETA** from 0.1 to 1.0 and increase **MAX_WEIGHT_VOTES** from 20 to 100, making the EEG‑level weight smaller for sparse counts. These minimal constant tweaks keep the core blending logic intact while moving the score toward the lower target.'
- What this solution (achieved 1.34508) has done: 'I lower the smoothing strength so the predictions rely more on the specific EEG‑ and patient‑level vote distributions, which historically reduces the KL‑divergence. This is done by setting `ALPHA` and `BETA` to a small value (0.05) and decreasing `MAX_WEIGHT_VOTES` to 10. The rest of the workflow stays unchanged, ensuring a valid CSV is still produced.'
- What this solution (achieved 1.41423) has done: 'I increase the smoothing strength and limit the influence of sparse EEG‑specific vote counts so the predictions rely more on the stable patient‑ and global‑level distributions. This should lower the KL‑divergence and move the score toward the target. The core workflow and blending logic remain unchanged.'
- What this solution (achieved 1.34508) has done: 'I lower the smoothing strengths (ALPHA, BETA) and the cap on vote weighting (MAX_WEIGHT_VOTES) so the model relies more on the EEG‑ and patient‑specific vote distributions, which historically reduces the KL‑divergence. I also shrink the epsilon to a very small value to avoid zero probabilities without noticeably altering calibration. These minimal constant tweaks keep the original workflow intact while moving the score closer to the target.'
- What this solution (achieved 1.41423) has done: 'I increase the smoothing strengths and reduce the influence of noisy EEG‑specific counts by raising `ALPHA` and `BETA` and by using a larger `MAX_WEIGHT_VOTES`. This makes the blended probabilities rely more on the stable global and patient‑level distributions, which should lower the KL‑divergence and move the score toward the target while keeping the core workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"  # 'local' or 'kaggle'

if PLATFORM == "local":
    train_path = "./input/hms-harmful-brain-activity-classification/train.csv"
    test_path = "./input/hms-harmful-brain-activity-classification/test.csv"
else:
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"

df = pd.read_csv(train_path)
test = pd.read_csv(test_path)

TARGETS = df.columns[-6:]

global_counts = df[TARGETS].sum()
global_probs = (global_counts / global_counts.sum()).astype(float)

ALPHA = 1.0  # stronger EEG‑level prior
BETA = 1.0  # stronger patient‑level prior
MAX_WEIGHT_VOTES = 100  # lower weight for EEG‑specific counts
EPS = 1e-12  # tiny epsilon to avoid zeros

patient_counts = df.groupby("patient_id")[list(TARGETS)].sum()
patient_counts_smooth = patient_counts.add(BETA * global_counts, axis=1)
patient_counts_smooth_sum = patient_counts_smooth.sum(axis=1)
patient_probs = patient_counts_smooth.div(patient_counts_smooth_sum, axis=0)

eeg_counts = df.groupby("eeg_id")[list(TARGETS)].sum()
eeg_counts_sum = eeg_counts.sum(axis=1)  # total votes per EEG
eeg_smooth_counts = eeg_counts + ALPHA * global_counts
eeg_smooth_sum = eeg_counts_sum + ALPHA * global_counts.sum()
eeg_smooth_probs = eeg_smooth_counts.div(eeg_smooth_sum, axis=0)


def get_probs(row):
    """Blend EEG‑level, patient‑level, and global probabilities for a test row."""
    eid = row["eeg_id"]
    pid = row["patient_id"]

    patient_vec = (
        patient_probs.loc[pid].values
        if pid in patient_probs.index
        else global_probs.values
    )

    if eid in eeg_smooth_probs.index:
        eeg_vec = eeg_smooth_probs.loc[eid].values
        total_votes = eeg_counts_sum.loc[eid]

        weight = total_votes / (total_votes + MAX_WEIGHT_VOTES)
        probs = weight * eeg_vec + (1.0 - weight) * patient_vec
    else:
        probs = patient_vec

    probs = probs + EPS
    probs = probs / probs.sum()
    return probs


sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = test.apply(get_probs, axis=1).tolist()

row_sum = sub[TARGETS].sum(axis=1)
sub[TARGETS] = sub[TARGETS].div(row_sum, axis=0)

output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
