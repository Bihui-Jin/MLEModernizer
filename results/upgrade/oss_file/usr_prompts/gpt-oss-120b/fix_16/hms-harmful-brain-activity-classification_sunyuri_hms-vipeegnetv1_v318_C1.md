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

0.2940860627808818

# 6. Current score

1.27225

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I set NEEDTRAIN to False to skip all model‑training code (which triggers the protobuf error) and replace the inference block with a simple baseline that uses the average class distribution from the training data for every test row. This removes the TensorFlow import when not needed, avoids the “MessageFactory…GetPrototype” error, and guarantees a valid `submission.csv` whose rows sum to 1, moving the score toward the target without altering core model logic.'
- What this solution (achieved 1.68479) has done: 'The fix corrects the faulty import statement that caused `np` to be undefined, allowing the baseline probability calculations and the submission generation to run without errors. No changes are made to the modeling logic, preserving the original approach while ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.85517) has done: 'We smooth the patient‑level vote proportions and blend them with the overall class baseline so the predictions are less extreme and better calibrated for KL‑divergence. A tiny Laplace‑style epsilon avoids zero probabilities, and a simple weighted average with the global baseline (α ≈ 0.6) keeps the original patient logic while moving the score toward the target. The rest of the pipeline stays unchanged, and the script still writes a valid `submission.csv` whose rows sum to 1.'
- What this solution (achieved 0.78698) has done: 'I increase the weight given to the patient‑specific vote distribution (alpha) so the predictions rely more on the per‑patient information, which is usually better calibrated for KL‑divergence. The change is minimal: only the `alpha` value is raised from 0.6 to 0.95, preserving all other logic, smoothing, fallback and final normalization, thereby keeping the core pipeline intact while moving the score closer to the target.'
- What this solution (achieved 0.90499) has done: 'I lower the patient‑specific blending weight from 0.95 to 0.5 so the global class baseline contributes more, which typically yields better calibration for KL‑divergence, and I clip the blended probabilities to a tiny epsilon before the final row‑wise normalization to avoid zeroes. These small adjustments keep the original pipeline intact while moving the score closer to the target.'
- What this solution (achieved 0.76992) has done: 'The update makes the blending weight adaptive: patients with many training votes rely more on their own distribution, while those with few votes fall back toward the global baseline. This typically yields better‑calibrated probabilities for KL‑divergence and moves the score closer to the target. The rest of the pipeline remains unchanged, and a valid `submission.csv` is still written.'
- What this solution (achieved 0.80228) has done: 'I correct the data path so the script locates the CSV files in the actual input folder and fix the typo where `patient_total_weights` was used instead of the defined `patient_total_votes`. These changes ensure the script runs without file‑not‑found or name errors and still produces a valid `submission.csv` whose rows sum to 1, moving the KL‑divergence score toward the target.'
- What this solution (achieved 0.77245) has done: 'I lower the smoothing constant that limits patient‑specific influence (weight_const = 5 instead of 50) so the model leans more on the per‑patient vote distribution, which tends to be better calibrated for KL‑divergence. I also remove the temperature scaling (set temperature = 1.0) and use a smaller epsilon to avoid unnecessary bias while still preventing zero probabilities. These tiny adjustments keep the original pipeline intact but are expected to move the KL score closer to the target.'
- What this solution (achieved 0.9146) has done: 'I reduce over‑confidence by flattening the blended predictions (temperature < 1) and rely more on the overall class baseline (larger weight_const) while keeping the same blending logic. This tiny adjustment is expected to lower the KL‑divergence, moving the score closer to the target without touching the core model pipeline.'
- What this solution (achieved 0.77245) has done: 'I keep the overall pipeline unchanged and only adjust the blending constants that control how much patient‑specific voting influences the final probabilities. By lowering `weight_const` from 50 to 5 the model relies more on the per‑patient vote distribution, which is usually better calibrated for KL‑divergence. I also set `temperature` to 1.0 so the blended probabilities are not artificially flattened. These minimal changes keep the core logic intact while moving the score closer to the target (lower is better).'
- What this solution (achieved 1.27225) has done: 'The update reduces the influence of patient‑specific vote distributions by using a much larger `weight_const` (so patient_weights become near 0) and adds a modest temperature scaling (`temperature=0.8`) to slightly flatten the probabilities. This drives the predictions closer to the global class baseline, which is better calibrated for KL‑divergence and moves the score toward the lower target while keeping the original pipeline intact.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

possible_dirs = [
    os.path.join("data", "hms-harmful-brain-activity-classification"),
    os.path.join("data", "input", "hms-harmful-brain-activity-classification"),
    "/kaggle/input/hms-harmful-brain-activity-classification",
]
LOAD_DATA_FROM = next((d for d in possible_dirs if os.path.isdir(d)), None)
if LOAD_DATA_FROM is None:
    raise FileNotFoundError("Could not locate the competition data directory.")
print(f"Using data directory: {LOAD_DATA_FROM}")




## === cell 1
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last six columns are the vote counts
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

vote_sums = df[TARGETS].sum(axis=0).astype(np.float64)
epsilon = 1e-6
baseline_probs = (vote_sums + epsilon) / (vote_sums.sum() + epsilon * len(vote_sums))
print("Baseline class probabilities:", baseline_probs.values)




## === cell 2
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})

patient_votes = df.groupby("patient_id")[list(TARGETS)].sum() + epsilon
patient_probs = patient_votes.div(patient_votes.sum(axis=1), axis=0)

patient_total_votes = patient_votes.sum(axis=1)  # total votes per patient

weight_const = 10000.0  # larger → rely much more on global baseline
patient_weights = patient_total_votes / (patient_total_votes + weight_const)
patient_weights = patient_weights.clip(0.0, 1.0)

patient_probs_test = patient_probs.reindex(test["patient_id"]).reset_index(drop=True)
patient_weights_test = patient_weights.reindex(test["patient_id"]).fillna(0.0).values

blended = patient_probs_test.fillna(0).values * patient_weights_test[
    :, None
] + baseline_probs.values * (1.0 - patient_weights_test[:, None])

blended = np.clip(blended, epsilon, None)

temperature = 0.8
blended = blended**temperature

row_sums = blended.sum(axis=1, keepdims=True)
blended = blended / row_sums

submission[TARGETS] = blended
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
