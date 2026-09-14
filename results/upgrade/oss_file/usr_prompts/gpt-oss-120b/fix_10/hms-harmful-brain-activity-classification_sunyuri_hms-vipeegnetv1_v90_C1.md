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

0.3464995089196601

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I guard the TensorFlow import to avoid the protobuf incompatibility and replace the inference part with a simple baseline that uses the overall class distribution from the training data, ensuring a valid *.csv* submission without needing TensorFlow or the heavy model code. This fixes the runtime error and still produces reasonable predictions for the KL‑divergence metric.'
- What this solution (achieved 1.41937) has done: 'I remove the TensorFlow import entirely to avoid the protobuf error and replace the naive global‑distribution baseline with a per‑eeg_id probability estimate (averaging the vote counts for each eeg_id in the training data). This keeps the same simple probabilistic approach but yields predictions that better match the training distribution, moving the KL‑divergence score toward the target while still producing a valid *.csv* file.'
- What this solution (achieved 1.41937) has done: 'I add a tiny Laplace smoothing (add 1 to every class count) when computing both the per‑`eeg_id` probabilities and the overall class distribution. This removes zero‑probability predictions that can heavily penalise KL‑divergence, while keeping the same simple baseline logic and structure of the original code.'
- What this solution (achieved 1.41937) has done: 'The update adds a simple blending step that mixes the per‑`eeg_id` probability estimates with the overall class distribution.  
A weight `alpha` (set to 0.6) gives more influence to the per‑`eeg_id` information while still pulling each prediction toward the global prior, which reduces overly confident or noisy rows and improves the KL‑divergence score toward the target. The rest of the pipeline (smoothing, normalization, and CSV output) stays unchanged.'
- What this solution (achieved 1.41937) has done: 'We lower the blending weight `alpha` to rely more on the stable global class distribution and increase the Laplace smoothing constant from 1 to 5 for both per‑eeg and overall counts. This makes each prediction less extreme and reduces KL‑divergence, moving the score nearer the target while keeping the original baseline logic unchanged.'
- What this solution (achieved 1.41936) has done: 'I reduce the model’s reliance on noisy per‑`eeg_id` estimates and increase smoothing so the predictions are closer to the stable global class distribution, which should lower the KL‑divergence toward the target. Specifically, I raise the Laplace smoothing count (`SMOOTH`) and set the blending weight `alpha` to 0 so the submission uses only the overall prior (still normalized per‑row).'
- What this solution (achieved 1.41937) has done: 'I lower the Laplace smoothing to 1 (so class probabilities are less overly uniform) and set a moderate blending weight `alpha = 0.5` to combine per‑`eeg_id` information with the global prior. This adds useful per‑record signals while still keeping the stable overall distribution, which should reduce the KL‑divergence score toward the target. I also add a tiny epsilon before the final normalization to avoid any zero‑probability rows.'
- What this solution (achieved 1.41937) has done: 'I lower the blending weight to 0 so the submission relies solely on the stable global class distribution, which reduces noisy per‑eeg estimates and generally lowers the KL‑divergence. I also increase the Laplace smoothing constant to 5 to make the global prior less extreme, keeping predictions well‑behaved while preserving the original pipeline structure.'
- What this solution (achieved 1.41937) has done: 'I reduce the Laplace smoothing from 5 to 1 so the global class prior better reflects the true training distribution, and set the blending weight `alpha` to 0.5 to combine the per‑`eeg_id` estimates (when available) with the global prior. This modest change keeps the original baseline logic but yields less uniform predictions and leverages any per‑eeg information, which should lower the KL‑divergence score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"  # "kaggle" or "local"
DATATYPE = ["eeg", "spe", "img"]
STAGE = 3

print("DATATYPE:", DATATYPE)

if PLATFORM == "local":
    TRAIN_PATH = "./input/hms-harmful-brain-activity-classification/train.csv"
    TEST_PATH = "./input/hms-harmful-brain-activity-classification/test.csv"
else:  # kaggle
    TRAIN_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TEST_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"

df = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGETS = df.columns[-6:]  # the six vote columns
print("Train shape:", df.shape)
print("Test shape :", test.shape)
print("Targets:", list(TARGETS))

SMOOTH = 1  # added count for each class to avoid zero‑probability

eeg_group = df.groupby("eeg_id")[list(TARGETS)].sum()
eeg_group += SMOOTH
eeg_probs = eeg_group.div(eeg_group.sum(axis=1), axis=0)

class_counts = df[TARGETS].sum()
class_counts += SMOOTH
total_counts = class_counts.sum()
overall_probs = class_counts / total_counts  # Series length = 6

alpha = 0.5  # weight for per‑eeg info; (1‑alpha) for global prior

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

mapped = sub["eeg_id"].map(eeg_probs.to_dict(orient="index"))
mapped_df = pd.json_normalize(mapped)

mapped_df = mapped_df.reindex(columns=TARGETS, fill_value=np.nan)

per_eeg_filled = mapped_df.fillna(0)

blended = per_eeg_filled.multiply(alpha).add(overall_probs.multiply(1 - alpha), axis=1)

blended += 1e-6  # tiny epsilon to avoid exact zeros

row_sums = blended.sum(axis=1)
sub[TARGETS] = blended.div(row_sums, axis=0)

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission saved as {submission_path} with shape {sub.shape}")
print("Row sums (should be 1.0):", sub[TARGETS].sum(axis=1).unique())
