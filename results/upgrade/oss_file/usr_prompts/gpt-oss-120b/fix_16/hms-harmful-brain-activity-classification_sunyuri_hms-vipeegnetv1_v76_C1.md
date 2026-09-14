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

0.4156363132542079

# 6. Current score

0.7949

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I fixed the import error caused by the external EfficientNet wheel by switching to TensorFlow’s built‑in EfficientNet implementation, added a safe fallback for model weight loading, and replaced the inference step with a simple baseline that uses the overall class distribution from the training set. This ensures the script runs end‑to‑end, creates a valid `submission.csv` where each row’s probabilities sum to 1, and avoids crashes when pretrained weights are missing.'
- What this solution (achieved 1.48867) has done: 'The script failed because `sys` was not imported, causing `NameError` before the training dataframe was built. Adding `import sys` resolves the error, allowing the training aggregation to run and the baseline mean‑probability submission to be created correctly. No changes to model logic are needed, and the output CSV now conforms to the required format.'
- What this solution (achieved 1.67064) has done: 'I replace the simple global‑mean baseline with a patient‑aware baseline: compute the mean class distribution for each patient in the training set and use it for test rows sharing the same patient_id, falling back to the overall mean when a patient is unseen. This small improvement keeps all core logic unchanged while providing more informative probabilities, which should lower the KL‑divergence toward the target.'
- What this solution (achieved 1.03903) has done: 'I replace the simple mean‑based patient baseline with a vote‑count‑based patient distribution (summing raw votes per patient and normalising) and add a tiny epsilon to avoid zero probabilities. This keeps the overall structure unchanged while providing a more accurate probability estimate, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.77501) has done: 'The script failed because the hard‑coded data paths do not exist in the execution environment. I added a tiny helper that searches for *train.csv* and *test.csv* under the current working directory, falling back to the original paths if they are found. This fixes the `FileNotFoundError` and lets the original patient‑wise probability baseline run unchanged, producing a valid `submission.csv` where each row sums to 1.'
- What this solution (achieved 0.77197) has done: 'The change adds a simple blending of patient‑specific vote distributions with the overall global distribution, weighting more heavily the patients that have many votes. This keeps the original patient‑wise baseline but makes unseen or low‑vote patients rely more on the global prior, which is expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.8267) has done: 'I increase the global‑patient blending factor (BLEND_K) so the model leans more on the stable overall class distribution, and add a mild temperature scaling step to soften the blended probabilities before the final renormalisation. These small post‑processing tweaks keep the core logic unchanged while reducing over‑confident patient‑specific predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.0158) has done: 'The update increases the global‑patient blending factor and the temperature scaling while also applying a slightly stronger Dirichlet smoothing. These tweaks keep the core baseline unchanged but push the predictions closer to the overall class distribution and make them less confident, which is expected to lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.7889) has done: 'I reduced the heavy global blending and temperature scaling that were driving the predictions toward a uniform distribution, and restored a modest smoothing factor. By setting `SMOOTH_ALPHA` back to 0.5, lowering `BLEND_K` to 30 (so patient‑specific distributions have more influence), and using a temperature of 1.0, the blended probabilities stay closer to the observed vote patterns while still remaining valid (sum = 1). These minimal tweaks keep the overall baseline logic unchanged but are expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.8575) has done: 'I slightly adjust the smoothing, blending strength and temperature‑scaling constants so the blended patient‑specific probabilities stay closer to the observed vote patterns while still falling back to a stable global prior for rare patients. Reducing `SMOOTH_ALPHA` makes distributions less overly smooth, lowering `BLEND_K` gives more weight to patient‑specific information, and setting `TEMPERATURE` below 1 sharpens the final probabilities – all minimal tweaks that keep the core baseline logic unchanged and are expected to move the KL‑divergence closer to the target value.'
- What this solution (achieved 1.40348) has done: 'I lower the KL‑divergence by making the predictions rely more on the stable global class distribution and by smoothing/flattening them: increase the Laplace smoothing (`SMOOTH_ALPHA`) to 0.5, raise the blending factor (`BLEND_K`) so patient‑specific weights become negligible, and set a higher temperature (2.0) to soften the final probabilities. These minimal tweaks keep the overall baseline logic unchanged while moving the score toward the lower target.'
- What this solution (achieved 0.7949) has done: 'I reduce the heavy global blending and temperature scaling that were causing overly uniform predictions. By lowering `SMOOTH_ALPHA` back to 0.3, setting a moderate `BLEND_K` (30) to give patient‑specific vote distributions more influence, and using a temperature < 1 (0.9) to sharpen the final probabilities, the predictions stay closer to the true class distribution while still summing to 1, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np


def locate_file(rel_path):
    if os.path.exists(rel_path):
        return rel_path
    filename = os.path.basename(rel_path)
    matches = list(glob.glob(f"**/{filename}", recursive=True))
    if matches:
        for m in matches:
            if "hms-harmful-brain-activity-classification" in m:
                return m
        return matches[0]
    return rel_path  # will raise later if truly missing


NEEDTRAIN = False
PLATFORM = "local"  # not used after simplification

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

SEED = 42
DATATYPE = ""
HIGH = LENGTH = IMG_HIGH = IMG_WIDE = EEG_LENGTH = SFREQ = 0

train_path = locate_file(
    os.path.join("data", "hms-harmful-brain-activity-classification", "train.csv")
)
test_path = locate_file(
    os.path.join("data", "hms-harmful-brain-activity-classification", "test.csv")
)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

SMOOTH_ALPHA = 0.3  # was 0.5

patient_vote_sums = train.groupby("patient_id")[TARGETS].sum()

patient_total_votes = patient_vote_sums.sum(axis=1).replace(0, np.finfo(float).eps)

patient_probs = (patient_vote_sums + SMOOTH_ALPHA) / (
    patient_total_votes.values[:, None] + SMOOTH_ALPHA * len(TARGETS)
)

global_vote_sum = train[TARGETS].sum()
global_total = global_vote_sum.sum()
global_prob = (
    (global_vote_sum + SMOOTH_ALPHA) / (global_total + SMOOTH_ALPHA * len(TARGETS))
).values  # shape (6,)

BLEND_K = 30  # was 1e6, now allows patient influence
patient_weights = patient_total_votes / (
    patient_total_votes + BLEND_K
)  # series indexed by patient_id


def get_blended_prob(pid):
    """Return a blended probability vector for a given patient ID.
    Patient‑specific distribution is combined with the global prior."""
    if pid in patient_probs.index:
        pw = patient_weights.loc[pid]
        p_prob = patient_probs.loc[pid].values
        blended = pw * p_prob + (1.0 - pw) * global_prob
        return blended
    else:
        return global_prob


prob_arrays = test["patient_id"].apply(get_blended_prob).to_list()
prob_matrix = np.vstack(prob_arrays)  # shape (n_test, 6)

epsilon = 1e-12
prob_matrix = prob_matrix + epsilon

TEMPERATURE = 0.9  # was 2.0, now <1 to reduce over‑softening
prob_matrix = prob_matrix ** (1.0 / TEMPERATURE)

row_sums = prob_matrix.sum(axis=1, keepdims=True)
prob_matrix = prob_matrix / row_sums

sub = pd.DataFrame(prob_matrix, columns=TARGETS)
sub.insert(0, "eeg_id", test["eeg_id"])

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)

print("Submission saved to", submission_path)
print("Submission shape:", sub.shape)
sub.head()
