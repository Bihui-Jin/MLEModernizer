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

cupy-cuda12x==13.6.0
fastai==2.8.5
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.4280755309867892

# 6. Current score

1.41677

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix removes the CUDA‑dependent spectrogram creation and the missing pretrained model loads, replacing them with safe fallbacks that generate a valid submission. A simple uniform probability vector (summing to 1) is written for each test `eeg_id`, ensuring the CSV complies with the required format and avoids runtime errors.'
- What this solution (achieved 1.41937) has done: 'I compute the overall class distribution from the training votes and use that empirical probability vector for every test row instead of a uniform one. This simple calibration typically reduces KL‑divergence, moving the score closer to the target (lower is better) while keeping the original pipeline unchanged and still producing a valid CSV submission.'
- What this solution (achieved 1.05318) has done: 'I replace the single global class‑probability baseline with a patient‑level calibrated baseline: for each test row I look up the aggregated vote distribution of the matching `patient_id` in the training set (falling back to the overall distribution when unseen) and apply a tiny smoothing ε to avoid zero probabilities. This modest, data‑driven adjustment keeps the original pipeline intact while expectedly lowering the KL‑divergence toward the target score.'
- What this solution (achieved 1.41832) has done: 'I smooth the patient‑level vote distributions toward the overall class distribution instead of using the raw patient proportions. By adding a small weight α of the global counts to each patient’s counts before normalising, we avoid overly extreme per‑patient probabilities, which typically lowers the KL‑divergence and moves the score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 1.41937) has done: 'I increase the smoothing strength `alpha` so the blended patient‑level probabilities are dominated by the global class distribution. This makes every test row’s prediction much closer to the overall vote proportions, which historically yields a lower KL‑divergence and therefore moves the score nearer to the target (lower is better). No other logic is altered.'
- What this solution (achieved 1.41885) has done: 'The patch reduces the excessive smoothing toward the global class distribution by changing `alpha` from a very large value (`1e6`) to a modest value (`10`). This lets patient‑specific vote patterns influence the predictions while still retaining a small amount of global regularisation, which should lower the KL‑divergence and move the score closer to the target. No other logic is altered, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.47425) has done: 'I reduce the smoothing strength to 0 so each patient uses its own raw vote distribution (fallback to the global distribution for unseen patients). This lets the predictions reflect the actual patient‑level class ratios more closely, which should lower the KL‑divergence toward the target score while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.41936) has done: 'I keep the overall workflow unchanged but replace the fixed‑alpha blending with a patient‑size‑aware smoothing: patients that have few annotated votes get a stronger pull toward the global class distribution, while well‑represented patients rely more on their own counts. This modest, data‑driven adjustment is expected to lower the KL‑divergence and move the score closer to the target without altering the core logic of the solution.'
- What this solution (achieved 0.77566) has done: 'I replace the overly‑aggressive smoothing that forces predictions toward the global class distribution with a modest additive (Laplace) smoothing that blends each patient’s own vote proportions with the global probabilities. This keeps the core patient‑level logic but uses a smaller smoothing strength (α ≈ 20), yielding more diverse but still regularised predictions, which should lower the KL‑divergence toward the target score while preserving a valid CSV output.'
- What this solution (achieved 0.77245) has done: 'I lower the smoothing strength `alpha` from 20 to 5 so each patient’s own vote distribution has more influence while still retaining the global regularisation. This small change keeps the overall pipeline intact but should produce predictions that better match the true per‑patient patterns, thereby reducing the KL‑divergence and moving the score nearer to the target.'
- What this solution (achieved 0.85553) has done: 'I replace the constant‑α smoothing with a patient‑specific smoothing that scales α inversely with the number of votes a patient has. This lets well‑represented patients rely more on their own vote distribution while still regularising scarce patients toward the global class probabilities, a change expected to lower the KL‑divergence and move the score closer to the target without altering the overall pipeline.'
- What this solution (achieved 0.85553) has done: 'I keep the overall baseline‑smoothing approach but add a more specific per‑eeg_id probability lookup (falling back to the patient‑level distribution and finally to the global one). This adds useful granularity without changing the core pipeline, and it should bring the KL‑divergence closer to the target lower score.'
- What this solution (achieved 1.47425) has done: 'I reduce the overly‑aggressive smoothing and instead use the raw patient‑ and EEG‑level vote distributions (with only a tiny epsilon for numerical safety). This change keeps the overall pipeline intact, removes the inverse‑scaled α that worsened the KL score, and is expected to move the validation metric closer to the target lower value.'
- What this solution (achieved 1.41677) has done: 'I replace the simple raw‑frequency lookup with a Bayesian‑smoothed blend of patient‑ (and EEG‑) level vote counts and the global class distribution. Each patient (and EEG) probability vector be computed as  

\[
p = \frac{c_{\text{local}} + \alpha \, c_{\text{global}}}{n_{\text{local}} + \alpha \, N_{\text{global}}}
\]

where α is a modest smoothing strength (set to 2.0). This keeps the original dictionary‑based logic but yields more reliable, less‑extreme probabilities, which should lower the KL‑divergence and move the score toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"



## === cell 1
PATH = {
    "test": "/kaggle/input/hms-harmful-brain-activity-classification/",
    "train": "/kaggle/input/hms-harmful-brain-activity-classification/",
}
sample_sub = pd.read_csv(os.path.join(PATH["test"], "sample_submission.csv"))
votes = [c for c in sample_sub.columns if "_vote" in c]



## === cell 2
train_df = pd.read_csv(os.path.join(PATH["train"], "train.csv"))

class_votes_sum = train_df[votes].sum()
global_counts = class_votes_sum.values.astype(np.float32)
global_total = global_counts.sum()
class_prob_global = (global_counts / global_total).astype(np.float32)

alpha = 2.0  # Bayesian smoothing factor

patient_group = train_df.groupby("patient_id")[votes].sum()
patient_prob_dict = {}
for pid, row in patient_group.iterrows():
    patient_counts = row.values.astype(np.float32)
    patient_total = patient_counts.sum()
    smoothed_counts = patient_counts + alpha * global_counts
    smoothed_total = patient_total + alpha * global_total
    probs = smoothed_counts / smoothed_total
    patient_prob_dict[pid] = probs.astype(np.float32)

eeg_group = train_df.groupby("eeg_id")[votes].sum()
eeg_prob_dict = {}
for eid, row in eeg_group.iterrows():
    eeg_counts = row.values.astype(np.float32)
    eeg_total = eeg_counts.sum()
    smoothed_counts = eeg_counts + alpha * global_counts
    smoothed_total = eeg_total + alpha * global_total
    probs = smoothed_counts / smoothed_total
    eeg_prob_dict[eid] = probs.astype(np.float32)



## === cell 3
test = pd.read_csv(os.path.join(PATH["test"], "test.csv"))
num_classes = len(votes)
epsilon = 1e-12  # avoid exact zeros

prob_matrix = np.empty((len(test), num_classes), dtype=np.float32)

for idx, (eid, pid) in enumerate(zip(test["eeg_id"].values, test["patient_id"].values)):
    prob = eeg_prob_dict.get(eid)
    if prob is None:
        prob = patient_prob_dict.get(pid, class_prob_global)
    prob = prob + epsilon
    prob = prob / prob.sum()
    prob_matrix[idx] = prob

submission = pd.DataFrame(
    np.column_stack([test["eeg_id"].values, prob_matrix]),
    columns=["eeg_id"] + votes,
)



## === cell 4
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())
