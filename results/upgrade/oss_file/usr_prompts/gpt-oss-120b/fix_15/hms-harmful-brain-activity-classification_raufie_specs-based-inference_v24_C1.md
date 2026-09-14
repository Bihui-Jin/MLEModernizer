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

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
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
PyWavelets==1.8.0
sklearn-pandas==2.2.0
timm==1.0.19
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

0.5169870009777799

# 6. Current score

1.38025

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I add the missing imports, load the test metadata into `test_df`, safely collect any model weight files (falling back to uniform predictions when none are found), and ensure the output directory exists before writing a correctly‑formatted CSV whose rows sum to 1. This resolves the NameError issues and guarantees a valid submission file.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform fallback with a simple but more informed baseline: compute the overall class vote distribution from the training data and use those priors as constant predictions for every test record. This keeps the existing structure intact, avoids heavy model loading, and should lower the KL‑divergence (moving the score from 1.4099 closer to the target 0.517).'
- What this solution (achieved 1.68479) has done: 'I keep the overall structure but replace the constant‑global prior with a simple per‑patient prior: for each patient appearing in the training set we compute the empirical vote distribution and use it for any test rows belonging to that patient. If a patient is unseen we fall back to the overall class priors. This small, data‑driven tweak keeps the same modelling pipeline while giving more informative probabilities, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41885) has done: 'I replace the raw per‑patient vote ratios with a smoothed version that blends each patient’s empirical distribution with the overall class priors (additive Dirichlet smoothing). This keeps the same overall pipeline but avoids zero‑probability issues and should produce probabilities that are closer to the true test distribution, reducing the KL‑divergence and moving the score toward the target.'
- What this solution (achieved 1.41423) has done: 'I lowered the Dirichlet smoothing constant from 10.0 to 1.0 so the per‑patient empirical vote distributions have more influence over the global priors. This makes the predictions more tailored to each patient while still keeping a safety fallback, which is expected to reduce the KL‑divergence and bring the score closer to the target. No other logic is changed, and the script still writes a valid CSV submission.'
- What this solution (achieved 1.41423) has done: 'I added a finer‑grained prior based on each `eeg_id` in addition to the existing per‑patient prior. For every test record the script now first looks up the `eeg_id`‑specific distribution, falls back to the per‑patient distribution, and finally to the global class priors. This keeps the original logic intact while providing more informative probabilities, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.37636) has done: 'I slightly reduce the Dirichlet smoothing strength so the empirical per‑patient and per‑eeg vote distributions influence the predictions more strongly, and I add a tiny epsilon after normalising each probability vector to avoid any zero‑probability issues that can inflate KL‑divergence. These tiny adjustments keep the original hierarchy (eeg → patient → global) unchanged while making the predictions a bit more confident and numerically stable, which should move the score closer to the target.'
- What this solution (achieved 1.41423) has done: 'I increase the Dirichlet smoothing constant from 0.1 to 1.0 so the per‑eeg and per‑patient empirical distributions are blended more strongly with the global class priors. This modest change keeps the original hierarchical prior logic intact while reducing overly confident, potentially noisy predictions, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.37636) has done: 'I lower the Dirichlet smoothing constant from 1.0 to 0.1 so that the per‑patient and per‑eeg empirical vote distributions have a stronger influence on the predicted probabilities. This small tweak keeps the original hierarchical prior logic unchanged while making the predictions more data‑driven, which is expected to reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.40504) has done: 'I increase the Dirichlet smoothing constant to 10.0 so the per‑patient / per‑eeg priors are blended more strongly with the global class distribution, and I add a mild temperature‑scaling step (sqrt) after normalisation to make the predictions less over‑confident. Both tweaks keep the original hierarchical prior logic intact while smoothing the output probabilities, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.21844) has done: 'I reduced the Dirichlet smoothing from 10.0 to 0.01 so the per‑eeg and per‑patient vote counts dominate the prior, and I removed the extra √‑transform and redundant epsilon‑normalisation that distorted the probability vectors. After selecting the hierarchical prior (eeg → patient → global) I clip any near‑zero values to a tiny epsilon and renormalise once, guaranteeing a valid probability distribution while moving the KL‑divergence closer to the target.'
- What this solution (achieved 1.34188) has done: 'I keep the original hierarchical prior computation but blend each sample’s prior with the global class distribution. By mixing a modest proportion of the global priors (which are less noisy) into the per‑eeg / per‑patient predictions, the output probabilities become smoother and should reduce the KL‑divergence, moving the score closer to the target. The change is limited to the inference function and preserves all existing logic and output format.'
- What this solution (achieved 1.38025) has done: 'I increase the influence of the data‑driven hierarchical priors by raising `blend_alpha` from 0.3 to 0.9, and make the Dirichlet smoothing a bit stronger (set `smoothing` to 0.1). This keeps the overall logic unchanged while giving predictions that are more tailored to each `eeg_id`/`patient_id`, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import gc
from glob import glob

import numpy as np
import pandas as pd
import torch


class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b0_epoch_8.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


test_df = pd.read_csv(paths.TEST_CSV)

model_weights = [x for x in glob("/kaggle/input/hba-efficientnet-weights/*.pth")]

device = torch.device("cpu")




## === cell 1
def run_inference_and_save():
    """
    Returns a (n_test, 6) array of class probabilities.
    Uses per‑patient and per‑eeg empirical vote distributions from the training data,
    smoothed with the global class priors (additive Dirichlet smoothing).
    Then blends each prior heavily with the global distribution (blend_alpha=0.9)
    to obtain a smoother prediction, which is expected to lower the KL‑divergence
    (lower‑is‑better metric). Hierarchy: eeg_id → patient_id → global priors.
    """
    num_classes = 6
    n_samples = len(test_df)

    train_df = pd.read_csv(paths.TRAIN_CSV)

    TARGETS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    class_votes = train_df[TARGETS].sum().astype(np.float64)
    total_votes = class_votes.sum()
    if total_votes == 0:
        global_priors = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
    else:
        global_priors = (class_votes / total_votes).values.astype(np.float32)

    patient_votes = train_df.groupby("patient_id")[TARGETS].sum()
    patient_totals = patient_votes.sum(axis=1)

    smoothing = 0.1  # slightly stronger Dirichlet smoothing

    smooth_global = smoothing * class_votes.values  # (6,)

    smoothed_numer_pat = patient_votes.values + smooth_global  # (n_patients, 6)
    smoothed_denom_pat = (
        patient_totals.values[:, None] + smoothing * total_votes
    )  # (n_patients, 1)

    patient_priors_arr = smoothed_numer_pat / smoothed_denom_pat
    patient_priors_arr = np.nan_to_num(patient_priors_arr, nan=global_priors)

    patient_priors = {
        pid: row.astype(np.float32)
        for pid, row in zip(patient_votes.index.astype(int), patient_priors_arr)
    }

    eeg_votes = train_df.groupby("eeg_id")[TARGETS].sum()
    eeg_totals = eeg_votes.sum(axis=1)

    smoothed_numer_eeg = eeg_votes.values + smooth_global  # (n_eegs, 6)
    smoothed_denom_eeg = (
        eeg_totals.values[:, None] + smoothing * total_votes
    )  # (n_eegs, 1)

    eeg_priors_arr = smoothed_numer_eeg / smoothed_denom_eeg
    eeg_priors_arr = np.nan_to_num(eeg_priors_arr, nan=global_priors)

    eeg_priors = {
        eid: row.astype(np.float32)
        for eid, row in zip(eeg_votes.index.astype(int), eeg_priors_arr)
    }

    predictions = np.empty((n_samples, num_classes), dtype=np.float32)

    eps = 1e-6  # avoid exact zeros
    blend_alpha = 0.9  # give strong weight to hierarchical priors

    for idx, (eid, pid) in enumerate(
        zip(test_df["eeg_id"].values, test_df["patient_id"].values)
    ):
        priors = eeg_priors.get(int(eid))
        if priors is None:
            priors = patient_priors.get(int(pid), global_priors)

        priors = np.maximum(priors, eps)
        priors = priors / priors.sum()

        blended = blend_alpha * priors + (1.0 - blend_alpha) * global_priors
        blended = np.maximum(blended, eps)  # keep epsilon after blend
        blended = blended / blended.sum()  # final normalisation

        predictions[idx] = blended.astype(np.float32)

    return predictions


predictions = run_inference_and_save()

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

os.makedirs(paths.OUTPUT_DIR, exist_ok=True)

sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
submission_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(f"Submission shape: {sub.shape}")
sub.head()
