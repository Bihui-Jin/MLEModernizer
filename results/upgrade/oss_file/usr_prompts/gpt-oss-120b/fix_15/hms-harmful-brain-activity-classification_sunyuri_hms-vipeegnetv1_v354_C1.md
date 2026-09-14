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

0.2939802417846457

# 6. Current score

1.1548

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I make the script robust by avoiding the failing TensorFlow import, adding safe detection of the data directory, and keeping the fallback prior‑based prediction logic unchanged. This ensures the notebook runs without errors and always writes a valid `submission.csv` with correctly normalised probabilities.'
- What this solution (achieved 1.41937) has done: 'I replace the simple prior‑only fallback with a per‑`eeg_id` vote distribution derived from the training data: for each test row, if its `eeg_id` appears in the training set we use the normalized vote counts for that `eeg_id`; otherwise we fall back to the overall class priors. This keeps the original structure while providing more informative probabilities, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.68479) has done: 'I keep the original fallback logic but add a second‑level fallback: if a test `eeg_id` is not present in the training aggregation, I look up the normalized vote distribution for its `patient_id`. This uses more information without changing the core model, and should lower the KL‑divergence, moving the score from 1.41937 closer to the target 0.29398. The rest of the script remains unchanged.'
- What this solution (achieved 0.76744) has done: 'I add Laplace smoothing to all probability estimates (global priors, per‑eeg_id and per‑patient_id distributions) by adding a tiny count before normalising. This removes zero probabilities that can inflate KL‑divergence, while keeping the overall fallback logic unchanged. The change is minimal, preserves the core workflow, and is expected to lower the score toward the target.'
- What this solution (achieved 0.809) has done: 'I reduce the Laplace smoothing constant to lessen over‑smoothing and combine the per‑eeg ID and per‑patient ID vote distributions with a small weight on the patient level (instead of using a strict fallback). This yields slightly more informative probability estimates while preserving the original fallback structure, and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.87464) has done: 'I reduce the Laplace smoothing constant to 0.01 and make the blending weight between the per‑eeg and per‑patient distributions depend on how many votes an eeg_id has (more votes → trust the eeg‑level distribution more). This keeps the same fallback logic but provides more informative probabilities, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.809) has done: 'I increase Laplace smoothing to 0.1, give the per‑eeg distribution more influence (BASE_EEG_WEIGHT = 0.9) and let that influence grow faster by lowering the vote‑threshold to 10. These minimal tweaks keep the original fallback logic intact while producing smoother, better‑calibrated probabilities that should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.94432) has done: 'We will reduce the Laplace‑smoothing constant (to limit over‑smoothing) and simplify the fallback logic: when an eeg_id has a calibrated distribution we now use it directly instead of blending with the patient‑level distribution. This keeps the overall fallback structure unchanged, avoids unnecessary mixing that can inflate KL‑divergence, and is expected to move the score closer to the target lower‑is‑better metric.'
- What this solution (achieved 0.87464) has done: 'I increase the Laplace smoothing constant to 0.01 and replace the strict fallback with a modest blending of the per‑`eeg_id` and per‑`patient_id` distributions. The blend weight grows with the number of votes for that `eeg_id` (up to a maximum of 0.8), so rows with many observed votes rely more on their own distribution while still benefiting from patient‑level information. This slight calibration should lower the KL‑divergence and move the score closer to the target without altering the overall logic.'
- What this solution (achieved 1.1548) has done: 'I reduce the Laplace smoothing to a negligible value and simplify the probability logic so that when an eeg_id has any training data we use its own vote distribution directly (no blending with patient or prior). If the eeg_id is unseen we fall back to the patient‑level distribution, and finally to the global prior. This tighter use of the observed votes should lower the KL‑divergence and move the score closer to the target while keeping the overall workflow unchanged.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

NEEDTRAIN = True  # retained but unused in fallback
LOAD_MODELS_FROM = "modelsxxxxxxx"  # unused
SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5
SPLITS = 5

import os, warnings, gc, time
import numpy as np, pandas as pd

TF_AVAILABLE = False
print("TensorFlow import skipped – using fallback model.")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

warnings.filterwarnings("ignore")


def find_data_root() -> str:
    """
    Return the directory that contains the competition folders.
    Checks common Kaggle and local locations.
    """
    candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "./data/hms-harmful-brain-activity-classification",
        "./hms-harmful-brain-activity-classification",
        "./data",
        ".",
    ]
    for p in candidates:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError("train.csv not found in any expected location.")


LOAD_DATA_FROM = find_data_root()
print(f"Data root resolved to: {LOAD_DATA_FROM}")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last six columns are the votes
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

alpha = 1e-6

class_counts = df[TARGETS].sum().astype(np.float64)
prior_probs = (class_counts + alpha) / (class_counts.sum() + alpha * len(TARGETS))
print("Class prior probabilities (tiny smoothing):", prior_probs.to_dict())

if not TF_AVAILABLE:
    test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
    test_df = pd.read_csv(test_path)
    print("Test shape:", test_df.shape)

    eeg_votes_sum = df.groupby("eeg_id")[list(TARGETS)].sum()
    eeg_probs = (eeg_votes_sum + alpha).div(
        eeg_votes_sum.sum(axis=1) + alpha * len(TARGETS), axis=0
    )

    patient_votes_sum = df.groupby("patient_id")[list(TARGETS)].sum()
    patient_probs = (patient_votes_sum + alpha).div(
        patient_votes_sum.sum(axis=1) + alpha * len(TARGETS), axis=0
    )

    def get_probs(eeg_id, patient_id):
        """
        Return a probability vector for a given test row.
        • If the eeg_id exists in training, use its own distribution directly.
        • Else if the patient_id exists, use the patient‑level distribution.
        • Otherwise fall back to the global prior.
        The result is clipped and re‑normalised to guarantee a valid probability
        vector (no zeros, sums to 1).
        """
        if eeg_id in eeg_probs.index:
            combined = eeg_probs.loc[eeg_id].values
        elif patient_id in patient_probs.index:
            combined = patient_probs.loc[patient_id].values
        else:
            combined = prior_probs.values

        combined = np.clip(combined, 1e-12, None)
        combined = combined / combined.sum()
        return combined

    probs = np.vstack(
        [
            get_probs(eid, pid)
            for eid, pid in zip(test_df["eeg_id"].values, test_df["patient_id"].values)
        ]
    )

    probs = probs / probs.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    sub[TARGETS] = probs
    submission_path = "submission.csv"
    sub.to_csv(submission_path, index=False)
    print(f"Fallback submission written to {submission_path} with shape {sub.shape}")
else:
    print("TensorFlow path would be executed here if TF were available.")
