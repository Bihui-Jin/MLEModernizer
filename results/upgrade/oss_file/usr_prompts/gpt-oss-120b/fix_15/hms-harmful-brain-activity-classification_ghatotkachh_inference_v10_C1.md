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

0.5626450365366341

# 6. Current score

1.06527

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I replace the model‑loading loop with a safe fallback that creates uniform predictions when the expected checkpoint directory is missing, and keep the rest of the pipeline unchanged. This fixes the FileNotFoundError and the shape mismatch when assigning predictions to the submission DataFrame, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 1.39779) has done: 'I replace the uniform‑prediction fallback with a simple baseline that uses the average class distribution from the training data. This gives more realistic probabilities than a flat 1/6, which should reduce the KL‑divergence and move the score closer to the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 1.39779) has done: 'I keep the overall pipeline unchanged but improve the fallback predictions: instead of using a single global class average, I compute the average class distribution for each `spectrogram_id` in the training set and use it when the same ID appears in the test set, falling back to the global average only when the ID is unseen. This leverages available metadata to give more tailored probabilities and should reduce the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 1.39692) has done: 'I adjust the fallback prediction logic to combine the spectrogram‑specific averages with the global class distribution, apply a mild temperature scaling for smoother probabilities, and re‑normalize each row so they sum to one. This should lower the KL‑divergence and move the score closer to the target while leaving the rest of the pipeline unchanged.'
- What this solution (achieved 1.39779) has done: 'The fix restores proper path handling by moving the base‑directory lookup out of the class body (preventing the NameError) and then creates a `paths` instance that other cells can use. No modelling changes are made; the baseline fallback remains unchanged, so the only effect is a valid run that writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'I enhance the fallback prediction logic to use a more specific per‑eeg_id class distribution first, then fall back to the per‑spectrogram distribution, and finally to the global average. This adds useful metadata without altering the core model pipeline and should reduce the KL‑divergence, moving the score closer to the lower target value.'
- What this solution (achieved 1.67825) has done: 'I fixed the TypeError caused by storing group‑means as dictionaries of Series, converting them to plain NumPy arrays, and added a safeguard to avoid division‑by‑zero when normalising the final predictions. This restores correct fallback prediction logic and guarantees each row sums to 1, allowing a valid submission.csv to be written.'
- What this solution (achieved 1.06527) has done: 'I tighten the fallback prediction logic by smoothing the hierarchical averages: for each specific group (eeg ID, patient ID, spectrogram ID) the prediction is blended with the global class distribution, with a weight that grows with the number of training samples supporting that group. After the blend I apply a mild temperature scaling to soften overly‑confident probabilities and re‑normalise each row. This keeps the overall pipeline unchanged while producing more calibrated probabilities, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader


class Paths:
    _candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/working/data/hms-harmful-brain-activity-classification",
        "data/hms-harmful-brain-activity-classification",
    ]

    @staticmethod
    def _choose_base():
        for p in Paths._candidates:
            if os.path.isdir(p):
                return p
        raise FileNotFoundError(
            "Cannot locate the dataset folder. Checked: " + ", ".join(Paths._candidates)
        )


Paths.base = Paths._choose_base()
Paths.train_csv = os.path.join(Paths.base, "train.csv")
Paths.test_csv = os.path.join(Paths.base, "test.csv")
Paths.out = "/kaggle/working"

paths = Paths()


class Config:
    batchsize = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
device = config.device

test_df = pd.read_csv(paths.test_csv)




## === cell 1
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _baseline_predictions():
    """
    Compute calibrated baseline predictions using a hierarchy of fallback averages
    with count‑based smoothing and temperature scaling.
    """
    train_df = pd.read_csv(paths.train_csv)

    train_votes = train_df[TARGETS].astype(float).values
    row_sums = train_votes.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    train_probs = train_votes / row_sums

    global_mean = train_probs.mean(axis=0)
    global_mean = global_mean / global_mean.sum()

    def _group_stats(g):
        summed = g[TARGETS].sum()
        total = summed.sum()
        probs = summed / (total if total != 0 else 1.0)
        return probs.values.astype(np.float32), len(g)

    spec_group = train_df.groupby("spectrogram_id")
    spec_means_dict = {sid: _group_stats(g) for sid, g in spec_group}

    patient_group = train_df.groupby("patient_id")
    patient_means_dict = {pid: _group_stats(g) for pid, g in patient_group}

    eeg_group = train_df.groupby("eeg_id")
    eeg_means_dict = {eid: _group_stats(g) for eid, g in eeg_group}

    preds = np.empty((len(test_df), len(TARGETS)), dtype=np.float32)

    test_spec_ids = test_df["spectrogram_id"].values
    test_eeg_ids = test_df["eeg_id"].values
    test_patient_ids = test_df["patient_id"].values

    thresh_eeg = 10
    thresh_patient = 20
    thresh_spec = 30

    for i, (eeg_id, patient_id, spec_id) in enumerate(
        zip(test_eeg_ids, test_patient_ids, test_spec_ids)
    ):
        if eeg_id in eeg_means_dict:
            probs, cnt = eeg_means_dict[eeg_id]
            w = min(1.0, cnt / thresh_eeg)
            preds[i] = w * probs + (1 - w) * global_mean
        elif patient_id in patient_means_dict:
            probs, cnt = patient_means_dict[patient_id]
            w = min(1.0, cnt / thresh_patient)
            preds[i] = w * probs + (1 - w) * global_mean
        elif spec_id in spec_means_dict:
            probs, cnt = spec_means_dict[spec_id]
            w = min(1.0, cnt / thresh_spec)
            preds[i] = w * probs + (1 - w) * global_mean
        else:
            preds[i] = global_mean

    row_sums = preds.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-8
    preds = preds / row_sums

    temperature = 1.5
    preds = np.power(preds, 1.0 / temperature)
    row_sums = preds.sum(axis=1, keepdims=True)
    preds = preds / row_sums

    return preds


checkpoint_dir = "/kaggle/input/resnet50"
predictions = None

if os.path.isdir(checkpoint_dir):
    predictions_list = []
    for fname in os.listdir(checkpoint_dir):
        ckpt_path = os.path.join(checkpoint_dir, fname)
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            raise RuntimeError("Model code not provided – using fallback.")
        except Exception as e:
            print(f"Warning: failed to load {ckpt_path}: {e}")

    if predictions_list:
        predictions = np.mean(np.stack(predictions_list), axis=0)

if predictions is None:
    predictions = _baseline_predictions()




## === cell 2
os.makedirs(paths.out, exist_ok=True)

submission_path = os.path.join(paths.out, "submission.csv")
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print(f"Submission shape: {sub.shape}")
sub.head()
