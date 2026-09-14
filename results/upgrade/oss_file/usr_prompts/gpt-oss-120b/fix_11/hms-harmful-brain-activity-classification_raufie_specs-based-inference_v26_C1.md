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

0.551701239162851

# 6. Current score

0.77875

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix adds graceful handling when the pretrained weight file is missing (skipping loading and using a pretrained EfficientNet or uniform predictions), corrects the EEG spectrogram visualisation call to avoid a NameError, ensures the dataset does not reference undefined variables, and guarantees that the final prediction array matches the submission format so a valid `submission.csv` is written.'
- What this solution (achieved 1.41937) has done: 'The fix adds the missing imports, loads the test metadata, computes class‑prior probabilities from the training votes, builds a prediction matrix that sums to 1 for each row, and writes a proper `submission.csv` with the required columns. All unused model‑related code is removed, so the script runs end‑to‑end and produces a valid submission file.'
- What this solution (achieved 1.68479) has done: 'The update adds a simple patient‑level prior: for each test record we use the normalized vote distribution of its patient from the training data when available, otherwise we fall back to the overall class‑prior. This keeps the original workflow but provides more tailored probabilities, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 1.0698) has done: 'I blend the patient‑level prior with the global prior, weighting the patient information by how many training samples it has (more samples → higher weight). This adds a modest, data‑driven adjustment that should lower the KL‑divergence without altering the overall architecture or training loop.'
- What this solution (achieved 1.05318) has done: 'The update removes the gradual blending of patient‑level and global priors and instead uses the full patient prior whenever a patient appears in the training data (fallback to the global prior otherwise). A tiny epsilon smoothing is added before renormalising to avoid zero probabilities. This stronger use of patient‑specific information is expected to lower the KL‑divergence, moving the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.77566) has done: 'I keep the overall prior‑based approach but replace the “always use the patient prior when available” rule with a smooth blending that favours the global prior for patients with few training samples. By weighting the patient‑specific distribution according to the number of votes we have for that patient (count / (count + 20)), the predictions become less noisy for rare patients, which should lower the KL‑divergence and move the score closer to the target. The rest of the pipeline and submission format stay unchanged.'
- What this solution (achieved 0.77566) has done: 'I add a more specific spectrogram‑level prior, computed from the training votes, and use it (with the same smooth blending based on sample count) before falling back to the patient prior or the global prior. This gives predictions that are better calibrated for each test record while preserving the existing workflow and smoothing logic, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.77245) has done: 'I tighten the blending logic to use both spectrogram‑ and patient‑level priors together (when available) and lower the smoothing constant from 20 to 5 so that more reliable specific priors receive higher weight. This richer, more responsive blending is expected to produce better‑calibrated probabilities and move the KL‑divergence score closer to the target (lower is better). The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 0.77875) has done: 'I increase the blending constant so the global prior has a stronger influence (reducing noisy patient‑/spectrogram‑specific priors) and add a tiny Laplace smoothing to the class counts before normalising. These minimal tweaks should lower the KL‑divergence and move the score nearer the target while keeping the original pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd


class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b0_epoch_9.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"


os.makedirs(paths.OUTPUT_DIR, exist_ok=True)




## === cell 1
test_df = pd.read_csv(paths.TEST_CSV)

if os.path.isfile(paths.TRAIN_CSV):
    train_df = pd.read_csv(paths.TRAIN_CSV)
    TARGETS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    class_sums = train_df[TARGETS].sum().values.astype(np.float32) + 1.0
    global_prior = class_sums / class_sums.sum()
    print(
        "Using global class‑prior predictions based on training vote counts (with Laplace smoothing)."
    )

    patient_sums = train_df.groupby("patient_id")[TARGETS].sum() + 1.0
    patient_prior = patient_sums.div(patient_sums.sum(axis=1), axis=0)
    patient_prior_dict = patient_prior.to_dict(orient="index")
    patient_counts_dict = patient_sums.sum(axis=1).to_dict()

    spec_sums = train_df.groupby("spectrogram_id")[TARGETS].sum() + 1.0
    spectrogram_prior = spec_sums.div(spec_sums.sum(axis=1), axis=0)
    spectrogram_prior_dict = spectrogram_prior.to_dict(orient="index")
    spectrogram_counts_dict = spec_sums.sum(axis=1).to_dict()
else:
    global_prior = np.full(6, 1 / 6, dtype=np.float32)
    patient_prior_dict = {}
    patient_counts_dict = {}
    spectrogram_prior_dict = {}
    spectrogram_counts_dict = {}
    print("Training file missing – using uniform predictions.")

blend_k = 20.0
eps = 1e-6  # tiny smoothing to avoid exact zeros

pred_list = []
for _, row in test_df.iterrows():
    pid = row["patient_id"]
    sid = row["spectrogram_id"]

    spec_dist = None
    cnt_spec = 0.0
    if sid in spectrogram_prior_dict:
        spec_dist = np.array(
            [
                spectrogram_prior_dict[sid][t]
                for t in [
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ]
            ],
            dtype=np.float32,
        )
        cnt_spec = spectrogram_counts_dict.get(sid, 0.0)

    patient_dist = None
    cnt_pat = 0.0
    if pid in patient_prior_dict:
        patient_dist = np.array(
            [
                patient_prior_dict[pid][t]
                for t in [
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ]
            ],
            dtype=np.float32,
        )
        cnt_pat = patient_counts_dict.get(pid, 0.0)

    if spec_dist is not None and patient_dist is not None:
        total_cnt = cnt_spec + cnt_pat
        combined_specific = (cnt_spec * spec_dist + cnt_pat * patient_dist) / total_cnt
    elif spec_dist is not None:
        total_cnt = cnt_spec
        combined_specific = spec_dist
    elif patient_dist is not None:
        total_cnt = cnt_pat
        combined_specific = patient_dist
    else:
        total_cnt = 0.0
        combined_specific = None

    if combined_specific is not None:
        w = total_cnt / (total_cnt + blend_k)
        pred = w * combined_specific + (1.0 - w) * global_prior
    else:
        pred = global_prior.copy()

    pred = pred + eps
    pred = pred / pred.sum()
    pred_list.append(pred)

predictions = np.stack(pred_list, axis=0)




## === cell 2
submission = pd.DataFrame(
    predictions,
    columns=[
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ],
)
submission.insert(0, "eeg_id", test_df["eeg_id"])

row_sums = submission.iloc[:, 1:].sum(axis=1)
if not np.allclose(row_sums, 1.0, atol=1e-6):
    submission.iloc[:, 1:] = submission.iloc[:, 1:].div(row_sums, axis=0)

output_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
