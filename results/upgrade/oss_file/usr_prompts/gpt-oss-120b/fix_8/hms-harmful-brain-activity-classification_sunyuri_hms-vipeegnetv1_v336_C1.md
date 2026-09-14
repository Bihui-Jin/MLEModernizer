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

0.314108034732427

# 6. Current score

0.79979

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix adds a protobuf compatibility flag before importing TensorFlow to prevent the `MessageFactory` error, sets `NEEDTRAIN` to False to skip the heavy training steps, and replaces the inference section with a lightweight baseline that uses class‑frequency priors from the training data to generate a valid submission CSV where each row’s probabilities sum to one.'
- What this solution (achieved 1.68479) has done: 'The fix skips TensorFlow import when training is disabled, avoiding the protobuf error, and adds a lightweight patient‑level prior that replaces the uniform class prior where possible, improving the KL‑divergence while keeping the original pipeline intact.'
- What this solution (achieved 0.76992) has done: 'I make the data‑loading path robust by checking the common Kaggle input locations and falling back to the original relative path. This prevents the FileNotFoundError that stopped the notebook, allowing the prior‑based inference to run and generate a valid submission CSV. No other logic is changed, preserving the original model‑free approach and keeping the score‑related calculations intact.'
- What this solution (achieved 0.86281) has done: 'I add a lightweight validation step that searches for the smoothing factor α which best blends the patient‑level prior with the global prior (using a small hold‑out split of the training data). The best α is then used for the test‑set predictions, keeping the original prior‑based logic unchanged while moving the KL‑divergence toward the target score. This change only adds a few lines for the validation loop and replaces the fixed α = 10.0 with the data‑driven value.'
- What this solution (achieved 0.79979) has done: 'The fix converts stored patient priors from dictionaries to NumPy arrays so arithmetic with class priors works during blending. This resolves the TypeError, restores valid probability calculations, and ensures the script creates a correct submission CSV. No other logic is changed, keeping the original approach intact.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

NEEDTRAIN = False


def resolve_data_dir():
    candidates = [
        os.path.join("data", "hms-harmful-brain-activity-classification"),
        os.path.join("kaggle", "input", "hms-harmful-brain-activity-classification"),
        os.path.join("input", "hms-harmful-brain-activity-classification"),
        "/kaggle/input/hms-harmful-brain-activity-classification",
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("Could not locate the dataset directory.")


LOAD_DATA_FROM = resolve_data_dir()

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
train_df = pd.read_csv(train_path)

global_votes = train_df[TARGETS].sum().values.astype(np.float32)
class_prior = global_votes / global_votes.sum()

patient_groups = train_df.groupby("patient_id")[TARGETS].sum()
patient_prior_dict = (
    patient_groups.div(patient_groups.sum(axis=1), axis=0)
    .fillna(0)
    .to_dict(orient="index")
)
patient_prior_dict = {
    pid: np.array(list(prior_dict.values()), dtype=np.float32)
    for pid, prior_dict in patient_prior_dict.items()
}

patient_counts = train_df.groupby("patient_id")[TARGETS].sum()
patient_counts_sum = patient_counts.sum(axis=1).to_dict()

if not NEEDTRAIN:
    rng = np.random.default_rng(42)
    shuffled_idx = rng.permutation(len(train_df))
    split_point = int(0.8 * len(train_df))
    train_idx, val_idx = shuffled_idx[:split_point], shuffled_idx[split_point:]

    train_split = train_df.iloc[train_idx]
    val_split = train_df.iloc[val_idx]

    patient_groups_split = train_split.groupby("patient_id")[TARGETS].sum()
    patient_prior_dict_split = (
        patient_groups_split.div(patient_groups_split.sum(axis=1), axis=0)
        .fillna(0)
        .to_dict(orient="index")
    )
    patient_prior_dict_split = {
        pid: np.array(list(prior_dict.values()), dtype=np.float32)
        for pid, prior_dict in patient_prior_dict_split.items()
    }

    patient_counts_split = train_split.groupby("patient_id")[TARGETS].sum()
    patient_counts_sum_split = patient_counts_split.sum(axis=1).to_dict()

    val_votes = val_split[TARGETS].values.astype(np.float32)
    val_sums = val_votes.sum(axis=1, keepdims=True)
    val_probs = np.where(
        val_sums > 0,
        val_votes / val_sums,
        np.full_like(val_votes, 1.0 / len(TARGETS)),
    )

    def blended_pred(pid, count, patient_prior, alpha):
        if patient_prior is not None:
            blended = (patient_prior * count + class_prior * alpha) / (count + alpha)
        else:
            blended = class_prior
        return blended

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_alpha = alphas[0]
    best_kl = np.inf

    for alpha in alphas:
        preds = []
        for pid in val_split["patient_id"].values:
            prior = patient_prior_dict_split.get(pid)
            cnt = patient_counts_sum_split.get(pid, 0.0)
            pred = blended_pred(pid, cnt, prior, alpha)
            preds.append(pred)
        preds = np.stack(preds, axis=0).astype(np.float32)
        preds /= preds.sum(axis=1, keepdims=True)

        kl_matrix = np.where(
            val_probs > 0,
            val_probs * np.log(val_probs / preds),
            0.0,
        )
        kl = kl_matrix.mean()
        if kl < best_kl:
            best_kl = kl
            best_alpha = alpha

    alpha = best_alpha  # use the data‑driven smoothing factor
    print(f"Selected smoothing factor α = {alpha:.3f} (validation KL ≈ {best_kl:.5f})")
else:
    alpha = 10.0  # fallback (should never be used when NEEDTRAIN=False)




## === cell 1
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test = pd.read_csv(test_path)
test["sign_id"] = test.index.values
print("Test shape", test.shape)

preds_all = np.empty((len(test), len(TARGETS)), dtype=np.float32)

for idx, row in test.iterrows():
    pid = row["patient_id"]
    if pid in patient_prior_dict:
        patient_pred = patient_prior_dict[pid]  # already a numpy array
        count = patient_counts_sum.get(pid, 0.0)
        blended = (patient_pred * count + class_prior * alpha) / (count + alpha)
        preds_all[idx] = blended
    else:
        preds_all[idx] = class_prior

preds_all /= preds_all.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for i, col in enumerate(TARGETS):
    sub[col] = preds_all[:, i]

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)
print("Submission shape", sub.shape)
sub.head()
