# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3263711182346715

# 6. Current score

1.19354

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I wrap the TensorFlow import in a safe try/except so the notebook can run without the protobuf error (TensorFlow isn’t needed for inference), and I correct the Pandas `sum` call that used the unsupported `keepdims` argument. The division now keeps the column dimension explicitly, preventing shape mismatches. These minimal fixes let the script execute end‑to‑end and generate a valid `submission.csv` whose rows sum to one, moving the solution toward the target score.'
- What this solution (achieved 1.41937) has done: 'I replace the naive equal‑row averaging with a vote‑weighted class probability calculation, which uses the total number of annotator votes per class. This keeps the same overall workflow but yields a more accurate probability distribution, moving the KL‑divergence score closer to the target (lower is better). No other code changes are needed.'
- What this solution (achieved 1.68479) has done: 'I replace the naïve global‑average baseline with a patient‑wise probability estimate: for each patient in the training set I compute the vote‑based class distribution and use it for any test rows that share the same patient_id, falling back to the overall class distribution when the patient is unseen. This keeps the same simple inference pipeline while providing a more informative probability estimate, which should lower the KL‑divergence (lower is better) and still produce a valid `submission.csv` where each row sums to 1.'
- What this solution (achieved 1.41937) has done: 'I fix the script so it runs without errors and improves the probability estimates by first using per‑eeg‑id vote distributions (which are more specific than patient‑level) and falling back to patient‑level and finally the global distribution. This keeps the original workflow while yielding lower KL‑divergence and guarantees a valid `submission.csv` whose rows sum to 1.'
- What this solution (achieved 1.68479) has done: 'The fix disables TensorFlow configuration that crashes on import and adds proper patient‑level fallback when per‑eeg probabilities are missing, ensuring every test row gets a valid probability distribution and improving the KL‑divergence score. The rest of the pipeline and model architecture remain unchanged.'
- What this solution (achieved 0.78112) has done: 'I wrapped all TensorFlow‑related imports and model‑building code in a check that only runs when TensorFlow is successfully imported, preventing AttributeErrors when TF isn’t available. I also added a tiny Laplace‑style smoothing to the final probability matrix (adding 0.01 to every entry before renormalising) so that no class gets a zero probability, which reduces KL‑divergence and moves the score toward the target while keeping the original inference logic unchanged.'
- What this solution (achieved 1.05318) has done: 'The script was failing because it could not locate the CSV files using the hard‑coded `data/hms‑harmful‑brain‑activity‑classification` path. I added a small helper that searches several common Kaggle directories (the relative `data/…` folder, the standard `/kaggle/input` location, and the current working directory) and loads the files from the first place they exist. This fixes the `FileNotFoundError` and lets the pipeline run end‑to‑end, producing a valid `submission.csv` whose rows sum to one.'
- What this solution (achieved 0.78112) has done: 'The update fixes the probability blending logic: it now prefers the most specific per‑eeg distribution, falls back to the patient‑level distribution, and finally to the global class distribution, avoiding unnecessary averaging that degrades the KL score. A small Laplace‑style smoothing (ε = 0.01) is added before renormalisation to prevent zero probabilities, which further reduces the KL divergence while keeping the core workflow intact.'
- What this solution (achieved 1.19354) has done: 'I reduced the Laplace‑style smoothing constant from 0.01 to a negligible 1e‑8 so the predicted probabilities stay as close as possible to the empirically derived vote distributions, which lowers the KL‑divergence score while keeping the same fallback hierarchy and ensuring rows still sum to 1. The rest of the pipeline is unchanged, and the script now writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

_possible_data_dirs = [
    os.path.join("data", "hms-harmful-brain-activity-classification"),
    os.path.join("/kaggle", "input", "hms-harmful-brain-activity-classification"),
    os.path.join("/kaggle", "working", "hms-harmful-brain-activity-classification"),
    ".",
]


def _find_data_dir():
    for d in _possible_data_dirs:
        if os.path.isdir(d):
            if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                os.path.join(d, "test.csv")
            ):
                return d
    raise FileNotFoundError(
        "Could not locate competition data directory. Checked: "
        + ", ".join(_possible_data_dirs)
    )


LOAD_DATA_FROM = _find_data_dir()
NEEDTRAIN = False
DATATYPE = ""



## === cell 1
try:
    import tensorflow as tf
except Exception:
    tf = None

if tf is not None:
    from tensorflow.keras.applications import EfficientNetB0

    def temporal_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 3), strides=(1, 1), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 3), strides=(1, 2), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def external_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters,
            dilation_rate=(4, 1),
            kernel_size=(4, 1),
            strides=(1, 1),
            padding="valid",
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def internal_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(4, 1), strides=(4, 1), padding="valid"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def build_model():
        inp = list()
        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        return tf.keras.Model(inputs=inp, outputs=y)

else:
    print("TensorFlow not available – model architecture definitions are skipped.")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # the six vote columns
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

if not NEEDTRAIN:
    train_votes = df[TARGETS].astype(float)
    global_class_probs = train_votes.sum(axis=0)
    global_class_probs = global_class_probs / global_class_probs.sum()
    global_class_probs = global_class_probs.values  # numpy array

    patient_votes = train_votes.groupby(df["patient_id"]).sum()
    patient_totals = patient_votes.sum(axis=1).replace(0, np.nan)
    patient_probs = patient_votes.div(patient_totals, axis=0).fillna(0)

    eeg_votes = train_votes.groupby(df["eeg_id"]).sum()
    eeg_totals = eeg_votes.sum(axis=1).replace(0, np.nan)
    eeg_probs = eeg_votes.div(eeg_totals, axis=0).fillna(0)

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    print("Test shape:", test.shape)

    test_with_probs = test.merge(
        eeg_probs,
        left_on="eeg_id",
        right_index=True,
        how="left",
        suffixes=("", "_eeg"),
    )
    test_with_probs = test_with_probs.merge(
        patient_probs,
        left_on="patient_id",
        right_index=True,
        how="left",
        suffixes=("", "_patient"),
    )

    for idx, col in enumerate(TARGETS):
        col_patient = f"{col}_patient"
        if col not in test_with_probs.columns:
            test_with_probs[col] = np.nan

        test_with_probs[col] = test_with_probs[col].fillna(test_with_probs[col_patient])
        test_with_probs[col] = test_with_probs[col].fillna(global_class_probs[idx])

    prob_matrix = test_with_probs[TARGETS].values.astype(float)

    eps = 1e-8
    prob_matrix = prob_matrix + eps
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(prob_matrix, row_sums, where=row_sums != 0)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = prob_matrix
    sub.to_csv("submission.csv", index=False)
    print("Submission shape:", sub.shape)
    print(sub.head())
