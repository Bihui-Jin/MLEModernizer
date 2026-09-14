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

0.3315990003082694

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I safeguard the TensorFlow import (which fails due to a protobuf version mismatch) by loading it only when training is actually requested, and I make the test‑file loading robust by checking several possible locations before reading. This eliminates the runtime errors and ensures a valid `submission.csv` with correctly normalised probabilities is written.'
- What this solution (achieved 1.40995) has done: 'I replace the global‑class probability baseline with a uniform probability for each of the six vote columns. Using a less‑confident, equal distribution reduces over‑confident KL penalties and moves the score nearer the target while keeping all existing logic and file handling unchanged.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform probability baseline with the empirical class distribution derived from the training data (`class_probs`). Using the observed class frequencies gives a more realistic prior and should lower the KL divergence, moving the score closer to the target while keeping all existing logic unchanged.'
- What this solution (achieved 1.41937) has done: 'The change adds a per‑`eeg_id` probability estimate derived from the training votes, and uses it as the baseline for each test record. If a test `eeg_id` was not seen in training, the overall class distribution is used as before. This gives more informative priors, reduces the KL‑divergence, and still writes a valid normalized `submission.csv` without altering any model‑training logic.'
- What this solution (achieved 1.41937) has done: 'We blend the per‑`eeg_id` vote‑based probabilities with the overall class distribution instead of using either one alone. By giving the overall distribution a modest weight (e.g., 30 %) we reduce over‑confident priors for unseen ids, which tends to lower the KL‑divergence and move the score closer to the target. The rest of the script remains unchanged, preserving all original logic and file handling.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

PLATFORM = "local"
NEEDTRAIN = False  # no training, so TensorFlow is not required
MIX = False
DATATYPE = ""
BATCHSIZE = 32
NSPLIT = 1
LOAD_MODELS_FROM = ""
STAGETEST = 0

print("Running submission script – training disabled.")




## === cell 1
def locate_file(rel_path):
    candidates = [
        os.path.join("data", rel_path),
        os.path.join("input", rel_path),
        os.path.join("/kaggle/input", rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Unable to locate {rel_path} in any known location.")


train_path = locate_file(
    os.path.join("hms-harmful-brain-activity-classification", "train.csv")
)
df = pd.read_csv(train_path)
TARGETS = df.columns[-6:]  # the six vote columns
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

class_counts = df[TARGETS].sum()
class_probs = class_counts / class_counts.sum()
print("Baseline class probabilities:", class_probs.values)

eeg_class_probs = df.groupby("eeg_id")[TARGETS].sum()
row_sums = eeg_class_probs.sum(axis=1).replace(0, np.nan)
eeg_class_probs = eeg_class_probs.div(row_sums, axis=0)



## === cell 2
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    else:
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"

    from scipy import signal
    import time
    import tensorflow as tf  # imported lazily for training only



## === cell 3
if NEEDTRAIN:

    class DataGenerator(tf.keras.utils.Sequence):
        """Generates data for Keras (placeholder – not used in submission)"""

        def __init__(
            self,
            data,
            batch_size=32,
            shuffle=False,
            augment=False,
            mode="train",
            specs=None,
            eegs=None,
            imgs=None,
            stfts=None,
            targets=None,
        ):
            self.targets = targets
            self.cmin = -4
            self.cmax = 6
            self.cmaps = plt.cm.cividis(np.linspace(0, 1, 256))[:, :3]
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.imgs = imgs
            self.stfts = stfts
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.data) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[index * self.batch_size : (index + self.batch_size)]
            x, y = self.__data_generation(indexes)
            return x, y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            x = []
            y = np.zeros((len(indexes), len(self.targets)), dtype="float32")
            return x, y




## === cell 4
if NEEDTRAIN:
    LR_MAX1 = 1e-3
    EPOCHS1 = 6

    def lrfn1(epoch):
        xx = [LR_MAX1, LR_MAX1, LR_MAX1]
        return xx[epoch] if epoch < len(xx) else 1e-4

    LR1 = tf.keras.callbacks.LearningRateScheduler(lrfn1, verbose=True)

    LR_MAX2 = 1e-4
    EPOCHS2 = 2

    def lrfn2(epoch):
        xx = [LR_MAX2, LR_MAX2, LR_MAX2, 1e-4, 1e-4]
        return xx[epoch] if epoch < len(xx) else 1e-4

    LR2 = tf.keras.callbacks.LearningRateScheduler(lrfn2, verbose=True)

    LR_MAX3 = 1e-4
    EPOCHS3 = 6

    def lrfn3(epoch):
        xx = [LR_MAX3, LR_MAX3, LR_MAX3]
        return xx[epoch] if epoch < len(xx) else 1e-5

    LR3 = tf.keras.callbacks.LearningRateScheduler(lrfn3, verbose=True)



## === cell 5
if NEEDTRAIN:

    def build_model(TARGETS_PRETRAIN):
        inp = []
        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(None, None, 3, 4))
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [inp_spe[:, :, :, :, i] for i in range(4)]
            )
            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_shape=None
            )
            base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = tf.nn.l2_normalize(x_spe, -1)
            inp.append(inp_spe)
            y = x_spe
        else:
            y = tf.keras.layers.Input(shape=(1,))  # dummy placeholder
        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN), activation="softmax", dtype="float32"
        )(y)
        return tf.keras.Model(inputs=inp, outputs=y)




## === cell 6
if NEEDTRAIN:

    def my_loss(y_true, y_pred):
        y_pred1 = tf.reduce_sum(y_pred[:, 5:6], 1, keepdims=True)
        y_pred2 = y_pred[:, 0:5]
        y_pred = tf.concat((y_pred2, y_pred1), axis=1)
        return tf.keras.losses.KLD(y_true, y_pred)




## === cell 7
test_path = locate_file(
    os.path.join("hms-harmful-brain-activity-classification", "test.csv")
)
test = pd.read_csv(test_path)
print("Test shape:", test.shape)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

eeg_probs_df = eeg_class_probs.reset_index()
sub = sub.merge(eeg_probs_df, on="eeg_id", how="left")

baseline_probs = class_probs.values.astype(float)  # shape (6,)

for col, prob in zip(TARGETS, baseline_probs):
    if col not in sub.columns:
        sub[col] = np.nan
    sub[col].fillna(prob, inplace=True)

alpha = 0.3  # 30 % per‑eeg, 70 % overall
overall_array = np.tile(baseline_probs, (sub.shape[0], 1))
per_eeg_array = sub[TARGETS].values
blended = alpha * per_eeg_array + (1 - alpha) * overall_array
sub[TARGETS] = blended

row_sums = sub[TARGETS].sum(axis=1).replace(0, np.nan)
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("Submission head:")
print(sub.head())
