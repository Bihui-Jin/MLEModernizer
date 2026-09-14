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

0.2730836967957302

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The changes add a small multiprocessing helper to preprocess all test EEG files in parallel, replacing the original per‑sample serial loop. By loading and filtering each parquet file concurrently we eliminate the dominant I/O‑bound bottleneck while keeping the exact same filtering, padding, and slicing logic, so model predictions remain unchanged. The rest of the pipeline (model loading, data generation, prediction aggregation) is left untouched.'
- What this solution (achieved 1.41937) has done: 'I force TensorFlow to be disabled (so the script skips model loading), move the `DataGenerator` class before it is used, and keep the rest of the logic unchanged. This removes the import‑related crash and the `NameError` for `DataGenerator`, allowing the pipeline to run end‑to‑end and produce a valid `submission.csv` using the baseline average probabilities.'
- What this solution (achieved 1.68479) has done: 'The fix adds a robust lookup for the dataset directory so the code can locate train.csv (and the other CSV files) whether the data lives under `data/...` or the standard Kaggle `/kaggle/input/...` path. It falls back to the first existing directory that contains the required files. This resolves the FileNotFoundError and lets the baseline patient‑level averaging run end‑to‑end, producing a valid `submission.csv` with correctly normalised probabilities. No core modelling logic is altered, keeping the score unchanged while ensuring a proper submission file.'
- What this solution (achieved 1.68479) has done: 'I add a fallback that uses the exact per‑eeg vote distribution when the test eeg_id was seen in the training metadata, then fall back to the patient‑level average, and finally to the global average. This small lookup change keeps the core logic unchanged while giving more specific probabilities for any overlapping eeg_id, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.82922) has done: 'I keep the overall workflow unchanged but add a simple smoothing step when falling back to patient‑level averages. Instead of using the raw patient distribution (which can be noisy for patients with few annotations), I blend it with the global class distribution using a small smoothing constant. This modest change preserves the core lookup logic while making the predictions more calibrated, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.68479) has done: 'I lower the smoothing effect so the model relies more on patient‑specific vote distributions (or uses them outright). This reduces the overly‑uniform global prior that was inflating the KL‑divergence, moving the score closer to the lower target while keeping the original lookup logic intact.'
- What this solution (achieved 1.41937) has done: 'I replace the per‑eeg / per‑patient lookup with a single use of the global class probability distribution for every test record. This removes noisy patient‑specific noise and moves the KL‑divergence closer to the target lower score while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import scipy.signal as signal
import multiprocessing as mp


def _find_data_dir():
    possible_roots = [
        pathlib.Path("data/hms-harmful-brain-activity-classification"),
        pathlib.Path("/kaggle/input/hms-harmful-brain-activity-classification"),
        pathlib.Path("/kaggle/working/hms-harmful-brain-activity-classification"),
    ]
    for root in possible_roots:
        if (root / "train.csv").exists() and (root / "test.csv").exists():
            return str(root)
    cur = pathlib.Path(".")
    if (cur / "train.csv").exists() and (cur / "test.csv").exists():
        return str(cur)
    raise FileNotFoundError(
        "Could not locate train.csv and test.csv in any expected location."
    )


LOAD_DATA_FROM = _find_data_dir()

TF_AVAILABLE = False  # TensorFlow not required for this baseline
SFREQ = 200  # original EEG sampling frequency
RSFREQ = 200  # desired resample frequency (keep same to skip resampling)
filter_range = None  # no band‑pass filtering in the baseline

BRAIN = []

DATATYPE = "eeg"

EEG_CHANNEL_USED = 19
EEG_MULTIPLY = 1
EEG_LENGTH = 50  # seconds
EEG_LENGTH_USED = 50
SPE_HIGH = SPE_WIDE = IMG_HIGH = IMG_WIDE = 0  # not used

TEST_BATCHSIZE = 256

eegs_test = {}
spectrograms_test = {}
stfts_test = {}
imgs_test = {}




## === cell 1
def _process_single_eeg(args):
    """
    Helper for multiprocessing: loads a single EEG parquet, computes the
    channel differences, optional resampling, padding, filtering,
    and returns (eeg_id, processed_eeg_array).
    This function is retained for compatibility but will not be invoked
    in the simplified inference path.
    """
    eeg_id, path, b, a, SFREQ, RSFREQ = args
    eeg_default = pd.read_parquet(os.path.join(path, f"{eeg_id}.parquet"))

    eeg = []
    for channel in BRAIN:
        diff = (
            eeg_default.loc[:, channel.split("-")[0]]
            - eeg_default.loc[:, channel.split("-")[1]]
        ).values
        diff[np.isnan(diff)] = 0
        eeg.append(np.reshape(diff, (1, -1)))
    if eeg:
        eeg = np.concatenate(eeg, axis=0)
    else:
        eeg = np.zeros((EEG_CHANNEL_USED, int(EEG_LENGTH * RSFREQ)), dtype=np.float32)

    if SFREQ != RSFREQ:
        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

    eegshape = eeg.shape[1]
    eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
    if filter_range is not None:
        eeg = signal.filtfilt(b, a, eeg, axis=1)
    eeg = eeg[:, eegshape : eegshape * 2]

    eeg = np.array(eeg, dtype=np.float32)
    return eeg_id, eeg




## === cell 2
_BaseSequence = object  # TensorFlow is disabled, use simple base class


class DataGenerator(_BaseSequence):
    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=False,
        sample_weights=False,
        mode="train",
        eegs=None,
        stfts=None,
        specs=None,
        imgs=None,
        stage=2,
    ):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stfts = stfts
        self.specs = specs
        self.imgs = imgs
        self.stage = stage
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        return x, y, sample_weights

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if "spe" in DATATYPE:
            x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (
                    len(indexes),
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                dtype="float32",
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]

            if self.mode == "test":
                r_spe = r_eeg = r_stft = 0
            else:
                r_spe = r_eeg = r_stft = 0

            if "eeg" in DATATYPE:
                eeg = self.eegs[row.eeg_id][
                    :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]
                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )
                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]
                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eeg = eeg + 1024
                eeg = eeg / 2048 * 255
                x_eeg[j] = eeg

            if "spe" in DATATYPE:
                pass
            if "img" in DATATYPE:
                pass

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                sample_weights[j] = 1

        x = {}
        if "spe" in DATATYPE:
            x["spe"] = x_spe
        if "eeg" in DATATYPE:
            x["eeg"] = x_eeg
        if "stft" in DATATYPE:
            x["stft"] = None
        if "img" in DATATYPE:
            x["img"] = x_img

        return x, y, sample_weights




## === cell 3
if __name__ == "__main__":
    df_train_meta = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
    TARGETS = df_train_meta.columns[-6:]  # last six columns are vote counts

    global_votes = df_train_meta[TARGETS].sum().values.astype(np.float32)
    global_probs = global_votes / global_votes.sum()

    eeg_group = df_train_meta.groupby("eeg_id")[list(TARGETS)].sum()
    eeg_probs = eeg_group.div(eeg_group.sum(axis=1), axis=0).fillna(global_probs)

    patient_group = df_train_meta.groupby("patient_id")[list(TARGETS)].sum()
    patient_probs = patient_group.div(patient_group.sum(axis=1), axis=0).fillna(
        global_probs
    )

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    print("Test shape", test.shape)

    preds = []
    for _, row in test.iterrows():
        eid = row["eeg_id"]
        pid = row["patient_id"]
        if eid in eeg_probs.index:
            prob = eeg_probs.loc[eid].values.astype(np.float32)
        elif pid in patient_probs.index:
            prob = patient_probs.loc[pid].values.astype(np.float32)
        else:
            prob = global_probs
        prob = prob / prob.sum()
        preds.append(prob)

    preds_all = np.vstack(preds).astype(np.float32)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"]})
    sub[TARGETS] = preds_all
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3048844681.py in <cell line: 0>()
     10     # Per‑eeg distribution: average votes for each eeg_id
     11     eeg_group = df_train_meta.groupby("eeg_id")[list(TARGETS)].sum()
---> 12     eeg_probs = eeg_group.div(eeg_group.sum(axis=1), axis=0).fillna(global_probs)
     13 
     14     # Per‑patient distribution: average votes for each patient_id

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7438                 new_data = self.where(self.notna(), value)._mgr
   7439             else:
-> 7440                 raise ValueError(f"invalid fill value with a {type(value)}")
   7441 
   7442         result = self._constructor_from_mgr(new_data, axes=new_data.axes)

ValueError: invalid fill value with a <class 'numpy.ndarray'>
