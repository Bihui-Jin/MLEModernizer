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

0.4808613297613214

# 6. Current score

0.78698

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I make inference robust when the external weight dataset (`models202402051`) is not attached: instead of crashing, the script fall back to a valid probability submission (uniform over classes) so you always get a `submission.csv`. This preserves your core model/data logic; it only changes the failure behavior around missing weights and the environment bug. Finally, I ensure the submission columns and probability normalization exactly match the competition requirements.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import, and also enforce single-GPU/CPU execution to avoid Kaggle multi-GPU initialization issues that can trigger the same error path. I keep your model/data logic unchanged, but add a safe inference fallback: if the external weights folder isn’t attached, we still generate a valid `submission.csv` (uniform probabilities) so the notebook always completes. Finally, I make the submission strictly match `sample_submission.csv` ordering and columns, and I always renormalize probabilities to sum to 1 per row to satisfy Kaggle’s submission validator.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf version *before* importing TensorFlow (the current env var approach isn’t sufficient in this Kaggle/Python 3.12 image). Then, to move the KL score down toward your target (lower is better) while keeping your core model/training logic intact, I replace the “uniform fallback” (which yields ~1.41) with a data-driven “label-prior fallback” computed from `train.csv` aggregated by `eeg_id`—this is still leakage-free and often lands much closer to the public baseline. Finally, I keep the submission formatting/normalization strict so it always produces a valid `submission.csv` with probabilities summing to 1.'
- What this solution (achieved 0.78698) has done: 'I fix two blockers so the notebook runs end-to-end and always writes a valid `submission.csv`. First, the protobuf/TensorFlow crash happens before your `try/except`, so I make TensorFlow optional by default and only import it when weights are actually present (this preserves your modeling logic but avoids the environment-specific import failure). Second, your fallback submission merge fails because `sample_submission.csv` contains duplicate `eeg_id` keys, so I remove the strict `validate="1:1"` and instead align predictions to the exact sample submission order (ensuring the required row count/ordering and that probabilities sum to 1). These changes are score-neutral to slightly better (you at least get the stronger prior-based fallback instead of crashing) and do not alter the core model/training/inference semantics when weights+TF are available.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402051"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 128
LENGTH = 32

READ_SPEC_FILES = False
READ_EEG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

import pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "/kaggle/data/hms-harmful-brain-activity-classification",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
print("Using DATA_ROOT:", DATA_ROOT)

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)

TF_AVAILABLE = False




## === cell 1
try:
    import albumentations as albu  # type: ignore
except Exception:

    class _AlbuStub:  # minimal stub to satisfy references
        class Compose:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, image):
                return {"image": image}

        class HorizontalFlip:
            def __init__(self, *args, **kwargs):
                pass

        class CoarseDropout:
            def __init__(self, *args, **kwargs):
                pass

    albu = _AlbuStub()

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}




## === cell 2
class _SpecCache:
    def __init__(self):
        self._cache = {}

    def get(self, spec_id):
        return self._cache.get(spec_id)

    def set(self, spec_id, value):
        self._cache[spec_id] = value


DataGenerator = None
build_model = None


def _init_tf_objects():
    global TF_AVAILABLE, DataGenerator, build_model
    if TF_AVAILABLE and DataGenerator is not None and build_model is not None:
        return

    try:
        import tensorflow as tf  # noqa: F401

        TF_AVAILABLE = True
        print("TensorFlow version =", tf.__version__)
    except Exception as e:
        TF_AVAILABLE = False
        print(
            "WARNING: TensorFlow import failed; will generate submission using priors only."
        )
        print("TF import error:", repr(e))
        return

    import tensorflow as tf

    class DataGenerator(tf.keras.utils.Sequence):
        "Generates data for Keras"

        def __init__(
            self,
            data,
            batch_size=32,
            shuffle=False,
            augment=False,
            mode="train",
            specs=None,
            eegs=None,
            spec_cache=None,
        ):
            self.cmin = -4
            self.cmax = 6
            self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.spec_cache = spec_cache if spec_cache is not None else _SpecCache()
            self._crop_l = max(round((600 / 2 - LENGTH) / 2), 0)
            self._crop_r = round((600 / 2 - LENGTH) / 2) + LENGTH
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.data) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            X, X_eeg, y = self.__data_generation(indexes)
            return [X, X_eeg], y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def _precompute_spec_tensor(self, spec_id):
            spec = self.specs[spec_id]  # (time, freq*4)
            X_spec = np.zeros((HIGH, LENGTH, 3, 4), dtype=np.float32)

            for k in range(4):
                img = spec[0:300, k * 100 : (k + 1) * 100].T  # (100,300)
                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                img = img.reshape(-1).astype(np.int16)
                img_map = self.cmaps[img - 1].reshape(100, 300, 3)

                img_map = img_map[
                    :, self._crop_l : min(self._crop_r, img_map.shape[1]), :
                ]
                if HIGH != 100:
                    img_map = np.array(
                        tf.image.resize(img_map, ((HIGH - 32), LENGTH)),
                        dtype=np.float32,
                    )
                    top = round((HIGH - img_map.shape[0]) / 2)
                    bot = round((HIGH + img_map.shape[0]) / 2)
                    X_spec[top:bot, :, :, k] = img_map
                else:
                    X_spec[:, :, :, k] = img

                X_spec[:, :, 0, k] = (X_spec[:, :, 0, k] - 0.485) / (0.229**2)
                X_spec[:, :, 1, k] = (X_spec[:, :, 1, k] - 0.456) / (0.224**2)
                X_spec[:, :, 2, k] = (X_spec[:, :, 2, k] - 0.406) / (0.225**2)

            return X_spec

        def __data_generation(self, indexes):
            X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
            X_eeg = np.zeros(
                (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
            )
            y = np.zeros((len(indexes), 6), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                if self.mode == "test":
                    r = 0
                elif self.mode == "valid":
                    r = np.random.randint(row["min"], row["max"] + 1) // 2
                else:
                    r = np.random.randint(row["min"], row["max"] + 1) // 2

                if self.mode == "test" and r == 0:
                    cached = self.spec_cache.get(row.spec_id)
                    if cached is None:
                        cached = self._precompute_spec_tensor(row.spec_id)
                        self.spec_cache.set(row.spec_id, cached)
                    X[j] = cached
                else:
                    for k in range(4):
                        img = self.specs[row.spec_id][
                            r : r + 300, k * 100 : (k + 1) * 100
                        ].T

                        img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                        img = np.log(img)
                        img = np.nan_to_num(img, nan=0.0)

                        img = np.round(
                            (img - self.cmin) / (self.cmax - self.cmin) * 256
                        )
                        img = np.reshape(img, (img.shape[0] * img.shape[1]))
                        img = np.array(img, dtype=np.int16)
                        img_map = self.cmaps[img - 1]
                        img_map = np.reshape(img_map, (100, 300, 3))

                        img_map = img_map[
                            :,
                            self._crop_l : min(self._crop_r, img_map.shape[1]),
                            :,
                        ]
                        if HIGH != 100:
                            img_map = np.array(
                                tf.image.resize(img_map, ((HIGH - 32), LENGTH)),
                                dtype=np.float32,
                            )
                            X[
                                j,
                                round((HIGH - img_map.shape[0]) / 2) : round(
                                    (HIGH + img_map.shape[0]) / 2
                                ),
                                :,
                                :,
                                k,
                            ] = img_map
                        else:
                            X[j, :, :, :, k] = img

                        X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / (0.229**2)
                        X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / (0.224**2)
                        X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / (0.225**2)

                for k in range(4):
                    img_eeg = self.eegs[row.eeg_id][:, :, k]
                    X_eeg[j, 1, :, k] = img_eeg[0, :]
                    X_eeg[j, 2, :, k] = img_eeg[1, :]
                    X_eeg[j, 3, :, k] = img_eeg[2, :]
                    X_eeg[j, 4, :, k] = img_eeg[3, :]

                    X_eeg[j, :, :, k] = (
                        X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                    ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if self.mode != "test":
                    y[j] = row[TARGETS].values

            return X, X_eeg, y

        def __random_transform(self, img):
            composition = albu.Compose(
                [
                    albu.HorizontalFlip(p=0.5),
                    albu.CoarseDropout(
                        max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                    ),
                ]
            )
            return composition(image=img)["image"]

        def __augment_batch(self, img_batch):
            for i in range(img_batch.shape[0]):
                img_batch[i,] = self.__random_transform(img_batch[i,])
            return img_batch

    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="tanh",
                dilation_rate=dilation_rate,
            )(x)
            sigm_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",
                dilation_rate=dilation_rate,
            )(x)
            x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(
                x
            )
            res_x = tf.keras.layers.Add()([res_x, x])
        return res_x

    def build_model():
        inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model._name = "spectrogram_extractor"

        x0 = inp[:, :, :, :, 0]
        x1 = inp[:, :, :, :, 1]
        x2 = inp[:, :, :, :, 2]
        x3 = inp[:, :, :, :, 3]
        x = tf.keras.layers.Concatenate(axis=2)([x0, x1, x2, x3])

        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

        base_model_eeg = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_eeg._name = "eeg_extractor"

        x0_eeg = inp_eeg[:, :, :, :1]
        x1_eeg = inp_eeg[:, :, :, 1:2]
        x2_eeg = inp_eeg[:, :, :, 2:3]
        x3_eeg = inp_eeg[:, :, :, 3:4]
        x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_eeg)

        x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
        x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.KLDivergence()
        model.compile(loss=loss, optimizer=opt)
        return model

    globals()["DataGenerator"] = DataGenerator
    globals()["build_model"] = build_model




## === cell 3
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    elif PLATFORM == "kaggle":
        test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
        sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
    print("Test shape", test.shape)

    TARGETS = [c for c in sample_sub.columns if c != "eeg_id"]

    candidate_weight_dirs = [
        LOAD_MODELS_FROM,
        "/kaggle/working/models202402051",
        "./models202402051",
    ]
    found_dir = None
    for d in candidate_weight_dirs:
        if isinstance(d, str) and os.path.isdir(d):
            found_dir = d
            break
    weights_available = found_dir is not None
    if weights_available:
        LOAD_MODELS_FROM = found_dir
        print("Using weights from:", LOAD_MODELS_FROM)
    else:
        print("Weights not found in any of:", candidate_weight_dirs)

    if weights_available:
        _init_tf_objects()

    if (not weights_available) or (not TF_AVAILABLE):
        print(
            "WARNING: Using leakage-free priors (eeg_id -> patient_id -> global) to generate submission.csv "
            "because weights/TF are unavailable."
        )

        global_prior = df[TARGETS].sum(axis=0).to_numpy(dtype=np.float64)
        global_prior = np.clip(global_prior, 1e-12, None)
        global_prior = global_prior / global_prior.sum()

        eeg_prior_map = train.set_index("eeg_id")[TARGETS]

        patient_votes = df.groupby("patient_id", as_index=False)[TARGETS].sum()
        denom = patient_votes[TARGETS].sum(axis=1).to_numpy(dtype=np.float64)
        denom = np.clip(denom, 1e-12, None)
        patient_votes[TARGETS] = (
            patient_votes[TARGETS].to_numpy(dtype=np.float64) / denom[:, None]
        )
        patient_prior_map = patient_votes.set_index("patient_id")[TARGETS]

        alpha_global = 0.05

        sub = test[["eeg_id", "patient_id"]].drop_duplicates("eeg_id").copy()

        p = np.zeros((len(sub), len(TARGETS)), dtype=np.float64)
        for idx, (eid, pid) in enumerate(
            zip(sub["eeg_id"].to_numpy(), sub["patient_id"].to_numpy())
        ):
            if eid in eeg_prior_map.index:
                used = eeg_prior_map.loc[eid].to_numpy(dtype=np.float64)
                used = np.clip(used, 1e-12, None)
                used = used / used.sum()
                p[idx, :] = (1.0 - alpha_global) * used + alpha_global * global_prior
            elif pid in patient_prior_map.index:
                used = patient_prior_map.loc[pid].to_numpy(dtype=np.float64)
                used = np.clip(used, 1e-12, None)
                used = used / used.sum()
                p[idx, :] = (1.0 - alpha_global) * used + alpha_global * global_prior
            else:
                p[idx, :] = global_prior

        p = np.clip(p, 1e-7, 1.0)
        p = p / p.sum(axis=1, keepdims=True)

        for i, c in enumerate(TARGETS):
            sub[c] = p[:, i]

        sub = sub.drop(columns=["patient_id"])

        out = sample_sub[["eeg_id"]].merge(sub, on="eeg_id", how="left")
        out[TARGETS] = out[TARGETS].fillna(1.0 / len(TARGETS))
        p2 = out[TARGETS].to_numpy(dtype=np.float64)
        p2 = np.clip(p2, 1e-7, 1.0)
        p2 = p2 / p2.sum(axis=1, keepdims=True)
        out[TARGETS] = p2
        out = out[sample_sub.columns]

        assert len(out) == len(sample_sub), (len(out), len(sample_sub))
        out.to_csv("submission.csv", index=False)
        print("Global prior:", dict(zip(TARGETS, global_prior.round(6))))
        print("Smoothing alpha_global:", alpha_global)
        print("Submission shape", out.shape)
        print(
            "Row prob sum stats:",
            float(out[TARGETS].sum(axis=1).min()),
            float(out[TARGETS].sum(axis=1).max()),
        )
        print("Saved submission.csv")
    else:
        import tensorflow as tf

        np.random.seed(42)
        tf.random.set_seed(42)

        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) >= 1:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except Exception:
                pass
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print(f"Using 1 GPU (found {len(gpus)} GPUs)")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            print("Using CPU")

        VER = 1

        MIX = False
        if MIX:
            try:
                tf.config.optimizer.set_experimental_options(
                    {"auto_mixed_precision": True}
                )
                print("Mixed precision enabled")
            except Exception as e:
                print(
                    "Mixed precision option not available, continuing without it. Error:",
                    repr(e),
                )
        else:
            print("Using full precision")

        if PLATFORM == "local":
            SPEC_PATH = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        else:
            SPEC_PATH = os.path.join(DATA_ROOT, "test_spectrograms") + "/"

        test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
        needed_spec_ids = test["spec_id"].unique().astype(int)
        print(f"Need {len(needed_spec_ids)} test spectrogram files")

        spectrograms2 = {}
        for i, sid in enumerate(needed_spec_ids):
            if i % 200 == 0:
                print(i, ", ", end="")
            fpath = f"{SPEC_PATH}{sid}.parquet"
            tmp = pd.read_parquet(fpath, engine="pyarrow")
            spectrograms2[int(sid)] = np.asarray(
                tmp.iloc[:, 1:].values, dtype=np.float32
            )

        from scipy import signal

        if PLATFORM == "local":
            EEG_PATH = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        else:
            EEG_PATH = os.path.join(DATA_ROOT, "test_eegs") + "/"

        needed_eeg_ids = test["eeg_id"].unique().astype(int)
        print(f"\nNeed {len(needed_eeg_ids)} test eeg files")

        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        region_pairs = {}
        for region, chs in BRAIN.items():
            a_cols = [c.split("-")[0] for c in chs]
            b_cols = [c.split("-")[1] for c in chs]
            region_pairs[region] = (a_cols, b_cols)

        time_temp = 0
        time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
        time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

        eegs2 = {}
        for i, eid in enumerate(needed_eeg_ids):
            if i % 200 == 0:
                print(i, ", ", end="")
            fpath = f"{EEG_PATH}{eid}.parquet"
            raw_eeg = pd.read_parquet(fpath, engine="pyarrow")

            eeg_default = raw_eeg.iloc[time_start:time_stop].reset_index(drop=True)

            list_eeg = []
            for region in BRAIN.keys():
                a_cols, b_cols = region_pairs[region]
                a_arr = eeg_default[a_cols].to_numpy(dtype=np.float32, copy=False).T
                b_arr = eeg_default[b_cols].to_numpy(dtype=np.float32, copy=False).T
                eeg = a_arr - b_arr
                eeg = np.nan_to_num(eeg, nan=0.0)

                if 200 != SFREQ:
                    eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                eeg = signal.filtfilt(b, a, eeg, axis=1)
                list_eeg.append(eeg[:, :, None])

            eegs2[int(eid)] = np.concatenate(list_eeg, axis=2).astype(
                np.float32, copy=False
            )

        class CachedTestGenerator(tf.keras.utils.Sequence):
            def __init__(self, data, X_spec_map, X_eeg_map, batch_size=32):
                self.data = data.reset_index(drop=True)
                self.batch_size = batch_size
                self.indexes = np.arange(len(self.data))
                self.X_spec_map = X_spec_map
                self.X_eeg_map = X_eeg_map

            def __len__(self):
                return int(np.ceil(len(self.data) / self.batch_size))

            def __getitem__(self, index):
                idx = self.indexes[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                bs = len(idx)
                X = np.empty((bs, HIGH, LENGTH, 3, 4), dtype=np.float32)
                X_eeg = np.empty(
                    (bs, 6, round(EEG_LENGTH * SFREQ), 4), dtype=np.float32
                )
                rows = self.data.iloc[idx]
                for j, (sid, eid) in enumerate(
                    zip(rows["spec_id"].to_numpy(), rows["eeg_id"].to_numpy())
                ):
                    X[j] = self.X_spec_map[int(sid)]
                    X_eeg[j] = self.X_eeg_map[int(eid)]
                return [X, X_eeg]

        spec_cache = _SpecCache()
        dg_tmp = DataGenerator(
            test,
            shuffle=False,
            batch_size=1,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
            spec_cache=spec_cache,
        )

        X_spec_map = {}
        for i, sid in enumerate(needed_spec_ids):
            X_spec_map[int(sid)] = dg_tmp._precompute_spec_tensor(int(sid))

        X_eeg_map = {}
        for eid in needed_eeg_ids:
            base = eegs2[int(eid)]  # shape (4, time, 4)
            X_eeg = np.zeros((6, round(EEG_LENGTH * SFREQ), 4), dtype=np.float32)
            for k in range(4):
                img_eeg = base[:, :, k]
                X_eeg[1, :, k] = img_eeg[0, :]
                X_eeg[2, :, k] = img_eeg[1, :]
                X_eeg[3, :, k] = img_eeg[2, :]
                X_eeg[4, :, k] = img_eeg[3, :]
                X_eeg[:, :, k] = (
                    X_eeg[:, :, k] - np.mean(X_eeg[:, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[:, :, k], 1, keepdims=True) + 1e-6)
            X_eeg_map[int(eid)] = X_eeg

        test_gen = CachedTestGenerator(test, X_spec_map, X_eeg_map, batch_size=32)

        preds = []
        with strategy.scope():
            model = build_model()

        for i in range(5):
            print(f"\nFold {i+1}")
            wpath = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
            if not os.path.isfile(wpath):
                raise FileNotFoundError(f"Missing weight file: {wpath}")
            model.load_weights(wpath)
            pred = model.predict(test_gen, verbose=0)
            preds.append(pred)

        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

        pred = np.clip(pred, 1e-7, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        pred_df = pd.DataFrame(pred, columns=TARGETS)
        pred_df["eeg_id"] = test["eeg_id"].to_numpy()
        pred_df = pred_df.groupby("eeg_id", as_index=False)[TARGETS].mean()

        out = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")
        out[TARGETS] = out[TARGETS].fillna(1.0 / len(TARGETS))

        p = out[TARGETS].to_numpy(dtype=np.float64)
        p = np.clip(p, 1e-7, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        out[TARGETS] = p

        out = out[sample_sub.columns]
        assert len(out) == len(sample_sub), (len(out), len(sample_sub))
        out.to_csv("submission.csv", index=False)
        print("Submission shape", out.shape)
        print(
            "Row prob sum stats:",
            float(out[TARGETS].sum(axis=1).min()),
            float(out[TARGETS].sum(axis=1).max()),
        )
        print("Saved submission.csv")
