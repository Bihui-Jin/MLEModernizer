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

0.4660618806062922

# 6. Current score

0.81499

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing TensorFlow to use the pure-Python protobuf implementation before importing it. Then I make the inference path robust to missing external weight files (`models202402052`) by gracefully falling back to a valid, competition-format baseline prediction (class priors from `train.csv`) so the notebook always produces `submission.csv`. Finally, I keep the existing model/data pipeline intact (no architecture/training changes) and ensure the submission probabilities are clipped and row-normalized to satisfy the KL-divergence evaluation requirements.'
- What this solution (achieved 1.3976) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime and also pinning protobuf’s Python implementation version early (before importing TensorFlow), which avoids the `MessageFactory.GetPrototype` error on Kaggle/Python 3.12. I keep your existing model/data pipeline intact, including the weight-loading fallback behavior, so it still produces a valid `submission.csv` even when external weights are missing. To nudge the score down (better) from the current class-prior baseline, I make the baseline a bit more competition-aligned by using *eeg_id-level* aggregated label priors (your `train` table) rather than row-level `df` totals, then apply minimal Dirichlet/Laplace smoothing and strict row normalization/clipping for KL stability. All I/O paths and submission columns remain unchanged.'
- What this solution (achieved 1.25932) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by ensuring TensorFlow is imported only after forcing the pure-Python protobuf runtime and by avoiding the problematic protobuf runtime-version pin that can still leave an incompatible protobuf/TensorFlow pairing on Kaggle/Python 3.12. Then I make the baseline (used when external weights are missing) slightly better-aligned to the metric by using smoothed `patient_id`-conditioned priors with a safe global fallback, while keeping the same overall “no-weights → priors” core behavior. Finally, I keep the submission strictly valid for KL divergence by clipping and row-normalizing probabilities and writing `submission.csv` with the exact required columns.'
- What this solution (achieved 1.26939) has done: 'I fix the TensorFlow import crash by avoiding TensorFlow entirely when `NEEDTRAIN=False` and no model weights are present (your current path), since the fallback baseline doesn’t need TF. This preserves your core modeling/training code but ensures the notebook runs end-to-end on Python 3.12 without hitting the protobuf `MessageFactory.GetPrototype` issue. To improve the score (lower KL is better) from your current patient-prior baseline, I make the fallback baseline more competition-aligned by using a **mixture of patient prior + global prior + mild per-(patient, expert_consensus) conditional prior** computed only from train metadata (no leakage), with safe smoothing and strict row-normalization. The script always write a valid `submission.csv` with the exact required columns and row sums equal to 1.'
- What this solution (achieved 1.26483) has done: 'Your current score (1.26939; lower is better) is still far from the target (0.46606), and the biggest limitation is that you’re usually running the metadata-only fallback because weights aren’t present. To move the KL down with minimal logic change, I keep the same “no weights → priors” approach but make the prior more informative by conditioning on both `patient_id` and `spectrogram_id` when available in train, with safe count-based smoothing and a simple backoff mixture to patient/global when sparse. This stays fully within train.csv metadata (no leakage) and preserves the rest of your pipeline unchanged. I also keep strict clipping + row-normalization to avoid KL blow-ups and ensure a valid submission.'
- What this solution (achieved 0.78897) has done: 'Your current score is far worse than the target (lower is better), and this code almost certainly runs the “no weights → metadata prior” fallback, so the only realistic way to move KL down without changing the model/training logic is to make the fallback probabilities closer to the hidden test label distribution. I keep the same prior/backoff structure, but (1) switch from mean-of-distributions to vote-weighted priors (better aligned to KL against vote-distributions), (2) add a small, safe ID-level prior using `spec_id` alone (since `spec_id` exists in test) with smoothing/backoff, and (3) use a very mild temperature sharpening toward the informative priors to reduce overly-uniform predictions. All changes are confined to the fallback path, preserve submission semantics, and still guarantee clipped + row-normalized probabilities.'
- What this solution (achieved 0.79903) has done: 'Your current score (0.78897; lower is better) is still far from the target (0.46606), and the most likely limiter is that you’re running the “no weights → metadata prior” fallback. To move KL down without touching the model/training core, I only improve that fallback by (1) computing priors at the **same unit as submission** (eeg_id) and then aggregating to **spectrogram_id** and **patient_id** (reduces overlap/duplication bias), and (2) using a small **KNN over spectrogram_id** (from train) to provide a meaningful prior for unseen test spectrogram_ids (many be unseen). I keep your existing backoff/mixture structure, smoothing, and strict clip+row-normalization so the submission stays valid and stable.'
- What this solution (achieved 0.78227) has done: 'Your current score (0.79903; lower is better) is still far from the target (0.46606), and since weights are likely missing the only “minimal-change” lever is improving the metadata-only fallback without touching the model/training core. I keep your existing prior/backoff structure, but fix one key mismatch: `spectrogram_id` in `train.csv` is not reliably the same ID namespace as test’s `spectrogram_id`, so your spectrogram-based KNN prior is effectively noise; I replace it with a deterministic, test-available signal by using (a) patient-conditioned priors and (b) patient KNN over `patient_id` built from train vote distributions (similar patients → similar label distributions). I also align the fallback to the metric by mixing in the train marginal distribution of **expert_consensus labels mapped into the 6 targets** (a safe, global calibration prior) and keep strict clipping + renormalization so the submission is always valid. All changes stay confined to the “no weights” fallback path and keep your TF/model code untouched.'
- What this solution (achieved 0.81416) has done: 'Your current score (0.78227; lower is better) is still far from the target (0.46606), and since you’re likely in the “no weights → metadata-only prior” fallback, the only minimal-change lever is improving that fallback so its probabilities better match test-label structure. I keep your exact fallback strategy (mixtures of global/consensus/patient/KNN + temperature) but fix two issues that can hurt KL: (1) the “patient KNN” based on numeric distance between patient_id values is arbitrary, so I replace it with a distribution-similarity KNN over patient priors learned from train; (2) I recalibrate the final mixture with a tiny amount of shrinkage toward the global prior (and slightly adjust smoothing) to avoid overconfident wrong spikes that inflate KL. All TensorFlow/model code stays untouched, and the script still always writes a valid `submission.csv` with clipped, row-normalized probabilities.'
- What this solution (achieved 0.81499) has done: 'Your current KL (0.81416, lower is better) is still far above the target (0.46606), so we should cautiously improve (reduce) KL. Since this script almost certainly runs the “no weights → metadata-only prior” fallback, the smallest meaningful lever is to make that fallback more test-relevant without touching the TF model/training core. I keep your existing mixture structure but (1) add a simple, robust prior conditioned on **patient_id + spec_id** learned from train (with backoff to patient/global when sparse), and (2) fix the “patient KNN” so it uses similarity to the **known patient’s distribution** when available (instead of always comparing to global), which should reduce systematic miscalibration. All outputs remain clipped and row-normalized to guarantee a valid KL-safe submission.csv.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402052"
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

VER = 1

MIX = True

if PLATFORM == "kaggle" and not os.path.isdir(LOAD_MODELS_FROM):
    alt = f"/kaggle/input/hms-harmful-brain-activity-classification/{os.path.basename(LOAD_MODELS_FROM)}"
    if os.path.isdir(alt):
        LOAD_MODELS_FROM = alt
print("LOAD_MODELS_FROM =", LOAD_MODELS_FROM)



## === cell 1
if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

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




## === cell 2
def _maybe_import_tf():
    import tensorflow as tf
    import matplotlib

    return tf, matplotlib


TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}


def _build_datagen_class(tf, matplotlib):
    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            data,
            batch_size=32,
            shuffle=False,
            augment=False,
            mode="train",
            specs=None,
            eegs=None,
        ):
            self.cmin = -4
            self.cmax = 6
            self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]

            self.data = data.reset_index(drop=True)
            self.batch_size = batch_size
            self.shuffle = shuffle

            self.augment = False

            self.mode = mode
            self.specs = specs if specs is not None else {}
            self.eegs = eegs if eegs is not None else {}
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.data) / self.batch_size))

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

        def __data_generation(self, indexes):
            X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
            X_eeg = np.zeros(
                (len(indexes), 8, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
            )
            y = np.zeros((len(indexes), 6), dtype="float32")

            img_map = np.zeros((100 * 300, 3), dtype=np.float32)

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                spec_id = int(row.spec_id) if "spec_id" in row else None
                eeg_id = int(row.eeg_id) if "eeg_id" in row else None
                spec_arr = self.specs.get(spec_id, None)
                eeg_arr = self.eegs.get(eeg_id, None)

                if self.mode == "test":
                    r = 0
                else:
                    r = (
                        np.random.randint(int(row["min"]), int(row["max"]) + 1) // 2
                        if "min" in row and "max" in row
                        else 0
                    )

                for k in range(4):
                    if spec_arr is not None:
                        img = spec_arr[r : r + 300, k * 100 : (k + 1) * 100].T
                        img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                        img = np.log(img)
                        img = np.nan_to_num(img, nan=0.0)

                        img = np.round(
                            (img - self.cmin) / (self.cmax - self.cmin) * 256
                        )
                        img = np.reshape(img, (img.shape[0] * img.shape[1]))
                        img = np.array(img, dtype=np.int16)

                        img = np.clip(img, 1, 256)
                        img_map = self.cmaps[img - 1]
                        img_map = np.reshape(img_map, (100, 300, 3))

                        img_map = img_map[
                            :,
                            max(round((600 / 2 - LENGTH) / 2), 0) : min(
                                (round((600 / 2 - LENGTH) / 2) + LENGTH),
                                img_map.shape[1],
                            ),
                            :,
                        ]
                        if HIGH != 100:
                            img_map2 = np.array(
                                tf.image.resize(img_map, ((HIGH - 32), LENGTH)),
                                dtype=np.float32,
                            )
                            h0 = round((HIGH - img_map2.shape[0]) / 2)
                            h1 = round((HIGH + img_map2.shape[0]) / 2)
                            X[j, h0:h1, :, :, k] = img_map2
                        else:
                            X[j, :, :, :, k] = img_map
                    X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / (0.229**2)
                    X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / (0.224**2)
                    X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / (0.225**2)

                    if eeg_arr is not None:
                        img_eeg = eeg_arr[:, :, k]
                        X_eeg[j, 2, :, k] = img_eeg[0, :]
                        X_eeg[j, 3, :, k] = img_eeg[1, :]
                        X_eeg[j, 4, :, k] = img_eeg[2, :]
                        X_eeg[j, 5, :, k] = img_eeg[3, :]
                        X_eeg[j, :, :, k] = (
                            X_eeg[j, :, :, k]
                            - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                        ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if self.mode != "test":
                    y[j] = row[TARGETS].values

            return X, X_eeg, y

    return DataGenerator


def _build_model_fn(tf):
    def build_model():
        inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        inp_eeg = tf.keras.Input(shape=(8, round(EEG_LENGTH * SFREQ), 4))

        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=(HIGH, LENGTH * 4, 3)
        )
        base_model._name = "spectrogram_extractor"

        x0 = inp[:, :, :, :, 0]
        x1 = inp[:, :, :, :, 1]
        x2 = inp[:, :, :, :, 2]
        x3 = inp[:, :, :, :, 3]
        x = tf.keras.layers.Concatenate(axis=2)([x0, x1, x2, x3])

        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2n_spec"
        )(x)

        base_model_eeg = tf.keras.applications.EfficientNetB2(
            include_top=False,
            weights=None,
            input_shape=(8 * 4, round(EEG_LENGTH * SFREQ), 3),
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
        x_eeg = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2n_eeg"
        )(x_eeg)

        x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
        x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.KLDivergence()
        model.compile(loss=loss, optimizer=opt)
        return model

    return build_model




## === cell 3
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        PATH_SPEC = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPEC = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    print("Test shape", test.shape)
    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    have_weights = os.path.isdir(LOAD_MODELS_FROM) and any(
        os.path.isfile(os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5"))
        for i in range(5)
    )

    if not have_weights:
        print(
            "WARNING: Model weights not found under LOAD_MODELS_FROM. "
            "Falling back to metadata-only prior baseline submission (no TensorFlow)."
        )

        train_eeg = df.groupby("eeg_id", sort=False).agg(
            {
                "patient_id": "first",
                "spectrogram_id": "first",
                "expert_consensus": "first",
                **{t: "sum" for t in TARGETS},
            }
        )

        alpha = 0.8

        vote_sums_global = train_eeg[TARGETS].sum(axis=0).values.astype("float64")
        global_prior = (vote_sums_global + alpha) / (vote_sums_global.sum() + 6 * alpha)

        consensus_counts = train_eeg["expert_consensus"].value_counts(dropna=False)
        cons_to_target = {
            "Seizure": "seizure_vote",
            "LPD": "lpd_vote",
            "GPD": "gpd_vote",
            "LRDA": "lrda_vote",
            "GRDA": "grda_vote",
            "Other": "other_vote",
        }
        cons_vec = np.zeros(6, dtype="float64")
        for k, v in cons_to_target.items():
            if k in consensus_counts.index:
                cons_vec[list(TARGETS).index(v)] = float(consensus_counts.loc[k])
        if cons_vec.sum() > 0:
            consensus_prior = (cons_vec + alpha) / (cons_vec.sum() + 6 * alpha)
        else:
            consensus_prior = global_prior.copy()

        pat_vote_sums = train_eeg.groupby("patient_id")[TARGETS].sum()
        pat_counts = train_eeg.groupby("patient_id").size().astype("int64")

        pat_probs = (pat_vote_sums.values.astype("float64") + alpha) / (
            pat_vote_sums.values.astype("float64").sum(axis=1, keepdims=True)
            + 6 * alpha
        )
        pat_ids = pat_vote_sums.index.values.astype(np.int64)

        pat_probs_norm = pat_probs / (
            np.linalg.norm(pat_probs, axis=1, keepdims=True) + 1e-12
        )

        pat_spec_vote_sums = train_eeg.groupby(["patient_id", "spectrogram_id"])[
            TARGETS
        ].sum()
        pat_spec_counts = (
            train_eeg.groupby(["patient_id", "spectrogram_id"]).size().astype("int64")
        )

        def _patient_knn_prior(pid, k=50):
            if pat_ids.size == 0:
                return global_prior
            pid = int(pid)
            if pid in pat_vote_sums.index:
                v_pat = pat_vote_sums.loc[pid].values.astype("float64")
                n_pat = int(pat_counts.loc[pid])
                a_pat = alpha * (10.0 / np.sqrt(max(n_pat, 1)))
                p_pat = (v_pat + a_pat) / (v_pat.sum() + 6 * a_pat)
                p_pat = np.clip(p_pat, 1e-12, 1.0)
                return p_pat / p_pat.sum()

            g = (0.7 * global_prior + 0.3 * consensus_prior).astype("float64")
            g = g / (np.linalg.norm(g) + 1e-12)
            sims = pat_probs_norm @ g
            kk = min(int(k), sims.size)
            idx_hi = np.argpartition(-sims, kk - 1)[:kk]
            p = pat_probs[idx_hi].mean(axis=0)
            p = np.clip(p, 1e-12, 1.0)
            return p / p.sum()

        patient_cons_vote_sums = df.groupby(["patient_id", "expert_consensus"])[
            TARGETS
        ].sum()
        patient_cons_counts = (
            df.groupby(["patient_id", "expert_consensus"]).size().astype("int64")
        )

        pat_cons_freq = (
            df.groupby(["patient_id", "expert_consensus"])
            .size()
            .rename("n")
            .reset_index()
        )
        pat_tot = (
            pat_cons_freq.groupby("patient_id")["n"].sum().rename("tot").reset_index()
        )
        pat_cons_freq = pat_cons_freq.merge(pat_tot, on="patient_id", how="left")
        pat_cons_freq["w"] = pat_cons_freq["n"] / pat_cons_freq["tot"]
        pat_cons_weights = {
            pid: list(
                zip(
                    g["expert_consensus"].tolist(),
                    g["w"].astype("float64").tolist(),
                )
            )
            for pid, g in pat_cons_freq.groupby("patient_id")
        }

        pred = np.zeros((len(test), 6), dtype="float64")
        test_pat = test["patient_id"].values
        test_spec = test["spec_id"].values

        temperature = 0.975

        for i, (pid, sid) in enumerate(zip(test_pat, test_spec)):
            p = 0.70 * global_prior + 0.30 * consensus_prior

            if pid in pat_vote_sums.index:
                v_pat = pat_vote_sums.loc[int(pid)].values.astype("float64")
                n_pat = int(pat_counts.loc[int(pid)])
                a_pat = alpha * (10.0 / np.sqrt(max(n_pat, 1)))
                p_pat = (v_pat + a_pat) / (v_pat.sum() + 6 * a_pat)

                p = 0.82 * p_pat + 0.18 * p

                key_ps = (int(pid), int(sid))
                if key_ps in pat_spec_vote_sums.index:
                    v_ps = pat_spec_vote_sums.loc[key_ps].values.astype("float64")
                    n_ps = int(pat_spec_counts.loc[key_ps])
                    a_ps = alpha * (35.0 / np.sqrt(max(n_ps, 1)))
                    p_ps = (v_ps + a_ps) / (v_ps.sum() + 6 * a_ps)
                    w_ps = float(np.clip(np.log1p(n_ps) / np.log1p(30), 0.10, 0.45))
                    p = (1.0 - w_ps) * p + w_ps * p_ps

                if pid in pat_cons_weights:
                    mix = np.zeros(6, dtype="float64")
                    for cons, w in pat_cons_weights[int(pid)]:
                        key = (int(pid), cons)
                        if key in patient_cons_vote_sums.index:
                            v_pc = patient_cons_vote_sums.loc[key].values.astype(
                                "float64"
                            )
                            n_pc = int(patient_cons_counts.loc[key])
                            a_pc = alpha * (20.0 / np.sqrt(max(n_pc, 1)))
                            p_pc = (v_pc + a_pc) / (v_pc.sum() + 6 * a_pc)
                            mix += w * p_pc
                    if mix.sum() > 0:
                        p = 0.90 * p + 0.10 * mix
            else:
                p_knn = _patient_knn_prior(pid, k=75)
                p = 0.62 * p_knn + 0.38 * p

            p = 0.985 * p + 0.015 * global_prior

            p = np.clip(p, 1e-12, 1.0)
            p = p ** (1.0 / temperature)
            p = p / p.sum()
            pred[i] = p

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred.astype("float32")
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(
            "Row prob sum stats:",
            float(sub[TARGETS].sum(axis=1).min()),
            float(sub[TARGETS].sum(axis=1).max()),
        )
        print(sub.head())

    else:
        tf, matplotlib = _maybe_import_tf()
        import matplotlib.pyplot as plt  # noqa: F401

        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        print("TensorFlow version =", tf.__version__)

        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) <= 1:
            if len(gpus) == 0:
                strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            else:
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print(f"Using {len(gpus)} GPU")
        else:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {len(gpus)} GPUs")

        if MIX:
            try:
                from tensorflow.keras import mixed_precision

                mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision policy set to:", mixed_precision.global_policy())
            except Exception as e:
                print("Mixed precision could not be enabled:", repr(e))
        else:
            print("Using full precision")

        DataGenerator = _build_datagen_class(tf, matplotlib)
        build_model = _build_model_fn(tf)

        files2 = os.listdir(PATH_SPEC)
        print(f"There are {len(files2)} test spectrogram parquets")
        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()

        from scipy import signal

        files2 = os.listdir(PATH_EEG)
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH_EEG}{f}")
            name = int(f.split(".")[0])

            if len(test[test.eeg_id == name]) > 0:
                time_temp = 0
                time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

                list_eeg = []
                for region in BRAIN.keys():
                    eeg = np.zeros(
                        (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                    )
                    for chan_i, chan in enumerate(BRAIN[region]):
                        a_ch, b_ch = chan.split("-")
                        eeg[chan_i, :] = (
                            eeg_default.loc[:, a_ch] - eeg_default.loc[:, b_ch]
                        ).values

                    eeg[np.isnan(eeg)] = 0
                    if 200 != SFREQ:
                        eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)
                    eeg = signal.filtfilt(b, a, eeg, axis=1)

                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs2[name] = list_eeg
        print()

        preds = []
        with strategy.scope():
            model = build_model()

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
        )

        for i in range(5):
            print(f"Fold {i + 1}")
            wpath = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
            if not os.path.isfile(wpath):
                raise FileNotFoundError(
                    f"Could not find weights at {wpath}. Check LOAD_MODELS_FROM path."
                )
            model.load_weights(wpath)
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)

        pred = np.mean(preds, axis=0)
        print("Test preds shape", pred.shape)

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred.astype("float32")
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(
            "Row prob sum stats:",
            float(sub[TARGETS].sum(axis=1).min()),
            float(sub[TARGETS].sum(axis=1).max()),
        )
        print(sub.head())
