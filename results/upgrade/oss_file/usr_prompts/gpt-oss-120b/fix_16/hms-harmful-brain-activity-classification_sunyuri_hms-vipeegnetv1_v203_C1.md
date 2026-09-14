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

0.3588725925966945

# 6. Current score

0.80052

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I moved the EfficientNet import to occur right after TensorFlow is loaded and made it optional: if the import fails (which triggers the protobuf error), the script falls back to the baseline “average‑probability” submission that already works without any model. This ensures a valid `submission.csv` is always created and avoids the crashing import.'
- What this solution (achieved 1.41937) has done: 'I guard the second EfficientNet import so it only runs when TensorFlow and EfficientNet were successfully imported earlier, and I make the data generator return only inputs during prediction to avoid Keras‑sequence incompatibility. This removes the protobuf “MessageFactory” error and ensures a valid submission.csv is produced.'
- What this solution (achieved 1.41937) has done: 'I added a small compatibility shim that patches the protobuf MessageFactory to provide the missing GetPrototype method before TensorFlow is imported. This prevents the “MessageFactory has no attribute GetPrototype” error, allowing the EfficientNet‑based model to run and generate predictions instead of falling back to the baseline average‑probability submission, which should improve the KL‑divergence score toward the target.'
- What this solution (achieved 1.68479) has done: 'Implemented a fix for the syntax error and ensured the script exits after creating a baseline submission when TensorFlow isn’t available. The stray `else:` line was removed and a `sys.exit(0)` call was added right after the baseline CSV is written, preventing later TensorFlow‑dependent code from running and causing NameError issues.'
- What this solution (achieved 1.68479) has done: 'Implemented an enhanced fallback prediction when TensorFlow/EfficientNet is unavailable.  
The new logic first tries a per‑`eeg_id` probability (more specific than patient‑level), then falls back to patient‑level, and finally to the global class distribution. This improves the baseline KL‑Divergence while still guaranteeing a valid `submission.csv` without requiring any external deep‑learning libraries.'
- What this solution (achieved 1.1548) has done: 'Implemented fixes to resolve runtime errors and ensure a valid submission is always created:

- Added safe TensorFlow import handling and conditional definitions for functions, classes, and the learning‑rate scheduler that depend on TensorFlow.
- Wrapped `DataGenerator` definition in a try/except to avoid errors when TensorFlow isn’t available.
- Guarded the model‑inference block so it only runs when TensorFlow and EfficientNet are successfully imported.
- Applied a small epsilon smoothing to the aggregated class‑probability calculations in the fallback baseline to slightly improve KL‑divergence without altering core logic.
- Ensured the script always writes `submission.csv` with proper columns and normalized probabilities.'
- What this solution (achieved 1.1548) has done: 'The fix adds a safeguard around the model‑prediction block: if the `DataGenerator` class is missing (or TensorFlow/EfficientNet isn’t usable), the code now falls back to the previously‑implemented baseline probability calculation, ensuring a CSV submission is always written. This removes the `NameError` and guarantees a valid `submission.csv` without altering the core model logic.'
- What this solution (achieved 0.84381) has done: 'Implemented a safe TensorFlow import with a protobuf shim, guarded all TensorFlow‑dependent definitions, and enhanced the fallback baseline probability calculation (larger Laplace smoothing and explicit renormalisation). The script now always produces a valid `submission.csv` and avoids the earlier NameError and protobuf errors while keeping the core model logic unchanged.'
- What this solution (achieved 1.05318) has done: 'Implemented a finer Laplace smoothing (ε = 1e‑6 instead of 1e‑3) in both fallback paths. This reduces the artificial bias introduced by strong smoothing, producing probability estimates that better reflect the observed class frequencies and thereby lowering the KL‑divergence toward the target. No core modeling logic was altered, and the script still guarantees a valid `submission.csv`.'
- What this solution (achieved 0.82785) has done: 'Implemented a lightweight weighted blending for the fallback probability computation.  
Instead of selecting the most specific probability (eeg → patient → global), the new logic combines them with preset weights (eeg 70 %, patient 20 %, global 10 %). This modest calibration often yields probabilities closer to the true distribution, nudging the KL‑divergence score down toward the target while keeping all core model code untouched.'
- What this solution (achieved 0.86348) has done: 'Implemented a calibrated fallback prediction that better aligns probabilities with the target KL‑divergence.  
- Adjusted blending weights to give more balanced influence (eeg 30 %, patient 30 %, global 40 %).  
- Added temperature scaling ( T = 1.5 ) to flatten the blended distribution before normalisation, which typically reduces KL‑divergence.  
- Slightly increased Laplace smoothing to 1e‑5 for numerical stability.  
These changes keep the core model untouched and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.80052) has done: 'The fix tightens the fallback probability blending to rely more on specific EEG‑ and patient‑level statistics (weights 0.60, 0.30, 0.10) and removes the aggressive temperature scaling (set T = 1.0). It also reduces Laplace smoothing from 1e‑5 to 1e‑6 in both fallback paths, giving sharper probability estimates. These small adjustments keep the core model untouched while expectedly lowering the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # shim not needed or protobuf not installed

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    print("TensorFlow import failed (will use baseline):", e)

if tf is not None:

    def external_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1,
            dilation_rate=(4, 1),
            kernel_size=(4, 1),
            strides=(1, 1),
            padding="valid",
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

else:

    def external_spatial_block(x_eeg, filters=32):
        raise NotImplementedError("external_spatial_block requires TensorFlow")




## === cell 1
if tf is not None:

    def internal_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(4, 1), strides=(4, 1), padding="valid"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

else:

    def internal_spatial_block(x_eeg, filters=32):
        raise NotImplementedError("internal_spatial_block requires TensorFlow")




## === cell 2
if tf is not None:

    class CosineAnnealingLRScheduler(
        tf.keras.optimizers.schedules.LearningRateSchedule
    ):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super(CosineAnnealingLRScheduler, self).__init__()
            self.total_step = total_step
            self.warm_step = (
                1 if warmth_rate == 0 else int(self.total_step * warmth_rate)
            )
            self.lr_max = lr_max
            self.lr_min = lr_min

        @tf.function
        def __call__(self, step):
            step = step + 1
            if step < self.warm_step:
                lr = self.lr_max / self.warm_step * step
            else:
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1.0
                    + tf.cos(
                        (step - self.warm_step)
                        / (self.total_step - self.warm_step)
                        * np.pi
                    )
                )
            return lr


import os, sys, warnings, gc, time, itertools
import pandas as pd, numpy as np
from scipy import signal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLATFORM = "kaggle"
NEEDTRAIN = False
DATATYPE = ["eeg"]
LOAD_MODELS_FROM = "models20241102a"

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200
RSFREQ = 100
EEG_LENGTH = 50
EEG_LENGTH_USED = 50
filter_range = [0.5, 45]
filter_range2 = [0.1, 35]
SEED = 2024
BATCHSIZE = 16
SPLITS = 5

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
warnings.filterwarnings("ignore")

tf_available = tf is not None

if tf_available:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
else:
    np.random.seed(SEED)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

FALLBACK_W_EEG = 0.60
FALLBACK_W_PATIENT = 0.30
FALLBACK_W_GLOBAL = 0.10

FALLBACK_TEMPERATURE = 1.0  # no scaling

if not tf_available:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    eps = 1e-6  # reduced Laplace smoothing for sharper probabilities

    global_counts = df[TARGETS].sum() + eps
    global_probs = global_counts / global_counts.sum()

    patient_counts = df.groupby("patient_id")[list(TARGETS)].sum() + eps
    patient_probs = patient_counts.div(patient_counts.sum(axis=1), axis=0)

    eeg_counts = df.groupby("eeg_id")[list(TARGETS)].sum() + eps
    eeg_probs = eeg_counts.div(eeg_counts.sum(axis=1), axis=0)

    probs_list = []
    for _, row in test.iterrows():
        pid = row["patient_id"]
        eid = row["eeg_id"]

        has_eeg = eid in eeg_probs.index
        has_pat = pid in patient_probs.index

        prob_eeg = eeg_probs.loc[eid].values if has_eeg else None
        prob_pat = patient_probs.loc[pid].values if has_pat else None
        prob_glob = global_probs.values

        w_eeg = FALLBACK_W_EEG if has_eeg else 0.0
        w_pat = FALLBACK_W_PATIENT if has_pat else 0.0
        w_glob = FALLBACK_W_GLOBAL

        total_w = w_eeg + w_pat + w_glob
        blended = (
            (w_eeg * prob_eeg if prob_eeg is not None else 0.0)
            + (w_pat * prob_pat if prob_pat is not None else 0.0)
            + (w_glob * prob_glob)
        ) / total_w

        blended = blended**FALLBACK_TEMPERATURE
        blended = np.clip(blended, eps, None)
        blended = blended / blended.sum()
        probs_list.append(blended)

    pred = np.vstack(probs_list)
    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Baseline submission written, shape:", sub.shape)
    sys.exit(0)




## === cell 3
def build_model():
    inp = []
    y = None
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
        x_spe = tf.keras.layers.Concatenate(axis=1)(
            [
                inp_spe[:, 0, :, :],
                inp_spe[:, 1, :, :],
                inp_spe[:, 2, :, :],
                inp_spe[:, 3, :, :],
            ]
        )
        x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(x_spe)
        x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe] * 3)
        base_spe = efn.EfficientNetB0(include_top=False, weights=None)
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                if PLATFORM == "local"
                else "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            base_spe.load_weights(weight_path)
        x_spe = base_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=((4 * 4 + 2) * 1, round(EEG_LENGTH_USED * RSFREQ / 1))
        )
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )
        x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg] * 3)
        base_eeg = efn.EfficientNetB0(include_top=False, weights=None)
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                if PLATFORM == "local"
                else "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            base_eeg.load_weights(weight_path)
        x_eeg = base_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        inp.append(inp_eeg)
        y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg]) if y is not None else x_eeg

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    return tf.keras.Model(inputs=inp, outputs=y)




## === cell 4
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
test["sign_id"] = test.index.values
print("Test shape", test.shape)

PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
b2, a2 = signal.butter(3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass")
eegs_test = {}
for i, eeg_id in enumerate(test.eeg_id.unique()):
    if i % 100 == 0:
        print(i, ", ", end="")
    eeg_default = pd.read_parquet(os.path.join(PATH_test, f"{eeg_id}.parquet"))
    eeg = []
    for channel in BRAIN:
        eeg_temp = (
            eeg_default.loc[:, channel.split("-")[0]]
            - eeg_default.loc[:, channel.split("-")[1]]
        ).values
        eeg_temp[np.isnan(eeg_temp)] = 0
        eeg.append(np.reshape(eeg_temp, (1, -1)))
    eeg = np.concatenate(eeg, axis=0)
    if SFREQ != RSFREQ:
        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)
    eeg = signal.filtfilt(b, a, eeg, axis=1)
    eeg = np.clip(eeg, -1024, 1024)
    eegs_test[eeg_id] = eeg

if tf_available and "DataGenerator" in globals():
    preds = []
    model = build_model()
    test_gen = DataGenerator(
        test,
        shuffle=False,
        sample_weights=False,
        batch_size=512,
        mode="test",
        specs={},
        eegs=eegs_test,
        stfts={},
        imgs={},
    )
    for i in range(SPLITS):
        print(f"Fold {i+1}")
        weight_path = os.path.join(LOAD_MODELS_FROM, f"fold{i}_stage2.h5")
        try:
            model.load_weights(weight_path)
            print(f"Loaded weights from {weight_path}")
        except Exception as e:
            print(f"Failed to load weights for fold {i}: {e}")
            continue
        pred = model.predict(test_gen, verbose=1)
        preds.append(pred)
    if preds:
        pred = np.mean(preds, axis=0)
        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred
        sub.to_csv("submission.csv", index=False)
        print("Model submission written, shape:", sub.shape)
    else:
        print("No model predictions – fallback to baseline.")
else:
    print("TensorFlow or DataGenerator unavailable – using baseline fallback.")
    eps = 1e-6  # reduced Laplace smoothing for consistency

    global_counts = df[TARGETS].sum() + eps
    global_probs = global_counts / global_counts.sum()

    patient_counts = df.groupby("patient_id")[list(TARGETS)].sum() + eps
    patient_probs = patient_counts.div(patient_counts.sum(axis=1), axis=0)

    eeg_counts = df.groupby("eeg_id")[list(TARGETS)].sum() + eps
    eeg_probs = eeg_counts.div(eeg_counts.sum(axis=1), axis=0)

    probs_list = []
    for _, row in test.iterrows():
        pid = row["patient_id"]
        eid = row["eeg_id"]

        has_eeg = eid in eeg_probs.index
        has_pat = pid in patient_probs.index

        prob_eeg = eeg_probs.loc[eid].values if has_eeg else None
        prob_pat = patient_probs.loc[pid].values if has_pat else None
        prob_glob = global_probs.values

        w_eeg = FALLBACK_W_EEG if has_eeg else 0.0
        w_pat = FALLBACK_W_PATIENT if has_pat else 0.0
        w_glob = FALLBACK_W_GLOBAL

        total_w = w_eeg + w_pat + w_glob
        blended = (
            (w_eeg * prob_eeg if prob_eeg is not None else 0.0)
            + (w_pat * prob_pat if prob_pat is not None else 0.0)
            + (w_glob * prob_glob)
        ) / total_w

        blended = blended**FALLBACK_TEMPERATURE
        blended = np.clip(blended, eps, None)
        blended = blended / blended.sum()
        probs_list.append(blended)

    pred = np.vstack(probs_list)
    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Baseline submission written, shape:", sub.shape)
