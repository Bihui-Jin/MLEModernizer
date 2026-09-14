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

0.4305725754210282

# 6. Current score

1.68479

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The script crashed because it tried to load pre‑trained model weights that do not exist in the environment.  
I added a safe fallback: if the weight files cannot be found, the code now skips model inference and instead builds a very simple baseline prediction using the overall class‑vote frequencies from the training set. This guarantees a valid `submission.csv` with correctly summed probabilities, letting the notebook finish without errors while still providing a reasonable score.'
- What this solution (achieved 1.68479) has done: 'I add a lightweight patient‑level baseline: compute per‑patient vote distributions from the training set and use them for test rows when the patient appears in the training data, falling back to the global class frequencies otherwise. This keeps the original fallback logic but gives more specific probabilities, which should lower the KL‑divergence toward the target score. The rest of the script remains unchanged.'
- What this solution (achieved 1.31003) has done: 'I add a patient‑vote count‑based weighting so the fallback uses a blend of the patient‑specific distribution and the overall class frequencies (more votes ⇒ trust the patient distribution more). This keeps the same baseline logic but gives slightly better calibrated predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.32322) has done: 'We keep the existing logic but soften the fallback probabilities with a temperature scaling ( T > 1 ). This makes the predictions less extreme, which usually reduces KL‑divergence when the baseline is still far from the true distribution. The change is limited to the fallback branch, preserving the original model‑inference path.'
- What this solution (achieved 1.47425) has done: 'I simplify the fallback baseline: when a patient appears in the training data, use its exact vote‑derived probability distribution; otherwise fall back to the global class frequencies. This removes the arbitrary blending weight and temperature scaling that were soft‑ening the predictions and can make the probabilities better matched to the true label distribution, helping to lower the KL‑divergence toward the target score. The rest of the script stays unchanged.'
- What this solution (achieved 1.68479) has done: 'Implemented a robust fallback that avoids undefined generators and correctly merges patient‑level probabilities without column name collisions. The new logic:
- Catches any errors during model inference and directly uses a hierarchical baseline.
- Merges test data with patient probability estimates only.
- Falls back to global class frequencies when a patient is unseen.
- Renormalizes rows to guarantee probabilities sum to 1.
- Writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import io
from PIL import Image

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.metrics import confusion_matrix
import librosa

try:
    import albumentations as albu
except Exception:
    albu = None

try:
    import efficientnet.tfkeras as efn
except Exception:
    from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4

    class efn:
        EfficientNetB0 = EfficientNetB0
        EfficientNetB4 = EfficientNetB4


print("TensorFlow version =", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

np.random.seed(2024)
os.environ["PYTHONHASHSEED"] = str(2024)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(2024)
tf.keras.utils.set_random_seed(2024)
tf.config.experimental.enable_op_determinism()

MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled")
else:
    print("Using full precision")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024031101"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128  # 128
LENGTH = 256  # 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024

BATCHSIZE = 16

AMP = 200

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

spectrograms = {}
eegs = {}
imgs = {}
stfts = {}

spectrograms2 = {}
eegs2 = {}
imgs2 = {}
stfts2 = {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()

patient_vote_sums = df.groupby("patient_id")[list(TARGETS)].sum()
patient_probs_df = patient_vote_sums.div(
    patient_vote_sums.sum(axis=1), axis=0
).reset_index()
patient_probs_df.rename(columns={col: f"{col}_prob" for col in TARGETS}, inplace=True)

patient_counts = patient_vote_sums.sum(axis=1).rename("vote_count").reset_index()

eeg_vote_sums = df.groupby("eeg_id")[list(TARGETS)].sum()
eeg_probs_df = eeg_vote_sums.div(eeg_vote_sums.sum(axis=1), axis=0).reset_index()
eeg_probs_df.rename(columns={col: f"{col}_prob" for col in TARGETS}, inplace=True)




## === cell 1
def build_model():
    inp = []
    l2_norm = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_spe = tf.keras.layers.Concatenate(axis=1)(
            [inp_spe[:, :, :, :, i] for i in range(4)]
        )
        base_model_spe = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                if PLATFORM == "local"
                else "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            if os.path.exists(weight_path):
                base_model_spe.load_weights(weight_path)
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = l2_norm(x_spe)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))
        x_eeg = tf.keras.layers.Concatenate(axis=1)(
            [inp_eeg[:, :, :, i : i + 1] for i in range(4)]
        )
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])
        base_model_eeg = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                if PLATFORM == "local"
                else "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            if os.path.exists(weight_path):
                base_model_eeg.load_weights(weight_path)
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = l2_norm(x_eeg)
        inp.append(inp_eeg)
        y = (
            tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
            if "spe" in DATATYPE
            else x_eeg
        )

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4))
        x_img = tf.keras.layers.Concatenate(axis=1)(
            [inp_img[:, :, :, i : i + 1] for i in range(4)]
        )
        x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])
        base_model_img = efn.EfficientNetB4(
            include_top=False, weights=None, input_shape=None
        )
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                if PLATFORM == "local"
                else "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            if os.path.exists(weight_path):
                base_model_img.load_weights(weight_path)
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = l2_norm(x_img)

        has_previous = len(inp) > 0
        if has_previous:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
        else:
            y = x_img
        inp.append(inp_img)

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_stft = tf.keras.layers.Concatenate(axis=1)(
            [inp_stft[:, :, :, :, i] for i in range(4)]
        )
        base_model_stft = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                if PLATFORM == "local"
                else "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            if os.path.exists(weight_path):
                base_model_stft.load_weights(weight_path)
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = l2_norm(x_stft)
        inp.append(inp_stft)
        y = tf.keras.layers.Concatenate(axis=1)([y, x_stft]) if inp else x_stft

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 2
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )

    try:
        preds = []
        model = build_model()
        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=BATCHSIZE * 2,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
            imgs=imgs2,
            stfts=stfts2,
        )
        for i in range(5):
            print(f"Fold {i + 1}")
            weight_path = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Missing weight file: {weight_path}")
            model.load_weights(weight_path)
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)
        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred
    except Exception as e:
        print("Warning:", e)
        print("Falling back to a hierarchical baseline: patient → global.")

        class_sums = df[TARGETS].sum()
        total_votes = class_sums.sum()
        global_probs = (class_sums / total_votes).values.astype(np.float32)

        test_merge = test.merge(patient_probs_df, on="patient_id", how="left")

        prob_cols = [f"{c}_prob" for c in TARGETS]
        for idx, col in enumerate(prob_cols):
            test_merge[col] = test_merge[col].fillna(global_probs[idx])

        final_probs = test_merge[prob_cols].values.astype(np.float32)

        row_sums = final_probs.sum(axis=1, keepdims=True)
        final_probs = final_probs / np.clip(row_sums, 1e-12, None)

        sub = pd.DataFrame({"eeg_id": test_merge["eeg_id"]})
        sub[TARGETS] = final_probs

    prob_sum = sub[TARGETS].sum(axis=1)
    sub[TARGETS] = sub[TARGETS].div(prob_sum, axis=0)

    sub.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")
    print("Submission shape", sub.shape)
    print("First 5 rows:")
    print(sub.head())
    print("Row sums (first 5):", sub[TARGETS].sum(axis=1).head())
