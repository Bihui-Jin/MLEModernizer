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

3.12

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

1.7432482382120178

# 6. Current score

1.41202

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.42965) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf “cpp” override so TF 2.18 can import cleanly in this environment; this unblocks all later cells that currently fail with `tf` not defined. I also make the submission-writing logic robust to the fact that `sample_submission.csv` in this dataset contains duplicate `eeg_id` rows: instead of asserting uniqueness, we aggregate predictions per `eeg_id` (mean) and then align to the exact sample submission row order. Finally, I ensure the script always writes a valid `submission.csv` with correct columns and per-row probabilities summing to 1.'
- What this solution (achieved 1.42965) has done: 'I fix the TensorFlow/protobuf import crash that prevents the notebook from running by forcing protobuf to use the pure-Python implementation before importing TensorFlow (this is the known workaround for the `MessageFactory.GetPrototype` error in some Kaggle images). Because your current score (1.42965, lower-is-better) is already better than the target (1.7432), I avoid any modeling/prediction changes that might further improve or significantly degrade performance; the inference and submission logic remain the same. I also keep the submission aggregation/alignment safeguards so the output matches `sample_submission.csv` row order and probabilities sum to 1. The result run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 1.41202) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents the notebook from running by removing the protobuf environment override and (as a safe fallback) forcing TensorFlow to use the legacy Keras API if needed in this Kaggle image. I also make the model’s `call()` method create pooling/softmax layers once in `__init__` (not inside `call`) to avoid runtime graph/variable creation issues across repeated `predict()` calls. Because your current score (1.42965, lower-is-better) is already better than the target (1.7432), I not change any modeling/inference behavior that would materially alter the score; the prediction normalization and submission alignment stay the same. The result run end-to-end and always write a valid `submission.csv` with correct columns and per-row probabilities summing to 1.'
- What this solution (achieved 1.41202) has done: 'You’re hitting the known TensorFlow↔protobuf incompatibility in some Kaggle images (`MessageFactory` missing `GetPrototype`). I fix this by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow, which is the minimal, standard workaround and should unblock the entire pipeline. Because your current score (1.41202, lower-is-better) is already better than the target (1.7432), I not change the model, features, ensembling, or post-processing that could shift the score; the goal here is stability and producing a valid `submission.csv`. I also keep `TF_USE_LEGACY_KERAS=1` as you had, since it’s score-neutral and helps avoid Keras 3 incompatibilities.'
- What this solution (achieved 1.41202) has done: 'I fix the TensorFlow import crash caused by the protobuf `MessageFactory.GetPrototype` incompatibility by forcing the pure-Python protobuf backend *and* using the protobuf internal API implementation type, both set before importing TensorFlow. This is a runtime-only fix and does not change your model, preprocessing, ensembling, or post-processing, so it should be score-neutral (your current score is already better than the target). I also add a small, safe fallback to disable XLA if the environment still triggers protobuf-related initialization issues. The rest of the pipeline (reading parquet, predicting, row-normalizing, aggregating by `eeg_id`, and writing `submission.csv`) stays identical.'
- What this solution (achieved 1.41202) has done: 'I fix the TensorFlow import crash caused by the protobuf `MessageFactory.GetPrototype` incompatibility by forcing protobuf to use the pure-Python runtime *and* setting the recommended `protobuf` internal implementation selector **before any TensorFlow import**. This is a runtime-only change and keeps your model, preprocessing, and ensembling identical (score should remain essentially unchanged, and since your current score is already better than target, we avoid any score-improving changes). I also renumber the cells starting at 1 (your current script starts at cell 0) while preserving order and logic, so the notebook-style runner won’t reject the format. The rest of the pipeline remains the same and still write a valid `submission.csv` with correct columns and per-row probabilities summing to 1.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_TYPE", "python")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault(
    "TF_XLA_FLAGS", "--tf_xla_auto_jit=0"
)  # safe fallback; score-neutral

import time
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from sklearn.model_selection import KFold  # kept (core logic preserved)
import joblib  # kept (core logic preserved)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
target_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
test_eegs = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
test_spectrograms = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)

sample_sub_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)



## === cell 2
ds_test = pd.read_csv(test_path)
print("test rows:", len(ds_test), "unique eeg_id:", ds_test["eeg_id"].nunique())
ds_test.head(3)




## === cell 3
def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x




## === cell 4
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x




## === cell 5
class RnnModel(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer1_100 = tf.keras.layers.Dense(108, activation="relu")
        self.layer1_6 = tf.keras.layers.Dense(6, activation="relu")
        self.add = tf.keras.layers.Add()

        self.gap = tf.keras.layers.GlobalAveragePooling1D()
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, X):
        x1 = X["x1"]
        x2 = X["x2"]
        x = tf.concat([x1, x2], 2)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        x = self.gap(x)
        x = self.layer1_100(x)
        x = self.layer1_6(x)
        x = self.softmax(x)
        return x




## === cell 6
def loss_fn(labels, targets):
    loss = tf.math.abs(labels - targets)
    loss = tf.math.reduce_mean(loss)
    return loss


models_infor = [
    [0, 100],
    [0, 90],
    [0, 80],
    [0, 70],
    [0, 60],
    [0, 50],
]



## === cell 7
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local")
except Exception:
    tpu = None

if tpu:
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

weights_root = "/kaggle/input/hms-train-model-tf-transformer"
has_weights = os.path.isdir(weights_root)

model_list = []
missing_weights = 0

with strategy.scope():
    for m_i in models_infor:
        model = RnnModel(
            num_layers=3, d_model=420, num_heads=2, dff=400, dropout_rate=0.5
        )
        model.build(input_shape={"x1": [1, 300, 20], "x2": [1, 300, 400]})
        w_path = f"{weights_root}/{m_i[0]}_weights/model_epoch_{m_i[1]}.weights.h5"
        if has_weights and os.path.isfile(w_path):
            model.load_weights(w_path)
        else:
            missing_weights += 1
        model_list.append(model)

if len(model_list) == 0:
    raise RuntimeError("No models were constructed; cannot run inference.")

print(
    f"TPU: {bool(tpu)} | Models: {len(model_list)} | Missing weight files: {missing_weights}"
)
if missing_weights == len(model_list):
    print(
        "WARNING: All weight files are missing; predictions will come from randomly initialized models."
    )




## === cell 8
def prediction_re(ds):
    preds = None
    for model_p in model_list:
        p = model_p.predict(ds, verbose=0)
        preds = p if preds is None else (preds + p)
    preds = preds / max(1, len(model_list))
    return preds




## === cell 9
span = 1999
eps = 1e-6


def _safe_row_normalize(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


n = (ds_test.shape[0] // span) + int(ds_test.shape[0] % span != 0)

predictins_all = []
for i in range(n):
    temp_ds = ds_test.iloc[i * span : (i + 1) * span, :].reset_index(drop=True)

    list_x1, list_x2 = [], []
    for ii in range(temp_ds.shape[0]):
        eeg_id = temp_ds.loc[ii, "eeg_id"]
        spectrogram_id = temp_ds.loc[ii, "spectrogram_id"]

        eeg_arr = (
            pd.read_parquet(os.path.join(test_eegs, f"{eeg_id}.parquet"))
            .fillna(0)
            .values[:300, :]
        )
        spc_arr = (
            pd.read_parquet(
                os.path.join(test_spectrograms, f"{spectrogram_id}.parquet")
            )
            .fillna(-1)
            .values[:300, 1:]
        )

        eeg_arr = np.clip(eeg_arr, np.exp(-6), np.exp(10))
        x1 = np.log(eeg_arr)
        m1 = x1.mean(axis=0, keepdims=True)
        s1 = x1.std(axis=0, keepdims=True)
        x1 = (x1 - m1) / (s1 + eps)

        spc_arr = np.clip(spc_arr, np.exp(-6), np.exp(10))
        x2 = np.log(spc_arr)
        m2 = x2.mean(axis=0, keepdims=True)
        s2 = x2.std(axis=0, keepdims=True)
        x2 = (x2 - m2) / (s2 + eps)

        list_x1.append(x1.astype(np.float32))
        list_x2.append(x2.astype(np.float32))

    pre_data = {"x1": np.stack(list_x1, axis=0), "x2": np.stack(list_x2, axis=0)}
    batch_preds = prediction_re(pre_data)
    predictins_all.append(batch_preds)

predictins = np.concatenate(predictins_all, axis=0)
predictins = _safe_row_normalize(predictins, eps=1e-12)

assert (
    predictins.shape[0] == ds_test.shape[0]
), f"Pred rows ({predictins.shape[0]}) != test rows ({ds_test.shape[0]})."
assert predictins.shape[1] == len(
    target_cols
), f"Pred cols ({predictins.shape[1]}) != {len(target_cols)}."

print(
    "Pred shape:",
    predictins.shape,
    "Row sum min/max:",
    predictins.sum(1).min(),
    predictins.sum(1).max(),
)



## === cell 10
sub = pd.read_csv(sample_sub_path)
sub = sub[["eeg_id"] + target_cols].copy()

pred_df = pd.DataFrame(predictins, columns=target_cols)
pred_df.insert(0, "eeg_id", ds_test["eeg_id"].values)

pred_agg = pred_df.groupby("eeg_id", as_index=False)[target_cols].mean()

sub = sub.drop(columns=target_cols)
sub = sub.merge(pred_agg, on="eeg_id", how="left")

missing = sub[target_cols].isna().any(axis=1)
if missing.any():
    sub.loc[missing, target_cols] = 1.0 / len(target_cols)

pred_vals = _safe_row_normalize(sub[target_cols].to_numpy(), eps=1e-12)
sub[target_cols] = pred_vals

sub.to_csv("submission.csv", index=False)
print(sub.head())
print(
    "Wrote submission.csv with rows:",
    len(sub),
    "| unique eeg_id:",
    sub["eeg_id"].nunique(),
)



## === cell 11
check = pd.read_csv("submission.csv")
assert (
    list(check.columns) == ["eeg_id"] + target_cols
), f"Wrong columns: {check.columns.tolist()}"
assert len(check) == len(
    pd.read_csv(sample_sub_path)
), "Length mismatch vs sample submission"
row_sums = check[target_cols].sum(axis=1).values
assert np.all(np.isfinite(check[target_cols].values)), "Non-finite predictions"
assert (
    np.max(np.abs(row_sums - 1.0)) < 1e-6
), f"Row sums not 1; max dev: {np.max(np.abs(row_sums-1.0))}"
check.head()
