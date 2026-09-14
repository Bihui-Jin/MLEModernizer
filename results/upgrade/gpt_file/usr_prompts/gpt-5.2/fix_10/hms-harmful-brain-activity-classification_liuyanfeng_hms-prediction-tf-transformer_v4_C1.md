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

1.1505295325756062

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.44548) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this environment. Then I remove the hard dependency on missing external weights (`/kaggle/input/hms-train-model-tf/...`) by training the same model architecture on a small, deterministic subset of `train.csv` + corresponding parquet files so `model_list` is non-empty and predictions are produced. I also correct the submission creation to guarantee the exact required columns, row alignment to `test.csv`, and enforce per-row probability normalization (sums to 1) so Kaggle accepts the file. These changes preserve the core model architecture and inference semantics (softmax probabilities) while making the notebook run end-to-end and output a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate TensorFlow/protobuf crash by ensuring a compatible protobuf version is used at runtime (TensorFlow 2.18 is not compatible with protobuf 6.x here), instead of only forcing the pure-Python implementation. Then I keep your model and training loop intact but make a minimal, metric-aligned correction: use KL-divergence loss (with safe clipping) rather than mean absolute error, since the competition is scored by KL divergence on probability vectors. Finally, I keep the same submission assembly but add a last safety normalization to guarantee every row sums to 1 and all required columns exist in the right order.'
- What this solution (achieved 1.40995) has done: 'We’re currently worse than the target (1.40995 vs 1.15053, lower is better), so the smallest safe way to improve is to (1) stop training on randomly overlapping subsamples and instead aggregate to one target distribution per `eeg_id` (matching the submission/eval granularity), and (2) avoid label leakage across the same `eeg_id` by sampling unique `eeg_id`s for training. This preserves your exact model architecture, feature extraction, and training loop style, but makes the supervised signal better aligned with how the metric is computed. I also keep runtime bounded by still training on a capped subset, just at the `eeg_id` level, and keep the same submission assembly with strict probability normalization.'
- What this solution (achieved 1.40995) has done: 'To move your score down toward the target (lower is better), the smallest safe improvement is to make training labels match the evaluation granularity: compute the per-`eeg_id` target distribution as a **vote-count weighted mean** (not a simple mean) so EEGs with more annotator votes contribute proportionally. Then, to reduce overfitting without changing the model/loop, I keep your exact architecture and training procedure but slightly lower dropout from 0.5 to 0.3 (still the same model, just less underfitting), and increase the unique-`eeg_id` training subset modestly (512 → 768) while staying within the time budget. Finally, I keep your KL loss and submission normalization unchanged to preserve evaluation semantics and ensure valid probabilities.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import subprocess


def _ensure_protobuf_compatible():
    """
    Bugfix: TF 2.18 can crash with protobuf 6.x in some Kaggle runtimes.
    Keep this minimal compatibility shim so imports succeed.
    """
    try:
        import google.protobuf
        from packaging.version import Version

        pb_ver = Version(google.protobuf.__version__)
        if pb_ver >= Version("5.0.0"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            import importlib

            importlib.invalidate_caches()
            for k in list(sys.modules.keys()):
                if k.startswith("google.protobuf"):
                    del sys.modules[k]
    except Exception as e:
        print("Warning: protobuf compatibility check/install failed:", repr(e))


_ensure_protobuf_compatible()

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import gc

print("Python:", sys.version)
import google.protobuf  # noqa: E402

print("protobuf:", google.protobuf.__version__)
print("tensorflow:", tf.__version__)



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 2
target_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"

test_eegs = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
test_spectrograms = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
train_eegs = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
train_spectrograms = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)



## === cell 3
ds_test = pd.read_csv(test_path)
ds_train = pd.read_csv(train_path)

vote_sum = ds_train[target_cols].sum(axis=1).astype(np.float32)
ds_train[target_cols] = (
    ds_train[target_cols]
    .astype(np.float32)
    .div(vote_sum, axis=0)
    .fillna(1.0 / len(target_cols))
)

print(ds_test.shape, ds_train.shape)
ds_test.head()




## === cell 4
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




## === cell 5
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

    def call(self, X):
        x1 = X["x1"]
        x2 = X["x2"]
        x = tf.concat([x1, x2], 2)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = self.layer1_100(x)
        x = self.layer1_6(x)
        x = tf.keras.layers.Softmax(-1)(x)
        return x




## === cell 6
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0.0
)



## === cell 7
eps = 1e-6

import pyarrow.parquet as pq
from functools import lru_cache


def _read_parquet_values_fast(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        table = pq.read_table(f)
    return table.to_pandas(self_destruct=True).values


def _norm_per_feature(x):
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    return (x - m) / (s + eps)


@lru_cache(maxsize=8192)
def _read_eeg_x1_cached(eeg_id, base_path, n_steps=300):
    try:
        arr = _read_parquet_values_fast(
            os.path.join(base_path, f"{int(eeg_id)}.parquet")
        )
        arr = np.nan_to_num(arr, nan=0.0)
        arr = arr[:n_steps, :]
        arr = np.clip(arr, np.exp(-6), np.exp(10))
        x1 = np.log(arr)
        x1 = _norm_per_feature(x1)
        return x1.astype(np.float32)
    except Exception:
        return np.zeros((n_steps, 20), dtype=np.float32)


@lru_cache(maxsize=8192)
def _read_spc_x2_cached(spectrogram_id, base_path, n_steps=300):
    try:
        arr = _read_parquet_values_fast(
            os.path.join(base_path, f"{int(spectrogram_id)}.parquet")
        )
        arr = np.nan_to_num(arr, nan=-1.0)
        arr = arr[:n_steps, :]
        if arr.shape[1] > 400:
            arr = arr[:, :400]
        elif arr.shape[1] < 400:
            pad = np.zeros((arr.shape[0], 400 - arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)

        arr = np.clip(arr, np.exp(-6), np.exp(10))
        x2 = np.log(arr)
        x2 = _norm_per_feature(x2)
        return x2.astype(np.float32)
    except Exception:
        return np.zeros((n_steps, 400), dtype=np.float32)


def build_xy_from_df(df, eeg_base, spc_base, n_steps=300):
    n = len(df)
    x1_out = np.empty((n, n_steps, 20), dtype=np.float32)
    x2_out = np.empty((n, n_steps, 400), dtype=np.float32)
    for i, r in enumerate(df.itertuples(index=False)):
        x1_out[i] = _read_eeg_x1_cached(getattr(r, "eeg_id"), eeg_base, n_steps=n_steps)
        x2_out[i] = _read_spc_x2_cached(
            getattr(r, "spectrogram_id"), spc_base, n_steps=n_steps
        )
    return {"x1": x1_out, "x2": x2_out}




## === cell 8
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local")
    print("Running on TPU")
except Exception:
    tpu = None

if tpu:
    strategy = tf.distribute.TPUStrategy(tpu)
    print("REPLICAS:", strategy.num_replicas_in_sync)
else:
    strategy = tf.distribute.get_strategy()
    print("Running on GPU/CPU")



## === cell 9
BATCH_SIZE = 32
EPOCHS = 1

raw_votes = pd.read_csv(
    train_path, usecols=["eeg_id", "spectrogram_id", "patient_id"] + target_cols
)
raw_votes[target_cols] = raw_votes[target_cols].astype(np.float32)
raw_votes["n_votes"] = (
    raw_votes[target_cols].sum(axis=1).astype(np.float32).clip(lower=1.0)
)

row_dist = (
    raw_votes[target_cols]
    .div(raw_votes["n_votes"], axis=0)
    .fillna(1.0 / len(target_cols))
    .astype(np.float32)
)

weighted = row_dist.mul(raw_votes["n_votes"], axis=0)
weighted["eeg_id"] = raw_votes["eeg_id"].values
weighted["spectrogram_id"] = raw_votes["spectrogram_id"].values
weighted["patient_id"] = raw_votes["patient_id"].values
weighted["n_votes"] = raw_votes["n_votes"].values

spc_mode = (
    raw_votes.groupby("eeg_id")["spectrogram_id"]
    .agg(lambda s: s.value_counts().idxmax())
    .rename("spectrogram_id")
    .reset_index()
)

pat_first = (
    raw_votes.groupby("eeg_id", as_index=False)["patient_id"]
    .first()
    .rename(columns={"patient_id": "patient_id"})
)

agg_num = (
    weighted.groupby("eeg_id", as_index=False)
    .agg(
        seizure_vote=("seizure_vote", "sum"),
        lpd_vote=("lpd_vote", "sum"),
        gpd_vote=("gpd_vote", "sum"),
        lrda_vote=("lrda_vote", "sum"),
        grda_vote=("grda_vote", "sum"),
        other_vote=("other_vote", "sum"),
        n_votes=("n_votes", "sum"),
    )
    .merge(spc_mode, on="eeg_id", how="left")
    .merge(pat_first, on="eeg_id", how="left")
    .reset_index(drop=True)
)

p = agg_num[target_cols].values.astype(np.float32)
p = np.clip(p, 1e-7, 1.0)
p = p / p.sum(axis=1, keepdims=True)
agg_num[target_cols] = p

train_subset = agg_num.reset_index(drop=True)

X_train = build_xy_from_df(train_subset, train_eegs, train_spectrograms, n_steps=300)
y_train = train_subset[target_cols].values.astype(np.float32)

with strategy.scope():
    model = RnnModel(num_layers=3, d_model=420, num_heads=1, dff=400, dropout_rate=0.3)
    model.build(input_shape={"x1": [None, 300, 20], "x2": [None, 300, 400]})
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_fn)

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)
model_list = [model]

for _name in [
    "X_train",
    "y_train",
    "train_subset",
    "agg_num",
    "p",
    "raw_votes",
    "row_dist",
    "weighted",
    "spc_mode",
    "pat_first",
]:
    if _name in globals():
        del globals()[_name]
gc.collect()




## === cell 10
def prediction_re(ds):
    if len(model_list) == 0:
        n = ds["x1"].shape[0]
        return np.full((n, 6), 1.0 / 6.0, dtype=np.float32)
    preds = 0.0
    for model_p in model_list:
        preds += model_p.predict(ds, verbose=0, batch_size=256)
    preds = preds / float(len(model_list))
    return preds




## === cell 11
span = 256  # smaller batch to limit RAM
n = (ds_test.shape[0] + span - 1) // span

all_preds = []
for i in range(n):
    temp_ds = ds_test.iloc[i * span : (i + 1) * span, :].reset_index(drop=True)
    pre_data = build_xy_from_df(temp_ds, test_eegs, test_spectrograms, n_steps=300)
    batch_preds = prediction_re(pre_data)
    all_preds.append(batch_preds)

predictins = np.concatenate(all_preds, axis=0).astype(np.float32)
print("pred shape:", predictins.shape)



## === cell 12
predictins = np.clip(predictins, 1e-7, 1.0)
predictins = predictins / predictins.sum(axis=1, keepdims=True)

predic_ds = ds_test[["eeg_id"]].copy()
for i, c in enumerate(target_cols):
    predic_ds[c] = predictins[:, i]

sample_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_path)
predic_ds = sample_sub[["eeg_id"]].merge(predic_ds, on="eeg_id", how="left")

for c in target_cols:
    if c not in predic_ds.columns:
        predic_ds[c] = 1.0 / len(target_cols)
predic_ds[target_cols] = predic_ds[target_cols].fillna(1.0 / len(target_cols))

vals = predic_ds[target_cols].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-7, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
predic_ds[target_cols] = vals
predic_ds = predic_ds[["eeg_id"] + target_cols]



## === cell 13
predic_ds.to_csv("submission.csv", index=False)
print(predic_ds.head())



## === cell 14
sub = pd.read_csv("submission.csv")
print(sub.shape)
print(sub.columns.tolist())
print(
    "row sum stats:",
    sub[target_cols].sum(axis=1).min(),
    sub[target_cols].sum(axis=1).max(),
)
