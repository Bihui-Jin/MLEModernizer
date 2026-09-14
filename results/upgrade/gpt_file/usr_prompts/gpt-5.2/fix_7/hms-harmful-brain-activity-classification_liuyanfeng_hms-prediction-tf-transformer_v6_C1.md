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

2.069964948904714

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the protobuf environment overrides that are triggering the TensorFlow/protobuf `MessageFactory.GetPrototype` crash in this Kaggle environment. Then I fix the test/sample merge logic to guarantee a strict 1:1 alignment with `sample_submission.csv` by mapping metadata onto the sample submission index rather than doing a merge that can duplicate rows. Finally, I ensure the generated prediction array length always equals 9850 and that the written `submission.csv` contains all required target columns with per-row probabilities normalized to sum to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf environment variables are cleared before importing TensorFlow (your current cell imports TF first, so the override removal happens too late). I also make the weight-loading path robust by checking both possible dataset base directories so the model can actually load weights when available (score should remain consistent with your intended setup, otherwise it falls back to uniform). Finally, I keep your strict 1:1 alignment with `sample_submission.csv`, preserve your preprocessing/model logic, and ensure the submission probabilities are valid and normalized.'

# 9. Code solution

## === cell 0
import os

for k in [
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
]:
    if k in os.environ:
        os.environ.pop(k, None)

import time
import gc
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE1 = "/kaggle/input/hms-harmful-brain-activity-classification"
BASE2 = "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification"


def pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


target_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

test_path = pick_existing(f"{BASE1}/test.csv", f"{BASE2}/test.csv")
sample_sub_path = pick_existing(
    f"{BASE1}/sample_submission.csv", f"{BASE2}/sample_submission.csv"
)
test_eegs = pick_existing(f"{BASE1}/test_eegs/", f"{BASE2}/test_eegs/")
test_spectrograms = pick_existing(
    f"{BASE1}/test_spectrograms/", f"{BASE2}/test_spectrograms/"
)

assert os.path.exists(test_path), f"Missing test.csv at {test_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.isdir(test_eegs), f"Missing test_eegs dir at {test_eegs}"
assert os.path.isdir(
    test_spectrograms
), f"Missing test_spectrograms dir at {test_spectrograms}"



## === cell 2
ds_test_meta = pd.read_csv(
    test_path, usecols=["eeg_id", "spectrogram_id", "patient_id"]
)
sample_sub = pd.read_csv(sample_sub_path)

meta_idx = ds_test_meta.drop_duplicates(subset=["eeg_id"]).set_index("eeg_id")

ds_test = sample_sub[["eeg_id"]].copy()
ds_test["spectrogram_id"] = ds_test["eeg_id"].map(meta_idx["spectrogram_id"])
ds_test["patient_id"] = ds_test["eeg_id"].map(meta_idx["patient_id"])

print(ds_test.head())
print(
    "test rows (must equal sample_sub rows):",
    len(ds_test),
    "sample_sub rows:",
    len(sample_sub),
)
assert len(ds_test) == len(
    sample_sub
), "ds_test must be aligned 1:1 with sample_submission."
assert (
    ds_test["eeg_id"].isna().sum() == 0
), "sample_submission eeg_id should all exist in test.csv"
print("Missing spectrogram_id rows:", int(ds_test["spectrogram_id"].isna().sum()))




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
        self.softmax = tf.keras.layers.Softmax(-1)

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




## === cell 5
models_infor = [
    [0, 160],
]

WEIGHTS_BASE = pick_existing(
    "/kaggle/input/hms-train-model-tf-transformer",
    "/kaggle/input/hms-train-model-tf-transformer/hms-train-model-tf-transformer",
)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local")
    print("Running on TPU")
except Exception:
    tpu = None

if tpu:
    strategy = tf.distribute.TPUStrategy(tpu)
    print("on TPU")
    print("REPLICAS:", strategy.num_replicas_in_sync)
else:
    print("on GPU/CPU")
    strategy = tf.distribute.get_strategy()

model_list = []
missing_weights = []

with strategy.scope():
    for m_i in models_infor:
        model = RnnModel(
            num_layers=3, d_model=420, num_heads=2, dff=400, dropout_rate=0.5
        )
        model.build(input_shape={"x1": [1, 300, 20], "x2": [1, 300, 400]})
        w_path = f"{WEIGHTS_BASE}/{m_i[0]}_weights/model_epoch_{m_i[1]}.weights.h5"
        if os.path.exists(w_path):
            model.load_weights(w_path)
            model(
                tf.nest.map_structure(
                    lambda s: tf.zeros(s, tf.float32),
                    {"x1": (1, 300, 20), "x2": (1, 300, 400)},
                )
            )
            model_list.append(model)
        else:
            missing_weights.append(w_path)

print("Loaded models:", len(model_list))
if missing_weights:
    print("Missing weight files (will use uniform fallback if none loaded):")
    for p in missing_weights[:5]:
        print(" -", p)



## === cell 6
eps = 1e-6
from concurrent.futures import ThreadPoolExecutor

_PARQUET_ENGINE = None
try:
    import pyarrow  # noqa: F401

    _PARQUET_ENGINE = "pyarrow"
except Exception:
    try:
        import fastparquet  # noqa: F401

        _PARQUET_ENGINE = "fastparquet"
    except Exception:
        _PARQUET_ENGINE = None


def _read_parquet_fast(path):
    if _PARQUET_ENGINE is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, engine=_PARQUET_ENGINE)


def _load_and_preprocess_one(eeg_id, spectrogram_id):
    eeg_path = os.path.join(test_eegs, f"{int(eeg_id)}.parquet")
    spc_path = os.path.join(test_spectrograms, f"{int(spectrogram_id)}.parquet")

    values_eeg = _read_parquet_fast(eeg_path).fillna(0).to_numpy()[:300, :]
    values_spc = _read_parquet_fast(spc_path).fillna(-1).to_numpy()[:300, 1:]

    values_eeg = np.clip(values_eeg, np.exp(-6), np.exp(10))
    x1 = np.log(values_eeg)
    data_mean = x1.mean(axis=(0, 1))
    data_std = x1.std(axis=(0, 1))
    x1 = (x1 - data_mean) / (data_std + eps)

    values_spc = np.clip(values_spc, np.exp(-6), np.exp(10))
    x2 = np.log(values_spc)
    data_mean = x2.mean(axis=(0, 1))
    data_std = x2.std(axis=(0, 1))
    x2 = (x2 - data_mean) / (data_std + eps)

    return x1.astype(np.float32), x2.astype(np.float32)


def prediction_re_from_arrays(x1_arr, x2_arr, batch_size):
    n_rows = x1_arr.shape[0]
    if len(model_list) == 0:
        return np.full((n_rows, 6), 1.0 / 6.0, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices({"x1": x1_arr, "x2": x2_arr})
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    preds_sum = None
    for m in model_list:
        pm = m.predict(ds, verbose=0)
        preds_sum = pm if preds_sum is None else (preds_sum + pm)
    return (preds_sum / float(len(model_list))).astype(np.float32)


eeg_ids = ds_test["eeg_id"].to_numpy(dtype=np.int64)
spectrogram_ids = ds_test["spectrogram_id"].to_numpy(
    dtype=np.float64
)  # may contain NaN
valid_mask = np.isfinite(spectrogram_ids)

n = len(ds_test)
BATCH = 64

predictins = np.full((n, 6), 1.0 / 6.0, dtype=np.float32)

if valid_mask.any() and len(model_list) > 0:
    eeg_valid = eeg_ids[valid_mask]
    spc_valid = spectrogram_ids[valid_mask].astype(np.int64)

    x1_all = np.empty((len(eeg_valid), 300, 20), dtype=np.float32)
    x2_all = np.empty((len(eeg_valid), 300, 400), dtype=np.float32)

    max_workers = min(16, (os.cpu_count() or 4))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, (x1, x2) in enumerate(
            ex.map(_load_and_preprocess_one, eeg_valid, spc_valid)
        ):
            x1_all[i] = x1
            x2_all[i] = x2
    print(
        f"Loaded+preprocessed {len(eeg_valid)} rows in {time.time()-t0:.1f}s with {max_workers} workers"
    )

    preds_valid = prediction_re_from_arrays(x1_all, x2_all, BATCH)
    assert preds_valid.shape == (len(eeg_valid), 6)
    predictins[valid_mask] = preds_valid

assert predictins is not None and predictins.shape == (ds_test.shape[0], 6)



## === cell 7
predic_ds = sample_sub[["eeg_id"]].copy()

for i, c in enumerate(target_cols):
    predic_ds[c] = predictins[:, i].astype(np.float64)

row_sum = predic_ds[target_cols].sum(axis=1).to_numpy()
row_sum[row_sum == 0] = 1.0
predic_ds[target_cols] = predic_ds[target_cols].div(row_sum, axis=0)

assert len(predic_ds) == len(
    sample_sub
), "Submission must match sample_submission length."
assert list(predic_ds.columns) == ["eeg_id"] + target_cols
assert np.allclose(predic_ds[target_cols].sum(axis=1).values, 1.0, atol=1e-6)

predic_ds.to_csv("submission.csv", index=None)
print("Wrote submission.csv with shape:", predic_ds.shape)
print(pd.read_csv("submission.csv").head())
print("Columns:", list(pd.read_csv("submission.csv", nrows=1).columns))
