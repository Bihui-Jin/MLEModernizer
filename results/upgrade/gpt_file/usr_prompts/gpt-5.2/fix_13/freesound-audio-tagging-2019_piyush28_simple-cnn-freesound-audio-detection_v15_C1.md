# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Develop a model to tag audio data automatically using a diverse vocabulary of 80 categories.

## Metric
The task consists of predicting the audio labels (tags) for every test clip. Some test clips bear one label while others bear several labels. The predictions are to be done at the clip level, i.e., no start/end timestamps for the sound events are required.

The primary metric is label-weighted label-ranking average precision. 

The  "label-weighted" part means that the overall score is the average over all the *labels* in the test set, where each label receives equal weight (by contrast, plain *lrap* gives each *test item* equal weight).

## Submission Format
For each `fname` in the test set, you must predict the probability of each label. The file should contain a header and have the following format:

```
fname,Accelerating_and_revving_and_vroom,...Zipper_(clothing)
000ccb97.wav,0.1,....,0.3
0012633b.wav,0.0,...,0.8
```

## Dataset
The following 5 audio files in the curated train set have a wrong label, due to a bug in the file renaming process:\
`f76181c4.wav, 77b925c2.wav, 6a1f682a.wav, c7db12aa.wav, 7752cc8a.wav`

The audio file `1d44b0bd.wav` in the curated train set was found to be corrupted (contains no signal) due to an error in format conversion.

- **train_curated.csv** - ground truth labels for the curated subset of the training audio files (see Data Fields below)
- **train_noisy.csv** - ground truth labels for the noisy subset of the training audio files (see Data Fields below)
- **sample_submission.csv** - a sample submission file in the correct format, including the correct sorting of the sound categories; it contains the list of audio files found in the test.zip folder (corresponding to the public leaderboard)
- **train_curated.zip** - a folder containing the audio (.wav) training files of the curated subset
- **train_noisy.zip** - a folder containing the audio (.wav) training files of the noisy subset
- **test.zip** - a folder containing the audio (.wav) test files for the public leaderboard

### Columns
Each row of the train_curated.csv and train_noisy.csv files contains the following information:

- **fname**: the audio file name, eg, `0006ae4e.wav`
- **labels**: the audio classification label(s) (ground truth). Note that the number of labels per clip can be one, eg, `Bark` or more, eg, `"Walk_and_footsteps,Slam"`.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        input/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        working/
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
```

-> data/freesound-audio-tagging-2019/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/freesound-audio-tagging-2019/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/freesound-audio-tagging-2019/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> data/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.4151301852907714

# 6. Current score

0.06853

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06442) has done: 'I fix the TensorFlow import/runtime crash by removing the eager-disable path and forcing TF1-style graph mode in a way that’s compatible with the Kaggle TF/protobuf stack. Then I fix test file enumeration so directories like `test/test` are excluded (that caused the `ReadFile ... Is a directory` error and the submission length mismatch). Finally, I ensure the submission rows exactly match `sample_submission.csv` ordering/length by using its `fname` list to drive prediction, which is score-neutral but guarantees a valid submission file is produced.'
- What this solution (achieved 0.06533) has done: 'We fix the TensorFlow/protobuf crash by avoiding the TF1 `disable_v2_behavior()` path (which is triggering the `MessageFactory.GetPrototype` issue in this environment) and instead run in TF2 with `tf.compat.v1` graph execution using `disable_eager_execution()` only. We also ensure the graph is reset at the right times and that sessions use the compat-v1 API consistently, without changing your model, preprocessing, loss, or training loop semantics. Finally, we keep submission ordering driven strictly by `sample_submission.csv` so the output is always valid and aligned, while keeping the approach score-neutral aside from enabling training/inference to run correctly.'
- What this solution (achieved 0.07307) has done: 'We fix the immediate TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by switching the script to use `tf.compat.v1` graph mode only after importing TensorFlow through a safer Keras path and by avoiding the problematic mixed TF/Keras imports. Then we ensure the graph/session lifecycle is consistent (reset graph before building Trainor/Predictor, and avoid stale graphs) so training/inference run end-to-end. Finally, we keep submission ordering strictly driven by `sample_submission.csv` and always write `submission.csv` with the exact required columns, which is score-neutral but guarantees a valid file. These changes are minimal and do not alter your model architecture, preprocessing, loss, or training loop semantics.'
- What this solution (achieved 0.06911) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing a safe pure-TF2 import path and using only `tf.compat.v1` APIs (graph mode via `disable_eager_execution`) while avoiding mixed/legacy TF1 toggles that trigger the protobuf issue. Then I make training actually run by default (the current `do_run=False` causes random-weight inference and the very low score), while keeping your exact model, preprocessing, loss, and training loop intact. I also add small stability guards (skip the known corrupted curated wav, and ensure label binarization order matches `sample_submission.csv`) without changing core semantics. Finally, I keep test prediction strictly aligned to `sample_submission.csv` order and always write a valid `submission.csv`.'
- What this solution (achieved 0.06853) has done: 'I fix the two execution blockers without changing your model/training logic: (1) avoid the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by using the safe `tensorflow.compat.v1` import path and graph mode only, and (2) fix the `tf.data` iterator construction by switching from deprecated `output_types/output_shapes` to `element_spec`-derived types/shapes (TF2-compatible while still using v1 sessions). These changes should let training actually run end-to-end and then produce predictions aligned exactly to `sample_submission.csv` ordering/columns, which should move the score up toward your target. I also keep the known corrupted curated wav excluded as you already do, and keep submission writing unchanged except for ensuring it always completes.'
- What this solution (achieved 0.02197) has done: 'Main runtime bottlenecks are (1) the very large `train(10000)` step count and (2) the TFRecord cache creation doing expensive per-example Python loops and repeatedly spinning up TF sessions per shard. To fit within 600s without changing the model/training semantics, I keep the exact architecture/loss/training op but (a) cap training steps by remaining wall-clock time (so it always finishes) and (b) make TFRecord writing provably equivalent but much faster by serializing whole batches at once with `tf.io.serialize_tensor` + `tf.io.serialize_example` inside the TF graph (eliminating Python per-record loops) and by reusing a single session for all shards. I also avoid redoing cache work by writing a robust meta file and using `tf.io.gfile` checks, and I keep determinism/seeds intact. No sampling, no early stopping, no reduced precision; the only behavioral change is that training steps may be fewer if time run out, ensuring the notebook completes.'
- What this solution (achieved 0.06853) has done: 'Main bottleneck is the per-file TF1 `Session.run()` loop used to build TFRecord caches (both train/val and test), which repeatedly crosses the Python↔TF boundary and also does an extra `.eval()` per example; this dominates wall time and causes the timeout. I keep the exact spectrogram logic and model/training semantics, but make cache building run as a `tf.data` pipeline that computes STFT+resize in-graph with parallel mapping and writes TFRecords in large batches, eliminating millions of tiny session calls. I also remove redundant per-example tensor serialization evals, reuse a single session/graph for cache writing, and add deterministic options and efficient TFRecord read settings to keep results stable. Training loop, model, loss, and evaluation remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import json
import gc
import hashlib
import math
import time
import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf1

tf1.disable_eager_execution()
tf = tf1  # keep downstream references stable (tf.io, tf.data, etc.)

from sklearn.preprocessing import MultiLabelBinarizer

np.random.seed(42)
tf1.set_random_seed(42)

INPUT_ROOT = "/kaggle/input/freesound-audio-tagging-2019"
WORKING_DIR = "/kaggle/working"
os.makedirs(WORKING_DIR, exist_ok=True)
os.chdir(WORKING_DIR)

try:
    tf1.config.optimizer.set_jit(True)
except Exception:
    pass

RUN_START_TS = time.time()
TIME_LIMIT_SEC = 600
TIME_SAFETY_MARGIN_SEC = 120

print("Using TF compat.v1 version:", tf1.__version__)
print("INPUT_ROOT exists:", os.path.exists(INPUT_ROOT))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
do_run = True



## === cell 2
train_curated_csv = os.path.join(INPUT_ROOT, "train_curated.csv")
train_noisy_csv = os.path.join(INPUT_ROOT, "train_noisy.csv")
sample_sub_csv = os.path.join(INPUT_ROOT, "sample_submission.csv")

train_curated_path = os.path.join(INPUT_ROOT, "train_curated")
train_noisy_path = os.path.join(INPUT_ROOT, "train_noisy")
test_path = os.path.join(INPUT_ROOT, "test")

assert os.path.exists(train_curated_csv), train_curated_csv
assert os.path.exists(train_noisy_csv), train_noisy_csv
assert os.path.exists(sample_sub_csv), sample_sub_csv
assert os.path.isdir(train_curated_path), train_curated_path
assert os.path.isdir(train_noisy_path), train_noisy_path
assert os.path.isdir(test_path), test_path

print("n_test_entries_in_dir:", len(os.listdir(test_path)))



## === cell 3
sample_sub = pd.read_csv(sample_sub_csv)
label_cols = [c for c in sample_sub.columns if c != "fname"]
labels_dict = {label: i for i, label in enumerate(label_cols)}
labels_set = set(labels_dict.keys())

train_curated_df = pd.read_csv(train_curated_csv)
train_noisy_df = pd.read_csv(train_noisy_csv)

train_curated_df = train_curated_df[
    train_curated_df["fname"] != "1d44b0bd.wav"
].reset_index(drop=True)

train_curated_df["fname"] = train_curated_df["fname"].apply(
    lambda x: os.path.join(train_curated_path, x)
)
train_noisy_df["fname"] = train_noisy_df["fname"].apply(
    lambda x: os.path.join(train_noisy_path, x)
)

noisy_split = train_noisy_df["labels"].fillna("").str.split(",")
train_noisy_df["labels"] = noisy_split.apply(
    lambda labs: ",".join([lab for lab in labs if lab in labels_set])
)

split = int(0.2 * train_curated_df.shape[0])
indices = np.random.permutation(train_curated_df.shape[0])
val_indices = indices[:split]
train_indices = indices[split:]
val_curated_df = train_curated_df.iloc[val_indices].reset_index(drop=True)
train_curated_df = train_curated_df.iloc[train_indices].reset_index(drop=True)

train_df = pd.concat([train_curated_df, train_noisy_df], axis=0).reset_index(drop=True)

os.makedirs("./preprocessed", exist_ok=True)
train_curated_df.to_csv("./preprocessed/train_curated.csv", index=False)
train_noisy_df.to_csv("./preprocessed/train_noisy.csv", index=False)
val_curated_df.to_csv("./preprocessed/val_curated.csv", index=False)
train_df.to_csv("./preprocessed/train.csv", index=False)
with open("./preprocessed/labels_dict.json", "w") as fp:
    json.dump(labels_dict, fp)

print("Prepared:", train_df.shape, "labels:", len(labels_dict))



## === cell 4
for _name in [
    "train_curated_df",
    "train_noisy_df",
    "train_df",
    "train_indices",
    "val_indices",
    "indices",
    "noisy_split",
]:
    if _name in globals():
        del globals()[_name]
gc.collect()




## === cell 5
def wav_to_spectogram(wav_filename):
    wav_bytes = tf.io.read_file(wav_filename)
    audio_decoded, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    audio_decoded = tf.squeeze(audio_decoded, axis=-1)  # [samples]

    stft = tf.signal.stft(
        audio_decoded,
        frame_length=1024,
        frame_step=64,
        fft_length=1024,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    spec = tf.abs(stft)  # [time, freq]

    spec = tf.minimum(spec, 255.0)
    spec = tf.expand_dims(spec, axis=-1)  # [time, freq, 1]
    spec = tf.image.resize(spec, [300, 300], method=tf.image.ResizeMethod.BILINEAR)
    return spec  # [300, 300, 1]


_CSV_PIPE_CACHE = {}


def csv_pipe(csv, labels_dict):
    key = (os.path.abspath(csv), json.dumps(labels_dict, sort_keys=True))
    if key in _CSV_PIPE_CACHE:
        return _CSV_PIPE_CACHE[key]

    df = pd.read_csv(csv)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    files = df["fname"].values
    labels = df["labels"].fillna("").str.split(",")

    classes = list(labels_dict.keys())
    binarizer = MultiLabelBinarizer(classes=classes)
    labels_1hot = binarizer.fit_transform(labels).astype(np.int32)

    _CSV_PIPE_CACHE[key] = (files, labels_1hot)
    return files, labels_1hot


def _hash_for_cache_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def ensure_tfrecord_cache(
    csv, labels_dict, cache_root="./preprocessed/spec_cache", shards=64
):
    os.makedirs(cache_root, exist_ok=True)
    cache_id = _hash_for_cache_id(
        os.path.abspath(csv) + "|" + json.dumps(labels_dict, sort_keys=True)
    )
    cache_dir = os.path.join(cache_root, cache_id)
    os.makedirs(cache_dir, exist_ok=True)

    meta_path = os.path.join(cache_dir, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as fp:
                meta = json.load(fp)
            ok = True
            for p in meta.get("tfrecords", []):
                if not os.path.isfile(p):
                    ok = False
                    break
            if (
                ok
                and int(meta.get("n_classes", -1)) == int(len(labels_dict))
                and int(meta.get("n", -1)) > 0
            ):
                return meta
        except Exception:
            pass  # fall through to rebuild

    files, labels_1hot = csv_pipe(csv, labels_dict)
    n = len(files)
    shards = int(max(1, min(shards, n)))
    shard_sizes = [n // shards] * shards
    for i in range(n % shards):
        shard_sizes[i] += 1
    shard_offsets = np.cumsum([0] + shard_sizes)

    cfg = tf1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    cfg.intra_op_parallelism_threads = 0
    cfg.inter_op_parallelism_threads = 0

    tfrecord_paths = []

    g = tf1.Graph()
    with g.as_default():
        fn_ph = tf1.placeholder(tf.string, shape=[None], name="fn_batch")
        lab_ph = tf1.placeholder(
            tf.int64, shape=[None, len(labels_dict)], name="lab_batch"
        )

        ds = tf.data.Dataset.from_tensor_slices((fn_ph, lab_ph))

        options = tf.data.Options()
        options.experimental_deterministic = True
        ds = ds.with_options(options)

        def _map_one(fn, lab):
            spec = wav_to_spectogram(fn)  # [300,300,1] float32
            spec_bytes = tf.io.serialize_tensor(spec)
            return spec_bytes, lab

        ds = ds.map(_map_one, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(64, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)

        it = tf1.data.make_initializable_iterator(ds)
        spec_b, lab_b = it.get_next()

        def _py_write_tfrecord(path, specs, labs):
            writer = tf.io.TFRecordWriter(path)
            for sb, lb in zip(specs, labs):
                feat = {
                    "spec": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sb])),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=lb.tolist())
                    ),
                }
                ex = tf.train.Example(features=tf.train.Features(feature=feat))
                writer.write(ex.SerializeToString())
            writer.close()
            return np.int64(len(specs))

        out_path_ph = tf1.placeholder(tf.string, shape=(), name="out_path")
        written_t = tf1.py_func(
            _py_write_tfrecord, [out_path_ph, spec_b, lab_b], tf.int64
        )

    with tf1.Session(graph=g, config=cfg) as sess:
        sess.run(tf1.global_variables_initializer())
        sess.run(tf1.local_variables_initializer())

        for shard_idx in range(shards):
            start = int(shard_offsets[shard_idx])
            end = int(shard_offsets[shard_idx + 1])
            shard_files = files[start:end].astype(np.str_)
            shard_labels = labels_1hot[start:end].astype(np.int64, copy=False)

            out_path = os.path.join(
                cache_dir, f"data-{shard_idx:03d}-{shards:03d}.tfrecord"
            )

            sess.run(
                it.initializer, feed_dict={fn_ph: shard_files, lab_ph: shard_labels}
            )

            total_written = 0
            while True:
                try:
                    w = sess.run(written_t, feed_dict={out_path_ph: out_path})
                    total_written += int(w)
                except tf.errors.OutOfRangeError:
                    break

            assert total_written == (end - start), (total_written, end - start)
            tfrecord_paths.append(out_path)

    meta = {
        "cache_id": cache_id,
        "csv": os.path.abspath(csv),
        "n": int(n),
        "n_classes": int(len(labels_dict)),
        "tfrecords": tfrecord_paths,
        "spec_shape": [300, 300, 1],
        "spec_dtype": "float32",
        "label_dtype": "int64",
        "encoding": "tf.io.serialize_tensor(spec)",
    }
    with open(meta_path, "w") as fp:
        json.dump(meta, fp)
    return meta


def _parse_cached_example(n_classes):
    def _fn(x):
        feat = {
            "spec": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([n_classes], tf.int64),
        }
        ex = tf.io.parse_single_example(x, feat)
        spec = tf.io.parse_tensor(ex["spec"], out_type=tf.float32)
        spec = tf.reshape(spec, [300, 300, 1])
        label = tf.cast(ex["label"], tf.int32)
        return spec, label

    return _fn


def wav_data_generators(csv, labels_dict, batch_size=32, shuffle=False, repeat=True):
    meta = ensure_tfrecord_cache(csv, labels_dict)
    n_classes = int(meta["n_classes"])
    tfrecord_paths = meta["tfrecords"]

    dataset = tf.data.TFRecordDataset(
        tfrecord_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    dataset = dataset.map(
        _parse_cached_example(n_classes),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(1000, seed=42, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    if repeat:
        dataset = dataset.repeat()

    return dataset




## === cell 6
def conv_relu_bn(filters, kernels, strides, padding="valid", in_shape=None):
    ls = tf.keras.layers
    if in_shape is not None:
        conv = ls.Conv2D(
            filters, kernels, strides, padding=padding, input_shape=in_shape
        )
    else:
        conv = ls.Conv2D(filters, kernels, strides, padding=padding)
    return tf.keras.models.Sequential([conv, ls.ReLU(), ls.BatchNormalization()])


class Model(tf.keras.Model):
    def __init__(self, in_shape, n_classes=80):
        super(Model, self).__init__()
        ls = tf.keras.layers
        self.conv_relu_bn1 = conv_relu_bn(64, 3, 1, padding="same", in_shape=in_shape)
        self.conv_relu_bn2 = conv_relu_bn(64, 3, 2)
        self.conv_relu_bn3 = conv_relu_bn(128, 3, 1, padding="same")
        self.conv_relu_bn4 = conv_relu_bn(128, 3, 2)
        self.conv_relu_bn5 = conv_relu_bn(256, 3, 1, padding="same")
        self.conv_relu_bn6 = conv_relu_bn(256, 3, 2)
        self.conv_relu_bn7 = conv_relu_bn(512, 3, 1, padding="same")
        self.conv_relu_bn8 = conv_relu_bn(512, 3, 2)
        self.conv_relu_bn9 = conv_relu_bn(1024, 3, 2)
        self.gpool = ls.GlobalAveragePooling2D()
        self.classifier = ls.Dense(n_classes)

    def call(self, inputs):
        x = self.conv_relu_bn1(inputs)
        x = self.conv_relu_bn2(x)
        x = self.conv_relu_bn3(x)
        x = self.conv_relu_bn4(x)
        x = self.conv_relu_bn5(x)
        x = self.conv_relu_bn6(x)
        x = self.conv_relu_bn7(x)
        x = self.conv_relu_bn8(x)
        x = self.conv_relu_bn9(x)
        x = self.gpool(x)
        return self.classifier(x)




## === cell 7
class Trainor:
    def __init__(
        self,
        train_csv,
        val_csv,
        labels_dict,
        lr=1e-3,
        batch_size=16,
        shuffle=True,
        model_dir="ds2",
    ):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = Model(in_shape=(300, 300, 1), n_classes=len(labels_dict))
        self._define_dataset_iterators(
            train_csv, val_csv, labels_dict, batch_size, shuffle
        )

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)
        preds = self.probas > 0.5

        self.loss = tf1.losses.sigmoid_cross_entropy(self.labels, logits)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(preds, tf.int32), self.labels), tf.float32)
        )

        optimizer = tf1.train.AdamOptimizer(lr)
        self.global_step = tf1.train.get_or_create_global_step()
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        cfg = tf1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.intra_op_parallelism_threads = 0
        cfg.inter_op_parallelism_threads = 0
        self.sess = tf1.Session(config=cfg)

        self._define_summaries()
        self.sess.run(tf1.global_variables_initializer())
        self.saver = tf1.train.Saver()
        self._may_be_load_model()

    def _define_summaries(self):
        self.summary_loss = tf1.summary.scalar("loss", self.loss)
        self.summary_acc = tf1.summary.scalar("accuracy", self.accuracy)
        self.scalar_summaries = tf1.summary.merge([self.summary_loss, self.summary_acc])

        self.image_summary = tf1.summary.merge([])
        self.summary_writer = tf1.summary.FileWriter(self.model_dir, self.sess.graph)

    def _define_dataset_iterators(
        self, train_csv, val_csv, labels_dict, batch_size, shuffle
    ):
        train_dataset = wav_data_generators(
            train_csv, labels_dict, batch_size, shuffle, repeat=True
        )
        val_dataset = wav_data_generators(
            val_csv, labels_dict, batch_size, shuffle=False, repeat=True
        )

        spec = train_dataset.element_spec
        output_types = tf.nest.map_structure(lambda s: s.dtype, spec)
        output_shapes = tf.nest.map_structure(lambda s: s.shape, spec)

        it = tf1.data.Iterator.from_structure(output_types, output_shapes)
        self.features, self.labels = it.get_next()
        self.train_init_op = it.make_initializer(train_dataset)
        self.val_init_op = it.make_initializer(val_dataset)

    def _may_be_load_model(self):
        if os.path.isdir(self.model_dir):
            files = os.listdir(self.model_dir)
            steps = []
            for f in files:
                if "model.ckpt-" in f and f.endswith(".index"):
                    try:
                        steps.append(int(f.split("model.ckpt-")[-1].split(".")[0]))
                    except Exception:
                        pass
            if len(steps) > 0:
                self._load_model(os.path.join(self.model_dir, "model.ckpt"), max(steps))

    def _save_model(self):
        return self.saver.save(
            self.sess,
            os.path.join(self.model_dir, "model.ckpt"),
            global_step=self.global_step,
        )

    def _load_model(self, ckpt_path, step):
        self.saver.restore(self.sess, ckpt_path + "-{}".format(step))

    def train(
        self,
        steps=500,
        ckpt_every_n_steps=100,
        log_every_n_steps=1500,
        log_val_n_steps=5,
        max_wall_time_sec=None,
    ):
        self.sess.run(self.train_init_op)

        last_ckpt_save_step = 0
        for step in range(1, steps + 1):
            if (
                max_wall_time_sec is not None
                and (time.time() - RUN_START_TS) > max_wall_time_sec
            ):
                print("Stopping training due to wall-time limit at local step:", step)
                break

            if step % log_every_n_steps == 0:
                scalar_sum, _ = self.sess.run([self.scalar_summaries, self.train_op])
                self.summary_writer.add_summary(scalar_sum, step)
                self.summary_writer.flush()

                self.sess.run(self.val_init_op)
                val_losses, val_accs = [], []
                for _ in range(log_val_n_steps):
                    l, a = self.sess.run([self.loss, self.accuracy])
                    val_losses.append(l)
                    val_accs.append(a)
                print(
                    "STEP: {}, Loss: {:.4f}, Acc: {:.4f}".format(
                        step, np.mean(val_losses), np.mean(val_accs)
                    )
                )
                self.sess.run(self.train_init_op)
            else:
                _ = self.sess.run(self.train_op)

            if step % ckpt_every_n_steps == 0:
                path = self._save_model()
                last_ckpt_save_step = step
                print("STEP: {}, Model saved at: {}".format(step, path))

        if last_ckpt_save_step == 0:
            path = self._save_model()
            print("Model saved at end (no periodic save happened):", path)

    def __del__(self):
        try:
            self.sess.close()
        except Exception:
            pass




## === cell 8
with open("./preprocessed/labels_dict.json") as fp:
    labels_dict = json.load(fp)

sample_sub = pd.read_csv(sample_sub_csv)
label_cols = [c for c in sample_sub.columns if c != "fname"]
labels_dict = {label: i for i, label in enumerate(label_cols)}

print("Loaded labels:", len(labels_dict))



## === cell 9
if do_run:
    tf1.reset_default_graph()
    trainor = Trainor(
        "./preprocessed/train.csv", "./preprocessed/val_curated.csv", labels_dict
    )



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2528418343.py in <cell line: 0>()
      1 if do_run:
      2     tf1.reset_default_graph()
----> 3     trainor = Trainor(
      4         "./preprocessed/train.csv", "./preprocessed/val_curated.csv", labels_dict
      5     )

/tmp/ipykernel_11/1034561670.py in __init__(self, train_csv, val_csv, labels_dict, lr, batch_size, shuffle, model_dir)
     36         self.sess = tf1.Session(config=cfg)
     37 
---> 38         self._define_summaries()
     39         self.sess.run(tf1.global_variables_initializer())
     40         self.saver = tf1.train.Saver()

/tmp/ipykernel_11/1034561670.py in _define_summaries(self)
     46         self.scalar_summaries = tf1.summary.merge([self.summary_loss, self.summary_acc])
     47 
---> 48         self.image_summary = tf1.summary.merge([])
     49         self.summary_writer = tf1.summary.FileWriter(self.model_dir, self.sess.graph)
     50 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/summary/summary.py in merge(inputs, collections, name)
    686   name = _summary_op_util.clean_tag(name)
    687   with _ops.name_scope(name, 'Merge', inputs):
--> 688     val = _gen_logging_ops.merge_summary(inputs=inputs, name=name)
    689     _summary_op_util.collect(val, collections, [])
    690   return val

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_logging_ops.py in merge_summary(inputs, name)
    471         "'merge_summary' Op, not %r." % inputs)
    472   _attr_N = len(inputs)
--> 473   _, _, _op, _outputs = _op_def_library._apply_op_helper(
    474         "MergeSummary", inputs=inputs, name=name)
    475   _result = _outputs[:]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    610         num_attr = _Attr(op_def, input_arg.number_attr)
    611         if num_attr.has_minimum and len(values) < num_attr.minimum:
--> 612           raise ValueError(
    613               f"List argument '{input_name}' to '{op_type_name}' Op with "
    614               f"length {len(values)} shorter than minimum length "

ValueError: List argument 'inputs' to 'MergeSummary' Op with length 0 shorter than minimum length 1.

## === cell 10
if do_run:
    remaining_for_training = max(0, TIME_LIMIT_SEC - TIME_SAFETY_MARGIN_SEC)
    trainor.train(10000, max_wall_time_sec=remaining_for_training)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2980080963.py in <cell line: 0>()
      1 if do_run:
      2     remaining_for_training = max(0, TIME_LIMIT_SEC - TIME_SAFETY_MARGIN_SEC)
----> 3     trainor.train(10000, max_wall_time_sec=remaining_for_training)
      4 
      5 

NameError: name 'trainor' is not defined

## === cell 11
class Predictor:
    def __init__(self, files, batch_size=16, model_dir="ds2"):
        self.model_dir = model_dir
        self.model = Model((300, 300, 1), n_classes=len(labels_dict))
        self._define_dataset_iterator(files, batch_size)

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)

        cfg = tf1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.intra_op_parallelism_threads = 0
        cfg.inter_op_parallelism_threads = 0
        self.sess = tf1.Session(config=cfg)
        self.sess.run(tf1.global_variables_initializer())
        self.saver = tf1.train.Saver()

        self._maybe_load_model()

    def _define_dataset_iterator(self, files, batch_size):
        cache_dir = "./preprocessed/spec_cache_test"
        os.makedirs(cache_dir, exist_ok=True)
        cache_id = _hash_for_cache_id(
            os.path.abspath(test_path) + "|" + str(len(files))
        )
        tfrec_path = os.path.join(cache_dir, f"test-{cache_id}.tfrecord")
        meta_path = os.path.join(cache_dir, f"test-{cache_id}.json")

        if not os.path.isfile(meta_path) or not os.path.isfile(tfrec_path):
            cfg = tf1.ConfigProto()
            cfg.gpu_options.allow_growth = True
            cfg.intra_op_parallelism_threads = 0
            cfg.inter_op_parallelism_threads = 0

            g = tf1.Graph()
            with g.as_default():
                fn_ph = tf1.placeholder(tf.string, shape=[None], name="fn_batch")

                ds = tf.data.Dataset.from_tensor_slices(fn_ph)

                options = tf.data.Options()
                options.experimental_deterministic = True
                ds = ds.with_options(options)

                def _map_one(fn):
                    spec = wav_to_spectogram(fn)
                    spec_bytes = tf.io.serialize_tensor(spec)
                    return spec_bytes

                ds = ds.map(_map_one, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                ds = ds.batch(64, drop_remainder=False).prefetch(
                    tf.data.experimental.AUTOTUNE
                )

                it = tf1.data.make_initializable_iterator(ds)
                spec_b = it.get_next()

            writer = tf.io.TFRecordWriter(tfrec_path)
            with tf1.Session(graph=g, config=cfg) as sess:
                sess.run(tf1.global_variables_initializer())
                sess.run(tf1.local_variables_initializer())
                sess.run(
                    it.initializer,
                    feed_dict={fn_ph: np.asarray(list(files), dtype=np.str_)},
                )

                while True:
                    try:
                        specs = sess.run(spec_b)
                        for sb in specs:
                            feat = {
                                "spec": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[sb])
                                )
                            }
                            ex = tf.train.Example(
                                features=tf.train.Features(feature=feat)
                            )
                            writer.write(ex.SerializeToString())
                    except tf.errors.OutOfRangeError:
                        break

            writer.close()
            with open(meta_path, "w") as fp:
                json.dump(
                    {
                        "tfrecord": tfrec_path,
                        "n": int(len(files)),
                        "encoding": "tf.io.serialize_tensor(spec)",
                    },
                    fp,
                )

        def _parse_test(x):
            feat = {"spec": tf.io.FixedLenFeature([], tf.string)}
            ex = tf.io.parse_single_example(x, feat)
            spec = tf.io.parse_tensor(ex["spec"], out_type=tf.float32)
            spec = tf.reshape(spec, [300, 300, 1])
            return spec

        dataset = tf.data.TFRecordDataset([tfrec_path])

        options = tf.data.Options()
        options.experimental_deterministic = True
        dataset = dataset.with_options(options)

        dataset = dataset.map(
            _parse_test, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.iterator = tf1.data.make_one_shot_iterator(dataset)
        self.features = self.iterator.get_next()

    def _maybe_load_model(self):
        if not os.path.isdir(self.model_dir):
            print("No model_dir found, using random weights:", self.model_dir)
            return
        files = os.listdir(self.model_dir)
        steps = []
        for f in files:
            if "model.ckpt-" in f and f.endswith(".index"):
                try:
                    steps.append(int(f.split("model.ckpt-")[-1].split(".")[0]))
                except Exception:
                    pass
        if len(steps) == 0:
            print(
                "No checkpoint found in model_dir, using random weights:",
                self.model_dir,
            )
            return
        ckpt_path = os.path.join(self.model_dir, "model.ckpt")
        step = max(steps)
        self.saver.restore(self.sess, ckpt_path + "-{}".format(step))
        print("Loaded checkpoint:", ckpt_path + "-{}".format(step))

    def pred_generator(self):
        while True:
            try:
                yield self.sess.run(self.probas)
            except tf.errors.OutOfRangeError:
                break
        print("Prediction over")

    def __del__(self):
        try:
            self.sess.close()
        except Exception:
            pass




## === cell 12
tf1.reset_default_graph()

sample_sub = pd.read_csv(sample_sub_csv)
orig_files = sample_sub["fname"].tolist()
files = [os.path.join(test_path, f) for f in orig_files]

bad = [p for p in files if (not os.path.isfile(p))]
if bad:
    raise FileNotFoundError(
        "Some test paths are not files (first 5 shown):\n{}".format("\n".join(bad[:5]))
    )

predictor = Predictor(files, batch_size=16, model_dir="ds2")



## === cell 13
pred_gen = predictor.pred_generator()



## === cell 14
n_test = len(orig_files)
n_classes = len(labels_dict)
all_preds = np.zeros((n_test, n_classes), dtype=np.float32)

offset = 0
for probs in pred_gen:
    bs = probs.shape[0]
    all_preds[offset : offset + bs, :] = probs
    offset += bs

assert offset == n_test, (offset, n_test)



## === cell 15
sub_df = pd.DataFrame(all_preds, columns=label_cols)
sub_df.insert(0, "fname", orig_files)
sub_df = sub_df[["fname"] + label_cols]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_df.shape)
print(sub_df.head())
