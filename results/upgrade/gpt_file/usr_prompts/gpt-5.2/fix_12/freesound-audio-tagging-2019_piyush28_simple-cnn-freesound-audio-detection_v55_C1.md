# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import gc
import hashlib
import threading
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as ls
from sklearn.preprocessing import MultiLabelBinarizer

np.random.seed(1337)
tf.random.set_seed(1337)

print("TF version:", tf.__version__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.run_functions_eagerly(False)
except Exception:
    pass




## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/freesound-audio-tagging-2019",
    "/kaggle/data/freesound-audio-tagging-2019",
    "../input/freesound-audio-tagging-2019",
]

DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if os.path.isdir(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(f"Could not find dataset root. Tried: {CANDIDATE_ROOTS}")

train_curated_csv = os.path.join(DATA_ROOT, "train_curated.csv")
train_noisy_csv = os.path.join(DATA_ROOT, "train_noisy.csv")
sample_submission_csv = os.path.join(DATA_ROOT, "sample_submission.csv")

train_curated_path = os.path.join(DATA_ROOT, "train_curated")
train_noisy_path = os.path.join(DATA_ROOT, "train_noisy")
test_path = os.path.join(DATA_ROOT, "test")

print("DATA_ROOT:", DATA_ROOT)
print("train_curated_csv exists:", os.path.exists(train_curated_csv))
print("train_noisy_csv exists:", os.path.exists(train_noisy_csv))
print("sample_submission_csv exists:", os.path.exists(sample_submission_csv))
print("test_path exists:", os.path.isdir(test_path))




## === cell 2
do_run = True

PREP_DIR = "./preprocessed"
os.makedirs(PREP_DIR, exist_ok=True)




## === cell 3
sample_sub = pd.read_csv(sample_submission_csv)
label_cols = [c for c in sample_sub.columns if c != "fname"]
labels_dict = {label: i for i, label in enumerate(label_cols)}

with open(os.path.join(PREP_DIR, "labels_dict.json"), "w") as fp:
    json.dump(labels_dict, fp)

print("n_labels:", len(labels_dict))
print("First 5 labels:", label_cols[:5])




## === cell 4
train_curated_df = pd.read_csv(train_curated_csv)
train_noisy_df = pd.read_csv(train_noisy_csv)

train_curated_df["fname"] = (
    train_curated_path + "/" + train_curated_df["fname"].astype(str)
)
train_noisy_df["fname"] = train_noisy_path + "/" + train_noisy_df["fname"].astype(str)

train_curated_df = train_curated_df[
    ~train_curated_df["fname"].str.endswith("1d44b0bd.wav")
].reset_index(drop=True)

split = int(0.2 * train_curated_df.shape[0])
indices = np.random.permutation(train_curated_df.shape[0])
val_mask = np.zeros(train_curated_df.shape[0], dtype=bool)
val_mask[indices[:split]] = True

val_curated_df = train_curated_df[val_mask].reset_index(drop=True)
train_curated_df_split = train_curated_df[~val_mask].reset_index(drop=True)

valid_labels = set(labels_dict.keys())
noisy_split = train_noisy_df["labels"].astype(str).str.split(",")
noisy_filtered = noisy_split.explode()
noisy_filtered = noisy_filtered[noisy_filtered.isin(valid_labels)]
train_noisy_df["labels"] = (
    noisy_filtered.groupby(level=0)
    .agg(",".join)
    .reindex(train_noisy_df.index)
    .fillna("")
)
train_noisy_df = train_noisy_df[train_noisy_df["labels"].str.len() > 0].reset_index(
    drop=True
)

train_df = pd.concat([train_curated_df_split, train_noisy_df], axis=0).reset_index(
    drop=True
)

train_curated_df_split.to_csv(os.path.join(PREP_DIR, "train_curated.csv"), index=False)
train_noisy_df.to_csv(os.path.join(PREP_DIR, "train_noisy.csv"), index=False)
val_curated_df.to_csv(os.path.join(PREP_DIR, "val_curated.csv"), index=False)
train_df.to_csv(os.path.join(PREP_DIR, "train.csv"), index=False)

print("Saved preprocessed CSVs:")
print("train.csv rows:", len(train_df))
print("val_curated.csv rows:", len(val_curated_df))




## === cell 5
del train_curated_df, train_noisy_df, train_df
gc.collect()




## === cell 6
TARGET_SR = 44100
TARGET_SECONDS = 5
TARGET_SAMPLES = TARGET_SR * TARGET_SECONDS

CACHE_WAV_DIR = os.path.join(PREP_DIR, "wav_cache_sr44100_len5")
CACHE_SPEC_DIR = os.path.join(PREP_DIR, "spec_cache_300x300_stft1024_step64")
os.makedirs(CACHE_WAV_DIR, exist_ok=True)
os.makedirs(CACHE_SPEC_DIR, exist_ok=True)

_cache_locks = {}
_cache_locks_guard = threading.Lock()


def _get_lock_for_path(path: str) -> threading.Lock:
    with _cache_locks_guard:
        lk = _cache_locks.get(path)
        if lk is None:
            lk = threading.Lock()
            _cache_locks[path] = lk
        return lk


def _safe_cache_key(path_str: str) -> str:
    return hashlib.sha1(path_str.encode("utf-8")).hexdigest()


def _wav_cache_path(file_path: str) -> str:
    return os.path.join(CACHE_WAV_DIR, _safe_cache_key(file_path) + ".npy")


def _spec_cache_path(file_path: str) -> str:
    return os.path.join(CACHE_SPEC_DIR, _safe_cache_key(file_path) + ".npy")


@tf.function
def _fix_audio_length(sound_1d):
    n = tf.shape(sound_1d)[0]
    sound_1d = tf.cond(
        n >= TARGET_SAMPLES,
        lambda: sound_1d[:TARGET_SAMPLES],
        lambda: tf.pad(sound_1d, [[0, TARGET_SAMPLES - n]]),
    )
    sound_1d.set_shape([TARGET_SAMPLES])
    return sound_1d


@tf.function
def random_rolls(audio_1d):
    random_num = tf.random.uniform((1,), 0, 10)[0]
    cond = random_num >= 5
    roll_amount = tf.random.uniform((1,), 1200, 2000)[0]
    new_audio = tf.cond(
        cond,
        lambda: tf.roll(audio_1d, shift=tf.cast(roll_amount, tf.int32), axis=0),
        lambda: audio_1d,
    )
    return new_audio


@tf.function
def random_speedx(audio_1d):
    def get_audio():
        factor = tf.random.uniform((1,), 0.7, 1.7)[0]
        indices = tf.round(
            tf.range(0.0, tf.cast(tf.shape(audio_1d)[0], tf.float32), factor)
        )
        indices = tf.boolean_mask(
            indices, indices < tf.cast(tf.shape(audio_1d)[0], tf.float32)
        )
        sound = tf.gather(audio_1d, tf.cast(indices, tf.int32))
        return sound

    random_num = tf.random.uniform((1,), 0, 10)[0]
    cond = random_num >= 5
    return tf.cond(cond, get_audio, lambda: audio_1d)


@tf.function
def _audio_to_spectrogram_from_fixedlen(sound_fixedlen_1d):
    stft = tf.signal.stft(
        sound_fixedlen_1d,
        frame_length=1024,
        frame_step=64,
        fft_length=1024,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    spectogram = tf.abs(stft)
    minimum = tf.minimum(spectogram, 255.0)
    expand_dims = tf.expand_dims(minimum, -1)
    expand_dims = tf.expand_dims(expand_dims, 0)
    resize = tf.image.resize(
        expand_dims, [300, 300], method=tf.image.ResizeMethod.BILINEAR
    )
    squeeze = tf.squeeze(resize, axis=0)
    squeeze.set_shape([300, 300, 1])
    return squeeze


@tf.function
def wav_to_spectogram(wav_filename, random_perturbs=False):
    wav_bytes = tf.io.read_file(wav_filename)
    audio_tensor, sample_rate = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    sound = tf.squeeze(audio_tensor, axis=-1)

    if random_perturbs:
        sound = random_speedx(random_rolls(sound))

    sound = _fix_audio_length(sound)
    return _audio_to_spectrogram_from_fixedlen(sound)


def _maybe_load_cached_spec(file_path: str):
    sp = _spec_cache_path(file_path)
    if os.path.exists(sp):
        arr = np.load(sp, allow_pickle=False, mmap_mode="r")
        return np.asarray(arr, dtype=np.float32)
    return None


def _atomic_save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)


def _compute_and_cache_fixed_wav_and_spec(file_path: str):
    wav_p = _wav_cache_path(file_path)
    spec_p = _spec_cache_path(file_path)

    wav_lock = _get_lock_for_path(wav_p)
    spec_lock = _get_lock_for_path(spec_p)

    wav_bytes = tf.io.read_file(file_path)
    audio_tensor, _ = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    sound = tf.squeeze(audio_tensor, axis=-1)
    sound = _fix_audio_length(sound)
    sound_np = sound.numpy().astype(np.float32, copy=False)

    if not os.path.exists(wav_p):
        with wav_lock:
            if not os.path.exists(wav_p):
                _atomic_save_npy(wav_p, sound_np)

    if not os.path.exists(spec_p):
        with spec_lock:
            if not os.path.exists(spec_p):
                spec_np = (
                    _audio_to_spectrogram_from_fixedlen(sound)
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                _atomic_save_npy(spec_p, spec_np)
                return spec_np

    arr = np.load(spec_p, allow_pickle=False, mmap_mode="r")
    return np.asarray(arr, dtype=np.float32)


def _cached_labels_npz_path(csv_path):
    base = os.path.basename(csv_path)
    return os.path.join(PREP_DIR, base.replace(".csv", "") + "__labels_cache.npz")


def csv_pipe(csv_path, labels_dict):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=1337).reset_index(drop=True)

    files = df["fname"].values.astype(str)

    cache_path = _cached_labels_npz_path(csv_path)
    if os.path.exists(cache_path):
        labels_1hot = np.load(cache_path)["labels"].astype(np.int32, copy=False)
        return files, labels_1hot

    labels = df["labels"].astype(str).str.split(",")
    class_order = [k for k, _ in sorted(labels_dict.items(), key=lambda kv: kv[1])]
    binarizer = MultiLabelBinarizer(classes=class_order)
    labels_1hot = binarizer.fit_transform(labels).astype(np.int32)

    np.savez_compressed(cache_path, labels=labels_1hot)
    return files, labels_1hot


def wav_data_generators(
    csv_path, labels_dict, batch_size=32, shuffle=False, repeat=True
):
    files, labels = csv_pipe(csv_path, labels_dict)

    def _load_spec_np_maybe_cache(fbytes):
        f = fbytes.decode("utf-8")
        cached = _maybe_load_cached_spec(f)
        if cached is not None:
            return cached
        return _compute_and_cache_fixed_wav_and_spec(f)

    def _load_wav_np_maybe_cache(fbytes):
        f = fbytes.decode("utf-8")
        wp = _wav_cache_path(f)
        if os.path.exists(wp):
            arr = np.load(wp, allow_pickle=False, mmap_mode="r")
            return np.asarray(arr, dtype=np.float32)
        _ = _compute_and_cache_fixed_wav_and_spec(f)
        arr = np.load(wp, allow_pickle=False, mmap_mode="r")
        return np.asarray(arr, dtype=np.float32)

    def _map_fn(file, label):
        if shuffle:
            wav = tf.numpy_function(_load_wav_np_maybe_cache, [file], Tout=tf.float32)
            wav.set_shape([TARGET_SAMPLES])
            wav = random_speedx(random_rolls(wav))
            wav = _fix_audio_length(wav)
            spec = _audio_to_spectrogram_from_fixedlen(wav)
        else:
            spec = tf.numpy_function(_load_spec_np_maybe_cache, [file], Tout=tf.float32)
            spec.set_shape([300, 300, 1])

        return spec, tf.cast(label, tf.float32)

    options = tf.data.Options()
    options.experimental_deterministic = True

    dataset = tf.data.Dataset.from_tensor_slices((files, labels)).with_options(options)

    if shuffle:
        dataset = dataset.shuffle(800, seed=1337, reshuffle_each_iteration=True)

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset




## === cell 7
def conv_relu_bn(filters, kernels, strides, padding="valid", in_shape=None):
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
        self.conv_relu_bn1 = conv_relu_bn(64, 3, 1, padding="same", in_shape=in_shape)
        self.conv_relu_bn2 = conv_relu_bn(64, 3, 2)
        self.conv_relu_bn3 = conv_relu_bn(128, 3, 1, padding="same")
        self.conv_relu_bn4 = conv_relu_bn(128, 3, 2)
        self.conv_relu_bn5 = conv_relu_bn(256, 3, 1, padding="same")
        self.conv_relu_bn6 = conv_relu_bn(256, 3, 2)
        self.conv_relu_bn7 = conv_relu_bn(512, 3, 1, padding="same")
        self.conv_relu_bn8 = conv_relu_bn(512, 3, 2)
        self.gpool = ls.GlobalAveragePooling2D()
        self.classifier = ls.Dense(n_classes)

    def call(self, inputs, training=False):
        x = self.conv_relu_bn1(inputs, training=training)
        x = self.conv_relu_bn2(x, training=training)
        x = self.conv_relu_bn3(x, training=training)
        x = self.conv_relu_bn4(x, training=training)
        x = self.conv_relu_bn5(x, training=training)
        x = self.conv_relu_bn6(x, training=training)
        x = self.conv_relu_bn7(x, training=training)
        x = self.conv_relu_bn8(x, training=training)
        x = self.gpool(x)
        return self.classifier(x)




## === cell 8
with open(os.path.join(PREP_DIR, "labels_dict.json")) as fp:
    labels_dict = json.load(fp)

ordered_labels = [c for c in sample_sub.columns if c != "fname"]
assert len(labels_dict) == len(ordered_labels) == 80
for i, lab in enumerate(ordered_labels):
    assert labels_dict[lab] == i

print("labels_dict OK")




## === cell 9
train_csv_p = os.path.join(PREP_DIR, "train.csv")
val_csv_p = os.path.join(PREP_DIR, "val_curated.csv")


class Trainor:
    def __init__(
        self,
        train_csv,
        val_csv,
        labels_dict,
        lr=1e-3,
        batch_size=8,
        shuffle=True,
        model_dir="ds2",
    ):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.labels_dict = labels_dict
        self.batch_size = batch_size

        self.model = Model(in_shape=(300, 300, 1), n_classes=len(labels_dict))

        self.train_dataset = wav_data_generators(
            train_csv, labels_dict, batch_size=batch_size, shuffle=True, repeat=True
        )
        self.val_dataset = wav_data_generators(
            val_csv, labels_dict, batch_size=batch_size, shuffle=False, repeat=True
        )

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

        self.train_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        self.val_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=self.model_dir, max_to_keep=5
        )
        self._maybe_restore()

    def _maybe_restore(self):
        latest = self.ckpt_manager.latest_checkpoint
        if latest is not None:
            self.ckpt.restore(latest).expect_partial()
            print("Restored checkpoint:", latest)

    @tf.function
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        probs = tf.nn.sigmoid(logits)
        self.train_acc.update_state(y, probs)
        return loss

    @tf.function
    def _val_step(self, x, y):
        logits = self.model(x, training=False)
        loss = self.loss_fn(y, logits)
        probs = tf.nn.sigmoid(logits)
        self.val_acc.update_state(y, probs)
        return loss

    def train(
        self,
        steps=500,
        ckpt_every_n_steps=100,
        log_every_n_steps=500,
        log_val_n_steps=5,
    ):
        train_it = iter(self.train_dataset)
        val_it = iter(self.val_dataset)

        for step in range(1, steps + 1):
            x, y = next(train_it)
            loss = self._train_step(x, y)

            if step % log_every_n_steps == 0:
                self.val_acc.reset_state()
                val_losses = []
                for _ in range(log_val_n_steps):
                    vx, vy = next(val_it)
                    vl = self._val_step(vx, vy)
                    val_losses.append(vl.numpy())
                print(
                    "STEP: {}, Loss: {:.4f}, TrainAcc: {:.4f}, ValLoss: {:.4f}, ValAcc: {:.4f}".format(
                        step,
                        float(loss.numpy()),
                        float(self.train_acc.result().numpy()),
                        float(np.mean(val_losses)),
                        float(self.val_acc.result().numpy()),
                    )
                )

            if step % ckpt_every_n_steps == 0:
                path = self.ckpt_manager.save(checkpoint_number=step)
                print("STEP: {}, Model saved at: {}".format(step, path))

        path = self.ckpt_manager.save(checkpoint_number=steps)
        print("Final Model saved at:", path)


train_steps = 10000 if do_run else 700

trainor = Trainor(
    os.path.join(PREP_DIR, "train.csv"),
    os.path.join(PREP_DIR, "val_curated.csv"),
    labels_dict,
    model_dir="ds2",
)
trainor.train(steps=train_steps)




## === cell 10
class Predictor:
    def __init__(self, files, batch_size=16, model_dir="ds2"):
        self.model_dir = model_dir
        self.model = Model((300, 300, 1), n_classes=80)

        options = tf.data.Options()
        options.experimental_deterministic = True

        def _load_spec_np_maybe_cache(fbytes):
            f = fbytes.decode("utf-8")
            cached = _maybe_load_cached_spec(f)
            if cached is not None:
                return cached
            return _compute_and_cache_fixed_wav_and_spec(f)

        ds = tf.data.Dataset.from_tensor_slices(files.astype(str)).with_options(options)
        ds = ds.map(
            lambda f: tf.numpy_function(
                _load_spec_np_maybe_cache, [f], Tout=tf.float32
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds = ds.map(
            lambda x: tf.ensure_shape(x, [300, 300, 1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.dataset = ds

        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=self.model_dir, max_to_keep=5
        )
        self._load_model()

    def _load_model(self):
        ckpt_path = self.ckpt_manager.latest_checkpoint
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {self.model_dir}. Ensure training ran and saved checkpoints."
            )
        self.ckpt.restore(ckpt_path).expect_partial()
        print("Loaded checkpoint:", ckpt_path)

    def predict_probas(self):
        for x in self.dataset:
            logits = self.model(x, training=False)
            yield tf.nn.sigmoid(logits).numpy()




## === cell 11
test_fnames = sample_sub["fname"].values.astype(str)
test_files = np.array([os.path.join(test_path, f) for f in test_fnames])

predictor = Predictor(test_files, batch_size=16, model_dir="ds2")

n_test = len(test_fnames)
preds = np.zeros((n_test, 80), dtype=np.float32)

offset = 0
for probs in predictor.predict_probas():
    bs = probs.shape[0]
    preds[offset : offset + bs, :] = probs
    offset += bs

assert offset == n_test, f"Predicted {offset} rows, expected {n_test}"

sub_df = pd.DataFrame(preds, columns=ordered_labels)
sub_df.insert(0, "fname", test_fnames)

assert list(sub_df.columns) == list(sample_sub.columns)
sub_df = sub_df.fillna(0.0)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "shape:", sub_df.shape)
print(sub_df.head())
