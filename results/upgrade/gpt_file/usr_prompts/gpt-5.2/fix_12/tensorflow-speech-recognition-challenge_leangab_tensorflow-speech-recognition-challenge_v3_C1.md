# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Build an algorithm that understands simple spoken commands.

## Metric
Multiclass Accuracy.

There are 12 possible labels for the Test set: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `silence`, `unknown`.

The `unknown` label should be used for a command that is not one one of the first 10 labels or that is not `silence`.

## Submission Format
For audio clip in the test set, you must predict the correct `label`. The submission file should contain a header and have the following format:

```
fname,label
clip_000044442.wav,silence
clip_0000adecb.wav,left
clip_0000d4322.wav,unknown
etc.
```

## Dataset
- **train.7z** - Contains a few informational files and a folder of audio files. The audio folder contains subfolders with 1 second clips of voice commands, with the folder name being the label of the audio clip. There are more labels that should be predicted. The labels you will need to predict in Test are `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`. Everything else should be considered either `unknown` or `silence`. The folder `_background_noise_` contains longer clips of "silence" that you can break up and use as training input.
    
    The files contained in the training audio are not uniquely named across labels, but they are unique if you include the label folder. For example, `00f0204f_nohash_0.wav` is found in 14 folders, but that file is a different speech command in each folder.
    
    The files are named so the first element is the subject id of the person who gave the voice command, and the last element indicated repeated commands. Repeated commands are when the subject repeats the same word multiple times. Subject id is not provided for the test data, and you can assume that the majority of commands in the test data were from subjects not seen in train.
    
    You can expect some inconsistencies in the properties of the training data (e.g., length of the audio).
    
- **test.7z** - Contains an audio folder with 150,000+ files in the format `clip_000044442.wav`. The task is to predict the correct label. Not all of the files are evaluated for the leaderboard score.
- **sample_submission.csv** - A sample submission file in the correct format.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (328 lines)
            sample_submission.csv (6474 lines)
            sample_submission.csv.zip (16.6 kB)
            test.zip (149.1 MB)
            train.zip (1.3 GB)
            tensorflow-speech-recognition-challenge/
                description.md (328 lines)
                sample_submission.csv (6474 lines)
                ... and 3 other files
                tensorflow-speech-recognition-challenge/
                test/
                    audio/
                        clip_00000000.wav (32.0 kB)
                        clip_00000001.wav (32.0 kB)
                        ... and 6471 other files
                    test/
                train/
                    audio/
                        _background_noise_/
                            ... (max depth reached)
                        bed/
                            ... (max depth reached)
                        ... and 29 other folders
                    train/
            test/
                audio/
                    clip_00000000.wav (32.0 kB)
                    clip_00000001.wav (32.0 kB)
                    ... and 6471 other files
                test/
            train/
                audio/
                    _background_noise_/
                        doing_the_dishes.wav (3.0 MB)
                        dude_miaowing.wav (2.0 MB)
                        ... and 3 other files
                    bed/
                        00176480_nohash_0.wav (32.0 kB)
                        004ae714_nohash_0.wav (32.0 kB)
                        ... and 1534 other files
                    ... and 29 other folders
                train/
        input/
            description.md (328 lines)
            sample_submission.csv (6474 lines)
            sample_submission.csv.zip (16.6 kB)
            test.zip (149.1 MB)
            train.zip (1.3 GB)
            tensorflow-speech-recognition-challenge/
                description.md (328 lines)
                sample_submission.csv (6474 lines)
                ... and 3 other files
                tensorflow-speech-recognition-challenge/
                test/
                    audio/
                        clip_00000000.wav (32.0 kB)
                        clip_00000001.wav (32.0 kB)
                        ... and 6471 other files
                    test/
                train/
                    audio/
                        _background_noise_/
                            ... (max depth reached)
                        bed/
                            ... (max depth reached)
                        ... and 29 other folders
                    train/
            test/
                audio/
                    clip_00000000.wav (32.0 kB)
                    clip_00000001.wav (32.0 kB)
                    ... and 6471 other files
                test/
                    audio/
                        clip_00000000.wav (32.0 kB)
                        clip_00000001.wav (32.0 kB)
                        ... and 6471 other files
                    test/
            train/
                audio/
                    _background_noise_/
                        doing_the_dishes.wav (3.0 MB)
                        dude_miaowing.wav (2.0 MB)
                        ... and 3 other files
                    bed/
                        00176480_nohash_0.wav (32.0 kB)
                        004ae714_nohash_0.wav (32.0 kB)
                        ... and 1534 other files
                    ... and 29 other folders
                train/
                    audio/
                        _background_noise_/
                            ... (max depth reached)
                        bed/
                            ... (max depth reached)
                        ... and 29 other folders
                    train/
        working/
            tensorflow-speech-recognition-challenge/
                description.md (328 lines)
                sample_submission.csv (6474 lines)
                ... and 3 other files
                tensorflow-speech-recognition-challenge/
                test/
                    audio/
                        clip_00000000.wav (32.0 kB)
                        clip_00000001.wav (32.0 kB)
                        ... and 6471 other files
                    test/
                train/
                    audio/
                        _background_noise_/
                            ... (max depth reached)
                        bed/
                            ... (max depth reached)
                        ... and 29 other folders
                    train/
```

-> data/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

-> data/tensorflow-speech-recognition-challenge/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

-> input/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

-> input/tensorflow-speech-recognition-challenge/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

-> working/tensorflow-speech-recognition-challenge/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

# 5. Code solution

## === cell 0
import os, glob, math, gc
from collections import Counter
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "9"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.utils import Sequence

import matplotlib.pyplot as plt

tf.random.set_seed(9)
np.random.seed(9)

try:
    cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, cpu))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
from tensorflow.python.client import device_lib

device_lib.list_local_devices()



## === cell 2
root_path = "/kaggle"
base_input = os.path.join(root_path, "input", "tensorflow-speech-recognition-challenge")

train_path = os.path.join(base_input, "train", "audio")
test_audio_path = os.path.join(base_input, "test", "audio")
sample_sub_path = os.path.join(base_input, "sample_submission.csv")

assert os.path.isdir(train_path), f"Train path not found: {train_path}"
assert os.path.isdir(test_audio_path), f"Test audio path not found: {test_audio_path}"
assert os.path.isfile(
    sample_sub_path
), f"Sample submission not found: {sample_sub_path}"

sample_sub = pd.read_csv(sample_sub_path)
sample_sub.head(), sample_sub.shape



## === cell 3
train_audio_sample = os.path.join(
    train_path, "yes", os.listdir(os.path.join(train_path, "yes"))[0]
)

wav_bytes = tf.io.read_file(train_audio_sample)
audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
audio = tf.squeeze(audio, axis=-1)
int(audio.shape[0]), int(sr.numpy())




## === cell 4
def pad_audio_np(samples, L):
    if len(samples) >= L:
        return samples[:L]
    return np.pad(
        samples,
        pad_width=(L - len(samples), 0),
        mode="constant",
        constant_values=(0, 0),
    )


def chop_audio_np(samples, L=16000):
    if len(samples) <= L:
        while True:
            yield pad_audio_np(samples, L)
    while True:
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg : beg + L]


def choose_background_generator_np(sound, backgrounds, max_alpha=0.7):
    if backgrounds is None:
        return sound
    my_gen = backgrounds[np.random.randint(len(backgrounds))]
    background = next(my_gen) * np.random.uniform(0, max_alpha)
    augmented_data = sound + background
    augmented_data = augmented_data.astype(sound.dtype, copy=False)
    return augmented_data


def random_shift_np(sound, shift_max=0.2, sampling_rate=16000):
    shift = np.random.randint(int(sampling_rate * shift_max))
    out = np.roll(sound, shift)
    if shift > 0:
        out[:shift] = 0
    else:
        out[shift:] = 0
    return out


def random_change_pitch_np(x, sr=16000):
    pitch_factor = np.random.randint(1, 4)  # mimic original selection
    rate = 1.0 + 0.03 * pitch_factor
    idx = (np.arange(len(x)) * rate).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    return x[idx]


def random_speed_up_np(x):
    where = ["start", "end"][np.random.randint(0, 2)]
    speed_factor = np.random.uniform(0, 0.5)
    rate = 1.0 + speed_factor
    idx = (np.arange(int(len(x) / rate)) * rate).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    up = x[idx]
    up_len = up.shape[0]
    if up_len >= x.shape[0]:
        return up[: x.shape[0]]
    if where == "end":
        up = np.concatenate((up, np.zeros((x.shape[0] - up_len,), dtype=x.dtype)))
    else:
        up = np.concatenate((np.zeros((x.shape[0] - up_len,), dtype=x.dtype), up))
    return up


def get_image_list(train_audio_path):
    classes = os.listdir(train_audio_path)
    classes = [thisclass for thisclass in classes if thisclass != "_background_noise_"]
    index = [i for i, j in enumerate(classes)]
    outlist = []
    labels = []
    for thisindex, thisclass in zip(index, classes):
        filelist = [
            f
            for f in os.listdir(os.path.join(train_audio_path, thisclass))
            if f.endswith(".wav")
        ]
        filelist = [os.path.join(train_audio_path, thisclass, x) for x in filelist]
        outlist.append(filelist)
        labels.append(np.full(len(filelist), fill_value=thisindex))
    return outlist, labels, dict(zip(classes, index))


def split_train_test_stratified_shuffle(images_list, labels, train_size=0.9):
    classes_size = np.array([len(x) for x in images_list])
    classes_vector = [np.arange(x) for x in classes_size]
    total = np.sum(classes_size)
    total_train = [int(train_size * total * x) for x in (classes_size / total)]
    train_index = [
        np.random.choice(x, y, replace=False) for x, y in zip(classes_size, total_train)
    ]
    validation_index = [np.setdiff1d(i, j) for i, j in zip(classes_vector, train_index)]

    train_set = [np.array(x)[idx] for x, idx in zip(images_list, train_index)]
    validation_set = [np.array(x)[idx] for x, idx in zip(images_list, validation_index)]
    train_labels = [np.array(x)[idx] for x, idx in zip(labels, train_index)]
    validation_labels = [np.array(x)[idx] for x, idx in zip(labels, validation_index)]

    train_set = np.array([element for array in train_set for element in array])
    validation_set = np.array(
        [element for array in validation_set for element in array]
    )
    train_labels = np.array([element for array in train_labels for element in array])
    validation_labels = np.array(
        [element for array in validation_labels for element in array]
    )

    train_shuffle = np.random.permutation(len(train_set))
    validation_shuffle = np.random.permutation(len(validation_set))

    train_set = train_set[train_shuffle]
    validation_set = validation_set[validation_shuffle]
    train_labels = train_labels[train_shuffle]
    validation_labels = validation_labels[validation_shuffle]

    return train_set, train_labels, validation_set, validation_labels


def _standardize_like_sklearn_np(X, eps=1e-12):
    mu = X.mean(axis=0, keepdims=True)
    var = ((X - mu) ** 2).mean(axis=0, keepdims=True)
    scale = np.sqrt(var)
    scale = np.where(scale < eps, 1.0, scale)
    return (X - mu) / scale


_AUDIO_CACHE = {}
_FEAT_CACHE = {}
_CACHE_LOCK = threading.Lock()

_MFCC_PARAMS = {
    "sr": 16000,
    "frame_length": 640,
    "frame_step": 320,
    "fft_length": 1024,
    "n_mels": 40,
}
_NUM_SPEC_BINS = _MFCC_PARAMS["fft_length"] // 2 + 1

_MEL_W = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=_MFCC_PARAMS["n_mels"],
    num_spectrogram_bins=_NUM_SPEC_BINS,
    sample_rate=_MFCC_PARAMS["sr"],
    lower_edge_hertz=20.0,
    upper_edge_hertz=_MFCC_PARAMS["sr"] / 2.0,
)

_MFCC_FRAMES = int(math.ceil(_MFCC_PARAMS["sr"] / _MFCC_PARAMS["frame_step"]))  # 50


@tf.function(reduce_retracing=True)
def _mfcc_tf(x_1d_float32, n_mfcc: tf.Tensor):
    stft = tf.signal.stft(
        x_1d_float32,
        frame_length=_MFCC_PARAMS["frame_length"],
        frame_step=_MFCC_PARAMS["frame_step"],
        fft_length=_MFCC_PARAMS["fft_length"],
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    spectrogram = tf.abs(stft)
    mel = tf.matmul(tf.square(spectrogram), _MEL_W)
    log_mel = tf.math.log(mel + 1e-6)
    mfcc_all = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)
    mfcc = mfcc_all[..., :n_mfcc]
    mfcc = tf.ensure_shape(mfcc, [_MFCC_FRAMES, None])
    return mfcc


@tf.function(reduce_retracing=True)
def _load_audio_tf_tensor(path, target_sr=tf.constant(16000, tf.int32)):
    wav_bytes = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # float32
    sr = tf.cast(sr, tf.int32)

    def _resample():
        xlen = tf.shape(audio)[0]
        new_len = tf.cast(
            tf.round(
                tf.cast(xlen, tf.float32)
                * (tf.cast(target_sr, tf.float32) / tf.cast(sr, tf.float32))
            ),
            tf.int32,
        )
        idx_f = tf.linspace(0.0, tf.cast(xlen - 1, tf.float32), new_len)
        idx = tf.cast(tf.round(idx_f), tf.int32)
        idx = tf.clip_by_value(idx, 0, xlen - 1)
        return tf.gather(audio, idx)

    audio = tf.cond(tf.not_equal(sr, target_sr), _resample, lambda: audio)

    L = target_sr
    xlen = tf.shape(audio)[0]
    audio = tf.cond(
        xlen >= L,
        lambda: audio[:L],
        lambda: tf.pad(audio, paddings=[[L - xlen, 0]], constant_values=0.0),
    )
    audio = tf.ensure_shape(audio, [None])
    return audio


def _load_audio_tf(path, target_sr=16000):
    wav_bytes = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # float32 [-1,1]
    sr_val = int(sr.numpy())
    x = audio.numpy()
    if sr_val != target_sr:
        xlen = len(x)
        new_len = int(round(xlen * (target_sr / sr_val)))
        idx = (np.linspace(0, xlen - 1, new_len)).astype(np.int64)
        x = x[idx]
    x = pad_audio_np(x, target_sr).astype(np.float32, copy=False)
    return x


def _tf_mfcc_from_audio(x_np, sr=16000, n_mfcc=40, n_mels=40):
    x = tf.convert_to_tensor(x_np, dtype=tf.float32)
    mfcc = _mfcc_tf(x, tf.constant(n_mfcc, dtype=tf.int32))
    mfcc = mfcc.numpy().astype(np.float32, copy=False)
    mfcc = _standardize_like_sklearn_np(mfcc)
    return mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1).astype(np.float32, copy=False)


@tf.function(reduce_retracing=True)
def _tf_mfcc_from_path_tensor(path, n_mfcc=tf.constant(40, tf.int32)):
    x = _load_audio_tf_tensor(path, target_sr=tf.constant(_MFCC_PARAMS["sr"], tf.int32))
    mfcc = _mfcc_tf(x, n_mfcc)
    return mfcc


def preprocess_data(
    file,
    background_generator,
    target_sr=16000,
    n_mfcc=40,
    threshold=0.7,
    use_cache=False,
):
    cache_key = (file, target_sr, n_mfcc)
    if use_cache and background_generator is None:
        with _CACHE_LOCK:
            cached = _FEAT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if use_cache and background_generator is None:
        with _CACHE_LOCK:
            arr = _AUDIO_CACHE.get((file, target_sr))
        if arr is None:
            arr = _load_audio_tf(file, target_sr=target_sr)
            with _CACHE_LOCK:
                _AUDIO_CACHE[(file, target_sr)] = arr
        x = arr
    else:
        x = _load_audio_tf(file, target_sr=target_sr)

    if np.random.uniform(0, 1) > threshold:
        x = choose_background_generator_np(x, background_generator)
    if np.random.uniform(0, 1) > threshold:
        x = random_shift_np(x, sampling_rate=target_sr)
    if np.random.uniform(0, 1) > threshold:
        x = random_change_pitch_np(x, sr=target_sr)
    if np.random.uniform(0, 1) > threshold:
        x = random_speed_up_np(x)

    out = _tf_mfcc_from_audio(x, sr=target_sr, n_mfcc=n_mfcc, n_mels=40)

    if use_cache and background_generator is None:
        with _CACHE_LOCK:
            _FEAT_CACHE[cache_key] = out
    return out


class data_generator(Sequence):
    def __init__(
        self,
        x_set,
        y_set,
        batch_size,
        background_generator,
        use_cache=False,
        n_workers=None,
    ):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.background_generator = background_generator
        self.use_cache = use_cache
        if n_workers is None:
            cpu = os.cpu_count() or 4
            n_workers = min(8, max(2, cpu // 2))
        self.n_workers = int(n_workers)

        self._executor = None
        if self.n_workers > 1:
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        idx_from = idx * self.batch_size
        idx_to = (idx + 1) * self.batch_size
        batch_x = self.x[idx_from:idx_to]
        batch_y = self.y[idx_from:idx_to]

        if len(batch_x) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        if self._executor is None or len(batch_x) == 1:
            feats = [
                preprocess_data(
                    elem, self.background_generator, use_cache=self.use_cache
                )
                for elem in batch_x
            ]
        else:
            feats = list(
                self._executor.map(
                    lambda p: preprocess_data(
                        p, self.background_generator, use_cache=self.use_cache
                    ),
                    batch_x,
                    chunksize=1,
                )
            )

        return np.asarray(feats, dtype=np.float32), np.asarray(batch_y)

    def __del__(self):
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass


def build_model(n_classes, input_shape):
    model_input = keras.Input(shape=input_shape)
    img_1 = layers.Convolution2D(
        filters=32, kernel_size=(3, 3), padding="same", activation=activations.relu
    )(model_input)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(
        filters=64, kernel_size=(3, 3), padding="same", activation=activations.relu
    )(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(
        filters=128, kernel_size=(3, 3), padding="same", activation=activations.relu
    )(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(
        filters=256, kernel_size=(3, 3), padding="same", activation=activations.relu
    )(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Dropout(rate=0.25)(img_1)
    img_1 = layers.Flatten()(img_1)
    img_1 = layers.Dense(128, activation=activations.relu)(img_1)
    img_1 = layers.Dropout(rate=0.5)(img_1)
    model_output = layers.Dense(n_classes, activation=activations.softmax)(img_1)
    model = keras.Model(model_input, model_output)
    return model




## === cell 5
wavfiles = glob.glob(os.path.join(train_path, "_background_noise_", "*wav"))
wavfiles_np = [_load_audio_tf(elem, target_sr=16000) for elem in wavfiles]
background_generator = [chop_audio_np(x) for x in wavfiles_np]
len(background_generator), [len(w) for w in wavfiles_np[:2]]



## === cell 6
images_list, labels, classes_map = get_image_list(train_path)

train_set, train_labels, validation_set, validation_labels = (
    split_train_test_stratified_shuffle(images_list, labels)
)

train_datagen = data_generator(
    train_set, train_labels, 40, background_generator, use_cache=False, n_workers=None
)
validation_datagen = data_generator(
    validation_set, validation_labels, 40, None, use_cache=True, n_workers=None
)

inv_map = {v: k for k, v in classes_map.items()}
len(classes_map), list(classes_map.items())[:5]



## === cell 7
target_commands = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]
folder_to_comp = {}
for folder in classes_map.keys():
    if folder in target_commands:
        folder_to_comp[folder] = folder
    elif folder == "_background_noise_":
        folder_to_comp[folder] = "silence"
    else:
        folder_to_comp[folder] = "unknown"

comp_labels = target_commands + ["silence", "unknown"]



## === cell 8
_probe = preprocess_data(train_set[0], background_generator=None, use_cache=False)
rows, columns = int(_probe.shape[0]), int(_probe.shape[1])
del _probe
gc.collect()

batch_size = 100
epochs = 50
base_path = root_path + "/working/models"
os.makedirs(base_path, exist_ok=True)

train_size = train_set.shape[0]
validation_size = validation_set.shape[0]
steps_per_epoch = max(1, train_size // batch_size)

lr = 1e-3

tensorboard_dir = base_path + "/logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

checkpoint_filepath = os.path.join(base_path, "cp-{epoch:04d}.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    save_weights_only=True,
    monitor="val_sparse_categorical_accuracy",
    mode="max",
    verbose=1,
)

BEST_CKPT_PATH = {"path": None, "best": -np.inf}


class BestCkptTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get("val_sparse_categorical_accuracy", None)
        if val is None:
            return
        if float(val) > float(BEST_CKPT_PATH["best"]):
            BEST_CKPT_PATH["best"] = float(val)
            BEST_CKPT_PATH["path"] = checkpoint_filepath.format(epoch=epoch + 1)


reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5, verbose=1
)

earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=1e-3, patience=5, verbose=1
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

model = build_model(len(classes_map), (rows, columns, 1))
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_metric])
model.summary()

(rows, columns, len(classes_map), train_size, validation_size, steps_per_epoch)



## === cell 9
AUTO = tf.data.AUTOTUNE


@tf.function(reduce_retracing=True)
def _standardize_like_sklearn_tf(X, eps=tf.constant(1e-12, tf.float32)):
    mu = tf.reduce_mean(X, axis=0, keepdims=True)
    var = tf.reduce_mean(tf.square(X - mu), axis=0, keepdims=True)
    scale = tf.sqrt(var)
    scale = tf.where(scale < eps, tf.ones_like(scale), scale)
    return (X - mu) / scale


@tf.function(reduce_retracing=True)
def _mfcc_from_path_standardized(path, label, n_mfcc=tf.constant(40, tf.int32)):
    x = _load_audio_tf_tensor(path, target_sr=tf.constant(_MFCC_PARAMS["sr"], tf.int32))
    mfcc = _mfcc_tf(x, n_mfcc)
    mfcc = _standardize_like_sklearn_tf(mfcc)
    mfcc = tf.expand_dims(mfcc, axis=-1)  # (time, n_mfcc, 1)
    mfcc = tf.ensure_shape(mfcc, [rows, columns, 1])
    label = tf.cast(label, tf.int64)
    return mfcc, label


def _mfcc_from_path_augmented_py(path_bytes):
    path = path_bytes.decode("utf-8")
    feat = preprocess_data(
        path,
        background_generator=background_generator,
        target_sr=16000,
        n_mfcc=40,
        threshold=0.7,
        use_cache=False,
    )
    return feat.astype(np.float32, copy=False)


def make_train_dataset(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths.astype("U"), labels.astype(np.int64))
    )
    ds = ds.shuffle(buffer_size=min(len(paths), 20000), reshuffle_each_iteration=True)

    def _map_train(p, y):
        x = tf.numpy_function(_mfcc_from_path_augmented_py, [p], Tout=tf.float32)
        x = tf.ensure_shape(x, [rows, columns, 1])
        y = tf.cast(y, tf.int64)
        return x, y

    ds = ds.map(_map_train, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_eval_dataset(paths, labels, batch_size, cache=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths.astype("U"), labels.astype(np.int64))
    )
    ds = ds.map(_mfcc_from_path_standardized, num_parallel_calls=AUTO)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_train_dataset(train_set, train_labels, batch_size=batch_size)
val_ds = make_eval_dataset(
    validation_set, validation_labels, batch_size=batch_size, cache=True
)

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_ds,
    validation_steps=max(1, validation_size // batch_size),
    callbacks=[
        earlystop_callback,
        reduce_lr_callback,
        checkpoint_callback,
        BestCkptTracker(),
        tensorboard_callback,
    ],
    verbose=1,
)



## === cell 10
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.plot(history.history.get("loss", []), label="loss")
plt.title("Loss history")
plt.ylabel("Loss value")
plt.xlabel("No. epoch")
plt.legend()
plt.show()

plt.plot(history.history.get("sparse_categorical_accuracy", []), label="accuracy")
plt.plot(
    history.history.get("val_sparse_categorical_accuracy", []), label="val_accuracy"
)
plt.title("Accuracy history")
plt.ylabel("Accuracy value")
plt.xlabel("No. epoch")
plt.legend()
plt.show()



## === cell 11
best_ckpt = BEST_CKPT_PATH.get("path")
if best_ckpt and os.path.exists(best_ckpt):
    model.load_weights(best_ckpt)
    print("Loaded best weights:", best_ckpt)
else:
    ckpts = sorted(glob.glob(os.path.join(base_path, "cp-*.weights.h5")))
    if ckpts:
        model.load_weights(ckpts[-1])
        print("Loaded last weights:", ckpts[-1])
    else:
        print("No checkpoint found; using current in-memory weights.")

test_df = pd.read_csv(sample_sub_path)
test_fnames = test_df["fname"].tolist()
test_full_paths = np.array([os.path.join(test_audio_path, f) for f in test_fnames])
dummy_labels = np.zeros(len(test_full_paths), dtype=np.int64)

test_ds = make_eval_dataset(
    test_full_paths, dummy_labels, batch_size=batch_size, cache=True
)
test_steps = int(np.ceil(len(test_full_paths) / batch_size))

y_pred = model.predict(test_ds, steps=test_steps, verbose=1)
y_labs = np.argmax(y_pred, axis=1)
y_labs[:10], y_pred.shape



## === cell 12
inv_map = {v: k for k, v in classes_map.items()}
pred_folders = [inv_map[int(i)] for i in y_labs]

pred_comp = []
for folder in pred_folders:
    if folder in target_commands:
        pred_comp.append(folder)
    else:
        pred_comp.append("unknown")

assert set(pred_comp).issubset(set(comp_labels))

submission = pd.DataFrame({"fname": test_fnames, "label": pred_comp})
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape



## === cell 13
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["fname", "label"]
assert len(check) == len(sample_sub)
check["label"].value_counts().head()
