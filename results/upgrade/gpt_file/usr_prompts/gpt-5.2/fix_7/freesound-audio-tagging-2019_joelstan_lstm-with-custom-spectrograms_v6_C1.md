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

0.3334805259547154

# 6. Current score

0.26028

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.07169) has done: 'I fix the TensorFlow/Keras import crash by avoiding the standalone `keras` package (which is incompatible in this environment) and using `tf.keras` everywhere. Since the external pretrained model path doesn’t exist, I replace that loading step with a small tf.keras model that preserves the same overall pipeline (spectrogram extraction → neural net → per-class probabilities) and ensure it outputs exactly 80 columns matching `sample_submission.csv`. I also make spectrogram building robust to corrupted/short wav files (returning zeros instead of crashing) so it always runs end-to-end. Finally, I write a valid `submission.csv` with the exact required column order.'
- What this solution (achieved 0.22174) has done: 'I fix the TensorFlow import crash (`MessageFactory`/protobuf mismatch) by forcing the pure-Python protobuf implementation before importing TensorFlow, which is the standard Kaggle workaround and is score-neutral. Then I minimally improve the score toward your target by adding a lightweight training step on the curated training set (same spectrogram features, same model architecture/loss), and use the trained model for test inference instead of untrained random weights. I also keep the submission column order exactly as in `sample_submission.csv`, and make train/test path handling robust to either `/kaggle/input/...` or `../input/...` layouts. No changes are made to the model layers, loss function, or feature extraction logic beyond what’s required to run and to train the existing model.'
- What this solution (achieved 0.22511) has done: 'I fix the TensorFlow/protobuf crash by setting both protobuf environment variables *before* importing anything that might load protobuf/TensorFlow, which resolves the `MessageFactory.GetPrototype` error in Kaggle. I also make the dataset base-path selection more robust by checking both the competition subfolder and the top-level `/kaggle/input` layout so CSV/audio paths always resolve. These changes are execution/stability fixes and keep your feature extraction, model, training loop, and submission formatting unchanged, so score behavior should only change negligibly (if at all). The script run end-to-end and write a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.26028) has done: 'The timeout is dominated by feature extraction: you are computing spectrograms for ~25k training clips + 3.3k test clips in pure-Python loops, plus an O(len(y)) sliding “volume” scan for long audio. I keep the exact model/training logic intact and focus on making spectrogram building much faster by (1) caching and parallelizing audio→spectrogram computation across CPU cores, (2) preallocating output arrays to avoid list growth and extra copies, and (3) vectorizing the “volume” scan with NumPy stride tricks to remove the inner Python loop while preserving the same argmax-based crop behavior. I also avoid unnecessary work (e.g., converting `zip(...)` to a list) and keep all paths and evaluation semantics unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf

keras = tf.keras


def pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


BASE = pick_existing(
    "/kaggle/input/freesound-audio-tagging-2019",
    "/kaggle/input/freesound-audio-tagging-2019/freesound-audio-tagging-2019",
    "../input/freesound-audio-tagging-2019",
    "../input/freesound-audio-tagging-2019/freesound-audio-tagging-2019",
)

Files = {
    "Test": {
        "csv": os.path.join(BASE, "sample_submission.csv"),
        "wav_dir": os.path.join(BASE, "test"),
    },
    "TrainCurated": {
        "csv": os.path.join(BASE, "train_curated.csv"),
        "wav_dir": os.path.join(BASE, "train_curated"),
    },
    "TrainNoisy": {
        "csv": os.path.join(BASE, "train_noisy.csv"),
        "wav_dir": os.path.join(BASE, "train_noisy"),
    },
}

print("BASE:", BASE)
print("TF:", tf.__version__)
print("Test CSV exists:", os.path.exists(Files["Test"]["csv"]))
print("Train curated CSV exists:", os.path.exists(Files["TrainCurated"]["csv"]))
print("Train noisy CSV exists:", os.path.exists(Files["TrainNoisy"]["csv"]))
print("Test dir exists:", os.path.isdir(Files["Test"]["wav_dir"]))
print("Train curated dir exists:", os.path.isdir(Files["TrainCurated"]["wav_dir"]))
print("Train noisy dir exists:", os.path.isdir(Files["TrainNoisy"]["wav_dir"]))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_sub = pd.read_csv(Files["Test"]["csv"])
test_wavs = df_sub.loc[:, "fname"].tolist()
TEST_PATH = Files["Test"]["wav_dir"]

print(df_sub.shape)
print("Num test wavs:", len(test_wavs))
print("First test wav:", test_wavs[0])
print("Test dir exists:", os.path.isdir(TEST_PATH))



## === cell 2
assert df_sub.columns[0] == "fname", "First column must be fname"
label_cols = list(df_sub.columns[1:])
n_classes = len(label_cols)
label_to_idx = {c: i for i, c in enumerate(label_cols)}
print("Num classes:", n_classes)



## === cell 3
from scipy.fftpack import rfft
from scipy.io.wavfile import read as read_wav


def spectrogram(y, sr, N=50):
    window_length = 2048
    num_windows = len(y) // window_length

    if num_windows == N:
        y = y[: N * window_length]
    elif num_windows < N:
        diff = N * window_length - len(y)
        before = diff // 2
        after = diff - before
        y = np.pad(y, (before, after), mode="constant", constant_values=0)
    else:
        total_len = window_length * N
        max_start = len(y) - total_len
        k = max_start // window_length + 1

        s0 = y.strides[0]
        windows = np.lib.stride_tricks.as_strided(
            y,
            shape=(k, total_len),
            strides=(window_length * s0, s0),
            writeable=False,
        )
        volume = np.abs(windows).sum(axis=1)

        m = max(int(volume.argmax()) - 5, 0)
        y = y[window_length * m : window_length * (m + N)]

    y = y.reshape((N, window_length))
    Y = abs(rfft(y, axis=1)).T

    Y = Y - Y.min()
    mmax = Y.max()
    if mmax != 0:
        Y = Y / mmax

    return Y[1:150, :].astype(np.float32)


def make_spec(path, filename, N=50):
    """
    Robustness: ensure we always return a valid fixed-shape spectrogram (N, 149).
    Handles read errors, empty/corrupt wav, and multi-channel wav.
    """
    target = (N, 149)
    fname = os.path.join(path, filename)
    try:
        sr, y = read_wav(fname)

        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.mean(axis=1)

        y = np.asarray(y, dtype=np.float32)

        if y.size == 0:
            return np.zeros(target, dtype=np.float32)

        spec = np.flip(spectrogram(y, sr, N).T, 0)  # (N, 149)
        if spec.shape != target:
            out = np.zeros(target, dtype=np.float32)
            n0 = min(target[0], spec.shape[0])
            n1 = min(target[1], spec.shape[1])
            out[:n0, :n1] = spec[:n0, :n1]
            return out
        return spec.astype(np.float32, copy=False)
    except Exception:
        return np.zeros(target, dtype=np.float32)




## === cell 4
train_curated_df = pd.read_csv(Files["TrainCurated"]["csv"])
train_noisy_df = pd.read_csv(Files["TrainNoisy"]["csv"])

TRAIN_CURATED_PATH = Files["TrainCurated"]["wav_dir"]
TRAIN_NOISY_PATH = Files["TrainNoisy"]["wav_dir"]

bad_files = set(
    [
        "f76181c4.wav",
        "77b925c2.wav",
        "6a1f682a.wav",
        "c7db12aa.wav",
        "7752cc8a.wav",
        "1d44b0bd.wav",
    ]
)
train_curated_df = train_curated_df[
    ~train_curated_df["fname"].isin(bad_files)
].reset_index(drop=True)
train_noisy_df = train_noisy_df[~train_noisy_df["fname"].isin(bad_files)].reset_index(
    drop=True
)

print("Train curated rows:", len(train_curated_df))
print("Train noisy rows:", len(train_noisy_df))
print("Train curated dir exists:", os.path.isdir(TRAIN_CURATED_PATH))
print("Train noisy dir exists:", os.path.isdir(TRAIN_NOISY_PATH))


def encode_labels(label_str):
    y = np.zeros(n_classes, dtype=np.float32)
    for lab in str(label_str).split(","):
        lab = lab.strip()
        if lab in label_to_idx:
            y[label_to_idx[lab]] = 1.0
    return y


train_all = pd.concat(
    [
        train_curated_df.assign(_wav_dir=TRAIN_CURATED_PATH),
        train_noisy_df.assign(_wav_dir=TRAIN_NOISY_PATH),
    ],
    axis=0,
    ignore_index=True,
)

train_wavs = train_all["fname"].tolist()
train_dirs = train_all["_wav_dir"].tolist()
train_labels_raw = train_all["labels"].tolist()

Y_train = np.stack([encode_labels(s) for s in train_labels_raw], axis=0).astype(
    np.float32
)
print("Y_train shape:", Y_train.shape, "pos_rate:", float(Y_train.mean()))



## === cell 5
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import multiprocessing


@lru_cache(maxsize=4096)
def _make_spec_cached(wav_dir, fname, N=50):
    return make_spec(wav_dir, fname, N=N)


def _build_specs_parallel(wav_dirs, fnames, N=50, desc="Building spectrograms"):
    n = len(fnames)
    out = np.empty((n, N, 149), dtype=np.float32)

    max_workers = min(8, (multiprocessing.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(
            lambda p: _make_spec_cached(p[0], p[1], N),
            zip(wav_dirs, fnames),
            chunksize=64,
        )
        for i, spec in enumerate(tqdm(it, total=n, desc=desc)):
            out[i] = spec
    return out


X_train = _build_specs_parallel(
    train_dirs, train_wavs, N=50, desc="Building TRAIN spectrograms"
)
print("X_train shape:", X_train.shape)

X_test = _build_specs_parallel(
    [TEST_PATH] * len(test_wavs), test_wavs, N=50, desc="Building TEST spectrograms"
)
print("X_test shape:", X_test.shape)



## === cell 6
try:
    for n in range(1):
        plt.figure(figsize=(12, 2.5))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(_make_spec_cached(TEST_PATH, test_wavs[i + n * 5], 50))
            plt.yticks([])
            plt.xticks([])
            plt.title(test_wavs[i + n * 5], fontsize=8)
        plt.tight_layout()
        plt.show()
except Exception as e:
    print("Visualization skipped due to:", repr(e))



## === cell 7
tf.random.set_seed(42)
np.random.seed(42)

inp = keras.Input(shape=X_test.shape[1:], dtype=tf.float32)
x = keras.layers.Reshape((X_test.shape[1], X_test.shape[2], 1))(inp)
x = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(n_classes, activation="sigmoid")(x)

LSTM_model = keras.Model(inp, out)
LSTM_model.compile(optimizer="adam", loss="binary_crossentropy")
print(LSTM_model.summary())



## === cell 8
history = LSTM_model.fit(
    X_train,
    Y_train,
    epochs=3,
    batch_size=32,
    verbose=1,
    shuffle=True,
)

y_pred = LSTM_model.predict(X_test, batch_size=32, verbose=1)
y_pred = np.asarray(y_pred)
if y_pred.ndim > 2:
    y_pred = y_pred.reshape((y_pred.shape[0], -1))

print("y_pred shape:", y_pred.shape)
if y_pred.shape[1] != n_classes:
    raise ValueError(
        f"Model output classes ({y_pred.shape[1]}) != submission classes ({n_classes})"
    )



## === cell 9
sub = pd.DataFrame(y_pred, columns=label_cols)
sub.insert(0, "fname", test_wavs)

for c in label_cols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0).astype(np.float32)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print(sub.shape)
