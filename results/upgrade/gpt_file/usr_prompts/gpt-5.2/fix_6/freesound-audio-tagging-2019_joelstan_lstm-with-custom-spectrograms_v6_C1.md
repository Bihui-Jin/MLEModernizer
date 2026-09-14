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
        volume = []
        for i in range(0, len(y) - window_length * N + 1, window_length):
            volume.append(abs(y[i : i + window_length * N]).sum())
        volume = np.array(volume)

        m = max(int(volume.argmax()) - 5, 0)
        y = y[window_length * m : window_length * (m + N)]

    y = y.reshape((N, window_length))
    Y = abs(rfft(y, axis=1)).T

    Y = Y - Y.min()
    if Y.max() != 0:
        Y = Y / Y.max()

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
X_train = []
for fn, d in tqdm(
    list(zip(train_wavs, train_dirs)),
    desc="Building TRAIN spectrograms",
    total=len(train_wavs),
):
    X_train.append(make_spec(d, fn))
X_train = np.asarray(X_train, dtype=np.float32)
print("X_train shape:", X_train.shape)

X_test = []
for fn in tqdm(test_wavs, desc="Building TEST spectrograms"):
    X_test.append(make_spec(TEST_PATH, fn))
X_test = np.asarray(X_test, dtype=np.float32)
print("X_test shape:", X_test.shape)



## === cell 6
try:
    for n in range(1):
        plt.figure(figsize=(12, 2.5))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(make_spec(TEST_PATH, test_wavs[i + n * 5]))
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
