# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.16341

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07169) has done: 'I fix the TensorFlow/Keras import crash by avoiding the standalone `keras` package (which is incompatible in this environment) and using `tf.keras` everywhere. Since the external pretrained model path doesn’t exist, I replace that loading step with a small tf.keras model that preserves the same overall pipeline (spectrogram extraction → neural net → per-class probabilities) and ensure it outputs exactly 80 columns matching `sample_submission.csv`. I also make spectrogram building robust to corrupted/short wav files (returning zeros instead of crashing) so it always runs end-to-end. Finally, I write a valid `submission.csv` with the exact required column order.'
- What this solution (achieved 0.22174) has done: 'I fix the TensorFlow import crash (`MessageFactory`/protobuf mismatch) by forcing the pure-Python protobuf implementation before importing TensorFlow, which is the standard Kaggle workaround and is score-neutral. Then I minimally improve the score toward your target by adding a lightweight training step on the curated training set (same spectrogram features, same model architecture/loss), and use the trained model for test inference instead of untrained random weights. I also keep the submission column order exactly as in `sample_submission.csv`, and make train/test path handling robust to either `/kaggle/input/...` or `../input/...` layouts. No changes are made to the model layers, loss function, or feature extraction logic beyond what’s required to run and to train the existing model.'
- What this solution (achieved 0.22511) has done: 'I fix the TensorFlow/protobuf crash by setting both protobuf environment variables *before* importing anything that might load protobuf/TensorFlow, which resolves the `MessageFactory.GetPrototype` error in Kaggle. I also make the dataset base-path selection more robust by checking both the competition subfolder and the top-level `/kaggle/input` layout so CSV/audio paths always resolve. These changes are execution/stability fixes and keep your feature extraction, model, training loop, and submission formatting unchanged, so score behavior should only change negligibly (if at all). The script run end-to-end and write a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.26028) has done: 'The timeout is dominated by feature extraction: you are computing spectrograms for ~25k training clips + 3.3k test clips in pure-Python loops, plus an O(len(y)) sliding “volume” scan for long audio. I keep the exact model/training logic intact and focus on making spectrogram building much faster by (1) caching and parallelizing audio→spectrogram computation across CPU cores, (2) preallocating output arrays to avoid list growth and extra copies, and (3) vectorizing the “volume” scan with NumPy stride tricks to remove the inner Python loop while preserving the same argmax-based crop behavior. I also avoid unnecessary work (e.g., converting `zip(...)` to a list) and keep all paths and evaluation semantics unchanged.'
- What this solution (achieved 0.24934) has done: 'I fix the runtime crash happening before any training by applying the standard Kaggle workaround for the protobuf/TensorFlow `MessageFactory.GetPrototype` mismatch: force the pure-Python protobuf implementation and (critically) disable the C++ protobuf backend *before* importing TensorFlow. This is an execution/stability fix and does not change your model, features, training loop, or submission formatting. I also keep the existing robust path selection and ensure the submission is written as `submission.csv` with the exact sample-submission column order.'
- What this solution (achieved 0.02267) has done: 'I fix the TensorFlow/protobuf crash by switching from the current environment-variable workaround (which still triggers `MessageFactory.GetPrototype`) to a Kaggle-safe approach that avoids importing TensorFlow entirely by using a pure NumPy training/inference fallback. This keeps your core pipeline intact (same spectrogram extraction, same train+noisy usage, same multi-label setup, same submission formatting/column order) while making the notebook run end-to-end reliably. To move the score up toward your target with minimal change, the fallback uses simple per-class centroid prototypes in the same feature space and outputs calibrated sigmoid-like probabilities rather than random/untrained predictions. The script still write `submission.csv` with the exact columns from `sample_submission.csv`.'
- What this solution (achieved 0.16809) has done: 'Your current score (0.02267) is far below the target (0.33348), so we should increase performance with minimal, low-risk changes while keeping the same core “prototype in spectrogram feature space → sigmoid probabilities” logic. The biggest issue is that raw Euclidean-distance-to-centroid sigmoid outputs are poorly calibrated for label-weighted LRAP; a small, legitimate improvement is to (1) L2-normalize features/prototypes and use cosine similarity (same prototype model, just a scale-invariant distance), and (2) add per-class weighting to reduce domination by frequent labels, which better matches label-weighted scoring. We also clip probabilities away from exact 0/1 for numerical stability and keep submission column order identical to `sample_submission.csv`. Everything still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.16532) has done: 'We’re still far below the target (0.16809 vs 0.33348; higher is better), so we should improve performance with the smallest changes that keep your “cosine-similarity to per-class prototypes → sigmoid probabilities” core logic intact. The main low-risk improvement is to compute two separate prototype sets (curated-only and noisy-only) and ensemble their cosine-similarity logits; this preserves the same feature extraction, normalization, prototype model, and sigmoid mapping, but reduces noise sensitivity and typically improves LRAP. Additionally, we fit a per-class temperature (scale) from training data using only the positives-vs-negatives separation in cosine space (still the same model, just calibrated scaling), which tends to improve ranking-based metrics without changing semantics. Finally, we keep the same submission schema/order and write `submission.csv` as before.'
- What this solution (achieved 0.16617) has done: 'We’re currently below the target (0.16532 vs 0.33348; higher is better), so we should improve ranking quality with minimal risk while keeping your same core pipeline (spectrogram → flatten/standardize → cosine-to-prototypes → sigmoid probabilities). The smallest high-leverage change is to improve the prototype estimation without changing the model family: compute *soft* (responsibility-weighted) prototypes using both positive and negative evidence per class, rather than using only positives (which is weak for LRAP). Then we keep your curated/noisy two-prototype ensemble and per-class temperature calibration exactly as-is, but recompute temps from the new logits (still train-only) for better calibration. This should move the score upward toward the target while preserving evaluation semantics and producing the same `submission.csv` format.'
- What this solution (achieved 0.16341) has done: 'Your current score (0.16617) is well below the target (0.33348), so we should increase LRAP with minimal, low-risk changes while keeping the same prototype + cosine-similarity core. The biggest issue is that a single global sigmoid calibration per class can still leave the within-clip ranking weak; we can improve ranking by adding a small “query expansion” refinement: use the initial predictions to build a per-clip mixture of top predicted class prototypes and recompute logits (same prototypes, same cosine model, just a second pass). Additionally, we tune the curated/noisy ensemble weight using a small validation split and directly optimize label-weighted LRAP on that split (no architecture change; only mixing weight selection). These changes are lightweight, deterministic, and should move the score upward toward the target while preserving submission format and core semantics.'
- What this solution (achieved 0.16341) has done: 'We’re far below the target (0.16341 vs 0.33348; higher-is-better), so we should legitimately improve ranking quality with minimal changes while preserving your prototype+cosine+sigmoid core. The biggest low-risk gain is to make the “query expansion” refinement class-aware: instead of adding the same residual bias to all classes, compute a second-pass cosine similarity between each clip’s top-K mixture prototype and every class prototype, then add it as a small refinement term (still the same ingredients: prototypes + cosine + a second pass). I also choose the refinement strength `beta` by maximizing label-weighted LRAP on the same holdout split you already use for mixing, so we move upward toward the target without changing the model family. Finally, I keep submission formatting identical and ensure everything still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.16341) has done: 'We’re far below the target (0.16341 vs 0.33348; higher-is-better), so the smallest safe way to move upward is to fix a key mismatch between what your metric wants (label-weighted ranking) and what the current tuning optimizes. I keep your exact feature extraction, prototype computation, cosine scoring, temperature scaling, and query-expansion refinement, but change the holdout selection/tuning to be label-stratified (so rare labels are represented) and tune `best_w`/`best_beta` on the same *post-sigmoid refined probabilities* that you actually submit. This preserves the model family and semantics, but makes the calibration/ensemble weights better aligned with the Kaggle evaluation. Finally, I make the split deterministic and reuse it consistently to avoid score regressions from randomness.'
- What this solution (achieved 0.16341) has done: 'We’re far below the target (0.16341 vs 0.33348; higher-is-better), so we should improve ranking quality with the smallest calibration/tuning changes while keeping your prototype+cosine+sigmoid pipeline intact. The main issue is that the current holdout tuning optimizes lwLRAP on similarities/probabilities, but the final submission is still a single global sigmoid of refined logits; for LRAP, per-clip ranking is what matters, so we add a minimal per-clip monotonic normalization (rank-based) that preserves ordering but improves comparability across clips. We also tune the top-K used by query expansion (K) on the same stratified holdout (no architecture/model change; it’s just a hyperparameter of your existing refinement), and we tune the refinement strength `beta` on the *final post-processed predictions* that you submit. These changes are deterministic, keep feature extraction/prototypes/temps intact, and should move the score upward toward the target while still producing the same `submission.csv` schema.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


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


train_curated_df = train_curated_df.assign(_wav_dir=TRAIN_CURATED_PATH, _src=0)
train_noisy_df = train_noisy_df.assign(_wav_dir=TRAIN_NOISY_PATH, _src=1)

train_all = pd.concat([train_curated_df, train_noisy_df], axis=0, ignore_index=True)

train_wavs = train_all["fname"].tolist()
train_dirs = train_all["_wav_dir"].tolist()
train_src = train_all["_src"].values.astype(np.int32, copy=False)
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
def _flatten_feats(X):
    return X.reshape((X.shape[0], -1)).astype(np.float32, copy=False)


Xtr = _flatten_feats(X_train)
Xte = _flatten_feats(X_test)
Ytr = Y_train.astype(np.float32, copy=False)

mu = Xtr.mean(axis=0, keepdims=True)
sigma = Xtr.std(axis=0, keepdims=True)
sigma = np.where(sigma > 1e-6, sigma, 1.0).astype(np.float32)
Xtrn = (Xtr - mu) / sigma
Xten = (Xte - mu) / sigma

pos_counts = Ytr.sum(axis=0)  # (C,)
w_per_class = 1.0 / np.sqrt(pos_counts + 1.0)  # (C,)

sample_w = (Ytr * w_per_class[None, :]).sum(axis=1)  # (n,)
sample_w = sample_w + 0.05  # small floor so all clips contribute a bit
sample_w = sample_w.astype(np.float32)


def _l2_normalize(A, eps=1e-6):
    nrm = np.sqrt((A * A).sum(axis=1, keepdims=True)).astype(np.float32)
    nrm = np.maximum(nrm, np.float32(eps))
    return (A / nrm).astype(np.float32, copy=False)


def _compute_prototypes_soft(X, Y, sw, alpha=0.35):
    """
    For each class c, prototype is a weighted difference between positive and negative means:
      proto_c = mean_pos - alpha * mean_neg
    using the same sample weights sw. alpha is kept small for stability.
    """
    sw_sum = float(sw.sum())
    global_mean = (X * sw[:, None]).sum(axis=0) / max(sw_sum, 1e-6)

    prot = np.empty((n_classes, X.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mpos = Y[:, c] > 0.5
        mneg = ~mpos

        if mpos.any():
            ww = sw[mpos]
            sww = float(ww.sum())
            mu_pos = (X[mpos] * ww[:, None]).sum(axis=0) / max(sww, 1e-6)
        else:
            mu_pos = global_mean

        if mneg.any():
            ww = sw[mneg]
            sww = float(ww.sum())
            mu_neg = (X[mneg] * ww[:, None]).sum(axis=0) / max(sww, 1e-6)
        else:
            mu_neg = global_mean

        prot[c] = (mu_pos - np.float32(alpha) * mu_neg).astype(np.float32, copy=False)

    return prot


m_cur = train_src == 0
m_noi = train_src == 1

if not m_cur.any():
    m_cur = np.ones(len(train_src), dtype=bool)
if not m_noi.any():
    m_noi = np.ones(len(train_src), dtype=bool)

protos_cur = _compute_prototypes_soft(
    Xtrn[m_cur], Ytr[m_cur], sample_w[m_cur], alpha=0.30
)
protos_noi = _compute_prototypes_soft(
    Xtrn[m_noi], Ytr[m_noi], sample_w[m_noi], alpha=0.40
)

Xten_u = _l2_normalize(Xten)
protos_cur_u = _l2_normalize(protos_cur)
protos_noi_u = _l2_normalize(protos_noi)

Xtrn_u = _l2_normalize(Xtrn)
sim_cur_tr = Xtrn_u @ protos_cur_u.T  # (n, C)
sim_noi_tr = Xtrn_u @ protos_noi_u.T  # (n, C)


def _label_weighted_lrap(y_true, y_score, eps=1e-12):
    n, C = y_true.shape
    per_label = np.zeros((C,), dtype=np.float64)
    for c in range(C):
        yt = y_true[:, c]
        if yt.sum() <= 0:
            per_label[c] = np.nan
            continue
        s = y_score[:, c]
        order = np.argsort(-s, kind="mergesort")
        yt_sorted = yt[order]
        cum_tp = np.cumsum(yt_sorted, dtype=np.float64)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        prec_at_k = cum_tp / ranks
        per_label[c] = float((prec_at_k * yt_sorted).sum() / max(float(yt.sum()), eps))
    m = np.nanmean(per_label)
    return float(m) if np.isfinite(m) else 0.0


def _make_label_stratified_split(Y, val_frac=0.15, seed=123, min_pos_per_label=2):
    rng = np.random.RandomState(seed)
    n = Y.shape[0]
    C = Y.shape[1]

    all_idx = np.arange(n)
    val_set = set()

    for c in range(C):
        pos_idx = np.flatnonzero(Y[:, c] > 0.5)
        if pos_idx.size >= (min_pos_per_label * 2):
            k = max(min_pos_per_label, int(round(val_frac * pos_idx.size)))
            k = min(k, pos_idx.size // 2)
            if k > 0:
                pick = rng.choice(pos_idx, size=k, replace=False)
                val_set.update(map(int, pick))

    target_val_n = int(round(val_frac * n))
    if len(val_set) < target_val_n:
        remaining = np.array(sorted(set(all_idx.tolist()) - val_set), dtype=np.int64)
        need = target_val_n - len(val_set)
        if remaining.size > 0 and need > 0:
            pick = rng.choice(remaining, size=min(need, remaining.size), replace=False)
            val_set.update(map(int, pick))

    val_idx = np.array(sorted(val_set), dtype=np.int64)
    if val_idx.size == 0:
        rng.shuffle(all_idx)
        val_idx = all_idx[: int(round(val_frac * n))]

    tr_mask = np.ones(n, dtype=bool)
    tr_mask[val_idx] = False
    tr_idx = np.flatnonzero(tr_mask).astype(np.int64)
    return tr_idx, val_idx


def _rank_normalize_per_clip(P, eps=1e-6):
    n, C = P.shape
    order = np.argsort(P, axis=1, kind="mergesort")  # ascending indices
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(C, dtype=np.int32)[None, :]
    out = (ranks.astype(np.float32) + 1.0) / (float(C) + 1.0)
    return np.clip(out, eps, 1.0 - eps).astype(np.float32, copy=False)


tr_idx, val_idx = _make_label_stratified_split(
    Ytr, val_frac=0.15, seed=123, min_pos_per_label=2
)
Y_val = Ytr[val_idx]

sim_cur_val = sim_cur_tr[val_idx]
sim_noi_val = sim_noi_tr[val_idx]

mix_grid = np.array([0.50, 0.55, 0.60, 0.65, 0.70], dtype=np.float32)
best_w = 0.60
best_score = -1.0
for w in mix_grid:
    sim_ens_val = float(w) * sim_cur_val + (1.0 - float(w)) * sim_noi_val
    sc = _label_weighted_lrap(Y_val, sim_ens_val)
    if sc > best_score:
        best_score = sc
        best_w = float(w)

print("Chosen mix weight (curated):", best_w, "holdout lwLRAP(sim):", best_score)

sim_ens_tr = best_w * sim_cur_tr + (1.0 - best_w) * sim_noi_tr

temps = np.empty((n_classes,), dtype=np.float32)
for c in range(n_classes):
    pos = sim_ens_tr[Ytr[:, c] > 0.5, c]
    neg = sim_ens_tr[Ytr[:, c] <= 0.5, c]
    if pos.size == 0 or neg.size == 0:
        temps[c] = np.float32(0.25)
        continue
    gap = float(np.median(pos) - np.median(neg))
    t = 0.25 / max(gap, 1e-3)
    temps[c] = np.float32(np.clip(t, 0.12, 0.60))

logits_cur = Xten_u @ protos_cur_u.T
logits_noi = Xten_u @ protos_noi_u.T
logits = best_w * logits_cur + (1.0 - best_w) * logits_noi  # cosine logits in [-1, 1]
logits = logits / temps[None, :]

y_pred_base = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

logits_tr = (
    best_w * (Xtrn_u @ protos_cur_u.T) + (1.0 - best_w) * (Xtrn_u @ protos_noi_u.T)
) / temps[None, :]
y_pred_tr_base = (1.0 / (1.0 + np.exp(-logits_tr))).astype(np.float32)

protos_ens_u = _l2_normalize(
    best_w * protos_cur + (1.0 - best_w) * protos_noi
)  # (C, D)

K_grid = [3, 5, 7, 10]
beta_grid = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25], dtype=np.float32)

best_K = 5
best_beta = 0.10
best_post_score = -1.0

for K in K_grid:
    topk_idx_tr = np.argpartition(-y_pred_tr_base, K - 1, axis=1)[:, :K]
    P_tr = protos_ens_u[topk_idx_tr]  # (n, K, D)
    W_tr = y_pred_tr_base[
        np.arange(y_pred_tr_base.shape[0])[:, None], topk_idx_tr
    ].astype(
        np.float32
    )  # (n, K)
    mix_proto_tr = _l2_normalize((P_tr * W_tr[:, :, None]).sum(axis=1))  # (n, D)
    refine_tr = (mix_proto_tr @ protos_ens_u.T).astype(np.float32)  # (n, C)

    for b in beta_grid:
        val_logits = logits_tr[val_idx] + float(b) * refine_tr[val_idx]
        val_prob = (1.0 / (1.0 + np.exp(-val_logits))).astype(np.float32)

        val_prob_post = _rank_normalize_per_clip(val_prob)
        sc = _label_weighted_lrap(Y_val, val_prob_post)

        if sc > best_post_score:
            best_post_score = sc
            best_K = int(K)
            best_beta = float(b)

print(
    "Chosen K:",
    best_K,
    "Chosen beta:",
    best_beta,
    "holdout lwLRAP(post):",
    best_post_score,
)

topk_idx = np.argpartition(-y_pred_base, best_K - 1, axis=1)[:, :best_K]  # (n, K)
P = protos_ens_u[topk_idx]  # (n, K, D)
W = y_pred_base[np.arange(y_pred_base.shape[0])[:, None], topk_idx].astype(
    np.float32
)  # (n,K)
mix_proto = (P * W[:, :, None]).sum(axis=1)  # (n, D)
mix_proto = _l2_normalize(mix_proto)  # (n, D)
refine = (mix_proto @ protos_ens_u.T).astype(np.float32)  # (n, C)

logits_ref = logits + np.float32(best_beta) * refine
y_pred = (1.0 / (1.0 + np.exp(-logits_ref))).astype(np.float32)

y_pred = _rank_normalize_per_clip(y_pred)

y_pred = np.clip(y_pred, 1e-4, 1.0 - 1e-4).astype(np.float32)

print("y_pred shape:", y_pred.shape)
if y_pred.shape[1] != n_classes:
    raise ValueError(
        f"Model output classes ({y_pred.shape[1]}) != submission classes ({n_classes})"
    )




## === cell 8
sub = pd.DataFrame(y_pred, columns=label_cols)
sub.insert(0, "fname", test_wavs)

for c in label_cols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0).astype(np.float32)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print(sub.shape)
