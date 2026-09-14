# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Identify technosignature signals in cadence snippets taken from a digital spectrometer.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
00034abb3629,0.5
0004be0baf70,0.5
0005be4d0752,0.5
etc.

```

## Dataset
The data is from a digital spectrometer, which takes incoming raw data from the telescope (amounting to hundreds of TB per day) and performs a Fourier Transform to generate a spectrogram. These spectrograms, also referred to as filterbank files, or dynamic spectra, consist of measurements of signal intensity as a function of frequency and time.

Below is an example of an FM radio signal. This is not from the GBT, but from a small antenna attached to a software defined radio dongle (a $20 piece of kit that you can plug into your laptop to pick up signals). The data we get from the GBT are very similar, but split into larger numbers of frequency channels, covering a much broader instantaneous frequency range, and with much better sensitivity.

![frequency-time-plot](https://prod-files-secure.s3.us-west-2.amazonaws.com/667f1cbf-826f-4641-a321-96054292638d/b59a57f3-11a7-4493-8268-55c3fa632f7e/Untitled.png)

The screenshot above shows frequency on the horizontal axis (running from around 88.2 to 89.8 MHz) and time on the vertical axis. The bright orange feature at 88.5 MHz is the FM signal from KQED, a radio station in the San Francisco Bay Area. The solid yellow blocks on either side (one highlighted by the pointer in the screenshot) are the KQED “HD radio” signal (the same data as the FM signal, but encoded digitally). Additional FM stations are visible at different frequencies, including another obvious FM signal (without the corresponding digital sidebands) at 89.5 MHz.

The spectrometer generates similar spectrograms to the one shown above, but typically spanning several GHz of the radio spectrum (rather than the approx. 2 MHz shown above). The data are stored either as filterbank format or HDF5 format files, but essentially are arrays of intensity as a function of frequency and time, accompanied by headers containing metadata such as the direction the telescope was pointed in, the frequency scale, and so on. We generate over 1 PB of spectrograms per year; individual filterbank files can be tens of GB in size. We have discarded the majority of the metadata and are simply presenting numpy arrays consisting of small regions of the spectrograms that we refer to as “snippets”.

The spectrometer is searching for candidate signatures of extraterrestrial technology - so-called technosignatures. The main obstacle to doing so is that our own human technology (not just radio stations, but wifi routers, cellphones, and even electronics that are not deliberately designed to transmit radio signals) also gives off radio signals. We refer to these human-generated signals as “radio frequency interference”, or RFI.

One method we use to isolate candidate technosignatures from RFI is to look for signals that appear to be coming from particular positions on the sky. Typically we do this by alternating observations of our primary target star with observations of three nearby stars: 5 minutes on star “A”, then 5 minutes on star “B”, then back to star “A” for 5 minutes, then “C”, then back to “A”, then finishing with 5 minutes on star “D”. One set of six observations (ABACAD) is referred to as a “cadence”. Since we're just giving you a small range of frequencies for each cadence, we refer to the datasets you'll be analyzing as “cadence snippets”.

An example of an extraterrestrial signal:

![voyager-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.39.42.png)

As the plot title suggests, this is the Voyager 1 spacecraft. Even though it's 20 billion kilometers from Earth, it's picked up clearly by the GBT. The first, third, and fifth panels are the “A” target (the spacecraft, in this case). The yellow diagonal line is the radio signal coming from Voyager. It's detected when we point at the spacecraft, and it disappears when we point away. It's a diagonal line in this plot because the relative motion of the Earth and the spacecraft imparts a Doppler drift, causing the frequency to change over time. As it happens, that's another possible way to reject RFI, which has a higher tendency to remain at a fixed frequency over time.

While it would be nice to train our algorithms entirely on observations of interplanetary spacecraft, there are not many examples of them, and we also want to be able to find a wider range of signal types. So we've turned to simulating technosignature candidates.

We've taken tens of thousands of cadence snippets, which we're calling the haystack, and we've hidden needles among them. Some of these needles look similar to the Voyager 1 signal above and should be easy to detect, even with classical detection algorithms. Others are hidden in noisy regions of the spectrum and will be harder, even though they might be relatively obvious on visual inspection:

![needle-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.34.06.png)

After we perform the signal injections, we normalize each snippet, so you probably can't identify most of the needles just by looking for excess energy in the corresponding array. You'll likely need a more subtle algorithm that looks for patterns that appear only in the on-target observations.

Not all of the “needle” signals look like diagonal lines, and they may not be present for the entirety of all three “A” observations, but what they do have in common is that they are only present in some or all of the “A” observations (panels 1, 3, and 5 in the cadence snippets). Your challenge is to train an algorithm to find as many needles as you can, while minimizing the number of false positives from the haystack.

- **train/** - a training set of cadence snippet files stored in `numpy` `float16` format (v1.20.1), one file per cadence snippet `id`, with corresponding labels found in the `train_labels.csv` file. Each file has dimension `(6, 273, 256)`, with the 1st dimension representing the 6 positions of the cadence, and the 2nd and 3rd dimensions representing the 2D spectrogram.
- **test/** - the test set cadence snippet files; you must predict whether or not the cadence contains a "needle", which is the `target` for this competition
- **sample_submission.csv** - a sample submission file in the correct format
- **train_labels** - targets corresponding (by `id`) to the cadence snippet files found in the `train/` folder
- **old_leaky_data** - full pre-relaunch data, including test labels; you should not assume this data is helpful (it may or may not be).

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        input/
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        working/
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
```

-> data/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> data/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(42)

BASE_INPUT = "/kaggle/input"
DATA_ROOT = os.path.join(BASE_INPUT, "seti-breakthrough-listen")

TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

assert os.path.exists(TRAIN_LABELS_CSV), f"Missing: {TRAIN_LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

assert list(sample_sub.columns) == [
    "id",
    "target",
], f"Unexpected sample_submission columns: {sample_sub.columns.tolist()}"
assert list(train_labels.columns) == [
    "id",
    "target",
], f"Unexpected train_labels columns: {train_labels.columns.tolist()}"

train_labels.head(), sample_sub.head()




## === cell 1
def _stats_1d(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    m = np.float32(v.mean())
    s = np.float32(v.std())
    mn = np.float32(v.min())
    mx = np.float32(v.max())

    q10, q50, q90 = np.quantile(v, (0.1, 0.5, 0.9), method="linear").astype(
        np.float32, copy=False
    )
    return np.array([m, s, mn, mx, q10, q50, q90], dtype=np.float32)


def _stats_2d(z2d: np.ndarray) -> np.ndarray:
    return _stats_1d(z2d)


def _grad_mag_stats(
    z2d: np.ndarray, _buf_gx: np.ndarray, _buf_gy: np.ndarray, _buf_g: np.ndarray
) -> np.ndarray:
    z2d = np.asarray(z2d, dtype=np.float32, order="C")
    gx = _buf_gx
    gy = _buf_gy
    g = _buf_g

    gx.fill(0.0)
    gy.fill(0.0)

    gx[:, :-1] = np.diff(z2d, axis=1)
    gy[:-1, :] = np.diff(z2d, axis=0)

    np.multiply(gx, gx, out=g)
    g += gy * gy
    np.sqrt(g, out=g)
    return _stats_2d(g)


def extract_features_from_snippet(
    x: np.ndarray,
    buf_gx: np.ndarray = None,
    buf_gy: np.ndarray = None,
    buf_g: np.ndarray = None,
) -> np.ndarray:
    """
    x shape: (6, 273, 256) float16/float32
    Cadence order: A, B, A, C, A, D
    Minimal, deterministic features:
      - global stats per panel
      - contrasts: A_panels vs off-target (B/C/D) panels
      - simple gradient magnitude stats to capture line-like structures
    """
    x = np.asarray(x, dtype=np.float32, order="C")
    if x.shape != (6, 273, 256):
        raise ValueError(f"Unexpected snippet shape {x.shape}; expected (6, 273, 256)")

    A1, B, A2, C, A3, D = x[0], x[1], x[2], x[3], x[4], x[5]
    A = (A1 + A2 + A3) / 3.0
    Off = (B + C + D) / 3.0
    Diff = A - Off
    AbsDiff = np.abs(Diff)

    feats = []
    for i in range(6):
        feats.append(_stats_2d(x[i]))
    feats.append(_stats_2d(A))
    feats.append(_stats_2d(Off))
    feats.append(_stats_2d(Diff))
    feats.append(_stats_2d(AbsDiff))

    if buf_gx is None:
        buf_gx = np.empty((273, 256), dtype=np.float32)
    if buf_gy is None:
        buf_gy = np.empty((273, 256), dtype=np.float32)
    if buf_g is None:
        buf_g = np.empty((273, 256), dtype=np.float32)

    feats.append(_grad_mag_stats(A, buf_gx, buf_gy, buf_g))
    feats.append(_grad_mag_stats(Off, buf_gx, buf_gy, buf_g))
    feats.append(_grad_mag_stats(Diff, buf_gx, buf_gy, buf_g))
    feats.append(_grad_mag_stats(AbsDiff, buf_gx, buf_gy, buf_g))

    A_time = A.mean(axis=1)
    A_freq = A.mean(axis=0)
    Off_time = Off.mean(axis=1)
    Off_freq = Off.mean(axis=0)
    Diff_time = Diff.mean(axis=1)
    Diff_freq = Diff.mean(axis=0)

    feats.append(_stats_1d(A_time))
    feats.append(_stats_1d(A_freq))
    feats.append(_stats_1d(Off_time))
    feats.append(_stats_1d(Off_freq))
    feats.append(_stats_1d(Diff_time))
    feats.append(_stats_1d(Diff_freq))

    out = np.concatenate(feats, axis=0)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite values encountered in extracted features.")
    return out


def list_npy_files(root_dir: str):
    files = glob.glob(os.path.join(root_dir, "*", "*.npy"))
    files.sort()
    return files


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

train_id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in train_files}
test_id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in test_files}

missing_train = [i for i in train_labels["id"].tolist() if i not in train_id_to_path]
if len(missing_train) > 0:
    raise FileNotFoundError(
        f"Some labeled train ids are missing .npy files. Example: {missing_train[:5]}"
    )

missing_test = [i for i in sample_sub["id"].tolist() if i not in test_id_to_path]
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Some sample_submission test ids are missing .npy files. Example: {missing_test[:5]}"
    )

len(train_files), len(test_files), len(train_id_to_path), len(test_id_to_path)


## === cell 2
from concurrent.futures import ThreadPoolExecutor
import threading

_tls = threading.local()


def _get_thread_buffers():
    bgx = getattr(_tls, "bgx", None)
    bgy = getattr(_tls, "bgy", None)
    bg = getattr(_tls, "bg", None)
    if bgx is None or bgy is None or bg is None:
        _tls.bgx = np.empty((273, 256), dtype=np.float32)
        _tls.bgy = np.empty((273, 256), dtype=np.float32)
        _tls.bg = np.empty((273, 256), dtype=np.float32)
    return _tls.bgx, _tls.bgy, _tls.bg


FDIM = (
    20 * 7
)  # 6 panels + A/Off/Diff/AbsDiff + 4 grad-mag + 6 1D projections, each with 7 stats


def make_feature_matrix(
    ids, id_to_path, batch_size=512, max_workers=None, cache_path=None
):
    if cache_path is not None and os.path.exists(cache_path):
        X = np.load(cache_path)
        if X.shape[0] == len(ids) and X.shape[1] == FDIM and X.dtype == np.float32:
            return X

    n = len(ids)
    X = np.empty((n, FDIM), dtype=np.float32)

    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 2))

    def _load_and_featurize(_id: str) -> np.ndarray:
        p = id_to_path.get(_id, None)
        if p is None:
            raise FileNotFoundError(f"Missing path for id={_id}")
        arr = np.load(p, mmap_mode="r")
        bgx, bgy, bg = _get_thread_buffers()
        return extract_features_from_snippet(arr, buf_gx=bgx, buf_gy=bgy, buf_g=bg)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i0 in range(0, n, batch_size):
            batch_ids = ids[i0 : i0 + batch_size]
            chunksize = max(1, len(batch_ids) // (max_workers * 4))
            for j, feat in enumerate(
                ex.map(_load_and_featurize, batch_ids, chunksize=chunksize)
            ):
                X[i0 + j] = feat

    if cache_path is not None:
        cache_dir = os.path.dirname(cache_path) or "."
        os.makedirs(cache_dir, exist_ok=True)

        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, X)
        os.replace(tmp, cache_path)

    return X


train_ids = train_labels["id"].tolist()
y = train_labels["target"].values.astype(np.int32)

X = make_feature_matrix(
    train_ids,
    train_id_to_path,
    batch_size=768,
    max_workers=None,
    cache_path="/kaggle/working/X_train_features.npy",
)
X.shape


## === cell 3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(train_ids), dtype=np.float32)

pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                C=1.0,
                class_weight=None,
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr = y[tr_idx]
    pipe.fit(Xtr, ytr)
    oof[va_idx] = pipe.predict_proba(Xva)[:, 1].astype(np.float32)

pipe.fit(X, y)

oof[:5], oof.mean(), oof.min(), oof.max()


## === cell 4
test_ids = sample_sub["id"].tolist()
X_test = make_feature_matrix(
    test_ids,
    test_id_to_path,
    batch_size=768,
    max_workers=None,
    cache_path="/kaggle/working/X_test_features.npy",
)

test_pred = pipe.predict_proba(X_test)[:, 1].astype(np.float32)

sub = pd.DataFrame({"id": test_ids, "target": test_pred})
sub.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["id", "target"]

sub.head()
