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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import hashlib
import numpy as np
import pandas as pd

BASE = "/kaggle/input"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
TRAIN_LABELS_PATH = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_ids = train_labels["id"].astype(str).tolist()
y = train_labels["target"].astype(int).to_numpy()
test_ids = sample_sub["id"].astype(str).tolist()

print("Train labels:", len(train_ids))
print("Test ids:", len(test_ids))

_A_IDX = np.array([0, 2, 4], dtype=np.int64)
_O_IDX = np.array([1, 3, 5], dtype=np.int64)

_F_BOUNDS = np.array([0, 64, 128, 192, 256], dtype=np.int64)
_T_BOUNDS = np.array([0, 91, 182, 273], dtype=np.int64)

_PIX_GLOBAL = np.float32(273 * 256)
_PIX_BAND = np.float32(273 * 64)
_PIX_SEG = np.float32(91 * 256)
_DX_PIX_GLOBAL = np.float32((273 - 1) * 256)  # 272*256
_DX_PIX_BAND = np.float32((273 - 1) * 64)  # 272*64
_DX_PIX_SEG = np.float32((91 - 1) * 256)  # 90*256


def id_to_path(root_dir: str, _id: str):
    return os.path.join(root_dir, _id[0], f"{_id}.npy")


train_paths = [id_to_path(TRAIN_DIR, _id) for _id in train_ids]
test_paths = [id_to_path(TEST_DIR, _id) for _id in test_ids]

_missing = [p for p in (train_paths[:10] + test_paths[:10]) if not os.path.exists(p)]
if _missing:
    raise FileNotFoundError(f"Example missing paths: {_missing[:3]}")


def _a_off_features(v6: np.ndarray) -> np.ndarray:
    a = v6[_A_IDX].mean()
    o = v6[_O_IDX].mean()
    diff = a - o
    ratio = a / (o + 1e-6)
    return np.array([a, o, diff, ratio], dtype=np.float32)


def extract_features_from_arr(x: np.ndarray) -> np.ndarray:
    """
    x: (6, 273, 256), float16/float32
    Same features as original:
      - per-panel mean/std/max/min (6 each)
      - per-panel time-gradient mean over |diff along time|
      - A-vs-OFF features for global stats and for 4 freq bands and 3 time segments, including tg
    """
    x = x.astype(
        np.float32, copy=False
    )  # keep semantics; avoid extra copy when already float32

    sum_g = x.sum(axis=(1, 2))
    sumsq_g = np.square(x).sum(axis=(1, 2))
    max_g = x.max(axis=(1, 2))
    min_g = x.min(axis=(1, 2))

    mean_g = (sum_g / _PIX_GLOBAL).astype(np.float32, copy=False)
    var_g = (sumsq_g / _PIX_GLOBAL) - np.square(mean_g)
    var_g = np.maximum(var_g, 0.0)
    std_g = np.sqrt(var_g, dtype=np.float32)

    dx = np.abs(x[:, 1:, :] - x[:, :-1, :])  # (6,272,256)
    tg_g = (dx.sum(axis=(1, 2)) / _DX_PIX_GLOBAL).astype(np.float32, copy=False)

    a_range = max_g[_A_IDX].mean() - min_g[_A_IDX].mean()
    o_range = max_g[_O_IDX].mean() - min_g[_O_IDX].mean()

    global_ao = np.concatenate(
        [
            _a_off_features(mean_g),
            _a_off_features(std_g),
            _a_off_features(min_g),
            _a_off_features(max_g),
            _a_off_features(tg_g),
            np.array(
                [a_range, o_range, a_range - o_range, a_range / (o_range + 1e-6)],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )

    band_ao_list = []
    for b in range(4):
        f0, f1 = int(_F_BOUNDS[b]), int(_F_BOUNDS[b + 1])
        xb = x[:, :, f0:f1]
        sum_b = xb.sum(axis=(1, 2))
        sumsq_b = np.square(xb).sum(axis=(1, 2))
        max_b = xb.max(axis=(1, 2))
        min_b = xb.min(axis=(1, 2))

        mean_b = (sum_b / _PIX_BAND).astype(np.float32, copy=False)
        var_b = (sumsq_b / _PIX_BAND) - np.square(mean_b)
        var_b = np.maximum(var_b, 0.0)
        std_b = np.sqrt(var_b, dtype=np.float32)

        dxb = dx[:, :, f0:f1]
        tg_b = (dxb.sum(axis=(1, 2)) / _DX_PIX_BAND).astype(np.float32, copy=False)

        band_ao_list.append(_a_off_features(mean_b))  # mean
        band_ao_list.append(_a_off_features(std_b))  # std
        band_ao_list.append(_a_off_features(min_b))  # min
        band_ao_list.append(_a_off_features(max_b))  # max
        band_ao_list.append(_a_off_features(tg_b))  # tg
    band_ao = np.concatenate(band_ao_list, axis=0)

    seg_ao_list = []
    for s in range(3):
        t0, t1 = int(_T_BOUNDS[s]), int(_T_BOUNDS[s + 1])
        xs = x[:, t0:t1, :]
        sum_s = xs.sum(axis=(1, 2))
        sumsq_s = np.square(xs).sum(axis=(1, 2))
        max_s = xs.max(axis=(1, 2))
        min_s = xs.min(axis=(1, 2))

        mean_s = (sum_s / _PIX_SEG).astype(np.float32, copy=False)
        var_s = (sumsq_s / _PIX_SEG) - np.square(mean_s)
        var_s = np.maximum(var_s, 0.0)
        std_s = np.sqrt(var_s, dtype=np.float32)

        tg_s = (dx[:, t0 : t1 - 1, :].sum(axis=(1, 2)) / _DX_PIX_SEG).astype(
            np.float32, copy=False
        )

        seg_ao_list.append(_a_off_features(mean_s))  # mean
        seg_ao_list.append(_a_off_features(std_s))  # std
        seg_ao_list.append(_a_off_features(min_s))  # min
        seg_ao_list.append(_a_off_features(max_s))  # max
        seg_ao_list.append(_a_off_features(tg_s))  # tg
    seg_ao = np.concatenate(seg_ao_list, axis=0)

    feats = np.concatenate(
        [
            mean_g,  # 6
            std_g,  # 6
            max_g,  # 6
            min_g,  # 6  -> 24
            tg_g,  # 6
            global_ao,  # 24
            band_ao,  # 80
            seg_ao,  # 60
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    return feats




## === cell 1
import multiprocessing as mp


def _cache_key_from_paths(paths, prefix: str) -> str:
    h = hashlib.sha1()
    h.update(prefix.encode("utf-8"))
    h.update(b"|")
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _featurize_one_path(p: str) -> np.ndarray:
    arr = np.load(p, mmap_mode="r", allow_pickle=False)
    return extract_features_from_arr(arr)


def build_feature_matrix_from_paths(
    paths,
    verbose_every: int = 5000,
    cache_prefix: str = "",
    n_feat: int = None,
    n_workers: int = None,
) -> np.ndarray:
    if len(paths) == 0:
        return np.empty((0, 0), dtype=np.float32)

    if n_feat is None:
        arr0 = np.load(paths[0], mmap_mode="r", allow_pickle=False)
        n_feat = int(extract_features_from_arr(arr0).shape[0])

    X_shape = (len(paths), n_feat)
    cache_dir = "/kaggle/working"
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key_from_paths(paths, cache_prefix + f"_nf{n_feat}")
    cache_path = os.path.join(cache_dir, f"features_{cache_prefix}_{key}.npy")

    if os.path.exists(cache_path):
        X_cached = np.load(cache_path, allow_pickle=False)
        if X_cached.shape == X_shape and X_cached.dtype == np.float32:
            print(f"Loaded cached {cache_prefix} features from {cache_path}")
            return X_cached

    X = np.empty(X_shape, dtype=np.float32)

    if n_workers is None:
        cpu = os.cpu_count() or 2
        n_workers = max(1, min(4, cpu // 2))

    chunksize = 256 if len(paths) >= 10000 else 64

    if n_workers == 1:
        for i, p in enumerate(paths):
            X[i] = _featurize_one_path(p)
            if verbose_every and (
                ((i + 1) % verbose_every == 0) or ((i + 1) == len(paths))
            ):
                print(f"Featurizing {cache_prefix} {i+1}/{len(paths)}")
    else:
        ctx = mp.get_context(
            "fork"
        )  # Kaggle linux; faster than spawn and deterministic here.
        with ctx.Pool(processes=n_workers, maxtasksperchild=1000) as pool:
            for i, feat in enumerate(
                pool.imap(_featurize_one_path, paths, chunksize=chunksize)
            ):
                X[i] = feat
                if verbose_every and (
                    ((i + 1) % verbose_every == 0) or ((i + 1) == len(paths))
                ):
                    print(f"Featurizing {cache_prefix} {i+1}/{len(paths)}")

    tmp_path = cache_path + ".tmp"
    np.save(tmp_path, X, allow_pickle=False)
    os.replace(tmp_path, cache_path)
    print(f"Saved cached {cache_prefix} features to {cache_path}")
    return X


X_train = build_feature_matrix_from_paths(
    train_paths, verbose_every=20000, cache_prefix="train", n_workers=None
)
X_test = build_feature_matrix_from_paths(
    test_paths,
    verbose_every=4000,
    cache_prefix="test",
    n_feat=X_train.shape[1],
    n_workers=None,
)

print("Shapes:", X_train.shape, X_test.shape)



## === cell 2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

test_pred_sum = np.zeros(len(test_ids), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train[tr_idx], y[tr_idx])
    p_test = model.predict_proba(X_test)[:, 1].astype(np.float64)
    test_pred_sum += p_test
    print(f"Fold {fold}: done")

final_target = (test_pred_sum / skf.get_n_splits()).astype(np.float32)
final_target = np.clip(final_target, 0.0, 1.0)

submission = sample_sub.copy()
submission["target"] = final_target
print(submission.head())



## === cell 3
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

sub = pd.read_csv(out_path)
assert list(sub.columns) == ["id", "target"]
assert len(sub) == len(sample_sub)
assert sub["id"].is_unique
assert (sub["id"].astype(str).values == sample_sub["id"].astype(str).values).all()
print("Wrote", out_path, "with shape", sub.shape)
print(sub.head())
