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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
import os
import hashlib
from functools import lru_cache


def _id_hash_to_uniform_01(id_str: str, salt: str) -> float:
    h = hashlib.md5((salt + "::" + id_str).encode("utf-8")).digest()
    u = int.from_bytes(h[:8], byteorder="little", signed=False)
    return (u % (10**12)) / float(10**12)


def _index_npy_paths(root: str) -> dict:
    out = {}
    if not os.path.isdir(root):
        return out
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        subdir = entry.path
        for f in os.scandir(subdir):
            if f.is_file() and f.name.endswith(".npy"):
                out[f.name[:-4]] = f.path
    return out


_TRAIN_ROOT = "/kaggle/data/train"
_TEST_ROOT = "/kaggle/data/test"

_TRAIN_PATHS = _index_npy_paths(_TRAIN_ROOT)
_TEST_PATHS = _index_npy_paths(_TEST_ROOT)


def _discover_test_file(id_str: str) -> str:
    p = _TEST_PATHS.get(id_str)
    if p is not None:
        return p
    sub = id_str[0].lower()
    return os.path.join(_TEST_ROOT, sub, f"{id_str}.npy")


def _discover_train_file(id_str: str) -> str:
    p = _TRAIN_PATHS.get(id_str)
    if p is not None:
        return p
    sub = id_str[0].lower()
    return os.path.join(_TRAIN_ROOT, sub, f"{id_str}.npy")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@lru_cache(maxsize=100_000)
def _heuristics_from_npy(path: str) -> dict:
    """
    Keep the same heuristic feature extraction core logic.
    """
    x = np.load(path).astype(np.float32, copy=False)  # (6,273,256)

    on = x[[0, 2, 4]]
    off = x[[1, 3, 5]]

    on_abs_mean = float(np.mean(np.abs(on)))
    off_abs_mean = float(np.mean(np.abs(off)))
    on_std = float(np.std(on))
    off_std = float(np.std(off))

    def _peakiness(a: np.ndarray) -> float:
        aa = a.reshape(-1, a.shape[-2], a.shape[-1])  # (...,273,256)
        row_max = np.max(aa, axis=-1)  # (...,273)
        row_med = np.median(aa, axis=-1)
        return float(np.mean(row_max - row_med))

    on_peak = _peakiness(on)
    off_peak = _peakiness(off)

    on_dt = float(np.mean(np.abs(np.diff(on, axis=-2))))
    off_dt = float(np.mean(np.abs(np.diff(off, axis=-2))))

    on_df = float(np.mean(np.abs(np.diff(on, axis=-1))))
    off_df = float(np.mean(np.abs(np.diff(off, axis=-1))))

    feats = {
        "diff_abs_mean": on_abs_mean - off_abs_mean,
        "diff_std": on_std - off_std,
        "diff_peak": on_peak - off_peak,
        "diff_dt": on_dt - off_dt,
        "diff_df": on_df - off_df,
        "combo": (on_abs_mean - off_abs_mean) + 0.5 * (on_std - off_std),
    }
    return feats


def _collect_train_feature_matrix(
    keys,
    max_files: int = 54000,
    seed: int = 1337,
):
    labels = pd.read_csv("/kaggle/data/train_labels.csv")
    ids = labels["id"].astype(str).values
    y_all = labels["target"].astype(np.int32).values

    rng = np.random.RandomState(seed)
    if (max_files is not None) and (len(ids) > max_files):
        sel = rng.choice(len(ids), size=max_files, replace=False)
        ids = ids[sel]
        y_all = y_all[sel]

    rows = []
    y = []
    for id_str, yy in zip(ids, y_all):
        p = _TRAIN_PATHS.get(id_str)
        if p is None:
            p = _discover_train_file(id_str)
            if not os.path.exists(p):
                continue
        feats = _heuristics_from_npy(p)
        rows.append([float(feats[k]) for k in keys])
        y.append(int(yy))

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return X, y


def _fit_logistic_calibrators_from_train_precomputed(
    keys,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 1337,
    ridge_c: float = 2.0,
) -> dict:
    from sklearn.linear_model import LogisticRegression

    params = {}
    for j, k in enumerate(keys):
        xk = X[:, j].astype(np.float32, copy=False)
        if xk.size < 500:
            params[k] = {"a": 1.0, "b": 0.0, "loc": 0.0, "scale": 1.0}
            continue

        loc = float(np.median(xk))
        mad = float(np.median(np.abs(xk - loc)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = float(np.std(xk) + 1e-6)

        z = ((xk - loc) / (scale + 1e-6)).reshape(-1, 1)

        lr = LogisticRegression(
            penalty="l2",
            C=ridge_c,
            solver="lbfgs",
            max_iter=200,
            random_state=seed,
        )
        lr.fit(z, y)

        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])
        params[k] = {"a": a, "b": b, "loc": loc, "scale": scale}
    return params


def _fit_multifeature_blend_from_train_precomputed(
    keys,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 1337,
    ridge_c: float = 1.0,
) -> dict:
    """
    Preserves the same blend logic; now consumes precomputed X/y to avoid recomputation.
    """
    from sklearn.linear_model import LogisticRegression

    if X.shape[0] < 1000:
        return {
            "loc": np.zeros(len(keys), dtype=np.float32),
            "scale": np.ones(len(keys), dtype=np.float32),
            "w": np.zeros(len(keys), dtype=np.float32),
            "b": 0.0,
        }

    loc = np.median(X, axis=0).astype(np.float32)
    mad = np.median(np.abs(X - loc), axis=0).astype(np.float32)
    scale = (1.4826 * mad).astype(np.float32)
    scale = np.where(
        (~np.isfinite(scale)) | (scale <= 1e-8),
        np.std(X, axis=0).astype(np.float32) + 1e-6,
        scale,
    )

    Z = (X - loc) / (scale + 1e-6)

    lr = LogisticRegression(
        penalty="l2",
        C=ridge_c,
        solver="lbfgs",
        max_iter=200,
        random_state=seed,
    )
    lr.fit(Z, y)

    w = lr.coef_.ravel().astype(np.float32)
    b = float(lr.intercept_.ravel()[0])
    return {"loc": loc, "scale": scale, "w": w, "b": b}


_HEUR_KEYS = ["combo", "diff_peak", "diff_dt", "diff_df", "diff_abs_mean", "diff_std"]

_X_train, _y_train = _collect_train_feature_matrix(
    _HEUR_KEYS, max_files=54000, seed=1337
)

_CALIB_1D = _fit_logistic_calibrators_from_train_precomputed(
    _HEUR_KEYS, _X_train, _y_train, seed=1337, ridge_c=2.0
)

_CALIB_BLEND = _fit_multifeature_blend_from_train_precomputed(
    _HEUR_KEYS, _X_train, _y_train, seed=1337, ridge_c=1.0
)


def _predict_blend_logit_from_feats(feats: dict) -> float:
    loc = _CALIB_BLEND["loc"]
    scale = _CALIB_BLEND["scale"]
    w = _CALIB_BLEND["w"]
    b = _CALIB_BLEND["b"]

    x = np.asarray([float(feats[k]) for k in _HEUR_KEYS], dtype=np.float32)
    z = (x - loc) / (scale + 1e-6)
    return float(np.dot(w, z) + b)


def _predict_blend_prob_from_feats(feats: dict) -> float:
    return float(_sigmoid(_predict_blend_logit_from_feats(feats)))


@lru_cache(maxsize=1)
def _precompute_test_feats() -> dict:
    sample = pd.read_csv("/kaggle/data/sample_submission.csv")
    out = {}
    for id_str in sample["id"].astype(str).values:
        p = _TEST_PATHS.get(id_str)
        if p is None:
            p = _discover_test_file(id_str)
            if not os.path.exists(p):
                continue
        out[id_str] = _heuristics_from_npy(p)
    return out


def _fallback_submission_from_test(salt: str) -> pd.DataFrame:
    sample = pd.read_csv("/kaggle/data/sample_submission.csv")
    ids = sample["id"].astype(str).values
    preds = np.empty(len(sample), dtype=np.float32)

    test_feats = _precompute_test_feats()

    use_blend = salt in ("data4", "data5", "data6")
    blend_logit_shift = {"data4": -0.35, "data5": 0.0, "data6": +0.35}.get(salt, 0.0)

    salt_to_key = {
        "data1": "combo",
        "data2": "diff_peak",
        "data3": "diff_dt",
        "data4": "diff_dt",
        "data5": "combo",
        "data6": "diff_df",
    }
    key = salt_to_key.get(salt, "combo")

    cal = _CALIB_1D.get(key, {"a": 1.0, "b": 0.0, "loc": 0.0, "scale": 1.0})
    a = float(cal.get("a", 1.0))
    b1 = float(cal.get("b", 0.0))
    loc1 = float(cal.get("loc", 0.0))
    scale1 = float(cal.get("scale", 1.0))

    eps = 1e-6
    for i, id_str in enumerate(ids):
        feats = test_feats.get(id_str)
        if feats is not None:
            if use_blend:
                logit = _predict_blend_logit_from_feats(feats) + blend_logit_shift
                prob = float(_sigmoid(logit))
            else:
                x = float(feats[key])
                z = (x - loc1) / (scale1 + 1e-6)
                prob = float(_sigmoid(a * z + b1))

            if prob < eps:
                prob = eps
            elif prob > 1.0 - eps:
                prob = 1.0 - eps
            preds[i] = prob
        else:
            preds[i] = _id_hash_to_uniform_01(id_str, salt)

    out = sample.copy()
    out["target"] = preds.astype(float)
    return out


def _safe_read_submission(path: str, salt: str) -> pd.DataFrame:
    sample = pd.read_csv("/kaggle/data/sample_submission.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        if "id" not in df.columns or "target" not in df.columns:
            raise ValueError(f"{path} must contain columns ['id','target']")
        df = df[["id", "target"]].copy()
        df = sample[["id"]].merge(df, on="id", how="left")

        if df["target"].isna().any():
            fb = _fallback_submission_from_test(salt=salt)[["id", "target"]]
            df = df.drop(columns=["target"]).merge(fb, on="id", how="left")

        df["target"] = df["target"].astype(float)
        return df

    return _fallback_submission_from_test(salt=salt)


data1 = _safe_read_submission(
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    salt="data1",
)
data2 = _safe_read_submission(
    "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    salt="data2",
)
data3 = _safe_read_submission(
    "../input/seti-bl-tf-starter-tpu/submission.csv",
    salt="data3",
)
data4 = _safe_read_submission(
    "../input/seti-learned-image-resizing/submission.csv",
    salt="data4",
)
data5 = _safe_read_submission(
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    salt="data5",
)
data6 = _safe_read_submission(
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
    salt="data6",
)



## === cell 2
data1.head()



## === cell 3
data2.head()



## === cell 4
data6 = data1.copy()
data6["target"] = (
    0.8 * data5["target"]
    + 0.115 * data4["target"]
    + 0.085 * data6["target"]
    + 0.0 * data2["target"]
    + 0.0 * data3["target"]
)



## === cell 5
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(data6), "rows and columns:", list(data6.columns))
print(
    "target summary:", data6["target"].describe(percentiles=[0.01, 0.5, 0.99]).to_dict()
)
