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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

SAMPLE_SUB_PATHS = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/sample_submission.csv",
]

TRAIN_LABELS_PATHS = [
    "/kaggle/input/train_labels.csv",
    "/kaggle/data/train_labels.csv",
    "../input/train_labels.csv",
]

OLD_TRAIN_LABELS_PATHS = [
    "/kaggle/input/old_leaky_data/train_labels_old.csv",
    "/kaggle/data/old_leaky_data/train_labels_old.csv",
    "../input/old_leaky_data/train_labels_old.csv",
]

TRAIN_DIR_CANDIDATES = [
    "/kaggle/input/train",
    "/kaggle/data/train",
    "../input/train",
]

TEST_DIR_CANDIDATES = [
    "/kaggle/input/test",
    "/kaggle/data/test",
    "../input/test",
]

OLD_TRAIN_DIR_CANDIDATES = [
    "/kaggle/input/old_leaky_data/train_old",
    "/kaggle/data/old_leaky_data/train_old",
    "../input/old_leaky_data/train_old",
]


def _read_first_existing_csv(paths):
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find required CSV in: {paths}")


def _find_first_existing_dir(candidates):
    for d in candidates:
        if os.path.exists(d) and os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Could not find required directory in: {candidates}")


def _read_sample_submission():
    df = _read_first_existing_csv(SAMPLE_SUB_PATHS)
    if not (set(df.columns) >= {"id", "target"}):
        raise ValueError(
            f"sample_submission must contain id,target. Got {df.columns.tolist()}"
        )
    return df[["id", "target"]].copy()


sample_sub = _read_sample_submission()


def align_on_id(df, ref_ids):
    """Align df to ref_ids order by id; requires all ids present."""
    out = df.set_index("id").reindex(ref_ids)
    if out["target"].isna().any():
        missing = int(out["target"].isna().sum())
        raise ValueError(
            f"Alignment produced {missing} missing targets; id sets do not match."
        )
    return out.reset_index()


_HEX_PREFIXES = tuple("0123456789abcdef")


def _resolve_npy_path(root_dir, id_):
    p = os.path.join(root_dir, id_[0], f"{id_}.npy")
    if os.path.exists(p):
        return p
    for h in _HEX_PREFIXES:
        pp = os.path.join(root_dir, h, f"{id_}.npy")
        if os.path.exists(pp):
            return pp
    raise FileNotFoundError(f"Could not find .npy for id={id_} under {root_dir}")


def _extract_features_from_path(npy_path):
    """
    Same feature computations (no approximations), kept identical for correctness.
    Input array shape: (6, 273, 256)
    """
    x = np.load(npy_path, mmap_mode="r")  # stored float16, memmap read-only

    a = x[[0, 2, 4]]
    off = x[[1, 3, 5]]

    a32 = a.astype(np.float32, copy=False)
    off32 = off.astype(np.float32, copy=False)

    n = a32.size  # equals off32.size
    inv_n = 1.0 / float(n)

    a_sum = float(a32.sum(dtype=np.float32))
    off_sum = float(off32.sum(dtype=np.float32))
    a_sumsq = float(np.square(a32, dtype=np.float32).sum(dtype=np.float32))
    off_sumsq = float(np.square(off32, dtype=np.float32).sum(dtype=np.float32))

    a_mean = a_sum * inv_n
    off_mean = off_sum * inv_n

    a_var = a_sumsq * inv_n - a_mean * a_mean
    off_var = off_sumsq * inv_n - off_mean * off_mean
    if a_var < 0.0:
        a_var = 0.0
    if off_var < 0.0:
        off_var = 0.0

    a_std = float(np.sqrt(a_var, dtype=np.float32))
    off_std = float(np.sqrt(off_var, dtype=np.float32))

    a_abs = float(np.abs(a32).mean(dtype=np.float32))
    off_abs = float(np.abs(off32).mean(dtype=np.float32))

    a_mean_0 = a32.mean(axis=0, dtype=np.float32)
    off_mean_0 = off32.mean(axis=0, dtype=np.float32)
    diff = a_mean_0 - off_mean_0

    diff_mean = float(diff.mean(dtype=np.float32))
    diff_sumsq = float(np.square(diff, dtype=np.float32).sum(dtype=np.float32))
    diff_n = diff.size
    diff_inv_n = 1.0 / float(diff_n)
    diff_var = diff_sumsq * diff_inv_n - diff_mean * diff_mean
    if diff_var < 0.0:
        diff_var = 0.0
    diff_std = float(np.sqrt(diff_var, dtype=np.float32))
    diff_abs = float(np.abs(diff).mean(dtype=np.float32))

    a_max = float(a32.max())
    off_max = float(off32.max())
    max_gap = a_max - off_max

    eps = 1e-6
    std_ratio = (a_std + eps) / (off_std + eps)
    abs_ratio = (a_abs + eps) / (off_abs + eps)

    a_t = a32.mean(axis=(0, 2), dtype=np.float32)  # (273,)
    off_t = off32.mean(axis=(0, 2), dtype=np.float32)
    dt = a_t - off_t
    dt_mean = float(dt.mean(dtype=np.float32))
    dt_std = float(dt.std(dtype=np.float32))
    dt_abs = float(np.abs(dt).mean(dtype=np.float32))
    dt_max = float(dt.max(initial=-np.inf))

    a_f = a32.mean(axis=(0, 1), dtype=np.float32)  # (256,)
    off_f = off32.mean(axis=(0, 1), dtype=np.float32)
    df = a_f - off_f
    df_mean = float(df.mean(dtype=np.float32))
    df_std = float(df.std(dtype=np.float32))
    df_abs = float(np.abs(df).mean(dtype=np.float32))
    df_max = float(df.max(initial=-np.inf))

    def _quantile_linear_flat(arr, q):
        flat = arr.reshape(-1)
        N = flat.size
        if N == 0:
            return np.nan
        pos = q * (N - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        if lo == hi:
            kth = np.partition(flat, lo)[lo]
            return float(kth)
        part = np.partition(flat, hi)
        x_lo = float(part[lo])
        x_hi = float(part[hi])
        w = float(pos - lo)
        return x_lo * (1.0 - w) + x_hi * w

    a_p99 = float(_quantile_linear_flat(a32, 0.99))
    off_p99 = float(_quantile_linear_flat(off32, 0.99))
    p99_gap = a_p99 - off_p99

    return np.array(
        [
            a_mean,
            off_mean,
            a_std,
            off_std,
            a_abs,
            off_abs,
            diff_mean,
            diff_std,
            diff_abs,
            a_max,
            off_max,
            max_gap,
            std_ratio,
            abs_ratio,
            dt_mean,
            dt_std,
            dt_abs,
            dt_max,
            df_mean,
            df_std,
            df_abs,
            df_max,
            a_p99,
            off_p99,
            p99_gap,
        ],
        dtype=np.float32,
    )


def _features_cache_path(root_dir, ids, n_features=25):
    root_dir = os.path.abspath(root_dir)
    cache_dir = "/kaggle/working"
    os.makedirs(cache_dir, exist_ok=True)
    ids0 = ids[0] if len(ids) else "none"
    ids1 = ids[-1] if len(ids) else "none"
    h = 2166136261
    for s in (ids0, ids1, str(len(ids))):
        for b in s.encode("utf-8"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
    key = f"feats_{os.path.basename(root_dir)}_{len(ids)}_{ids0}_{ids1}_{h}_{n_features}.npz"
    return os.path.join(cache_dir, key)


def _build_id_to_path_map(root_dir):
    id_to_path = {}
    for h in _HEX_PREFIXES:
        d = os.path.join(root_dir, h)
        if not (os.path.exists(d) and os.path.isdir(d)):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".npy"):
                id_to_path[fn[:-4]] = os.path.join(d, fn)
    return id_to_path


def _compute_features_parallel(paths, n_features=25):
    from concurrent.futures import ThreadPoolExecutor

    feats = np.empty((len(paths), n_features), dtype=np.float32)

    def _work(i_p):
        i, p = i_p
        return i, _extract_features_from_path(p)

    max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, f in ex.map(_work, enumerate(paths), chunksize=64):
            feats[i] = f
    return feats


def _build_feature_df(root_dir, ids):
    n_features = 25
    cache_path = _features_cache_path(root_dir, ids, n_features=n_features)
    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path)
            feats = z["feats"]
            if feats.shape == (len(ids), n_features):
                cols = [
                    "a_mean",
                    "off_mean",
                    "a_std",
                    "off_std",
                    "a_abs",
                    "off_abs",
                    "diff_mean",
                    "diff_std",
                    "diff_abs",
                    "a_max",
                    "off_max",
                    "max_gap",
                    "std_ratio",
                    "abs_ratio",
                    "dt_mean",
                    "dt_std",
                    "dt_abs",
                    "dt_max",
                    "df_mean",
                    "df_std",
                    "df_abs",
                    "df_max",
                    "a_p99",
                    "off_p99",
                    "p99_gap",
                ]
                return pd.DataFrame(feats, columns=cols)
        except Exception:
            pass  # fall through to recompute

    id_to_path = _build_id_to_path_map(root_dir)
    try:
        paths = [id_to_path[id_] for id_ in ids]
    except KeyError:
        paths = [_resolve_npy_path(root_dir, id_) for id_ in ids]

    feats = _compute_features_parallel(paths, n_features=n_features)

    try:
        np.savez_compressed(cache_path, feats=feats)
    except Exception:
        pass

    cols = [
        "a_mean",
        "off_mean",
        "a_std",
        "off_std",
        "a_abs",
        "off_abs",
        "diff_mean",
        "diff_std",
        "diff_abs",
        "a_max",
        "off_max",
        "max_gap",
        "std_ratio",
        "abs_ratio",
        "dt_mean",
        "dt_std",
        "dt_abs",
        "dt_max",
        "df_mean",
        "df_std",
        "df_abs",
        "df_max",
        "a_p99",
        "off_p99",
        "p99_gap",
    ]
    return pd.DataFrame(feats, columns=cols)


def _local_model_predictions():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    labels_new = _read_first_existing_csv(TRAIN_LABELS_PATHS)
    if not (set(labels_new.columns) >= {"id", "target"}):
        raise ValueError(
            f"train_labels must contain id,target. Got {labels_new.columns.tolist()}"
        )
    labels_new = labels_new[["id", "target"]].copy()
    labels_new["target"] = labels_new["target"].astype(int)

    train_dir_new = _find_first_existing_dir(TRAIN_DIR_CANDIDATES)
    test_dir = _find_first_existing_dir(TEST_DIR_CANDIDATES)

    use_old = True
    try:
        labels_old = _read_first_existing_csv(OLD_TRAIN_LABELS_PATHS)
        if not (set(labels_old.columns) >= {"id", "target"}):
            use_old = False
        else:
            labels_old = labels_old[["id", "target"]].copy()
            labels_old["target"] = labels_old["target"].astype(int)
            train_dir_old = _find_first_existing_dir(OLD_TRAIN_DIR_CANDIDATES)
    except Exception:
        use_old = False

    test_ids = sample_sub["id"].tolist()

    X_new = _build_feature_df(train_dir_new, labels_new["id"].tolist())
    y_new = labels_new["target"].values

    if use_old:
        X_old = _build_feature_df(train_dir_old, labels_old["id"].tolist())
        y_old = labels_old["target"].values
        X_all = pd.concat([X_new, X_old], axis=0, ignore_index=True)
        y_all = np.concatenate([y_new, y_old], axis=0)
    else:
        X_all = X_new
        y_all = y_new

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=0.15, random_state=0, stratify=y_all
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    max_iter=300,
                    solver="lbfgs",
                    random_state=0,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    clf.fit(X_tr, y_tr)

    va_pred = clf.predict_proba(X_va)[:, 1]
    try:
        print(
            "Local validation AUC (fallback LR):", float(roc_auc_score(y_va, va_pred))
        )
        print("Used old_leaky_data augmentation:", bool(use_old))
        if use_old:
            print("Train sizes new/old/total:", len(y_new), len(y_old), len(y_all))
        else:
            print("Train size total:", len(y_all))
    except Exception as e:
        print("Could not compute local AUC:", repr(e))

    clf.fit(X_all, y_all)
    X_test = _build_feature_df(test_dir, test_ids)
    proba = clf.predict_proba(X_test)[:, 1].astype(np.float64)

    out = pd.DataFrame({"id": test_ids, "target": proba})
    out["target"] = out["target"].clip(0.0, 1.0)
    return out


_LOCAL_FALLBACK_DF = None


def read_submission_or_fallback(path):
    """
    Try reading a submission CSV from `path`. If not found, use local-model fallback.
    Always returns columns: id,target with float target aligned to sample_submission ids.

    --- Timeout fix: compute the fallback model at most once total and return already-aligned output.
    """
    global _LOCAL_FALLBACK_DF
    ref_ids = sample_sub["id"].tolist()

    if path is not None and os.path.exists(path):
        df = pd.read_csv(path)
        if not (set(df.columns) >= {"id", "target"}):
            raise ValueError(
                f"Submission at {path} must contain columns id,target. Got {df.columns.tolist()}"
            )
        df = df[["id", "target"]].copy()
        df["target"] = df["target"].astype(float)
        return align_on_id(df, ref_ids)

    if _LOCAL_FALLBACK_DF is None:
        _LOCAL_FALLBACK_DF = align_on_id(_local_model_predictions(), ref_ids)
    return _LOCAL_FALLBACK_DF.copy()




## === cell 1
data1 = read_submission_or_fallback(
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv"
)
data2 = read_submission_or_fallback(
    "../input/seti-bl-spatial-info-tf-tpu/submission.csv"
)
data3 = read_submission_or_fallback("../input/seti-bl-tf-starter-tpu/submission.csv")
data4 = read_submission_or_fallback(
    "../input/seti-learned-image-resizing/submission.csv"
)
data5 = read_submission_or_fallback(
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv"
)
data6 = read_submission_or_fallback(
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv"
)



## === cell 2
ref_ids = sample_sub["id"].tolist()



## === cell 3
blend = data1.copy()
blend["target"] = (
    0.795 * data5["target"]
    + 0.1025 * data4["target"]
    + 0.1025 * data6["target"]
    + 0.0 * data2["target"]
    + 0.0 * data3["target"]
)
blend["target"] = blend["target"].clip(0.0, 1.0)



## === cell 4
submission = blend[["id", "target"]].copy()
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
