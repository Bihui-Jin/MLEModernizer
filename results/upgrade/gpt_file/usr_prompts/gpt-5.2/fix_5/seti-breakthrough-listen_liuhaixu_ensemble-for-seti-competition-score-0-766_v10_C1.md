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
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)

INPUT_BASES = [
    "../input",  # typical Kaggle notebooks
    "/kaggle/input",  # some environments
    "/kaggle/data/input",  # as provided in this environment tree
]


def _find_existing_path(rel_path: str):
    for base in INPUT_BASES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None




## === cell 1
from sklearn.linear_model import LogisticRegression


def _first_existing_dir(rel_dir: str):
    for base in INPUT_BASES:
        p = os.path.join(base, rel_dir)
        if os.path.isdir(p):
            return p
    return None


def _collect_npy_paths(root_dir: str):
    out = []
    shard_dirs = [os.path.join(root_dir, str(i)) for i in range(16)]
    if all(os.path.isdir(d) for d in shard_dirs):
        for d in shard_dirs:
            try:
                for fn in os.listdir(d):
                    if fn.endswith(".npy"):
                        out.append(os.path.join(d, fn))
            except FileNotFoundError:
                continue
    else:
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.endswith(".npy"):
                    out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def _p99_equivalent(x_flat: np.ndarray) -> np.float32:
    n = x_flat.size
    if n == 0:
        return np.float32(np.nan)
    q = 0.99
    i = (n - 1) * q
    lo = int(np.floor(i))
    hi = int(np.ceil(i))
    if lo == hi:
        return np.float32(np.partition(x_flat, lo)[lo])
    part = np.partition(x_flat, (lo, hi))
    x_lo = part[lo]
    x_hi = part[hi]
    w = np.float32(i - lo)
    return np.float32((1.0 - w) * x_lo + w * x_hi)


def _extract_features_from_npy(path: str) -> np.ndarray:
    x = np.load(path)  # (6, 273, 256), float16
    x = x.astype(np.float32, copy=False)

    mu = x.mean()
    sd = x.std()

    on = x[[0, 2, 4]].mean()
    off = x[[1, 3, 5]].mean()
    contrast = on - off

    mx = x.max()

    x_flat = x.ravel()
    p99 = _p99_equivalent(x_flat)

    t_var = x.mean(axis=2).var()  # variability across time bins (over all panels)
    f_var = x.mean(axis=1).var()  # variability across frequency bins (over all panels)

    return np.array([mu, sd, contrast, mx, p99, t_var, f_var], dtype=np.float32)


def _extract_feature_matrix(
    paths, id_to_y=None, n_workers: int = 0, chunksize: int = 32
):
    if id_to_y is None:
        feats = np.empty((len(paths), 7), dtype=np.float32)
        if n_workers and n_workers > 1:
            import multiprocessing as mp

            with mp.get_context("fork").Pool(processes=n_workers) as pool:
                for i, f in enumerate(
                    pool.imap(_extract_features_from_npy, paths, chunksize=chunksize)
                ):
                    feats[i] = f
        else:
            for i, p in enumerate(paths):
                feats[i] = _extract_features_from_npy(p)
        return feats, None

    feats = np.empty((len(paths), 7), dtype=np.float32)
    ys = np.empty((len(paths),), dtype=np.int32)
    keep_paths = []
    for p in paths:
        _id = os.path.splitext(os.path.basename(p))[0]
        y = id_to_y.get(_id)
        if y is None:
            continue
        ys[len(keep_paths)] = int(y)
        keep_paths.append(p)

    keep = len(keep_paths)
    ys = ys[:keep]
    feats = feats[:keep]

    if keep == 0:
        return feats, ys

    if n_workers and n_workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(processes=n_workers) as pool:
            for i, f in enumerate(
                pool.imap(_extract_features_from_npy, keep_paths, chunksize=chunksize)
            ):
                feats[i] = f
    else:
        for i, p in enumerate(keep_paths):
            feats[i] = _extract_features_from_npy(p)

    return feats, ys


_HEX2INT = {c: i for i, c in enumerate("0123456789abcdef")}


def _resolve_id_to_npy_path(root_dir: str, _id: str) -> str:
    if not _id:
        return None
    c = _id[0].lower()
    if c in _HEX2INT:
        p = os.path.join(root_dir, str(_HEX2INT[c]), f"{_id}.npy")
        if os.path.exists(p):
            return p
    p2 = os.path.join(root_dir, f"{_id}.npy")
    if os.path.exists(p2):
        return p2
    return None


def build_fallback_predictions(sample_sub: pd.DataFrame) -> pd.DataFrame:
    train_labels_path = _find_existing_path("train_labels.csv") or _find_existing_path(
        "seti-breakthrough-listen/train_labels.csv"
    )
    train_dir = _first_existing_dir("train") or _first_existing_dir(
        "seti-breakthrough-listen/train"
    )
    test_dir = _first_existing_dir("test") or _first_existing_dir(
        "seti-breakthrough-listen/test"
    )
    if train_labels_path is None or train_dir is None or test_dir is None:
        df = sample_sub.copy()
        df["target"] = 0.5
        return df

    labels = pd.read_csv(train_labels_path, usecols=["id", "target"]).copy()
    label_map = dict(zip(labels["id"].astype(str), labels["target"].astype(int)))

    train_paths = _collect_npy_paths(train_dir)

    max_train = 12000  # unchanged
    if len(train_paths) > max_train:
        idx = np.linspace(0, len(train_paths) - 1, max_train, dtype=int)
        train_paths = [train_paths[i] for i in idx]

    cpu_cnt = os.cpu_count() or 2
    n_workers = max(1, min(8, cpu_cnt))  # cap to avoid oversubscription in Kaggle
    X_train, y_train = _extract_feature_matrix(
        train_paths, id_to_y=label_map, n_workers=n_workers, chunksize=32
    )

    if X_train.shape[0] < 100:
        df = sample_sub.copy()
        df["target"] = 0.5
        return df

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=300,
        n_jobs=1,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(X_train, y_train)

    ids_list = sample_sub["id"].astype(str).tolist()
    test_paths_for_ids = []
    missing_mask = np.zeros(len(ids_list), dtype=bool)
    for i, _id in enumerate(ids_list):
        p = _resolve_id_to_npy_path(test_dir, _id)
        if p is None:
            missing_mask[i] = True
            test_paths_for_ids.append(None)
        else:
            test_paths_for_ids.append(p)

    preds = np.full((len(ids_list),), 0.5, dtype=np.float64)

    exist_indices = [i for i, p in enumerate(test_paths_for_ids) if p is not None]
    exist_paths = [test_paths_for_ids[i] for i in exist_indices]
    if exist_paths:
        X_test, _ = _extract_feature_matrix(
            exist_paths, id_to_y=None, n_workers=n_workers, chunksize=32
        )
        proba = clf.predict_proba(X_test)[:, 1].astype(np.float64, copy=False)
        preds[np.array(exist_indices, dtype=int)] = proba

    df = sample_sub.copy()
    df["target"] = preds
    df["target"] = (
        df["target"].replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)
    )
    return df


def load_submission_or_fallback(
    rel_path: str, fallback_df: pd.DataFrame
) -> pd.DataFrame:
    p = _find_existing_path(rel_path)
    if p is None:
        return fallback_df.copy()
    df = pd.read_csv(p)
    if "id" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Submission file missing required columns id/target: {p}")
    df = df[["id", "target"]].copy()
    return df




## === cell 2
sample_path = _find_existing_path("sample_submission.csv") or _find_existing_path(
    "seti-breakthrough-listen/sample_submission.csv"
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under known input paths."
    )

sample_sub = pd.read_csv(sample_path)[["id", "target"]].copy()

fallback = build_fallback_predictions(sample_sub)

data1 = load_submission_or_fallback(
    "rerun-seti-e-t-volo-d1-baseline-inference/submission.csv", fallback
)
data2 = load_submission_or_fallback(
    "lb-0-980-efficientnet-b0-more-epoch/submission.csv", fallback
)
data3 = load_submission_or_fallback(
    "inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv", fallback
)
data4 = load_submission_or_fallback(
    "seti-learned-image-resizing/submission.csv", fallback
)
data5 = load_submission_or_fallback(
    "rerun-seti-e-t-resnet18d-baseline/submission.csv", fallback
)
data6 = load_submission_or_fallback(
    "ensemble-for-seti-competition/submission.csv", fallback
)
data7 = load_submission_or_fallback(
    "fixed-gradual-warmup-custom-head/submission.csv", fallback
)


def align_to_ids(df: pd.DataFrame, ids: pd.Series) -> pd.Series:
    s = df.set_index("id").reindex(ids)["target"]
    return s.fillna(0.5).astype(float)


ids = sample_sub["id"]
t1 = align_to_ids(data1, ids)
t2 = align_to_ids(data2, ids)
t3 = align_to_ids(data3, ids)
t4 = align_to_ids(data4, ids)
t5 = align_to_ids(data5, ids)
t6 = align_to_ids(data6, ids)
t7 = align_to_ids(data7, ids)




## === cell 3
data11 = sample_sub.copy()




## === cell 4
data11["target"] = 0.12 * t1 + 0.10 * t2 + 0.10 * t3 + 0.10 * t4 + 0.12 * t5 + 0.60 * t6

data11["target"] = data11["target"].replace([np.inf, -np.inf], np.nan).fillna(0.5)
data11["target"] = data11["target"].clip(0.0, 1.0)




## === cell 5
data11.to_csv("submission.csv", index=False)
print(data11.head())
print(
    f"Wrote submission.csv with shape {data11.shape} and columns {list(data11.columns)}"
)
