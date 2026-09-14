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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

INPUT_ROOTS = [
    "/kaggle/input",  # standard Kaggle path
    "../input",  # relative path sometimes used in notebooks
    "/kaggle/data",
]

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


def _read_sample_submission():
    for p in SAMPLE_SUB_PATHS:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if {"id", "target"}.issubset(df.columns):
                df = df[["id", "target"]].copy()
                df["id"] = df["id"].astype(str)
                return df
    for root in INPUT_ROOTS:
        if not os.path.exists(root):
            continue
        for p in glob.glob(
            os.path.join(root, "**", "sample_submission.csv"), recursive=True
        ):
            df = pd.read_csv(p)
            if {"id", "target"}.issubset(df.columns):
                df = df[["id", "target"]].copy()
                df["id"] = df["id"].astype(str)
                return df
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected locations."
    )


def _read_train_labels():
    for p in TRAIN_LABELS_PATHS:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if {"id", "target"}.issubset(df.columns):
                df = df[["id", "target"]].copy()
                df["id"] = df["id"].astype(str)
                df["target"] = df["target"].astype(int)
                return df
    for root in INPUT_ROOTS:
        if not os.path.exists(root):
            continue
        for p in glob.glob(
            os.path.join(root, "**", "train_labels.csv"), recursive=True
        ):
            df = pd.read_csv(p)
            if {"id", "target"}.issubset(df.columns):
                df = df[["id", "target"]].copy()
                df["id"] = df["id"].astype(str)
                df["target"] = df["target"].astype(int)
                return df
    raise FileNotFoundError("Could not locate train_labels.csv in expected locations.")


def _safe_read_submission_csv(path):
    df = pd.read_csv(path)
    if "id" not in df.columns or "target" not in df.columns:
        raise ValueError(
            f"Submission at {path} must contain columns ['id','target']. Got: {df.columns.tolist()}"
        )
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    return df


def _find_candidate_submissions_fast(max_candidates=50):
    candidates = []
    patterns = ("submission.csv",)

    for root in INPUT_ROOTS:
        if not os.path.isdir(root):
            continue
        try:
            with os.scandir(root) as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    for pat in patterns:
                        p = os.path.join(entry.path, pat)
                        if (
                            os.path.isfile(p)
                            and os.path.basename(p) != "sample_submission.csv"
                        ):
                            candidates.append(p)
                            if len(candidates) >= max_candidates:
                                return candidates
                    try:
                        with os.scandir(entry.path) as it2:
                            for sub in it2:
                                if not sub.is_dir():
                                    continue
                                for pat in patterns:
                                    p = os.path.join(sub.path, pat)
                                    if (
                                        os.path.isfile(p)
                                        and os.path.basename(p)
                                        != "sample_submission.csv"
                                    ):
                                        candidates.append(p)
                                        if len(candidates) >= max_candidates:
                                            return candidates
                    except OSError:
                        continue
        except OSError:
            continue

    seen, uniq = set(), []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _pick_existing_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    for root in INPUT_ROOTS:
        if not os.path.exists(root):
            continue
        for d in candidates:
            base = os.path.basename(d.rstrip("/"))
            for found in glob.glob(os.path.join(root, "**", base), recursive=True):
                if os.path.isdir(found):
                    return found
    return None


sample_sub = _read_sample_submission()
print("sample_submission shape:", sample_sub.shape)



## === cell 1
preferred_paths = [
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    "../input/seti-bl-tf-starter-tpu/submission.csv",
    "../input/seti-learned-image-resizing/submission.csv",
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
]

available = [p for p in preferred_paths if os.path.exists(p)]
if len(available) == 0:
    discovered = _find_candidate_submissions_fast(max_candidates=80)
    available = discovered

subs = []
sub_paths_used = []
MAX_USABLE = 6
for p in available:
    if len(subs) >= MAX_USABLE:
        break
    try:
        df = _safe_read_submission_csv(p)
        df = sample_sub[["id"]].merge(df, on="id", how="left")
        if df["target"].isna().mean() > 0.5:
            continue
        df["target"] = df["target"].fillna(0.5)
        subs.append(df)
        sub_paths_used.append(p)
    except Exception:
        continue

print(f"Found {len(subs)} usable submission file(s).")
for p in sub_paths_used[:10]:
    print(" -", p)

data1 = subs[0] if len(subs) > 0 else None
data2 = subs[1] if len(subs) > 1 else None
data3 = subs[2] if len(subs) > 2 else None
data4 = subs[3] if len(subs) > 3 else None
data5 = subs[4] if len(subs) > 4 else None
data6 = subs[5] if len(subs) > 5 else None



## === cell 2
from concurrent.futures import ThreadPoolExecutor


def _fast_percentile(a, q):
    """
    Exact equivalent of np.percentile(a, q) with default method='linear' for q in [0,100],
    implemented via np.partition to avoid full sort.
    """
    a = np.asarray(a, dtype=np.float32)
    n = a.size
    if n == 0:
        return np.float32(np.nan)

    r = (q / 100.0) * (n - 1)
    lo = int(np.floor(r))
    hi = int(np.ceil(r))
    flat = a.reshape(-1)

    if lo == hi:
        v = np.partition(flat, lo)[lo]
        return np.float32(v)

    part = np.partition(flat, (lo, hi))
    vlo = part[lo]
    vhi = part[hi]
    v = vlo + (r - lo) * (vhi - vlo)
    return np.float32(v)


def _extract_features_from_array(x):
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(dtype=np.float32)
    sig = x.std(dtype=np.float32) + np.float32(1e-6)
    x = (x - mu) / sig

    A = x[[0, 2, 4]]
    O = x[[1, 3, 5]]

    mean_all = x.mean(dtype=np.float32)
    std_all = x.std(dtype=np.float32) + 1e-6

    mean_A = A.mean(dtype=np.float32)
    mean_O = O.mean(dtype=np.float32)
    std_A = A.std(dtype=np.float32) + 1e-6
    std_O = O.std(dtype=np.float32) + 1e-6

    p95_A = _fast_percentile(A, 95)
    p95_O = _fast_percentile(O, 95)
    p99_A = _fast_percentile(A, 99)
    p99_O = _fast_percentile(O, 99)

    A_time_mean = A.mean(axis=2, dtype=np.float32)  # (3, 273)
    O_time_mean = O.mean(axis=2, dtype=np.float32)
    A_freq_mean = A.mean(axis=1, dtype=np.float32)  # (3, 256)
    O_freq_mean = O.mean(axis=1, dtype=np.float32)

    v_time_A = A_time_mean.var(dtype=np.float32)
    v_time_O = O_time_mean.var(dtype=np.float32)
    v_freq_A = A_freq_mean.var(dtype=np.float32)
    v_freq_O = O_freq_mean.var(dtype=np.float32)

    feats = np.array(
        [
            mean_all,
            std_all,
            mean_A - mean_O,
            std_A - std_O,
            p95_A - p95_O,
            p99_A - p99_O,
            v_time_A - v_time_O,
            v_freq_A - v_freq_O,
        ],
        dtype=np.float32,
    )
    return feats


def _iter_npy_ids(root_dir):
    out = []
    shard_dirs = []
    try:
        with os.scandir(root_dir) as it:
            for e in it:
                if e.is_dir():
                    shard_dirs.append(e.path)
    except OSError:
        shard_dirs = []

    if shard_dirs:
        for sd in shard_dirs:
            try:
                with os.scandir(sd) as it:
                    for e in it:
                        if e.is_file() and e.name.endswith(".npy"):
                            out.append((e.name[:-4], e.path))
            except OSError:
                continue
    else:
        try:
            with os.scandir(root_dir) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".npy"):
                        out.append((e.name[:-4], e.path))
        except OSError:
            pass
    return out


def _featurize_one(path):
    arr = np.load(path, mmap_mode="r")
    return _extract_features_from_array(arr)


def _build_features_for_ids(id_list, id_to_path, max_workers=None, chunksize=256):
    n = len(id_list)
    X = np.zeros((n, 8), dtype=np.float32)
    if n == 0:
        return X
    paths = [id_to_path[_id] for _id in id_list]

    if n < 256:
        for i, p in enumerate(paths):
            X[i] = _featurize_one(p)
        return X

    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 2))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in enumerate(ex.map(_featurize_one, paths, chunksize=chunksize)):
            X[i] = feats
    return X


def _make_fallback_lr_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=300,
                    n_jobs=1,
                    class_weight=None,
                    C=0.5,
                    random_state=42,
                ),
            ),
        ]
    )


def _train_and_predict_fallback(sample_sub_df):
    train_labels = _read_train_labels()

    train_dir = _pick_existing_dir(TRAIN_DIR_CANDIDATES)
    test_dir = _pick_existing_dir(TEST_DIR_CANDIDATES)
    if train_dir is None or test_dir is None:
        out = sample_sub_df.copy()
        out["target"] = 0.5
        return out

    train_pairs = _iter_npy_ids(train_dir)
    test_pairs = _iter_npy_ids(test_dir)

    train_id_to_path = {i: p for i, p in train_pairs}
    test_id_to_path = {i: p for i, p in test_pairs}
    train_ids_available = set(train_id_to_path.keys())
    test_ids_available = set(test_id_to_path.keys())

    train_df = train_labels[train_labels["id"].isin(train_ids_available)].copy()
    train_ids = train_df["id"].tolist()
    y = train_df["target"].to_numpy(dtype=int)

    test_ids = sample_sub_df["id"].astype(str).tolist()

    X_train = _build_features_for_ids(train_ids, train_id_to_path)

    clf = _make_fallback_lr_pipeline()
    clf.fit(X_train, y)

    preds = np.full(len(test_ids), 0.5, dtype=np.float32)

    have_set = test_ids_available.intersection(test_ids)
    if len(have_set) > 0:
        have_mask = np.fromiter(
            (tid in have_set for tid in test_ids), count=len(test_ids), dtype=bool
        )
        have_ids = [tid for tid in test_ids if tid in have_set]
        X_test = _build_features_for_ids(have_ids, test_id_to_path)
        proba = clf.predict_proba(X_test)[:, 1].astype(np.float32)
        preds[have_mask] = proba

    out = sample_sub_df.copy()
    out["target"] = np.clip(preds, 0.0, 1.0)
    return out


data_fallback = _train_and_predict_fallback(sample_sub)

data_fallback = sample_sub[["id"]].merge(
    data_fallback[["id", "target"]], on="id", how="left"
)
data_fallback["target"] = pd.to_numeric(
    data_fallback["target"], errors="coerce"
).fillna(0.5)

if data1 is None:
    data1 = data_fallback.copy()
else:
    data1 = sample_sub[["id"]].merge(data1[["id", "target"]], on="id", how="left")
    data1["target"] = pd.to_numeric(data1["target"], errors="coerce").fillna(0.5)

data1.head()



## === cell 3
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _submission_to_train_aligned_preds(sub_df, train_labels_df):
    tmp = train_labels_df[["id"]].merge(sub_df[["id", "target"]], on="id", how="left")
    p = (
        pd.to_numeric(tmp["target"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32, copy=False)
    )
    return p


def _oof_preds_for_fallback_lr(train_labels_df, n_splits=5, random_state=42):
    train_dir = _pick_existing_dir(TRAIN_DIR_CANDIDATES)
    if train_dir is None:
        return np.full(len(train_labels_df), 0.5, dtype=np.float32)

    train_pairs = _iter_npy_ids(train_dir)
    train_id_to_path = {i: p for i, p in train_pairs}
    train_ids_available = set(train_id_to_path.keys())

    df = train_labels_df[train_labels_df["id"].isin(train_ids_available)].copy()
    df = df.reset_index(drop=True)

    oof_small = np.full(len(df), 0.5, dtype=np.float32)

    ids = df["id"].tolist()
    y = df["target"].to_numpy(dtype=int, copy=False)

    X_all = _build_features_for_ids(ids, train_id_to_path)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in skf.split(X_all, y):
        clf = _make_fallback_lr_pipeline()
        clf.fit(X_all[tr_idx], y[tr_idx])
        oof_small[va_idx] = clf.predict_proba(X_all[va_idx])[:, 1].astype(np.float32)

    tmp = train_labels_df[["id"]].copy()
    tmp["pred"] = 0.5
    tmp2 = df[["id"]].copy()
    tmp2["pred"] = oof_small
    tmp = tmp.merge(tmp2, on="id", how="left", suffixes=("", "_y"))
    pred = tmp["pred_y"].fillna(0.5).to_numpy(dtype=np.float32, copy=False)
    return np.clip(pred, 0.0, 1.0)


def _fit_logit_calibrator(y, p, C=1.0):
    p = np.asarray(p, dtype=np.float32)
    eps = np.float32(1e-6)
    p = np.clip(p, eps, 1.0 - eps)
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    cal = LogisticRegression(solver="lbfgs", max_iter=200, C=C, random_state=42)
    cal.fit(z, y)
    return cal


def _apply_logit_calibrator(cal, p):
    p = np.asarray(p, dtype=np.float32)
    eps = np.float32(1e-6)
    p = np.clip(p, eps, 1.0 - eps)
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    return cal.predict_proba(z)[:, 1].astype(np.float32)


def _learn_convex_blend_weights_from_train_preds(
    y, P, names, min_w=0.0, grid_step=0.05
):
    P = np.asarray(P, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    m = P.shape[0]
    if m == 0:
        return {}
    if m == 1:
        return {names[0]: 1.0}

    w = np.full(m, 1.0 / m, dtype=np.float32)

    def auc_for(wvec):
        pred = np.clip(np.dot(wvec, P), 0.0, 1.0)
        return roc_auc_score(y, pred)

    best_auc = auc_for(w)

    grid = np.arange(min_w, 1.0 + 1e-9, grid_step, dtype=np.float32)
    for _ in range(4):
        improved = False
        for j in range(m):
            cur_best_w = w.copy()
            cur_best_auc = best_auc

            for cand in grid:
                w2 = w.copy()
                rem = 1.0 - float(cand)
                if rem < 0:
                    continue
                others = [k for k in range(m) if k != j]
                if len(others) == 0:
                    continue
                sum_others = float(w2[others].sum())
                if sum_others <= 0:
                    w2[others] = rem / len(others)
                else:
                    w2[others] = w2[others] / sum_others * rem
                w2[j] = float(cand)

                a = auc_for(w2)
                if a > cur_best_auc + 1e-6:
                    cur_best_auc = a
                    cur_best_w = w2

            if cur_best_auc > best_auc + 1e-6:
                w = cur_best_w
                best_auc = cur_best_auc
                improved = True
        if not improved:
            break

    return {n: float(wi) for n, wi in zip(names, w)}


def _maybe_load_train_predictions_for_external_submission(submission_path):
    """
    Only use external sources for weight learning if they provide train-id predictions.
    """
    base_dir = os.path.dirname(submission_path)
    candidates = [
        os.path.join(base_dir, "train_predictions.csv"),
        os.path.join(base_dir, "train_pred.csv"),
        os.path.join(base_dir, "train.csv"),
        os.path.join(base_dir, "oof.csv"),
        os.path.join(base_dir, "oof_predictions.csv"),
        os.path.join(base_dir, "submission_train.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = _safe_read_submission_csv(p)
                return df
            except Exception:
                continue

    try:
        for p in glob.glob(os.path.join(base_dir, "*.csv")):
            name = os.path.basename(p).lower()
            if ("oof" in name or "train" in name) and "submission" not in name:
                try:
                    df = _safe_read_submission_csv(p)
                    return df
                except Exception:
                    continue
    except Exception:
        pass

    return None


def _oof_calibrated_preds(y, p, n_splits=5, random_state=42, C=1.0):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=np.float32)
    if float(np.std(p)) < 1e-5:
        return np.clip(p, 0.0, 1.0), None  # nothing to calibrate stably

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    p_cal_oof = np.empty_like(p, dtype=np.float32)
    for tr_idx, va_idx in skf.split(p.reshape(-1, 1), y):
        cal_fold = _fit_logit_calibrator(y[tr_idx], p[tr_idx], C=C)
        p_cal_oof[va_idx] = _apply_logit_calibrator(cal_fold, p[va_idx])

    cal_full = _fit_logit_calibrator(y, p, C=C)
    return np.clip(p_cal_oof, 0.0, 1.0), cal_full


train_labels_for_weighting = _read_train_labels()
y_train_full = train_labels_for_weighting["target"].to_numpy(dtype=int, copy=False)

blend_candidates = [("fallback_lr", data_fallback.copy())]

for nm, df in [
    ("sub1", data1),
    ("sub2", data2),
    ("sub3", data3),
    ("sub4", data4),
    ("sub5", data5),
    ("sub6", data6),
]:
    if df is None:
        continue
    df2 = sample_sub[["id"]].merge(df[["id", "target"]], on="id", how="left")
    df2["target"] = pd.to_numeric(df2["target"], errors="coerce").fillna(0.5)
    blend_candidates.append((nm, df2))

fallback_oof = _oof_preds_for_fallback_lr(
    train_labels_for_weighting, n_splits=5, random_state=42
)

P_list = [fallback_oof]
names = ["fallback_lr"]

name_to_path = {}
for i, p in enumerate(sub_paths_used):
    name_to_path[f"sub{i+1}"] = p

for nm, _df_test_aligned in blend_candidates[1:]:
    sub_path = name_to_path.get(nm)
    if not sub_path:
        continue
    df_train_like = _maybe_load_train_predictions_for_external_submission(sub_path)
    if df_train_like is None:
        continue

    p_tr = _submission_to_train_aligned_preds(df_train_like, train_labels_for_weighting)
    frac_not_05 = float(np.mean(np.abs(p_tr - 0.5) > 1e-6))
    std_p = float(np.std(p_tr))
    if frac_not_05 < 0.01 or std_p < 1e-4:
        continue

    P_list.append(p_tr)
    names.append(nm)

P = np.vstack(P_list).astype(np.float32)

weights_by_name = _learn_convex_blend_weights_from_train_preds(
    y_train_full, P, names, min_w=0.0, grid_step=0.05
)
if not weights_by_name:
    weights_by_name = {"fallback_lr": 1.0}

print("Weight-learning sources used:", names)
print("Learned blend weights (OOF / train-preds):", weights_by_name)

wvec = np.array([weights_by_name.get(n, 0.0) for n in names], dtype=np.float32)
if float(wvec.sum()) <= 0:
    wvec = np.zeros_like(wvec)
    wvec[0] = 1.0
else:
    wvec = wvec / np.float32(wvec.sum())

oof_blend = np.clip(np.dot(wvec, P), 0.0, 1.0)

oof_blend_cal, cal = _oof_calibrated_preds(
    y_train_full, oof_blend, n_splits=5, random_state=42, C=1.0
)

print("OOF AUC before calib:", roc_auc_score(y_train_full, oof_blend))
print(
    "OOF AUC after  calib (leak-free OOF):", roc_auc_score(y_train_full, oof_blend_cal)
)

name_to_testdf = {nm: df for nm, df in blend_candidates}
blend_test = np.zeros(len(sample_sub), dtype=np.float32)
wsum = 0.0
for nm in names:
    w = float(weights_by_name.get(nm, 0.0))
    if w <= 0:
        continue
    df = name_to_testdf.get(nm)
    if df is None:
        continue
    blend_test += w * df["target"].to_numpy(dtype=np.float32, copy=False)
    wsum += w

if wsum <= 0:
    blend_test = name_to_testdf["fallback_lr"]["target"].to_numpy(
        dtype=np.float32, copy=False
    )
else:
    blend_test = blend_test / np.float32(wsum)

blend_test = np.clip(blend_test, 0.0, 1.0)
if cal is not None:
    blend_test = _apply_logit_calibrator(cal, blend_test)

data6 = sample_sub.copy()
data6["target"] = np.clip(blend_test, 0.0, 1.0)
data6.head()



## === cell 4
data6[["id", "target"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print(data6.head())
print(
    "Target stats: min/mean/max =",
    float(data6["target"].min()),
    float(data6["target"].mean()),
    float(data6["target"].max()),
)
