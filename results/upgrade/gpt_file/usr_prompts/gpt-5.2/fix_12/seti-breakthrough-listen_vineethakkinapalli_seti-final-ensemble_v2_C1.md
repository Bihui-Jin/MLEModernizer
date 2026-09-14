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



## === cell 1
BASE_INPUT = "/kaggle/input"
BASE_DATA = "/kaggle/data"


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


sample_path = _first_existing(
    [
        os.path.join(BASE_INPUT, "sample_submission.csv"),
        os.path.join(BASE_DATA, "sample_submission.csv"),
        os.path.join(BASE_INPUT, "seti-breakthrough-listen", "sample_submission.csv"),
        os.path.join(BASE_DATA, "seti-breakthrough-listen", "sample_submission.csv"),
    ]
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations under /kaggle/input or /kaggle/data"
    )

sample = pd.read_csv(sample_path)
if list(sample.columns)[:2] != ["id", "target"]:
    sample = sample[["id", "target"]].copy()


def load_submission_csv(path):
    """Load a submission file and align it to sample ids. Returns a DataFrame with columns [id, target]."""
    df = pd.read_csv(path)
    if "id" not in df.columns or "target" not in df.columns:
        raise ValueError(
            f"Submission at {path} must contain columns ['id','target']. Found: {df.columns.tolist()}"
        )
    df = df[["id", "target"]].copy()
    df = sample[["id"]].merge(df, on="id", how="left")
    df["target"] = df["target"].astype("float64")
    df["target"] = df["target"].fillna(0.5)
    return df


def find_candidate_submissions(search_root):
    """Find plausible submission-like csvs under a root folder."""
    out = []
    if search_root is None or not os.path.exists(search_root):
        return out
    for root, _, files in os.walk(search_root):
        for fn in files:
            lfn = fn.lower()
            if lfn.endswith(".csv") and ("submission" in lfn or "sub" in lfn):
                out.append(os.path.join(root, fn))
    return out


intended_paths = {
    "data1": "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "data2": "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "data3": "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
    "data4": "../input/seti-learned-image-resizing/submission.csv",
    "data5": "../input/rerun-seti-e-t-resnet18d-baseline/submission.csv",
    "data6": "../input/ensemble-for-seti-competition/submission.csv",
    "data7": "../input/fixed-gradual-warmup-custom-head/submission.csv",
    "data8": "../input/seti-results/learned_resizing_384.csv",
}

loaded = {}
for k, p in intended_paths.items():
    if os.path.exists(p):
        loaded[k] = load_submission_csv(p)

if "data6" not in loaded or "data8" not in loaded:
    candidate_paths = []
    candidate_paths.extend(find_candidate_submissions(BASE_INPUT))
    candidate_paths.extend(find_candidate_submissions(BASE_DATA))
    candidate_paths = sorted(set(candidate_paths))

    def choose_candidate(exclude_paths=()):
        for cp in candidate_paths:
            bn = os.path.basename(cp).lower()
            if bn == "sample_submission.csv":
                continue
            if cp in exclude_paths:
                continue
            return cp
        return None

    if "data6" not in loaded:
        p6 = choose_candidate()
        if p6 is not None:
            loaded["data6"] = load_submission_csv(p6)

    if "data8" not in loaded:
        p8 = choose_candidate()
        if p8 is not None:
            loaded["data8"] = load_submission_csv(p8)

if "data6" not in loaded:
    loaded["data6"] = sample.copy()
    loaded["data6"]["target"] = 0.5
if "data8" not in loaded:
    loaded["data8"] = sample.copy()
    loaded["data8"]["target"] = 0.5

data1 = loaded.get("data1", None)
data2 = loaded.get("data2", None)
data3 = loaded.get("data3", None)
data4 = loaded.get("data4", None)
data5 = loaded.get("data5", None)
data6 = loaded["data6"]
data7 = loaded.get("data7", None)
data8 = loaded["data8"]



## === cell 2
data11 = sample.copy()



## === cell 3
data11 = data11[["id"]].merge(
    data6[["id", "target"]], on="id", how="left", suffixes=("", "_6")
)
data11 = data11.merge(data8[["id", "target"]], on="id", how="left", suffixes=("", "_8"))

t6 = data11["target"].astype("float64").fillna(0.5)
t8 = data11["target_8"].astype("float64").fillna(0.5)

data11["target"] = 0.95 * t6 + 0.05 * t8
data11["target"] = data11["target"].clip(0.0, 1.0)
data11 = data11[["id", "target"]]




## === cell 4
def _resolve_comp_root():
    candidates = [
        os.path.join(BASE_INPUT, "seti-breakthrough-listen"),
        os.path.join(BASE_DATA, "seti-breakthrough-listen"),
        BASE_INPUT,
        BASE_DATA,
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train_labels.csv")) and (
            os.path.exists(os.path.join(c, "train"))
            and os.path.exists(os.path.join(c, "test"))
        ):
            return c
    for c in candidates:
        if os.path.exists(os.path.join(c, "train_labels.csv")) and (
            os.path.exists(os.path.join(c, "train"))
            or os.path.exists(os.path.join(c, "seti-breakthrough-listen", "train"))
        ):
            return c
    return None


def _baseline_is_constant_half(df):
    v = df["target"].to_numpy(dtype=np.float64)
    return np.all(np.isfinite(v)) and np.max(np.abs(v - 0.5)) < 1e-12


def _list_shards(folder):
    shards = []
    try:
        for name in os.listdir(folder):
            p = os.path.join(folder, name)
            if os.path.isdir(p) and name.isdigit():
                shards.append(int(name))
    except FileNotFoundError:
        return []
    shards = sorted(shards)
    return shards


def _find_npy_by_id(folder, id_, shards_cache=None):
    if shards_cache is None:
        shards_cache = list(range(16))
    for shard in shards_cache:
        p = os.path.join(folder, str(shard), f"{id_}.npy")
        if os.path.exists(p):
            return p
    p = os.path.join(folder, f"{id_}.npy")
    if os.path.exists(p):
        return p
    return None


def _extract_features(arr):
    """
    Keep the same engineered-feature approach; this function is unchanged to preserve core semantics.
    """
    x = arr.astype(np.float32)  # (6, 273, 256)
    a = x[[0, 2, 4]]
    b = x[[1, 3, 5]]

    a_mean = float(a.mean())
    b_mean = float(b.mean())
    a_std = float(a.std())
    b_std = float(b.std())
    a_p99 = float(np.quantile(a, 0.99))
    b_p99 = float(np.quantile(b, 0.99))

    diff = a.mean(axis=0) - b.mean(axis=0)  # (273, 256)
    absdiff = np.abs(diff)

    diff_mean = float(diff.mean())
    diff_std = float(diff.std())
    diff_p99 = float(np.quantile(absdiff, 0.99))

    t_profile = absdiff.mean(axis=1)  # (273,)
    f_profile = absdiff.mean(axis=0)  # (256,)
    diff_tmax = float(t_profile.max())
    diff_fmax = float(f_profile.max())

    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    a_intra = float(
        (np.mean(np.abs(a0 - a1)) + np.mean(np.abs(a0 - a2)) + np.mean(np.abs(a1 - a2)))
        / 3.0
    )
    b_intra = float(
        (np.mean(np.abs(b0 - b1)) + np.mean(np.abs(b0 - b2)) + np.mean(np.abs(b1 - b2)))
        / 3.0
    )

    a_p999 = float(np.quantile(a, 0.999))
    b_p999 = float(np.quantile(b, 0.999))

    thr = float(np.quantile(absdiff, 0.995))
    strong_frac = float((absdiff >= thr).mean())

    a_med = np.median(a)
    b_med = np.median(b)
    a_l1 = float(np.mean(np.abs(a - a_med)))
    b_l1 = float(np.mean(np.abs(b - b_med)))
    a_over_b_l1 = float(a_l1 / (b_l1 + 1e-6))

    def _corr(u, v):
        u = u.ravel()
        v = v.ravel()
        u = u - u.mean()
        v = v - v.mean()
        denom = np.sqrt((u * u).mean()) * np.sqrt((v * v).mean()) + 1e-6
        return float((u * v).mean() / denom)

    a_corr = float((_corr(a0, a1) + _corr(a0, a2) + _corr(a1, a2)) / 3.0)
    b_corr = float((_corr(b0, b1) + _corr(b0, b2) + _corr(b1, b2)) / 3.0)

    argmax_f = np.argmax(absdiff, axis=1).astype(np.float32)  # (273,)
    t_idx = np.arange(argmax_f.shape[0], dtype=np.float32)
    t0 = t_idx - t_idx.mean()
    f0 = argmax_f - argmax_f.mean()
    slope = float((t0 * f0).sum() / (t0 * t0).sum())
    drift_abs = float(abs(slope))

    peak_over_mean = float(absdiff.max() / (absdiff.mean() + 1e-6))

    t_line_strength = float(t_profile.max() / (t_profile.mean() + 1e-6))
    f_line_strength = float(f_profile.max() / (f_profile.mean() + 1e-6))

    mean_t_of_fmax = float(np.mean(np.max(absdiff, axis=1)))
    line_conc_t = float(mean_t_of_fmax / (absdiff.mean() + 1e-6))

    mean_f_of_tmax = float(np.mean(np.max(absdiff, axis=0)))
    line_conc_f = float(mean_f_of_tmax / (absdiff.mean() + 1e-6))

    upper_tail_ratio = float(
        (np.quantile(absdiff, 0.999) + 1e-6) / (np.quantile(absdiff, 0.95) + 1e-6)
    )

    a_off_diffs = np.array(
        [
            np.mean(np.abs(a0 - b0)),
            np.mean(np.abs(a0 - b1)),
            np.mean(np.abs(a0 - b2)),
            np.mean(np.abs(a1 - b0)),
            np.mean(np.abs(a1 - b1)),
            np.mean(np.abs(a1 - b2)),
            np.mean(np.abs(a2 - b0)),
            np.mean(np.abs(a2 - b1)),
            np.mean(np.abs(a2 - b2)),
        ],
        dtype=np.float32,
    )
    a_off_mean = float(a_off_diffs.mean())
    a_off_min = float(a_off_diffs.min())
    a_off_max = float(a_off_diffs.max())
    a_off_std = float(a_off_diffs.std())
    a_only_vs_off_ratio = float((a_off_mean + 1e-6) / (b_intra + 1e-6))

    cons_ratio = float((b_intra + 1e-6) / (a_intra + 1e-6))

    a_map = a.mean(axis=0)
    b_map = b.mean(axis=0)

    a_rowmax = a_map.max(axis=1)
    b_rowmax = b_map.max(axis=1)
    a_colmax = a_map.max(axis=0)
    b_colmax = b_map.max(axis=0)

    a_rowmax_strength = float(a_rowmax.max() / (a_rowmax.mean() + 1e-6))
    b_rowmax_strength = float(b_rowmax.max() / (b_rowmax.mean() + 1e-6))
    a_colmax_strength = float(a_colmax.max() / (a_colmax.mean() + 1e-6))
    b_colmax_strength = float(b_colmax.max() / (b_colmax.mean() + 1e-6))

    a_tracks = np.stack(
        [np.argmax(a0, axis=1), np.argmax(a1, axis=1), np.argmax(a2, axis=1)], axis=0
    )
    b_tracks = np.stack(
        [np.argmax(b0, axis=1), np.argmax(b1, axis=1), np.argmax(b2, axis=1)], axis=0
    )
    a_track_disp = float(np.mean(np.std(a_tracks.astype(np.float32), axis=0)))
    b_track_disp = float(np.mean(np.std(b_tracks.astype(np.float32), axis=0)))
    track_disp_ratio = float((b_track_disp + 1e-6) / (a_track_disp + 1e-6))

    return np.array(
        [
            a_mean,
            b_mean,
            a_std,
            b_std,
            a_p99,
            b_p99,
            a_mean - b_mean,
            a_std - b_std,
            a_p99 - b_p99,
            diff_mean,
            diff_std,
            diff_p99,
            diff_tmax,
            diff_fmax,
            a_intra,
            b_intra,
            a_intra - b_intra,
            a_p999,
            b_p999,
            a_p999 - b_p999,
            strong_frac,
            a_over_b_l1,
            a_corr,
            b_corr,
            a_corr - b_corr,
            drift_abs,
            peak_over_mean,
            t_line_strength,
            f_line_strength,
            line_conc_t,
            line_conc_f,
            upper_tail_ratio,
            a_off_mean,
            a_off_min,
            a_off_max,
            a_off_std,
            a_only_vs_off_ratio,
            cons_ratio,
            a_rowmax_strength,
            b_rowmax_strength,
            a_rowmax_strength - b_rowmax_strength,
            a_colmax_strength,
            b_colmax_strength,
            a_colmax_strength - b_colmax_strength,
            a_track_disp,
            b_track_disp,
            track_disp_ratio,
        ],
        dtype=np.float32,
    )


if _baseline_is_constant_half(data11):
    comp_root = _resolve_comp_root()
    if comp_root is None:
        print(
            "Could not resolve competition data root; keeping baseline 0.5 submission."
        )
    else:
        train_labels_path = os.path.join(comp_root, "train_labels.csv")
        if not os.path.exists(train_labels_path):
            train_labels_path = _first_existing(
                [
                    os.path.join(BASE_INPUT, "train_labels.csv"),
                    os.path.join(BASE_DATA, "train_labels.csv"),
                    os.path.join(
                        BASE_INPUT, "seti-breakthrough-listen", "train_labels.csv"
                    ),
                    os.path.join(
                        BASE_DATA, "seti-breakthrough-listen", "train_labels.csv"
                    ),
                ]
            )

        train_dir = os.path.join(comp_root, "train")
        test_dir = os.path.join(comp_root, "test")
        if not os.path.exists(train_dir):
            train_dir = os.path.join(comp_root, "seti-breakthrough-listen", "train")
        if not os.path.exists(test_dir):
            test_dir = os.path.join(comp_root, "seti-breakthrough-listen", "test")

        if (
            train_labels_path is None
            or (not os.path.exists(train_dir))
            or (not os.path.exists(test_dir))
        ):
            print("Missing train/test data or labels; keeping baseline 0.5 submission.")
        else:
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            labels = pd.read_csv(train_labels_path)[["id", "target"]]
            labels["id"] = labels["id"].astype(str)
            labels["target"] = labels["target"].astype(int)

            N_TRAIN_MAX = 30000
            rng = np.random.RandomState(42)

            labels_sub = labels.sample(
                n=min(N_TRAIN_MAX, len(labels)),
                replace=False,
                random_state=42,
            ).reset_index(drop=True)

            train_shards = _list_shards(train_dir)
            test_shards = _list_shards(test_dir)
            if len(train_shards) == 0:
                train_shards = list(range(16))
            if len(test_shards) == 0:
                test_shards = list(range(16))

            X_list = []
            y_list = []
            missing = 0
            for id_, y in zip(labels_sub["id"].values, labels_sub["target"].values):
                p = _find_npy_by_id(train_dir, id_, shards_cache=train_shards)
                if p is None:
                    missing += 1
                    continue
                arr = np.load(p)
                X_list.append(_extract_features(arr))
                y_list.append(y)

            if len(X_list) < 6000:
                print(
                    f"Too few training files found ({len(X_list)}); keeping baseline 0.5 submission."
                )
            else:
                X = np.vstack(X_list)
                y = np.asarray(y_list, dtype=int)

                n1 = max(1, int((y == 1).sum()))
                n0 = max(1, int((y == 0).sum()))
                w1 = 0.5 / n1
                w0 = 0.5 / n0
                sample_weight = np.where(y == 1, w1, w0).astype(np.float64)

                Cs = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)

                clf = Pipeline(
                    steps=[
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        (
                            "lr",
                            LogisticRegressionCV(
                                Cs=Cs,
                                cv=3,
                                scoring="roc_auc",
                                solver="lbfgs",
                                max_iter=800,
                                n_jobs=1,
                                refit=True,
                                random_state=42,
                            ),
                        ),
                    ]
                )
                clf.fit(X, y, lr__sample_weight=sample_weight)

                train_feat_mean = X.mean(axis=0).astype(np.float32)

                test_ids = sample["id"].astype(str).values
                Xte = np.zeros((len(test_ids), X.shape[1]), dtype=np.float32)
                missing_test = 0
                for i, id_ in enumerate(test_ids):
                    p = _find_npy_by_id(test_dir, id_, shards_cache=test_shards)
                    if p is None:
                        missing_test += 1
                        Xte[i, :] = train_feat_mean
                    else:
                        arr = np.load(p)
                        Xte[i, :] = _extract_features(arr)

                proba = clf.predict_proba(Xte)[:, 1].astype(np.float64)
                proba = np.clip(proba, 0.0, 1.0)

                data11 = pd.DataFrame({"id": test_ids, "target": proba})
                chosen_C = float(np.ravel(clf.named_steps["lr"].C_)[0])
                print(
                    "Replaced constant-0.5 baseline with scaled logistic-regression-CV feature model predictions."
                )
                print(
                    "Chosen C:",
                    chosen_C,
                    "Train subset requested:",
                    len(labels_sub),
                    "used:",
                    len(X_list),
                    "missing:",
                    missing,
                    "Test rows:",
                    len(data11),
                    "missing_test:",
                    missing_test,
                    "train_shards:",
                    train_shards,
                    "test_shards:",
                    test_shards,
                    "n_features:",
                    X.shape[1],
                )



## === cell 5
data11.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data11.shape)
print(data11.head())
