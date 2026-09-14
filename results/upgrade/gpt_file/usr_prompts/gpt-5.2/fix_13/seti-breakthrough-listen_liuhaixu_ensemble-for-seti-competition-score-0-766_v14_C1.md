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
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression




## === cell 1
def _read_submission_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = [c.strip() for c in df.columns.tolist()]
    if "id" not in cols or "target" not in cols:
        return None
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].isna().any():
        return None
    return df


SEARCH_ROOTS = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
]


def _candidate_submission_paths() -> list[str]:
    paths = []
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        direct = os.path.join(root, "submission.csv")
        if os.path.exists(direct):
            paths.append(direct)
        nested = os.path.join(root, "seti-breakthrough-listen", "submission.csv")
        if os.path.exists(nested):
            paths.append(nested)

        try:
            for name in os.listdir(root):
                p = os.path.join(root, name, "submission.csv")
                if os.path.exists(p):
                    paths.append(p)
        except Exception:
            pass

    if not paths:
        for root in SEARCH_ROOTS:
            if os.path.isdir(root):
                paths.extend(
                    glob.glob(
                        os.path.join(
                            root, "seti-breakthrough-listen", "**", "submission.csv"
                        ),
                        recursive=True,
                    )
                )
    return sorted(set(paths))


candidate_paths = _candidate_submission_paths()

loaded = []
loaded_paths = []
for p in candidate_paths:
    df = _read_submission_csv(p)
    if df is not None:
        loaded.append(df)
        loaded_paths.append(p)

data1 = data2 = data3 = data4 = data5 = data6 = data7 = None

for i, df in enumerate(loaded[:7], start=1):
    locals()[f"data{i}"] = df

print(f"Found {len(loaded)} valid submission-like CSV(s). Using up to 7 for blending.")
for i, p in enumerate(loaded_paths[:7], start=1):
    print(f"data{i} <- {p}")

sample_path = None
for root in SEARCH_ROOTS:
    p = os.path.join(root, "sample_submission.csv")
    if os.path.exists(p):
        sample_path = p
        break
    p2 = os.path.join(root, "seti-breakthrough-listen", "sample_submission.csv")
    if os.path.exists(p2):
        sample_path = p2
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under expected Kaggle paths."
    )

sample = pd.read_csv(sample_path)[["id", "target"]].copy()
sample["id"] = sample["id"].astype(str)




## === cell 2
if data1 is not None:
    data11 = data1.copy()
else:
    data11 = sample.copy()

data11 = data11.merge(sample[["id"]], on="id", how="right")




## === cell 3
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler


def _find_data_root_with_subdirs(name: str) -> str | None:
    """Find a root that contains `{name}/0`, `{name}/1`, ... subdirs."""
    for root in SEARCH_ROOTS:
        p = os.path.join(root, name)
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "0")):
            return p
        p2 = os.path.join(root, "seti-breakthrough-listen", name)
        if os.path.isdir(p2) and os.path.isdir(os.path.join(p2, "0")):
            return p2
    return None


def _id_to_npy_path(id_str: str, base_dir: str) -> str:
    s = str(id_str).strip()
    shard = s[0].lower()  # "0".."9","a".."f"
    return os.path.join(base_dir, shard, f"{s}.npy")


def _quantile_flat(x2d: np.ndarray, q: float) -> np.float32:
    """
    Exact quantile matching numpy's default method for 1D quantile with linear interpolation.
    Here x2d is any array; we work on a flattened copy for partition.
    """
    a = np.asarray(x2d, dtype=np.float32).ravel()
    n = a.size
    if n == 0:
        return np.float32(np.nan)
    idx = (n - 1) * q
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return np.float32(np.partition(a, lo)[lo])
    part = np.partition(a, hi)
    x_hi = float(part[hi])
    x_lo = float(np.partition(part[: hi + 1], lo)[lo])
    w = idx - lo
    return np.float32(x_lo + (x_hi - x_lo) * w)


def _median_flat(x2d: np.ndarray) -> np.float32:
    a = np.asarray(x2d, dtype=np.float32).ravel()
    n = a.size
    if n == 0:
        return np.float32(np.nan)
    mid = (n - 1) * 0.5
    lo = int(np.floor(mid))
    hi = int(np.ceil(mid))
    if lo == hi:
        return np.float32(np.partition(a, lo)[lo])
    part = np.partition(a, hi)
    x_hi = float(part[hi])
    x_lo = float(np.partition(part[: hi + 1], lo)[lo])
    w = mid - lo
    return np.float32(x_lo + (x_hi - x_lo) * w)


def _extract_features_from_npy(arr: np.ndarray) -> np.ndarray:
    """
    Same core idea and same feature set as before:
    per-panel means/std/min/max/median/energy/grad energies/q10/q90 + A vs O contrasts.
    """
    x = np.asarray(arr, dtype=np.float32)  # (6, 273, 256)

    means = x.mean(axis=(1, 2))
    stds = x.std(axis=(1, 2))
    maxs = x.max(axis=(1, 2))
    mins = x.min(axis=(1, 2))

    medians = np.empty((6,), dtype=np.float32)
    q10 = np.empty((6,), dtype=np.float32)
    q90 = np.empty((6,), dtype=np.float32)
    for i in range(6):
        xi = x[i]
        medians[i] = _median_flat(xi)
        q10[i] = _quantile_flat(xi, 0.10)
        q90[i] = _quantile_flat(xi, 0.90)

    energies = np.mean(x * x, axis=(1, 2)).astype(np.float32)

    dt = x[:, 1:, :] - x[:, :-1, :]
    df = x[:, :, 1:] - x[:, :, :-1]
    grad_t_energy = np.mean(dt * dt, axis=(1, 2)).astype(np.float32)
    grad_f_energy = np.mean(df * df, axis=(1, 2)).astype(np.float32)

    A_idx = np.array([0, 2, 4])
    O_idx = np.array([1, 3, 5])

    d_mean = float(means[A_idx].mean() - means[O_idx].mean())
    d_std = float(stds[A_idx].mean() - stds[O_idx].mean())
    d_max = float(maxs[A_idx].max() - maxs[O_idx].max())
    d_min = float(mins[A_idx].min() - mins[O_idx].min())

    A_energy = float(energies[A_idx].mean())
    O_energy = float(energies[O_idx].mean())
    d_energy = float(A_energy - O_energy)

    d_median = float(medians[A_idx].mean() - medians[O_idx].mean())

    A_q10 = float(q10[A_idx].mean())
    O_q10 = float(q10[O_idx].mean())
    d_q10 = float(A_q10 - O_q10)

    A_q90 = float(q90[A_idx].mean())
    O_q90 = float(q90[O_idx].mean())
    d_q90 = float(A_q90 - O_q90)

    A_dt = dt[A_idx]
    O_dt = dt[O_idx]
    A_df = df[A_idx]
    O_df = df[O_idx]
    A_grad_t = float(np.mean(A_dt * A_dt))
    O_grad_t = float(np.mean(O_dt * O_dt))
    d_grad_t = float(A_grad_t - O_grad_t)

    A_grad_f = float(np.mean(A_df * A_df))
    O_grad_f = float(np.mean(O_df * O_df))
    d_grad_f = float(A_grad_f - O_grad_f)

    A_means = means[A_idx]
    O_means = means[O_idx]
    A_mean_pairdiff = float(
        (
            abs(A_means[0] - A_means[1])
            + abs(A_means[0] - A_means[2])
            + abs(A_means[1] - A_means[2])
        )
        / 3.0
    )
    A_vs_O_mean_diff = float(abs(A_means.mean() - O_means.mean()))
    A_consistency_score = float(A_vs_O_mean_diff - A_mean_pairdiff)

    feat = np.concatenate(
        [
            means,
            stds,
            maxs,
            mins,
            medians,
            energies,
            grad_t_energy,
            grad_f_energy,
            q10,
            q90,
            np.array(
                [
                    d_mean,
                    d_std,
                    d_max,
                    d_min,
                    A_energy,
                    O_energy,
                    d_energy,
                    d_median,
                    A_q10,
                    O_q10,
                    d_q10,
                    A_q90,
                    O_q90,
                    d_q90,
                    A_grad_t,
                    O_grad_t,
                    d_grad_t,
                    A_grad_f,
                    O_grad_f,
                    d_grad_f,
                    A_mean_pairdiff,
                    A_vs_O_mean_diff,
                    A_consistency_score,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return feat.astype(np.float32)


_DUMMY = np.zeros((6, 273, 256), dtype=np.float32)
_N_FEATS = int(_extract_features_from_npy(_DUMMY).shape[0])


def _load_one_float32(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def _build_features_for_ids(
    ids: np.ndarray, base_dir: str, max_workers: int | None = None, chunksize: int = 64
) -> np.ndarray:
    n = len(ids)
    X = np.empty((n, _N_FEATS), dtype=np.float32)

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = min(8, cpu)

    paths = [_id_to_npy_path(id_str, base_dir) for id_str in ids]

    if n < 128 or max_workers <= 1:
        for i, p in enumerate(paths):
            arr = _load_one_float32(p)
            if arr is None:
                X[i] = 0.0
            else:
                X[i] = _extract_features_from_npy(arr)
        return X

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, arr in enumerate(ex.map(_load_one_float32, paths, chunksize=chunksize)):
            if arr is None:
                X[i] = 0.0
            else:
                X[i] = _extract_features_from_npy(arr)
    return X


def _missing_rate(ids: np.ndarray, base_dir: str, max_check: int = 2000) -> float:
    n = min(len(ids), max_check)
    if n == 0:
        return 1.0
    miss = 0
    for s in ids[:n]:
        if not os.path.exists(_id_to_npy_path(s, base_dir)):
            miss += 1
    return miss / n


def _train_predict_fallback(sample_df: pd.DataFrame) -> np.ndarray:
    """
    Train a simple model on train/ + train_labels.csv and predict on test/.
    Only used when no external submission-like CSVs were found.
    """
    train_dir = _find_data_root_with_subdirs("train")
    test_dir = _find_data_root_with_subdirs("test")

    labels_path = None
    for root in SEARCH_ROOTS:
        p = os.path.join(root, "train_labels.csv")
        if os.path.exists(p):
            labels_path = p
            break
        p2 = os.path.join(root, "seti-breakthrough-listen", "train_labels.csv")
        if os.path.exists(p2):
            labels_path = p2
            break

    if train_dir is None or test_dir is None or labels_path is None:
        return np.full(len(sample_df), 0.5, dtype=np.float64)

    ydf = pd.read_csv(labels_path, usecols=["id", "target"]).copy()
    ydf["id"] = ydf["id"].astype(str)
    y = ydf["target"].to_numpy(dtype=np.int64)

    train_ids = ydf["id"].to_numpy()
    test_ids = sample_df["id"].to_numpy()

    print("Diagnostic missing-rate (lower is better):")
    print("  train missing rate ~", round(_missing_rate(train_ids, train_dir), 4))
    print("  test  missing rate ~", round(_missing_rate(test_ids, test_dir), 4))
    print("  feature dimension  =", _N_FEATS)

    X_train = _build_features_for_ids(train_ids, train_dir)
    X_test = _build_features_for_ids(test_ids, test_dir)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        n_jobs=None,
        random_state=0,
        class_weight="balanced",
        C=1.0,
    )
    clf.fit(X_train_s, y)
    p = clf.predict_proba(X_test_s)[:, 1].astype(np.float64)
    return np.clip(p, 0.0, 1.0)


weights = {
    "data1": 0.12,
    "data2": 0.12,
    "data3": 0.12,
    "data4": 0.12,
    "data5": 0.15,
    "data6": 0.70,
}

sources = {}
for name in ["data1", "data2", "data3", "data4", "data5", "data6"]:
    df = locals().get(name, None)
    if df is None:
        continue
    aligned = sample[["id"]].merge(df[["id", "target"]], on="id", how="left")
    if aligned["target"].isna().any():
        aligned["target"] = aligned["target"].fillna(0.5)
    sources[name] = aligned["target"].to_numpy(dtype=np.float64)

if len(sources) == 0:
    data11["target"] = _train_predict_fallback(sample)
else:
    w = np.array([weights.get(k, 1.0) for k in sources.keys()], dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0:
        w = np.ones_like(w)
        w_sum = float(w.sum())
    w = w / w_sum
    preds = np.zeros(len(sample), dtype=np.float64)
    for (k, arr), wk in zip(sources.items(), w):
        preds += wk * arr
    preds = np.clip(preds, 0.0, 1.0)
    data11["target"] = preds

data11 = data11[["id", "target"]].copy()




## === cell 4
data11.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data11.shape)
print(data11.head())
print("target summary:", data11["target"].describe())
