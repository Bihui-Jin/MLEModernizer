# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.7571864047949978

# 6. Current score

0.50317

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to ensemble external “../input/…” submission files that don’t exist in this environment, so the very first `read_csv` throws `FileNotFoundError` and everything downstream is undefined. I keep the same core approach (weighted ensembling of multiple submission files) but make it robust: it automatically discover any available `submission.csv` files under `/kaggle/input`, fall back to a valid baseline using `sample_submission.csv` if none are found, and always align by `id` before averaging. This fixes runtime errors, guarantees a properly formatted `submission.csv`, and should move score upward from “no submission” to a reasonable ensemble (or safe baseline) without changing evaluation semantics.'
- What this solution (achieved 0.50193) has done: 'Main bottlenecks are (1) recursively globbing all of `/kaggle/input/**` for submissions (very slow) and (2) single-threaded per-file `np.load` feature extraction over ~60k train + 6k test snippets. I make the submission search O(known paths) by only scanning the immediate `/kaggle/input` root (non-recursive) and skipping heavy traversal, while keeping identical ensemble/ranking behavior when submission files are present. For the fallback model, I keep the exact same features and LogisticRegression CV logic, but speed up feature extraction by using `np.load(..., mmap_mode='r')`, precomputing the `id->path` vectorized, and parallelizing per-file feature extraction with a deterministic thread pool (safe since NumPy releases the GIL on heavy ops). These changes preserve evaluation semantics and should bring runtime under 600s primarily by eliminating recursive filesystem walks and by parallel I/O/compute for feature extraction.'
- What this solution (achieved 0.50193) has done: 'I keep your pipeline identical (ensemble if any valid external submissions are found; otherwise use your same 12 handcrafted features + LogisticRegression) and make one minimal scoring-oriented improvement: apply the same rank-based post-processing you already use for ensembling to the fallback model predictions, because ROC-AUC is rank-based and this often nudges the score upward without changing the model/training logic. I also fix a hidden compatibility risk in your threadpool feature extraction: `ThreadPoolExecutor.map` doesn’t support `chunksize` on some Python versions, which can silently break or slow execution; removing it preserves semantics and stabilizes runtime. Finally, I add a tiny safeguard to ensure feature extraction doesn’t crash the whole run if a single `.npy` is missing/corrupt (fill with per-feature means), keeping submission generation robust. These changes are minimal, keep evaluation semantics, and are aimed at moving your current ~0.50 AUC toward the 0.757 target.'
- What this solution (achieved 0.50317) has done: 'Your current ~0.50 AUC strongly suggests the fallback model is producing near-constant rankings; the smallest legitimate boost without changing the model/training core is to make the handcrafted features better reflect the competition’s “A vs not-A” cadence structure. I keep the same 12-feature LogisticRegression pipeline and training loop, but change the feature definitions to compare A panels against the average of B/C/D (off-target mean) rather than pairing A-O by index (which is not the real cadence logic). I also add two extra “needle-like” features (A-vs-off max and A-vs-off 90th percentile) while keeping the total feature count at 12 by replacing two weaker/duplicative ones, preserving the same model architecture and evaluation semantics. These changes should improve separability and move AUC upward toward your 0.757 target with minimal code edits and no new dependencies.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
SAMPLE_PATH_CANDIDATES = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
]

sample_path = None
for p in SAMPLE_PATH_CANDIDATES:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected locations: "
        + ", ".join(SAMPLE_PATH_CANDIDATES)
    )

sample = pd.read_csv(sample_path)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv must contain columns ['id','target']; got {list(sample.columns)}"
    )

sample["id"] = sample["id"].astype(str)
sample = sample[["id", "target"]].copy()




## === cell 2
def load_submission_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"id", "target"}.issubset(df.columns):
        raise ValueError(f"{path} missing required columns")
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    return df


def to_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n <= 1:
        return np.zeros_like(x, dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)

    xs = x[order]
    dif = np.diff(xs)
    tie_starts = np.r_[0, np.nonzero(dif != 0)[0] + 1]
    tie_ends = np.r_[tie_starts[1:], n]

    ranks_sorted = np.empty(n, dtype=np.float64)
    for s, e in zip(tie_starts, tie_ends):
        avg_rank_1based = 0.5 * ((s + 1) + e)  # positions s..e-1 => ranks (s+1)..e
        ranks_sorted[s:e] = avg_rank_1based
    ranks = ranks_sorted[inv]
    return (ranks - 1.0) / (n - 1.0)


exclude_names = {
    "sample_submission.csv",
    "train_labels.csv",
    "train_labels_old.csv",
    "test_labels_old.csv",
}

candidate_paths = set()

INPUT_ROOTS = [
    "/kaggle/input",
    "/kaggle/data",
]
for root in INPUT_ROOTS:
    if not os.path.isdir(root):
        continue
    try:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isfile(p) and name.lower().endswith(".csv"):
                candidate_paths.add(p)
            elif os.path.isdir(p):
                try:
                    for fn in os.listdir(p):
                        fp = os.path.join(p, fn)
                        if os.path.isfile(fp) and fn.lower().endswith(".csv"):
                            candidate_paths.add(fp)
                except OSError:
                    pass
    except OSError:
        pass

filtered_paths = []
for p in sorted(candidate_paths):
    if os.path.basename(p) in exclude_names:
        continue
    try:
        if os.path.getsize(p) > 50_000_000:  # 50MB
            continue
    except OSError:
        continue
    bn = os.path.basename(p).lower()
    if ("sub" not in bn) and ("submission" not in bn):
        continue
    filtered_paths.append(p)

loaded = []
loaded_paths = []
for p in filtered_paths:
    try:
        df = load_submission_csv(p)
        if len(df) == len(sample):
            loaded.append(df)
            loaded_paths.append(p)
    except Exception:
        continue

print(
    f"Found {len(loaded)} usable submission-like CSV files under /kaggle/input or /kaggle/data (shallow scan)"
)
if len(loaded_paths) > 0:
    print("Using:")
    for p in loaded_paths[:10]:
        print("  ", p)
    if len(loaded_paths) > 10:
        print(f"  ... (+{len(loaded_paths) - 10} more)")




## === cell 3
def find_existing_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


TRAIN_DIR_CANDIDATES = [
    "/kaggle/input/train",
    "/kaggle/data/train",
    "/kaggle/input/seti-breakthrough-listen/train",
    "/kaggle/data/seti-breakthrough-listen/train",
]
TEST_DIR_CANDIDATES = [
    "/kaggle/input/test",
    "/kaggle/data/test",
    "/kaggle/input/seti-breakthrough-listen/test",
    "/kaggle/data/seti-breakthrough-listen/test",
]
LABELS_CANDIDATES = [
    "/kaggle/input/train_labels.csv",
    "/kaggle/data/train_labels.csv",
    "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
    "/kaggle/data/seti-breakthrough-listen/train_labels.csv",
]

train_dir = find_existing_dir(TRAIN_DIR_CANDIDATES)
test_dir = find_existing_dir(TEST_DIR_CANDIDATES)

labels_path = None
for p in LABELS_CANDIDATES:
    if os.path.exists(p):
        labels_path = p
        break


def resolve_npy_path(root_dir: str, _id: str) -> str:
    shard = _id[0]
    return os.path.join(root_dir, shard, f"{_id}.npy")


def extract_features_from_npy(path: str) -> np.ndarray:
    x = np.load(path, mmap_mode="r")  # (6, 273, 256), float16
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # on-target (A)
    O = x[[1, 3, 5]]  # off-target (B,C,D)

    A_mean = A.mean()
    O_mean = O.mean()
    A_std = A.std()
    O_std = O.std()

    O_bar = O.mean(axis=0)  # (273,256)
    D = A - O_bar[None, :, :]  # (3,273,256)

    D_mean = D.mean()
    D_std = D.std()

    A_max = A.max()
    O_max = O.max()

    D_max = (A.max(axis=(1, 2)) - O_bar.max()).mean()

    A_q90 = np.quantile(A, 0.90)
    O_q90 = np.quantile(O, 0.90)

    D_q90 = np.quantile(D, 0.90)

    return np.array(
        [
            A_mean,
            O_mean,
            A_std,
            O_std,
            D_mean,
            D_std,
            A_max,
            O_max,
            D_max,
            A_q90,
            O_q90,
            D_q90,
        ],
        dtype=np.float32,
    )


def extract_features_for_ids(
    root_dir: str, ids: np.ndarray, n_features: int = 12
) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor

    paths = [resolve_npy_path(root_dir, _id) for _id in ids]
    X = np.empty((len(ids), n_features), dtype=np.float32)

    def _safe_extract(path: str) -> np.ndarray:
        try:
            return extract_features_from_npy(path)
        except Exception:
            return np.full((n_features,), np.nan, dtype=np.float32)

    def _worker(i_path):
        i, path = i_path
        return i, _safe_extract(path)

    max_workers = min(16, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in ex.map(_worker, enumerate(paths)):
            X[i] = feats

    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    return X




## === cell 4
sub = sample.copy()

if len(loaded) > 0:
    preds = []
    for df, p in zip(loaded, loaded_paths):
        m = sample[["id"]].merge(df, on="id", how="left", validate="one_to_one")
        fill_value = float(np.nanmean(m["target"].values))
        if not np.isfinite(fill_value):
            fill_value = 0.5
        m["target"] = m["target"].fillna(fill_value).astype(float).to_numpy()
        preds.append(m["target"])

    P = np.vstack(preds)  # (n_models, n_rows)
    R = np.vstack([to_ranks(P[i]) for i in range(P.shape[0])])
    avg_rank = R.mean(axis=0)

    sub["target"] = avg_rank.astype(np.float64)
else:
    if train_dir is None or test_dir is None or labels_path is None:
        sub["target"] = 0.5
    else:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        labels = pd.read_csv(labels_path)
        labels["id"] = labels["id"].astype(str)

        ids = labels["id"].values
        y = labels["target"].values.astype(int)

        X_train = extract_features_for_ids(train_dir, ids, n_features=12)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(ids), dtype=np.float64)

        for tr_idx, va_idx in skf.split(X_train, y):
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=300,
                    n_jobs=None,
                    class_weight=None,
                    random_state=42,
                ),
            )
            clf.fit(X_train[tr_idx], y[tr_idx])
            oof[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]

        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                solver="lbfgs",
                max_iter=300,
                n_jobs=None,
                class_weight=None,
                random_state=42,
            ),
        )
        clf.fit(X_train, y)

        test_ids = sample["id"].values
        X_test = extract_features_for_ids(test_dir, test_ids, n_features=12)

        raw_pred = clf.predict_proba(X_test)[:, 1].astype(np.float64)

        sub["target"] = to_ranks(raw_pred).astype(np.float64)

sub["target"] = (
    pd.to_numeric(sub["target"], errors="coerce").astype(float).clip(0.0, 1.0)
)
sub = sub[["id", "target"]].copy()

assert len(sub) == len(sample)
assert (sub["id"].values == sample["id"].values).all()
assert sub["target"].notna().all()



## === cell 5
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(sub.head())
print("target summary:", sub["target"].describe())
