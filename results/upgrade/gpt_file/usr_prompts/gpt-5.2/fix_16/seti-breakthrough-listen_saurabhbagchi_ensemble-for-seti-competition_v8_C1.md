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

0.7554475030351673

# 6. Current score

0.51444

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50362) has done: 'The timeout is almost certainly dominated by Python-level file discovery (`glob` over tens of thousands of files) and then many small `np.load` calls in Python loops. I keep the exact same feature extraction and LogisticRegression CV logic, but speed up I/O by (1) building id→path maps via `os.scandir` (much faster than `glob`), and (2) parallelizing feature extraction with a deterministic thread pool (NumPy file reads release the GIL, so threads help) while writing into preallocated arrays. I also avoid repeated Series→numpy conversions inside loops and keep all paths unchanged.'
- What this solution (achieved 0.50125) has done: 'Your current pipeline likely underperforms because the fixed `C=1.0` regularization strength is not tuned for these 24 handcrafted features, which can leave LogisticRegression underfit and near-random (AUC ~0.5). To move toward the target score with minimal change and identical core logic, I keep the same feature extraction and CV setup but select `C` via cross-validated AUC on the existing folds, then refit/predict with that `C`. I also add lightweight feature standardization (fit on each fold only) which is a safe, minimal calibration step for LogisticRegression and typically improves AUC without changing the modeling approach. Submission format and paths remain unchanged and a valid `submission.csv` is still written.'
- What this solution (achieved 0.50092) has done: 'Your CV-tuned LogisticRegression is already the right core approach for this baseline, so the minimal score gain likely comes from (1) removing the explicit `class_weight="balanced"` (AUC is threshold-free and balancing can distort probability ranking here), and (2) slightly broadening the `C` grid so the CV search can find a better regularization strength without changing the model family. I keep the exact same feature extraction, the same StratifiedKFold procedure, and the same submission writing logic. These changes are small, safe, and directly aimed at improving ranking quality (AUC) while staying well within Kaggle constraints and runtime.'
- What this solution (achieved 0.5147) has done: 'Your current AUC (~0.501) is far below the target (~0.755), so we should make the smallest change that legitimately improves ranking quality without changing your overall “handcrafted features + LogisticRegression + CV” core. The biggest likely issue is feature impoverishment: global means/stds and percentiles wash out the key cadence structure (A-only signals), so I add a minimal set of cadence-difference features computed from the same arrays (A−B summary stats and their max-projections), keeping everything else (model family, CV, submission writing, paths) the same. I also switch the solver to `lbfgs` (still LogisticRegression) to better handle correlated standardized features, while keeping the same CV selection of `C`. These changes directly target AUC improvement and should move you toward the target without rewriting the approach.'
- What this solution (achieved 0.5126) has done: 'Your current score (0.5147 AUC) is far below the target (0.7554), so we should increase AUC with the smallest legitimate change while keeping your “handcrafted features + LogisticRegression + CV” core intact. The highest-leverage minimal fix is to add a few more cadence-structure features that are still simple summary statistics of the same arrays, specifically capturing “signal present in A but not B” via per-pixel positive differences and how often A exceeds B (these tend to improve ranking for SETI). I also switch LogisticRegression to `class_weight="balanced"` (AUC is insensitive to prevalence but this often helps ranking when features are weak and classes are imbalanced) while keeping the same CV-tuned `C` selection and standardization. All paths and I/O stay the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.51587) has done: 'The main timeout driver here is heavy per-file numpy loading and repeated expensive quantile computations across ~54k train + 6k test arrays; the rest (logreg CV) is comparatively small. I keep the exact same features and modeling logic, but make feature extraction faster by computing all required quantiles via a single `np.partition`-based helper (equivalent to `np.quantile` for these probabilities) and by reusing intermediate flattened views to avoid repeated full-array passes. I also reduce Python overhead and improve I/O throughput by batching threadpool work, avoiding repeated dictionary lookups, and ensuring BLAS thread oversubscription doesn’t stall CPU when combined with your own threadpool. These changes are deterministic and preserve evaluation semantics (only negligible floating-point differences).'
- What this solution (achieved 0.51644) has done: 'The timeout is dominated by feature extraction repeatedly computing multiple quantiles via `np.partition` on ~200k-element arrays (many times per sample) and by redundant array reshapes/temporaries. I keep the exact feature set and model/CV logic the same, but make quantile computation faster by (a) using in-place `np.partition` on a reusable float32 buffer (provably equivalent for the same interpolation rule) and (b) computing paired quantiles (0.95/0.99) with a single partition call. I also reduce I/O overhead by using `np.load(..., allow_pickle=False)` and modestly increase thread parallelism for the I/O-bound extraction while keeping determinism intact. No approximations, no changes to folds, C-grid, or LogisticRegression settings.'
- What this solution (achieved 0.51444) has done: 'Your current AUC (0.51644) is far below the target (0.75545), so we should make a small, legitimate improvement that preserves your “handcrafted features + LogisticRegression + CV” core. The biggest low-risk gain here is to add just a few more cadence-structure summary features that capture “bright localized energy in A but not in B” using max-pooled profiles and robust quantiles, without changing the model family, training loops, or loss/metric semantics. I also expand the C search slightly upward (still the same CV selection logic) because your feature set is now a bit richer and may need weaker regularization. All paths remain unchanged and the script still writes a valid `submission.csv` with `id,target`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

BASE_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",  # in case dataset is mounted flat
    "/kaggle/data",
]


def _find_existing_path(rel_path: str) -> str:
    for base in BASE_CANDIDATES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    if os.path.exists(rel_path):
        return rel_path
    raise FileNotFoundError(
        f"Could not find {rel_path} under any known base paths: {BASE_CANDIDATES}"
    )


train_labels_path = _find_existing_path("train_labels.csv")
sample_sub_path = _find_existing_path("sample_submission.csv")
train_root = _find_existing_path("train")
test_root = _find_existing_path("test")

train_labels = pd.read_csv(train_labels_path)
sample_sub = pd.read_csv(sample_sub_path)

train_labels.head(), sample_sub.head(), train_root, test_root




## === cell 1
def _build_id_to_path_map(root_dir: str) -> dict:
    id2path = {}
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        fid = f.name[:-4]  # strip ".npy"
                        id2path[fid] = f.path
    return id2path


train_id2path = _build_id_to_path_map(train_root)
test_id2path = _build_id_to_path_map(test_root)

train_labels = train_labels[train_labels["id"].isin(train_id2path)].reset_index(
    drop=True
)

len(train_labels), len(train_id2path), len(test_id2path)



## === cell 2
import threading

_tls = threading.local()


def _get_workbuf(n: int) -> np.ndarray:
    buf = getattr(_tls, "buf", None)
    if buf is None or buf.size < n:
        buf = np.empty(n, dtype=np.float32)
        _tls.buf = buf
    return buf


def _quantile_linear_from_buf(buf: np.ndarray, n: int, q: float) -> np.float32:
    if n == 0:
        return np.float32(0.0)
    if q <= 0.0:
        return np.float32(buf[:n].min())
    if q >= 1.0:
        return np.float32(buf[:n].max())

    h = (n - 1) * float(q)
    i = int(np.floor(h))
    j = int(np.ceil(h))
    if i == j:
        np.partition(buf[:n], i)
        return np.float32(buf[i])

    np.partition(buf[:n], (i, j))
    ai = float(buf[i])
    aj = float(buf[j])
    w = h - i
    return np.float32(ai + w * (aj - ai))


def _quantile_linear_pair(a: np.ndarray, q1: float, q2: float) -> tuple:
    a = np.asarray(a, dtype=np.float32).ravel()
    n = a.size
    buf = _get_workbuf(n)
    buf[:n] = a  # copy once; partition happens in-place on the buffer

    if n == 0:
        z = np.float32(0.0)
        return z, z

    if q1 > q2:
        q1, q2 = q2, q1

    def _ij(q: float):
        h = (n - 1) * float(q)
        i = int(np.floor(h))
        j = int(np.ceil(h))
        return h, i, j

    h1, i1, j1 = _ij(q1)
    h2, i2, j2 = _ij(q2)

    if q1 <= 0.0 and q2 >= 1.0:
        mn = np.float32(buf[:n].min())
        mx = np.float32(buf[:n].max())
        return mn, mx
    if q1 <= 0.0:
        q2v = _quantile_linear_from_buf(buf, n, q2)
        return np.float32(buf[:n].min()), q2v
    if q2 >= 1.0:
        q1v = _quantile_linear_from_buf(buf, n, q1)
        return q1v, np.float32(buf[:n].max())

    kth = tuple(sorted(set([i1, j1, i2, j2])))
    np.partition(buf[:n], kth)

    def _interp(h, i, j):
        if i == j:
            return np.float32(buf[i])
        ai = float(buf[i])
        aj = float(buf[j])
        w = h - i
        return np.float32(ai + w * (aj - ai))

    return _interp(h1, i1, j1), _interp(h2, i2, j2)


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]
    B = x[[1, 3, 5]]

    A_mean = np.float32(A.mean())
    B_mean = np.float32(B.mean())
    A_std = np.float32(A.std())
    B_std = np.float32(B.std())

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    A_p95, A_p99 = _quantile_linear_pair(A_flat, 0.95, 0.99)
    B_p95, B_p99 = _quantile_linear_pair(B_flat, 0.95, 0.99)

    A_tmax = A.max(axis=2)  # (3,273)
    B_tmax = B.max(axis=2)
    A_fmax = A.max(axis=1)  # (3,256)
    B_fmax = B.max(axis=1)

    D = A - B
    D_mean = np.float32(D.mean())
    D_std = np.float32(D.std())

    D_flat = D.reshape(-1)
    D_p95, D_p99 = _quantile_linear_pair(D_flat, 0.95, 0.99)

    D_tmax = D.max(axis=2)  # (3,273)
    D_fmax = D.max(axis=1)  # (3,256)

    D_pos = np.maximum(D, 0.0)
    D_pos_mean = np.float32(D_pos.mean())
    D_pos_flat = D_pos.reshape(-1)
    D_pos_p99 = _quantile_linear_pair(D_pos_flat, 0.99, 0.99)[0]

    D_pos_tmax = D_pos.max(axis=2)
    D_pos_fmax = D_pos.max(axis=1)
    frac_A_gt_B = np.float32((D > 0.0).mean())

    A01 = A[0] - A[1]
    A12 = A[1] - A[2]
    A02 = A[0] - A[2]
    abs_A01 = np.abs(A01)
    abs_A12 = np.abs(A12)
    abs_A02 = np.abs(A02)
    A_cons_mean = np.float32((abs_A01.mean() + abs_A12.mean() + abs_A02.mean()) / 3.0)

    A01_flat = abs_A01.reshape(-1)
    A12_flat = abs_A12.reshape(-1)
    A02_flat = abs_A02.reshape(-1)

    A01_p95, A01_p99 = _quantile_linear_pair(A01_flat, 0.95, 0.99)
    A12_p95, A12_p99 = _quantile_linear_pair(A12_flat, 0.95, 0.99)
    A02_p95, A02_p99 = _quantile_linear_pair(A02_flat, 0.95, 0.99)
    A_cons_p95 = np.float32((A01_p95 + A12_p95 + A02_p95) / 3.0)
    A_cons_p99 = np.float32((A01_p99 + A12_p99 + A02_p99) / 3.0)

    A_time_profile = A.mean(axis=2)  # (3,273)
    B_time_profile = B.mean(axis=2)  # (3,273)
    A_freq_profile = A.mean(axis=1)  # (3,256)
    B_freq_profile = B.mean(axis=1)  # (3,256)

    D_time_profile = A_time_profile - B_time_profile
    D_freq_profile = A_freq_profile - B_freq_profile
    Dp_time_profile = np.maximum(D_time_profile, 0.0)
    Dp_freq_profile = np.maximum(D_freq_profile, 0.0)

    A_tmax_mean = np.float32(A_tmax.mean())
    B_tmax_mean = np.float32(B_tmax.mean())
    A_tmax_std = np.float32(A_tmax.std())
    B_tmax_std = np.float32(B_tmax.std())

    A_fmax_mean = np.float32(A_fmax.mean())
    B_fmax_mean = np.float32(B_fmax.mean())
    A_fmax_std = np.float32(A_fmax.std())
    B_fmax_std = np.float32(B_fmax.std())

    D_tmax_mean = np.float32(D_tmax.mean())
    D_tmax_std = np.float32(D_tmax.std())
    D_fmax_mean = np.float32(D_fmax.mean())
    D_fmax_std = np.float32(D_fmax.std())

    D_pos_tmax_mean = np.float32(D_pos_tmax.mean())
    D_pos_tmax_std = np.float32(D_pos_tmax.std())
    D_pos_fmax_mean = np.float32(D_pos_fmax.mean())
    D_pos_fmax_std = np.float32(D_pos_fmax.std())

    A_time_mean = np.float32(A_time_profile.mean())
    B_time_mean = np.float32(B_time_profile.mean())
    A_time_std = np.float32(A_time_profile.std())
    B_time_std = np.float32(B_time_profile.std())
    A_time_max = np.float32(A_time_profile.max())
    B_time_max = np.float32(B_time_profile.max())

    A_freq_mean = np.float32(A_freq_profile.mean())
    B_freq_mean = np.float32(B_freq_profile.mean())
    A_freq_std = np.float32(A_freq_profile.std())
    B_freq_std = np.float32(B_freq_profile.std())
    A_freq_max = np.float32(A_freq_profile.max())
    B_freq_max = np.float32(B_freq_profile.max())

    D_time_mean = np.float32(D_time_profile.mean())
    D_time_std = np.float32(D_time_profile.std())
    D_time_max = np.float32(D_time_profile.max())
    D_freq_mean = np.float32(D_freq_profile.mean())
    D_freq_std = np.float32(D_freq_profile.std())
    D_freq_max = np.float32(D_freq_profile.max())

    Dp_time_mean = np.float32(Dp_time_profile.mean())
    Dp_time_std = np.float32(Dp_time_profile.std())
    Dp_time_max = np.float32(Dp_time_profile.max())
    Dp_freq_mean = np.float32(Dp_freq_profile.mean())
    Dp_freq_std = np.float32(Dp_freq_profile.std())
    Dp_freq_max = np.float32(Dp_freq_profile.max())

    A_tmax_flat = A_tmax.reshape(-1)
    B_tmax_flat = B_tmax.reshape(-1)
    A_fmax_flat = A_fmax.reshape(-1)
    B_fmax_flat = B_fmax.reshape(-1)
    A_tmax_p99 = _quantile_linear_pair(A_tmax_flat, 0.99, 0.99)[0]
    B_tmax_p99 = _quantile_linear_pair(B_tmax_flat, 0.99, 0.99)[0]
    A_fmax_p99 = _quantile_linear_pair(A_fmax_flat, 0.99, 0.99)[0]
    B_fmax_p99 = _quantile_linear_pair(B_fmax_flat, 0.99, 0.99)[0]

    D_pos_tmax_p99 = _quantile_linear_pair(D_pos_tmax.reshape(-1), 0.99, 0.99)[0]
    D_pos_fmax_p99 = _quantile_linear_pair(D_pos_fmax.reshape(-1), 0.99, 0.99)[0]
    Dp_time_p99 = _quantile_linear_pair(Dp_time_profile.reshape(-1), 0.99, 0.99)[0]
    Dp_freq_p99 = _quantile_linear_pair(Dp_freq_profile.reshape(-1), 0.99, 0.99)[0]

    feat = np.array(
        [
            A_mean,
            B_mean,
            A_mean - B_mean,
            A_std,
            B_std,
            A_std - B_std,
            A_p95,
            B_p95,
            A_p95 - B_p95,
            A_p99,
            B_p99,
            A_p99 - B_p99,
            A_tmax_mean,
            B_tmax_mean,
            A_tmax_mean - B_tmax_mean,
            A_tmax_std,
            B_tmax_std,
            A_tmax_std - B_tmax_std,
            A_fmax_mean,
            B_fmax_mean,
            A_fmax_mean - B_fmax_mean,
            A_fmax_std,
            B_fmax_std,
            A_fmax_std - B_fmax_std,
            D_mean,
            D_std,
            D_p95,
            D_p99,
            D_tmax_mean,
            D_tmax_std,
            D_fmax_mean,
            D_fmax_std,
            np.float32(abs(D_mean)),
            np.float32(abs(D_p95)),
            np.float32(abs(D_p99)),
            np.float32(abs(D_tmax_mean)),
            D_pos_mean,
            D_pos_p99,
            D_pos_tmax_mean,
            D_pos_tmax_std,
            D_pos_fmax_mean,
            D_pos_fmax_std,
            frac_A_gt_B,
            A_cons_mean,
            A_cons_p95,
            A_cons_p99,
            A_time_mean,
            B_time_mean,
            A_time_mean - B_time_mean,
            A_time_std,
            B_time_std,
            A_time_std - B_time_std,
            A_time_max,
            B_time_max,
            A_time_max - B_time_max,
            A_freq_mean,
            B_freq_mean,
            A_freq_mean - B_freq_mean,
            A_freq_std,
            B_freq_std,
            A_freq_std - B_freq_std,
            A_freq_max,
            B_freq_max,
            A_freq_max - B_freq_max,
            D_time_mean,
            D_time_std,
            D_time_max,
            D_freq_mean,
            D_freq_std,
            D_freq_max,
            Dp_time_mean,
            Dp_time_std,
            Dp_time_max,
            Dp_freq_mean,
            Dp_freq_std,
            Dp_freq_max,
            A_tmax_p99,
            B_tmax_p99,
            A_tmax_p99 - B_tmax_p99,
            A_fmax_p99,
            B_fmax_p99,
            A_fmax_p99 - B_fmax_p99,
            D_pos_tmax_p99,
            D_pos_fmax_p99,
            Dp_time_p99,
            Dp_freq_p99,
        ],
        dtype=np.float32,
    )

    feat[~np.isfinite(feat)] = 0.0
    return feat


one_id = train_labels["id"].iloc[0]
arr = np.load(train_id2path[one_id], mmap_mode="r", allow_pickle=False)
extract_features_from_array(arr).shape



## === cell 3
from concurrent.futures import ThreadPoolExecutor

N_FEATS = int(extract_features_from_array(arr).shape[0])

X = np.zeros((len(train_labels), N_FEATS), dtype=np.float32)
y = train_labels["target"].astype(np.int8).to_numpy(copy=False)
train_ids = train_labels["id"].to_numpy(copy=False)

_np_load = np.load
_extract = extract_features_from_array


def _feat_from_path(p: str) -> np.ndarray:
    x = _np_load(p, mmap_mode="r", allow_pickle=False)
    return _extract(x)


cpu = os.cpu_count() or 2
max_workers = min(16, max(4, cpu))
train_paths = [train_id2path[fid] for fid in train_ids]

chunksize = 1024

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feat in enumerate(ex.map(_feat_from_path, train_paths, chunksize=chunksize)):
        X[i] = feat

X.shape, float(y.mean())



## === cell 4
from sklearn.metrics import roc_auc_score


def _fit_standardizer(Xtr: np.ndarray):
    mu = Xtr.mean(axis=0, dtype=np.float64)
    sigma = Xtr.std(axis=0, dtype=np.float64)
    sigma[sigma == 0] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def _standardize_inplace(
    Xin: np.ndarray, mu: np.ndarray, sigma: np.ndarray, out: np.ndarray
):
    np.subtract(Xin, mu, out=out)
    np.divide(out, sigma, out=out)
    return out


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

best_C = None
best_auc = -1.0

Xtr_buf = np.empty((0, N_FEATS), dtype=np.float32)
Xva_buf = np.empty((0, N_FEATS), dtype=np.float32)

for C in C_GRID:
    oof_c = np.zeros(len(train_labels), dtype=np.float32)
    for tr_idx, va_idx in skf.split(X, y):
        mu, sigma = _fit_standardizer(X[tr_idx])

        if Xtr_buf.shape[0] != tr_idx.size:
            Xtr_buf = np.empty((tr_idx.size, N_FEATS), dtype=np.float32)
        if Xva_buf.shape[0] != va_idx.size:
            Xva_buf = np.empty((va_idx.size, N_FEATS), dtype=np.float32)

        Xtr = _standardize_inplace(X[tr_idx], mu, sigma, Xtr_buf)
        Xva = _standardize_inplace(X[va_idx], mu, sigma, Xva_buf)

        model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
        model.fit(Xtr, y[tr_idx])
        oof_c[va_idx] = model.predict_proba(Xva)[:, 1].astype(np.float32)

    auc = roc_auc_score(y, oof_c)
    if auc > best_auc:
        best_auc = auc
        best_C = C

best_C, best_auc



## === cell 5
X_test = np.zeros((len(sample_sub), N_FEATS), dtype=np.float32)
test_ids = sample_sub["id"].to_numpy(copy=False)

_test_id2path = test_id2path
_zeros_feat = np.zeros(N_FEATS, dtype=np.float32)


def _feat_from_test_id(fid: str) -> np.ndarray:
    p = _test_id2path.get(fid)
    if p is None:
        return _zeros_feat
    x = _np_load(p, mmap_mode="r", allow_pickle=False)
    return _extract(x)


missing_test = int(sum((fid not in test_id2path) for fid in test_ids))

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feat in enumerate(ex.map(_feat_from_test_id, test_ids, chunksize=chunksize)):
        X_test[i] = feat

oof = np.zeros(len(train_labels), dtype=np.float32)
test_pred = np.zeros(len(sample_sub), dtype=np.float32)

Xte_buf = np.empty((X_test.shape[0], N_FEATS), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    mu, sigma = _fit_standardizer(X[tr_idx])

    if Xtr_buf.shape[0] != tr_idx.size:
        Xtr_buf = np.empty((tr_idx.size, N_FEATS), dtype=np.float32)
    if Xva_buf.shape[0] != va_idx.size:
        Xva_buf = np.empty((va_idx.size, N_FEATS), dtype=np.float32)

    Xtr = _standardize_inplace(X[tr_idx], mu, sigma, Xtr_buf)
    Xva = _standardize_inplace(X[va_idx], mu, sigma, Xva_buf)
    Xte = _standardize_inplace(X_test, mu, sigma, Xte_buf)

    model = LogisticRegression(
        C=best_C,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(Xtr, y[tr_idx])
    oof[va_idx] = model.predict_proba(Xva)[:, 1].astype(np.float32)
    test_pred += model.predict_proba(Xte)[:, 1].astype(np.float32) / skf.n_splits

oof_auc = roc_auc_score(y, oof)
missing_test, oof_auc, best_C, N_FEATS

sub = sample_sub.copy()
sub["target"] = np.clip(test_pred, 0.0, 1.0)
sub.to_csv("submission.csv", index=False)

sub.head(), sub.shape
