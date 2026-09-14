# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7561532991008403

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50055) has done: 'Main bottlenecks are (1) slow Python-side directory scanning and ID alignment, and (2) extremely expensive feature extraction dominated by repeated `np.quantile` calls (which sort/partition large arrays per sample). To fit within 600s without changing modeling logic, I keep the exact same features but compute the needed quantiles via `np.partition` (order-statistics) on flattened views, which is mathematically equivalent for the requested linear quantiles at fixed `q` (up to negligible floating-point differences) and avoids full sorts. I also speed up file discovery using a single `glob` pattern (still identical set of files) and reduce per-sample overhead by avoiding temporary arrays and redundant passes. Training loop/model stays the same.'
- What this solution (achieved 0.50055) has done: 'Your current AUC (~0.50055) is near-random, which strongly suggests the predictions in `submission.csv` are mostly misaligned with the true test `id` order (many IDs likely map to 0.5 due to missing keys), not that the model can’t learn. I make a minimal, score-relevant fix to guarantee `test_files_ordered` is built in exactly the same order as `sample_submission.csv` (and assert full coverage), so every row gets the intended prediction. I also add a hard check that train IDs align to labels (to avoid silent mis-ordering) while keeping the exact same features, model, CV, and training loop. This should move the score substantially upward toward the target without changing core modeling logic.'
- What this solution (achieved 0.50656) has done: 'Your current AUC is essentially random, so the highest-probability issue is that the model is training on features with extreme scale differences / outliers and LogisticRegression is failing to find a useful decision boundary (even though IDs are now aligned). To move the score upward toward the 0.756 target while preserving the same model and features, I add a `StandardScaler` fitted inside each CV fold (no leakage) and apply it to both validation and test predictions for that fold. This is a minimal change that keeps the architecture/training loop intact but makes LBFGS converge to a better solution for this kind of hand-crafted feature set. I also compute and print OOF AUC locally for sanity (doesn’t affect submission) and keep the exact submission ordering from `sample_submission.csv`.'
- What this solution (achieved 0.50687) has done: 'Your current AUC (0.50656) is still near-random versus the 0.756 target, so the most likely issue is that the fold-wise scaling of the *test* set (and therefore predictions) is inconsistent across folds and can wash out signal when you average probabilities. I keep the exact same features and LogisticRegression setup, but switch to a leakage-safe, score-relevant approach: generate test predictions by averaging *decision_function* across folds and applying a single sigmoid at the end (this preserves ranking better for AUC than averaging already-squashed probabilities from differently-scaled spaces). I also set `class_weight="balanced"` (still LogisticRegression, same loss/solver) to counter class imbalance, which typically gives a modest AUC lift without changing core logic. Submission ordering and ID alignment remain exactly tied to `sample_submission.csv`, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.51558) has done: 'The timeout is dominated by reading ~60k `.npy` files and repeatedly doing heavy per-file NumPy work; the fastest safe wins are to (1) avoid multiprocessing overhead and disk contention, (2) remove repeated full-array passes (especially multiple `np.abs/np.square` means), and (3) cut quantile work from multiple `np.partition` calls down to one per array while keeping the exact same “linear between order stats” semantics. I keep the model, CV, scaling, and training loop identical, but refactor feature extraction to reuse intermediate results and compute all needed quantiles via a single partition per vector. I also precompute the scaled `X_test` once and reuse it across folds (identical math), and enforce deterministic threading by limiting BLAS threads to 1 to avoid slow oversubscription. These changes preserve evaluation semantics and only reduce redundant work and overhead.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by feature extraction doing 60k+ individual `np.load` calls plus repeated per-sample quantile computations; the model training itself is fast. To finish under 600s without changing the algorithm, the main speedups are (1) replacing `glob` with much faster `os.scandir` traversal, (2) parallelizing feature extraction across CPU cores using a deterministic multiprocessing pool, and (3) reducing per-sample overhead in quantiles by using `np.quantile(method="linear")` (same linear interpolation semantics) instead of repeated `np.partition` bookkeeping. We also cache the per-file extracted features to disk (keyed by path+mtime+size and code signature) so reruns don’t recompute anything, while preserving identical outputs for a given input and feature function. All paths, features, model, CV scheme, and prediction logic remain unchanged.'

# 9. Code solution

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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_INPUT = "/kaggle/input"
DATA_ROOT_CANDIDATES = [
    os.path.join(BASE_INPUT, "seti-breakthrough-listen"),
    os.path.join(BASE_INPUT, "seti-breakthrough-listen", "seti-breakthrough-listen"),
    os.path.join(BASE_INPUT, "data"),
    os.path.join(BASE_INPUT, "input"),
]


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


DATA_ROOT = first_existing(*DATA_ROOT_CANDIDATES)
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find dataset root among: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_DIR = first_existing(os.path.join(DATA_ROOT, "train"))
TEST_DIR = first_existing(os.path.join(DATA_ROOT, "test"))
LABELS_PATH = first_existing(os.path.join(DATA_ROOT, "train_labels.csv"))
SAMPLE_SUB_PATH = first_existing(os.path.join(DATA_ROOT, "sample_submission.csv"))

for pth, nm in [
    (TRAIN_DIR, "TRAIN_DIR"),
    (TEST_DIR, "TEST_DIR"),
    (LABELS_PATH, "LABELS_PATH"),
    (SAMPLE_SUB_PATH, "SAMPLE_SUB_PATH"),
]:
    if pth is None:
        raise FileNotFoundError(
            f"Missing required path for {nm} under DATA_ROOT={DATA_ROOT}"
        )

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("LABELS_PATH:", LABELS_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 1
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def list_npy_files(root_dir):
    out = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        out.append(f.path)
    out.sort()
    return out


def ids_from_files(files):
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

train_ids = ids_from_files(train_files)

train_labels = pd.read_csv(LABELS_PATH)

train_labels_idx = train_labels.set_index("id")
missing_train_labels = [i for i in train_ids if i not in train_labels_idx.index]
if missing_train_labels:
    raise ValueError(
        f"Some train ids have no labels (count={len(missing_train_labels)}). "
        f"Example: {missing_train_labels[:5]}"
    )
train_labels_aligned = train_labels_idx.loc[train_ids].reset_index()

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

id_to_test_file = {os.path.splitext(os.path.basename(f))[0]: f for f in test_files}
test_ids_ordered = sample_sub["id"].tolist()

missing_test_files = [i for i in test_ids_ordered if i not in id_to_test_file]
if missing_test_files:
    raise ValueError(
        f"Some sample_submission ids are missing from test files (count={len(missing_test_files)}). "
        f"Example: {missing_test_files[:5]}"
    )

test_files_ordered = [id_to_test_file[i] for i in test_ids_ordered]

_Q_95 = 0.95
_Q_99 = 0.99
_Q_05 = 0.05
_Q_01 = 0.01


def _quantiles_linear_1d(a1d, qs):
    if a1d.size == 0:
        return [np.nan] * len(qs)
    return np.quantile(a1d, qs, method="linear").astype(np.float64, copy=False).tolist()


def extract_features_from_array(x):
    """
    x shape: (6, 273, 256)
    Cadence panels: A,B,A,C,A,D -> indices 0,1,2,3,4,5
    """
    x = x.astype(np.float32, copy=False)

    A0, B, A1, C, A2, Dp = x[0], x[1], x[2], x[3], x[4], x[5]
    A = (A0 + A1 + A2) * (1.0 / 3.0)
    O = (B + C + Dp) * (1.0 / 3.0)

    Diff = A - O
    AD = np.abs(Diff)

    diff_mean = float(Diff.mean())
    diff_std = float(Diff.std())
    ad_mean = float(AD.mean())
    ad_std = float(AD.std())
    ad_max = float(AD.max())

    ad_flat = AD.reshape(-1)
    diff_flat = Diff.reshape(-1)
    ad_q95, ad_q99 = _quantiles_linear_1d(ad_flat, (_Q_95, _Q_99))
    diff_q05, diff_q95, diff_q01, diff_q99 = _quantiles_linear_1d(
        diff_flat, (_Q_05, _Q_95, _Q_01, _Q_99)
    )

    feats = [
        diff_mean,
        diff_std,
        ad_mean,
        ad_std,
        ad_max,
        ad_q95,
        ad_q99,
    ]

    row = AD.mean(axis=1)  # (273,)
    col = AD.mean(axis=0)  # (256,)

    row_mean = float(row.mean())
    row_std = float(row.std())
    row_max = float(row.max())
    (row_q95,) = _quantiles_linear_1d(row, (_Q_95,))

    col_mean = float(col.mean())
    col_std = float(col.std())
    col_max = float(col.max())
    (col_q95,) = _quantiles_linear_1d(col, (_Q_95,))

    feats += [row_mean, row_std, row_max, row_q95]
    feats += [col_mean, col_std, col_max, col_q95]

    eps = 1e-6
    absA = np.abs(A)
    absO = np.abs(O)
    sqA = A * A
    sqO = O * O

    absA_mean = float(absA.mean())
    absO_mean = float(absO.mean())
    sqA_mean = float(sqA.mean())
    sqO_mean = float(sqO.mean())

    feats += [
        (absA_mean + eps) / (absO_mean + eps),
        (sqA_mean + eps) / (sqO_mean + eps),
    ]

    feats += [
        diff_q05,
        diff_q95,
        diff_q01,
        diff_q99,
    ]

    absA_pan = (np.abs(A0).mean() + np.abs(A1).mean() + np.abs(A2).mean()) / 3.0
    absO_pan = (np.abs(B).mean() + np.abs(C).mean() + np.abs(Dp).mean()) / 3.0
    sqA_pan = ((A0 * A0).mean() + (A1 * A1).mean() + (A2 * A2).mean()) / 3.0
    sqO_pan = ((B * B).mean() + (C * C).mean() + (Dp * Dp).mean()) / 3.0

    feats += [
        float(absA_pan - absO_pan),
        float(sqA_pan - sqO_pan),
    ]

    A_t = A.max(axis=1)  # (273,)
    O_t = O.max(axis=1)
    d = A_t - O_t
    feats += [
        float(d.mean()),
        float(np.abs(d).mean()),
        float(A_t.max()),
        float(O_t.max()),
    ]

    A_flat = A.reshape(-1)
    O_flat = O.reshape(-1)

    (A_q99,) = _quantiles_linear_1d(A_flat, (_Q_99,))
    (O_q99,) = _quantiles_linear_1d(O_flat, (_Q_99,))
    (A_q95,) = _quantiles_linear_1d(A_flat, (_Q_95,))
    (O_q95,) = _quantiles_linear_1d(O_flat, (_Q_95,))

    feats += [
        float((A_q99 + eps) / (O_q99 + eps)),
        float((A_q95 + eps) / (O_q95 + eps)),
    ]

    thr95 = ad_q95
    thr99 = ad_q99
    feats += [
        float((AD > thr95).mean()),
        float((AD > thr99).mean()),
    ]

    pos_mean = float(np.maximum(Diff, 0.0).mean())
    neg_mean = float(np.maximum(-Diff, 0.0).mean())
    feats += [
        float((pos_mean + eps) / (neg_mean + eps)),
        float(pos_mean - neg_mean),
    ]

    return np.asarray(feats, dtype=np.float32)


def _features_from_file(fpath):
    arr = np.load(fpath, mmap_mode="r")
    return extract_features_from_array(arr)


def _cache_dir():
    d = "/kaggle/working/seti_feat_cache"
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key_for_file(fpath, code_sig):
    st = os.stat(fpath)
    return f"{os.path.basename(fpath)}__{st.st_size}__{int(st.st_mtime)}__{code_sig}"


def _code_signature():
    return "v3_quantile_linear_mp_cache_33"


def _cache_path_for_key(key):
    return os.path.join(_cache_dir(), key + ".npy")


def _load_or_compute_feature(fpath):
    sig = _code_signature()
    key = _cache_key_for_file(fpath, sig)
    cpath = _cache_path_for_key(key)
    if os.path.exists(cpath):
        return np.load(cpath)
    feat = _features_from_file(fpath)
    tmp = cpath + ".tmp"
    np.save(tmp, feat)
    os.replace(tmp, cpath)
    return feat


def build_feature_matrix(files):
    n = len(files)
    X = np.empty((n, 33), dtype=np.float32)

    import multiprocessing as mp

    workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    chunksize = 64  # reduce scheduling overhead

    if workers == 1:
        for i, f in enumerate(files):
            X[i] = _load_or_compute_feature(f)
        return X

    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
        for i, feat in enumerate(
            pool.imap(_load_or_compute_feature, files, chunksize=chunksize)
        ):
            X[i] = feat
    return X


X_train = build_feature_matrix(train_files)
y_train = train_labels_aligned["target"].values.astype(np.int64)
X_test = build_feature_matrix(test_files_ordered)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "pos_rate:", y_train.mean())
print("X_test:", X_test.shape, "sample_sub:", sample_sub.shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/321638552.py", line 239, in _load_or_compute_feature
    os.replace(tmp, cpath)
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/seti_feat_cache/0000799a2b2c42d.npy__838784__1755236214__v3_quantile_linear_mp_cache_33.npy.tmp' -> '/kaggle/working/seti_feat_cache/0000799a2b2c42d.npy__838784__1755236214__v3_quantile_linear_mp_cache_33.npy'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/321638552.py in <cell line: 0>()
    267 
    268 
--> 269 X_train = build_feature_matrix(train_files)
    270 y_train = train_labels_aligned["target"].values.astype(np.int64)
    271 X_test = build_feature_matrix(test_files_ordered)

/tmp/ipykernel_11/321638552.py in build_feature_matrix(files)
    260     ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    261     with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
--> 262         for i, feat in enumerate(
    263             pool.imap(_load_or_compute_feature, files, chunksize=chunksize)
    264         ):

/usr/lib/python3.11/multiprocessing/pool.py in <genexpr>(.0)
    421                     result._set_length
    422                 ))
--> 423             return (item for chunk in result for item in chunk)
    424 
    425     def imap_unordered(self, func, iterable, chunksize=1):

/usr/lib/python3.11/multiprocessing/pool.py in next(self, timeout)
    871         if success:
    872             return value
--> 873         raise value
    874 
    875     __next__ = next                    # XXX

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/seti_feat_cache/0000799a2b2c42d.npy__838784__1755236214__v3_quantile_linear_mp_cache_33.npy.tmp' -> '/kaggle/working/seti_feat_cache/0000799a2b2c42d.npy__838784__1755236214__v3_quantile_linear_mp_cache_33.npy'

## === cell 2
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(X_train), dtype=np.float32)
test_logit_sum = np.zeros(len(X_test), dtype=np.float64)


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[tr_idx])
    X_va = scaler.transform(X_train[va_idx])

    X_te = scaler.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        n_jobs=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_tr, y_train[tr_idx])

    oof[va_idx] = model.predict_proba(X_va)[:, 1].astype(np.float32)
    test_logit_sum += model.decision_function(X_te).astype(np.float64) / skf.n_splits

test_pred = _sigmoid(test_logit_sum).astype(np.float32)

print("OOF pred summary:", float(oof.min()), float(oof.mean()), float(oof.max()))
print(
    "Test pred summary:",
    float(test_pred.min()),
    float(test_pred.mean()),
    float(test_pred.max()),
)
print("OOF AUC:", float(roc_auc_score(y_train, oof)))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3168075739.py in <cell line: 0>()
      1 skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
      2 
----> 3 oof = np.zeros(len(X_train), dtype=np.float32)
      4 test_logit_sum = np.zeros(len(X_test), dtype=np.float64)
      5 

NameError: name 'X_train' is not defined

## === cell 3
sub = sample_sub.copy()

sub["target"] = test_pred.astype(np.float32)
sub["target"] = sub["target"].clip(0.0, 1.0)

if len(sub) != len(sample_sub):
    raise ValueError("Submission row count mismatch.")
if sub["target"].isna().any():
    raise ValueError("Submission contains NaN predictions.")

print(sub.head())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1927242861.py in <cell line: 0>()
      1 sub = sample_sub.copy()
      2 
----> 3 sub["target"] = test_pred.astype(np.float32)
      4 sub["target"] = sub["target"].clip(0.0, 1.0)
      5 

NameError: name 'test_pred' is not defined

## === cell 4
data6 = sub



## === cell 5
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print(data6.head())
