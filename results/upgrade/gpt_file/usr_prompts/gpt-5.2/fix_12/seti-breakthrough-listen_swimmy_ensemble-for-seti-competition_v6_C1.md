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

0.7567603833591164

# 6. Current score

0.51543

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5063) has done: 'I remove the dependency on missing Kaggle Dataset inputs (those `../input/.../submission.csv` files don’t exist in your environment), and replace the broken ensemble with a small, deterministic local baseline that can run only with `numpy/pandas/sklearn` and the provided `.npy` snippets. To keep core semantics (predict `P(target=1)` and optimize for ROC AUC), the fix trains a simple logistic regression on lightweight, hand-crafted features extracted from each `(6,273,256)` snippet (no deep learning, no extra packages). I also ensure IDs are read and predictions are aligned to `sample_submission.csv`, then write a valid `submission.csv` with the required `id,target` columns. This should run end-to-end within the time limit and yield a non-trivial AUC (better than constant predictions), moving you toward the target score.'
- What this solution (achieved 0.51052) has done: 'Most of the timeout is coming from expensive per-file feature extraction across ~60k+ `.npy` files combined, with heavy overhead from `ProcessPoolExecutor` repeatedly calling `np.load` and computing multiple `percentile`/corr operations in Python for each snippet. I keep the exact same feature set and model training semantics, but reduce wall-time by (1) switching to a faster thread-based executor (NumPy releases the GIL in these ops and it avoids process spawn/pickle overhead), (2) using `os.scandir`-based direct filename-to-path mapping without building/sorting huge path lists, and (3) making feature extraction more allocation-efficient while keeping numerically equivalent results. I also add deterministic threading limits for BLAS/OpenMP to avoid oversubscription slowdowns. Paths and outputs remain unchanged.'
- What this solution (achieved 0.52002) has done: 'I keep your exact feature set and logistic-regression core logic, but fix two score-impacting issues: (1) the model is currently trained on ~54k examples while the competition’s full train set is ~54,000 ids but the snippet files on disk are ~60k+; ensuring we use *all* available labeled files (and only those) avoids silent row drops and improves generalization toward your target AUC. (2) I add a small, deterministic per-fold probability calibration step (Platt scaling via `CalibratedClassifierCV` with `cv="prefit"`) applied to the already-trained logistic regression; this preserves the architecture/training loop but typically improves ROC AUC modestly by correcting fold-wise probability scaling. All paths remain unchanged and the script still write a valid `submission.csv` with `id,target`.'
- What this solution (achieved 0.53092) has done: 'The timeout is dominated by slow per-file feature extraction and repeated disk I/O overhead across ~60k `.npy` files, plus very expensive percentile computations on full 273×256 arrays per file. I keep the exact model/training/calibration logic, but make feature extraction provably equivalent while cutting constants: compute all percentiles in one pass, avoid repeated ravel/mean/std work in `corr()`, and eliminate Python-loop overhead in top‑k and per-panel computations. I also replace the nested `ThreadPoolExecutor.map` batching with a single executor pass that preserves order, increases `chunksize`, and uses `os.scandir`-based path listing already present (no logic change) while reducing scheduling overhead. These changes keep the same features/semantics (only negligible FP ordering differences) and should bring runtime under 600s.'
- What this solution (achieved 0.5141) has done: 'I keep your exact feature set and logistic-regression + Platt calibration core logic, but fix a key AUC limiter: calibrating on the same validation fold you later score on leaks information and can hurt generalization to test. The minimal change is to switch calibration to an internal CV on the training fold only (no leakage), while still using sigmoid (Platt scaling) and the same base pipeline. I also make the logistic regression slightly more robust by enabling balanced class weights (still the same model/solver) since this competition is imbalanced and AUC typically benefits; this is a small, legitimate hyperparameter change aimed at improving toward your target. Everything else (paths, feature extraction, CV loop, and submission writing) stays the same.'
- What this solution (achieved 0.50588) has done: 'I keep your exact feature extraction and the same logistic-regression + Platt calibration approach, but make two minimal score-positive changes that typically help ROC AUC without changing core semantics. First, I switch the calibration CV to be stratified and use `ensemble=False` so the calibrator trains a single base model per fold (more stable and usually slightly better calibrated than averaging multiple internal models). Second, I tune only the regularization strength `C` (still LogisticRegression lbfgs, same pipeline) via a tiny fixed grid and pick the best by out-of-fold AUC inside the existing CV loop; this is a small hyperparameter adjustment aimed at moving your 0.5141 toward the 0.7568 target. The script still runs end-to-end and writes a valid `submission.csv` with `id,target` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.51543) has done: 'To move your AUC upward toward the 0.7568 target without changing the core “hand-crafted features → logistic regression” approach, I make two minimal score-relevant adjustments. First, I remove probability calibration (Platt scaling), because ROC AUC is ranking-based and calibration often adds noise/overfitting here—your current scores dropped after adding it. Second, I keep the same CV training loop but tune only `C` on out-of-fold AUC using the plain logistic-regression pipeline, then average fold predictions as before. Everything else (feature extraction, file/path handling, submission alignment/format) stays the same and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
BASE = "/kaggle/data"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
TRAIN_LABELS_PATH = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

if not os.path.exists(TRAIN_LABELS_PATH):
    BASE = "/kaggle/input"
    TRAIN_DIR = os.path.join(BASE, "train")
    TEST_DIR = os.path.join(BASE, "test")
    TRAIN_LABELS_PATH = os.path.join(BASE, "train_labels.csv")
    SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(
    TRAIN_LABELS_PATH
), f"Missing train_labels.csv at {TRAIN_LABELS_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert {"id", "target"}.issubset(train_labels.columns)
assert {"id", "target"}.issubset(sample_sub.columns)
print("train_labels:", train_labels.shape, "sample_submission:", sample_sub.shape)




## === cell 2
def _build_id_to_path(root_dir: str):
    out = {}
    stack = [root_dir]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.name.endswith(".npy"):
                        _id = os.path.splitext(e.name)[0]
                        out[_id] = e.path
        except FileNotFoundError:
            pass
    return out


train_path_by_id = _build_id_to_path(TRAIN_DIR)
test_path_by_id = _build_id_to_path(TEST_DIR)

assert len(train_path_by_id) > 0, f"No .npy files found under {TRAIN_DIR}"
assert len(test_path_by_id) > 0, f"No .npy files found under {TEST_DIR}"

train_labels = train_labels[train_labels["id"].isin(train_path_by_id)].reset_index(
    drop=True
)
train_labels = train_labels.drop_duplicates(subset=["id"]).reset_index(drop=True)

print(
    "Train ids with files:", train_labels.shape[0], "Test files:", len(test_path_by_id)
)




## === cell 3
def extract_features_from_snippet(x: np.ndarray) -> np.ndarray:
    """
    x: np.ndarray shape (6, 273, 256)
    returns: 1D float32 feature vector

    Core logic preserved: hand-crafted numeric features -> logistic regression.
    """
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # on-target
    B = x[[1, 3, 5]]  # off-target

    mean_all = float(x.mean())
    std_all = float(x.std()) + 1e-6
    max_all = float(x.max())
    min_all = float(x.min())

    mean_A = float(A.mean())
    mean_B = float(B.mean())
    std_A = float(A.std()) + 1e-6
    std_B = float(B.std()) + 1e-6

    d = A.mean(axis=0) - B.mean(axis=0)  # (273,256)
    abs_d = np.abs(d)

    l1 = float(abs_d.mean())
    l2 = float(np.sqrt((d * d).mean()))
    d_max = float(d.max())
    d_min = float(d.min())

    row_mean = d.mean(axis=1)  # (273,)
    col_mean = d.mean(axis=0)  # (256,)
    row_std = float(row_mean.std())
    col_std = float(col_mean.std())

    gy = np.diff(d, axis=0)
    gx = np.diff(d, axis=1)
    grad_energy = float((gx * gx).mean() + (gy * gy).mean())

    q90, q95, q99 = np.percentile(abs_d, [90, 95, 99], method="linear")
    q90 = float(q90)
    q95 = float(q95)
    q99 = float(q99)

    frac_q90 = float((abs_d > q90).mean())
    frac_q95 = float((abs_d > q95).mean())
    frac_q99 = float((abs_d > q99).mean())

    abs_row = abs_d.mean(axis=1)
    abs_col = abs_d.mean(axis=0)
    abs_row_max = float(abs_row.max())
    abs_col_max = float(abs_col.max())

    abs_row_p95 = float(np.percentile(abs_row, 95, method="linear"))
    abs_col_p95 = float(np.percentile(abs_col, 95, method="linear"))

    A_panel_mean = A.mean(axis=(1, 2))
    B_panel_mean = B.mean(axis=(1, 2))
    A_panel_std = A.std(axis=(1, 2))
    B_panel_std = B.std(axis=(1, 2))

    A_mean_std = float(A_panel_mean.std())
    B_mean_std = float(B_panel_mean.std())
    A_std_mean = float(A_panel_std.mean())
    B_std_mean = float(B_panel_std.mean())

    def corr(u2d: np.ndarray, v2d: np.ndarray) -> float:
        ur = u2d.reshape(-1).astype(np.float32, copy=False)
        vr = v2d.reshape(-1).astype(np.float32, copy=False)
        um = float(ur.mean())
        vm = float(vr.mean())
        uc = ur - um
        vc = vr - vm
        num = float((uc * vc).mean())
        denom = float(np.sqrt((uc * uc).mean()) * np.sqrt((vc * vc).mean()) + 1e-6)
        return num / denom

    A0, A1, A2 = A[0], A[1], A[2]
    B0, B1, B2 = B[0], B[1], B[2]

    corr_A01 = corr(A0, A1)
    corr_A12 = corr(A1, A2)
    corr_A02 = corr(A0, A2)

    corr_B01 = corr(B0, B1)
    corr_B12 = corr(B1, B2)
    corr_B02 = corr(B0, B2)

    A_mean_img = A.mean(axis=0)
    B_mean_img = B.mean(axis=0)
    abs_A = np.abs(A_mean_img)
    abs_B = np.abs(B_mean_img)

    def topk_mean(img_abs: np.ndarray, k: int) -> float:
        flat = img_abs.reshape(-1)
        if k <= 0:
            return 0.0
        k = min(k, flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        return float(flat[idx].mean())

    A_top64 = topk_mean(abs_A, 64)
    B_top64 = topk_mean(abs_B, 64)
    A_top256 = topk_mean(abs_A, 256)
    B_top256 = topk_mean(abs_B, 256)

    panel_diff_mean = float(A_panel_mean.mean() - B_panel_mean.mean())
    panel_diff_std = float(A_panel_std.mean() - B_panel_std.mean())

    A_panel_top64 = np.empty(3, dtype=np.float32)
    B_panel_top64 = np.empty(3, dtype=np.float32)
    for i in range(3):
        A_panel_top64[i] = topk_mean(np.abs(A[i]), 64)
        B_panel_top64[i] = topk_mean(np.abs(B[i]), 64)

    A_top64_mean = float(A_panel_top64.mean())
    B_top64_mean = float(B_panel_top64.mean())
    A_top64_std = float(A_panel_top64.std())
    B_top64_std = float(B_panel_top64.std())

    feats = np.array(
        [
            mean_all,
            std_all,
            max_all,
            min_all,
            mean_A,
            mean_B,
            (mean_A - mean_B),
            std_A,
            std_B,
            (std_A - std_B),
            l1,
            l2,
            d_max,
            d_min,
            row_std,
            col_std,
            grad_energy,
            q90,
            q95,
            q99,
            frac_q90,
            frac_q95,
            frac_q99,
            abs_row_max,
            abs_col_max,
            abs_row_p95,
            abs_col_p95,
            A_mean_std,
            B_mean_std,
            A_std_mean,
            B_std_mean,
            corr_A01,
            corr_A12,
            corr_A02,
            corr_B01,
            corr_B12,
            corr_B02,
            A_top64,
            B_top64,
            (A_top64 - B_top64),
            A_top256,
            B_top256,
            (A_top256 - B_top256),
            panel_diff_mean,
            panel_diff_std,
            A_top64_mean,
            B_top64_mean,
            (A_top64_mean - B_top64_mean),
            A_top64_std,
            B_top64_std,
            (A_top64_std - B_top64_std),
        ],
        dtype=np.float32,
    )

    feats[~np.isfinite(feats)] = 0.0
    return feats


def _features_from_path(p: str) -> np.ndarray:
    arr = np.load(p, mmap_mode="r")
    return extract_features_from_snippet(arr)


def build_feature_matrix(
    ids, path_by_id, verbose_every=5000, max_workers=None, chunksize=512
):
    from concurrent.futures import ThreadPoolExecutor

    n = len(ids)
    paths = [path_by_id[_id] for _id in ids]

    first = np.load(paths[0], mmap_mode="r")
    f0 = extract_features_from_snippet(first)
    m = f0.shape[0]

    X = np.empty((n, m), dtype=np.float32)
    X[0] = f0
    if n == 1:
        return X

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = min(cpu, 16)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for j, feat in enumerate(
            ex.map(_features_from_path, paths[1:], chunksize=chunksize), start=1
        ):
            X[j] = feat
            if verbose_every and (j % verbose_every == 0):
                print(f"Processed {j}/{n} files")

    return X




## === cell 4
train_ids_ordered = train_labels["id"].tolist()
y = train_labels["target"].astype(int).values

X_train = build_feature_matrix(train_ids_ordered, train_path_by_id, verbose_every=10000)

test_ids_ordered = sample_sub["id"].tolist()
missing_test = [i for i in test_ids_ordered if i not in test_path_by_id]
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Some sample_submission ids not found on disk (showing up to 5): {missing_test[:5]}"
    )

X_test = build_feature_matrix(test_ids_ordered, test_path_by_id, verbose_every=2000)

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y:", y.shape)



## === cell 5
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

C_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

oof = np.zeros(len(X_train), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)

fold_aucs = []
best_Cs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    best_auc = -1.0
    best_model = None
    best_c = None

    for C in C_GRID:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=C,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=800,
                        n_jobs=None,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        model.fit(X_tr, y_tr)
        va_pred_tmp = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, va_pred_tmp)

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_c = C

    oof[va_idx] = best_model.predict_proba(X_va)[:, 1].astype(np.float32)
    test_pred += best_model.predict_proba(X_test)[:, 1].astype(np.float32) / n_splits

    fold_aucs.append(best_auc)
    best_Cs.append(best_c)
    print(f"Fold {fold}/{n_splits} done | best C={best_c} | fold AUC={best_auc:.6f}")

print("Mean CV AUC:", float(np.mean(fold_aucs)), "Best Cs:", best_Cs)



## === cell 6
sub = pd.DataFrame({"id": test_ids_ordered, "target": np.clip(test_pred, 0.0, 1.0)})

assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["id", "target"]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", sub.shape)
print(sub.head())
