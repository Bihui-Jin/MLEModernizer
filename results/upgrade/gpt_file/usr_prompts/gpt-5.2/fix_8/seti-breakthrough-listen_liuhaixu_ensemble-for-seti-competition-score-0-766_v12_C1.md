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

0.7625291002892738

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49203) has done: 'The timeout is dominated by pure-Python feature extraction over ~60k `.npy` files plus repeated heavy reductions (mean/std/abs) per file. I keep the exact same features and model, but remove unnecessary work: avoid `glob` per-id path lookup, reduce the number of full-array passes inside `extract_features_from_npy` by reusing computed intermediates, and preallocate `X`/`X_test` instead of building large Python lists and `vstack`. I also use faster directory scanning to build the id→path maps and iterate with `itertuples()`/NumPy arrays to cut Pandas overhead. These changes are provably equivalent (same arithmetic in the same dtype) and only reduce overhead/duplicate computation.'
- What this solution (achieved 0.49097) has done: 'Your current score (0.492) is far below the target (0.7625), so we should improve generalization with minimal changes while keeping the same feature set and the same LogisticRegression approach. The biggest likely issue is that the default LogisticRegression regularization strength (C=1) is suboptimal for these handcrafted features; tuning only `C` and using a more appropriate `class_weight='balanced'` often yields a large AUC gain without changing core logic. I add a small internal CV over a short list of C values (still the same model, solver, and features) scored by ROC-AUC, then refit on all training data with the best C before generating `submission.csv`. This keeps evaluation semantics intact and should move the score upward toward your target band.'
- What this solution (achieved 0.49125) has done: 'Your current AUC (0.49097) is far below the target (0.76253), so we should make the smallest “same-core-logic” change that typically yields a large AUC jump: prevent data leakage in scaling/C selection and remove the destabilizing effect of `class_weight="balanced"` for an AUC metric (it often harms ranking calibration here). I keep the exact same handcrafted features and LogisticRegression (same solver/training flow), but (1) use a `Pipeline(StandardScaler -> LogisticRegression)` so scaling is always fit only on each training fold, (2) tune only `C` with CV AUC as before, and (3) set `class_weight=None` to improve probability ranking. Everything still runs end-to-end and writes a valid `submission.csv` with `id,target`.'
- What this solution (achieved 0.47797) has done: 'Your current AUC (0.49125) is far below the target (0.76253), so we need a meaningful lift without changing the core “handcrafted features + LogisticRegression” logic. The biggest low-risk gain here is to tune `penalty`/`solver` (still LogisticRegression) and widen the `C` search on a log scale, because your current narrow grid can easily miss a workable regularization regime for these features. I also switch the pipeline scaler to `with_mean=False` to better preserve feature sparsity/scale interactions (while still preventing leakage via the Pipeline), and keep everything else the same. These changes keep the same evaluation semantics and should move the score upward toward your target band.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = "/kaggle/input"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 2
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def extract_features_from_npy(path: str) -> np.ndarray:
    x = np.load(path, mmap_mode="r")  # float16 on disk; mmap reduces IO overhead
    x = x.astype(np.float32, copy=False)

    absx = np.abs(x)

    panel_mean = x.mean(axis=(1, 2))  # (6,)
    panel_std = x.std(axis=(1, 2))  # (6,)
    panel_absmean = absx.mean(axis=(1, 2))
    panel_absmax = absx.max(axis=(1, 2))

    A = x[[0, 2, 4]]
    BCD = x[[1, 3, 5]]

    A_mean = A.mean()
    BCD_mean = BCD.mean()
    A_std = A.std()
    BCD_std = BCD.std()

    mean_diff = A_mean - BCD_mean
    std_diff = A_std - BCD_std

    A_panel_mean = panel_mean[[0, 2, 4]]
    BCD_panel_mean = panel_mean[[1, 3, 5]]
    A_panel_std = panel_std[[0, 2, 4]]
    BCD_panel_std = panel_std[[1, 3, 5]]

    feats_tail = np.array(
        [
            A_mean,
            BCD_mean,
            mean_diff,
            A_std,
            BCD_std,
            std_diff,
            A_panel_mean.mean(),
            BCD_panel_mean.mean(),
            (A_panel_mean - BCD_panel_mean).mean(),
            A_panel_std.mean(),
            BCD_panel_std.mean(),
            (A_panel_std - BCD_panel_std).mean(),
            A_panel_mean.std(),
            BCD_panel_mean.std(),
            A_panel_std.std(),
            BCD_panel_std.std(),
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [panel_mean, panel_std, panel_absmean, panel_absmax, feats_tail]
    ).astype(np.float32, copy=False)

    return feats


def build_id_to_path_map(root_dir: str) -> dict:
    id2p = {}
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            with os.scandir(entry.path) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        _id = f.name[:-4]
                        id2p[_id] = f.path
    return id2p


def extract_features_for_ids(
    ids: np.ndarray, id2path: dict, n_feats: int, workers: int
) -> np.ndarray:
    paths = [id2path[_id] for _id in ids]
    X_out = np.empty((len(paths), n_feats), dtype=np.float32)
    if workers <= 1:
        for i, p in enumerate(paths):
            X_out[i] = extract_features_from_npy(p)
        return X_out

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for i, feats in enumerate(
            ex.map(extract_features_from_npy, paths, chunksize=64)
        ):
            X_out[i] = feats
    return X_out




## === cell 3
train_id_to_path = build_id_to_path_map(TRAIN_DIR)

train_ids = train_labels["id"].to_numpy()
train_targets = train_labels["target"].to_numpy(dtype=np.int64, copy=False)

missing = [_id for _id in train_ids if _id not in train_id_to_path]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} training npy files, e.g. {missing[:5]}"
    )

first_feat = extract_features_from_npy(train_id_to_path[train_ids[0]])
n_feats = first_feat.shape[0]

workers = min(8, (os.cpu_count() or 2))

X = np.empty((train_ids.shape[0], n_feats), dtype=np.float32)
y = train_targets

X[0] = first_feat
if len(train_ids) > 1:
    X[1:] = extract_features_for_ids(train_ids[1:], train_id_to_path, n_feats, workers)

X.shape, y.shape



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2436305573.py in <cell line: 0>()
     23 X[0] = first_feat
     24 if len(train_ids) > 1:
---> 25     X[1:] = extract_features_for_ids(train_ids[1:], train_id_to_path, n_feats, workers)
     26 
     27 X.shape, y.shape

/tmp/ipykernel_11/3563426260.py in extract_features_for_ids(ids, id2path, n_feats, workers)
     92         # executor.map preserves input order => row alignment is unchanged
     93         for i, feats in enumerate(
---> 94             ex.map(extract_features_from_npy, paths, chunksize=64)
     95         ):
     96             X_out[i] = feats

/usr/lib/python3.11/concurrent/futures/process.py in map(self, fn, timeout, chunksize, *iterables)
    835             raise ValueError("chunksize must be >= 1.")
    836 
--> 837         results = super().map(partial(_process_chunk, fn),
    838                               _get_chunks(*iterables, chunksize=chunksize),
    839                               timeout=timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in map(self, fn, timeout, chunksize, *iterables)
    606             end_time = timeout + time.monotonic()
    607 
--> 608         fs = [self.submit(fn, *args) for args in zip(*iterables)]
    609 
    610         # Yield must be hidden in closure so that the futures are submitted

/usr/lib/python3.11/concurrent/futures/_base.py in <listcomp>(.0)
    606             end_time = timeout + time.monotonic()
    607 
--> 608         fs = [self.submit(fn, *args) for args in zip(*iterables)]
    609 
    610         # Yield must be hidden in closure so that the futures are submitted

/usr/lib/python3.11/concurrent/futures/process.py in submit(self, fn, *args, **kwargs)
    806 
    807             if self._safe_to_dynamically_spawn_children:
--> 808                 self._adjust_process_count()
    809             self._start_executor_manager_thread()
    810             return f

/usr/lib/python3.11/concurrent/futures/process.py in _adjust_process_count(self)
    765             # we know a thread is running (self._executor_manager_thread).
    766             #assert self._safe_to_dynamically_spawn_children or not self._executor_manager_thread, 'https://github.com/python/cpython/issues/90622'
--> 767             self._spawn_process()
    768 
    769     def _launch_processes(self):

/usr/lib/python3.11/concurrent/futures/process.py in _spawn_process(self)
    783                   self._initargs,
    784                   self._max_tasks_per_child))
--> 785         p.start()
    786         self._processes[p.pid] = p
    787 

/usr/lib/python3.11/multiprocessing/process.py in start(self)
    119                'daemonic processes are not allowed to have children'
    120         _cleanup()
--> 121         self._popen = self._Popen(self)
    122         self._sentinel = self._popen.sentinel
    123         # Avoid a refcycle if the target function holds an indirect

/usr/lib/python3.11/multiprocessing/context.py in _Popen(process_obj)
    286         def _Popen(process_obj):
    287             from .popen_spawn_posix import Popen
--> 288             return Popen(process_obj)
    289 
    290         @staticmethod

/usr/lib/python3.11/multiprocessing/popen_spawn_posix.py in __init__(self, process_obj)
     30     def __init__(self, process_obj):
     31         self._fds = []
---> 32         super().__init__(process_obj)
     33 
     34     def duplicate_for_child(self, fd):

/usr/lib/python3.11/multiprocessing/popen_fork.py in __init__(self, process_obj)
     17         self.returncode = None
     18         self.finalizer = None
---> 19         self._launch(process_obj)
     20 
     21     def duplicate_for_child(self, fd):

/usr/lib/python3.11/multiprocessing/popen_spawn_posix.py in _launch(self, process_obj)
     45         try:
     46             reduction.dump(prep_data, fp)
---> 47             reduction.dump(process_obj, fp)
     48         finally:
     49             set_spawning_popen(None)

/usr/lib/python3.11/multiprocessing/reduction.py in dump(obj, file, protocol)
     58 def dump(obj, file, protocol=None):
     59     '''Replacement for pickle.dump() using ForkingPickler.'''
---> 60     ForkingPickler(file, protocol).dump(obj)
     61 
     62 #

/usr/lib/python3.11/multiprocessing/connection.py in reduce_connection(conn)
    983 else:
    984     def reduce_connection(conn):
--> 985         df = reduction.DupFd(conn.fileno())
    986         return rebuild_connection, (df, conn.readable, conn.writable)
    987     def rebuild_connection(df, readable, writable):

/usr/lib/python3.11/multiprocessing/connection.py in fileno(self)
    169     def fileno(self):
    170         """File descriptor or handle of the connection"""
--> 171         self._check_closed()
    172         return self._handle
    173 

/usr/lib/python3.11/multiprocessing/connection.py in _check_closed(self)
    135     def _check_closed(self):
    136         if self._handle is None:
--> 137             raise OSError("handle is closed")
    138 
    139     def _check_readable(self):

OSError: handle is closed

## === cell 4
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

C_CANDIDATES = np.logspace(-6, 3, 10).astype(float).tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_auc = -1.0
best_params = None

candidates = [
    ("l2", "lbfgs"),
    ("l2", "liblinear"),
    ("l1", "liblinear"),
]


def oof_auc_for_params(X, y, penalty, solver, C, skf):
    oof = np.empty(X.shape[0], dtype=np.float32)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        X_va = X[va_idx]

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        clf = LogisticRegression(
            solver=solver,
            penalty=penalty,
            max_iter=2000,
            n_jobs=None,
            class_weight=None,
            random_state=42,
            C=C,
        )
        clf.fit(X_tr_s, y_tr)
        oof[va_idx] = clf.predict_proba(X_va_s)[:, 1].astype(np.float32, copy=False)
    return roc_auc_score(y, oof)


for penalty, solver in candidates:
    for C in C_CANDIDATES:
        auc = oof_auc_for_params(X, y, penalty, solver, C, skf)
        print(
            f"penalty={penalty:<2} solver={solver:<9} C={C:<10.4g}  OOF AUC={auc:.6f}"
        )
        if auc > best_auc:
            best_auc = auc
            best_params = (penalty, solver, C)

best_penalty, best_solver, best_C = best_params
print(
    f"Selected penalty={best_penalty}, solver={best_solver}, C={best_C} with OOF AUC={best_auc:.6f}"
)

final_scaler = StandardScaler(with_mean=True, with_std=True)
X_s = final_scaler.fit_transform(X)
final_clf = LogisticRegression(
    solver=best_solver,
    penalty=best_penalty,
    max_iter=2000,
    n_jobs=None,
    class_weight=None,
    random_state=42,
    C=best_C,
)
final_clf.fit(X_s, y)



## === cell 5
test_id_to_path = build_id_to_path_map(TEST_DIR)

test_ids = sample_sub["id"].to_numpy()
missing_test = [_id for _id in test_ids if _id not in test_id_to_path]
if missing_test:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test npy files, e.g. {missing_test[:5]}"
    )

X_test = extract_features_for_ids(test_ids, test_id_to_path, n_feats, workers)
X_test_s = final_scaler.transform(X_test)

test_pred = final_clf.predict_proba(X_test_s)[:, 1].astype(np.float32)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/233771603.py in <cell line: 0>()
      9     )
     10 
---> 11 X_test = extract_features_for_ids(test_ids, test_id_to_path, n_feats, workers)
     12 X_test_s = final_scaler.transform(X_test)
     13 

/tmp/ipykernel_11/3563426260.py in extract_features_for_ids(ids, id2path, n_feats, workers)
     92         # executor.map preserves input order => row alignment is unchanged
     93         for i, feats in enumerate(
---> 94             ex.map(extract_features_from_npy, paths, chunksize=64)
     95         ):
     96             X_out[i] = feats

/usr/lib/python3.11/concurrent/futures/process.py in map(self, fn, timeout, chunksize, *iterables)
    835             raise ValueError("chunksize must be >= 1.")
    836 
--> 837         results = super().map(partial(_process_chunk, fn),
    838                               _get_chunks(*iterables, chunksize=chunksize),
    839                               timeout=timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in map(self, fn, timeout, chunksize, *iterables)
    606             end_time = timeout + time.monotonic()
    607 
--> 608         fs = [self.submit(fn, *args) for args in zip(*iterables)]
    609 
    610         # Yield must be hidden in closure so that the futures are submitted

/usr/lib/python3.11/concurrent/futures/_base.py in <listcomp>(.0)
    606             end_time = timeout + time.monotonic()
    607 
--> 608         fs = [self.submit(fn, *args) for args in zip(*iterables)]
    609 
    610         # Yield must be hidden in closure so that the futures are submitted

/usr/lib/python3.11/concurrent/futures/process.py in submit(self, fn, *args, **kwargs)
    806 
    807             if self._safe_to_dynamically_spawn_children:
--> 808                 self._adjust_process_count()
    809             self._start_executor_manager_thread()
    810             return f

/usr/lib/python3.11/concurrent/futures/process.py in _adjust_process_count(self)
    765             # we know a thread is running (self._executor_manager_thread).
    766             #assert self._safe_to_dynamically_spawn_children or not self._executor_manager_thread, 'https://github.com/python/cpython/issues/90622'
--> 767             self._spawn_process()
    768 
    769     def _launch_processes(self):

/usr/lib/python3.11/concurrent/futures/process.py in _spawn_process(self)
    783                   self._initargs,
    784                   self._max_tasks_per_child))
--> 785         p.start()
    786         self._processes[p.pid] = p
    787 

/usr/lib/python3.11/multiprocessing/process.py in start(self)
    119                'daemonic processes are not allowed to have children'
    120         _cleanup()
--> 121         self._popen = self._Popen(self)
    122         self._sentinel = self._popen.sentinel
    123         # Avoid a refcycle if the target function holds an indirect

/usr/lib/python3.11/multiprocessing/context.py in _Popen(process_obj)
    286         def _Popen(process_obj):
    287             from .popen_spawn_posix import Popen
--> 288             return Popen(process_obj)
    289 
    290         @staticmethod

/usr/lib/python3.11/multiprocessing/popen_spawn_posix.py in __init__(self, process_obj)
     30     def __init__(self, process_obj):
     31         self._fds = []
---> 32         super().__init__(process_obj)
     33 
     34     def duplicate_for_child(self, fd):

/usr/lib/python3.11/multiprocessing/popen_fork.py in __init__(self, process_obj)
     17         self.returncode = None
     18         self.finalizer = None
---> 19         self._launch(process_obj)
     20 
     21     def duplicate_for_child(self, fd):

/usr/lib/python3.11/multiprocessing/popen_spawn_posix.py in _launch(self, process_obj)
     45         try:
     46             reduction.dump(prep_data, fp)
---> 47             reduction.dump(process_obj, fp)
     48         finally:
     49             set_spawning_popen(None)

/usr/lib/python3.11/multiprocessing/reduction.py in dump(obj, file, protocol)
     58 def dump(obj, file, protocol=None):
     59     '''Replacement for pickle.dump() using ForkingPickler.'''
---> 60     ForkingPickler(file, protocol).dump(obj)
     61 
     62 #

/usr/lib/python3.11/multiprocessing/connection.py in reduce_connection(conn)
    983 else:
    984     def reduce_connection(conn):
--> 985         df = reduction.DupFd(conn.fileno())
    986         return rebuild_connection, (df, conn.readable, conn.writable)
    987     def rebuild_connection(df, readable, writable):

/usr/lib/python3.11/multiprocessing/connection.py in fileno(self)
    169     def fileno(self):
    170         """File descriptor or handle of the connection"""
--> 171         self._check_closed()
    172         return self._handle
    173 

/usr/lib/python3.11/multiprocessing/connection.py in _check_closed(self)
    135     def _check_closed(self):
    136         if self._handle is None:
--> 137             raise OSError("handle is closed")
    138 
    139     def _check_readable(self):

OSError: handle is closed
