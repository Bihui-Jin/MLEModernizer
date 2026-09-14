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

0.75713

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48796) has done: 'To cut the runtime we eliminate per‑item dictionary look‑ups and the overhead of lambda functions, and we run the file‑loading work in a process pool (which can read many files in parallel without being limited by Python’s GIL).  We also force NumPy to use a single thread per process to avoid oversubscribing the CPU.  The core logic – loading each .npy, computing its mean, and training/predicting with a logistic regression – remains unchanged.'
- What this solution (achieved 0.51911) has done: 'I fix the infinity/large‑value issue by ensuring the feature matrices contain only finite numbers and by keeping them as float64 (the default expected by scikit‑learn). After cleaning the training data the model fit, and the test predictions be generated so a proper `submission.csv` is written.'
- What this solution (achieved 0.49342) has done: 'I fixed the typo that prevented the test feature extraction (`exexecutor` → `executor`) and consequently the creation of `test_pred`. This resolves the NameError chain and ensures the pipeline writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.49867) has done: 'The changes switch from heavyweight process‑based parallelism to lightweight thread‑based parallelism, which is better for the I/O‑bound task of loading many small .npy files.  A larger chunksize and pre‑allocation of the feature matrix also cut down on Python‑level overhead while keeping the exact same feature calculations, scaling, and model‑training logic unchanged.'
- What this solution (achieved 0.50297) has done: 'I add simple but potentially informative statistics (overall minimum and per‑slot minima) to the feature vector and resize the feature matrices accordingly. These extra features keep the original logic unchanged while giving the logistic regression a richer representation, which should raise the AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import concurrent.futures  # retain import for ThreadPoolExecutor

os.environ["OMP_NUM_THREADS"] = "1"

TRAIN_DIR = Path("/kaggle/input/train")
TEST_DIR = Path("/kaggle/input/test")
TRAIN_LABELS_PATH = Path("/kaggle/input/train_labels.csv")
SAMPLE_SUB_PATH = Path("/kaggle/input/sample_submission.csv")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def build_id_path_map(root_dir: Path) -> dict:
    """Create a deterministic mapping from file stem (id) to full Path."""
    id_path = {}
    for p in root_dir.rglob("*.npy"):
        id_path[p.stem] = p
    return id_path


def compute_features_from_path(path: Path) -> np.ndarray:
    """
    Load a .npy file (shape (6, 273, 256)) and compute the original 39 statistics.
    Uses memory‑mapping to avoid copying the full array into memory.
    Non‑finite values are replaced with 0.
    Returns a NumPy array of shape (39,) with dtype float64.
    """
    arr = np.load(path, mmap_mode="r")
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    overall_mean = arr.mean()
    overall_std = arr.std()
    overall_max = arr.max()
    overall_min = arr.min()
    overall_median = np.median(arr)
    overall_sum = arr.sum()
    overall_range = overall_max - overall_min
    overall_p5 = np.percentile(arr, 5)
    overall_p95 = np.percentile(arr, 95)

    slot_flat = arr.reshape(6, -1)
    slot_means = slot_flat.mean(axis=1)
    slot_stds = slot_flat.std(axis=1)
    slot_maxs = slot_flat.max(axis=1)
    slot_mins = slot_flat.min(axis=1)
    slot_sums = slot_flat.sum(axis=1)

    feats = np.empty(39, dtype=np.float64)
    feats[0] = overall_mean
    feats[1] = overall_std
    feats[2] = overall_max
    feats[3] = overall_min
    feats[4] = overall_median
    feats[5] = overall_sum
    feats[6] = overall_range
    feats[7] = overall_p5
    feats[8] = overall_p95
    feats[9:15] = slot_means  # 6 values
    feats[15:21] = slot_stds  # 6 values
    feats[21:27] = slot_maxs  # 6 values
    feats[27:33] = slot_mins  # 6 values
    feats[33:39] = slot_sums  # 6 values
    return feats


def compute_features_from_path_safe(path):
    """Return a zero‑filled feature vector when the path is missing."""
    if path is None:
        return np.zeros(39, dtype=np.float64)
    return compute_features_from_path(path)


max_workers = min(os.cpu_count() or 4, 4)  # cap at 4 threads
chunksize = 256  # process many files per task




## === cell 1
train_path_map = build_id_path_map(TRAIN_DIR)

train_paths = [train_path_map.get(id_str) for id_str in train_labels["id"]]

train_X = np.empty((len(train_paths), 39), dtype=np.float64)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i, feats in enumerate(
        executor.map(compute_features_from_path_safe, train_paths, chunksize=chunksize)
    ):
        train_X[i] = feats

train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
train_y = train_labels["target"].values

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2041299445.py in <cell line: 0>()
      6 
      7 with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
----> 8     for i, feats in enumerate(
      9         executor.map(compute_features_from_path_safe, train_paths, chunksize=chunksize)
     10     ):

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/2198945111.py in compute_features_from_path_safe(path)
     80     if path is None:
     81         return np.zeros(39, dtype=np.float64)
---> 82     return compute_features_from_path(path)
     83 
     84 

/tmp/ipykernel_11/2198945111.py in compute_features_from_path(path)
     37     arr = np.load(path, mmap_mode="r")
     38     # Replace NaNs / Infs in‑place where possible.
---> 39     np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
     40 
     41     # Overall statistics (single pass per reduction).

/usr/local/lib/python3.11/dist-packages/numpy/lib/type_check.py in nan_to_num(x, copy, nan, posinf, neginf)
    515         idx_posinf = isposinf(d)
    516         idx_neginf = isneginf(d)
--> 517         _nx.copyto(d, nan, where=idx_nan)
    518         _nx.copyto(d, maxf, where=idx_posinf)
    519         _nx.copyto(d, minf, where=idx_neginf)

ValueError: assignment destination is read-only

## === cell 2
model = LogisticRegression(
    C=10.0, max_iter=1000, solver="lbfgs", class_weight="balanced"
)
model.fit(train_X, train_y)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/952276469.py in <cell line: 0>()
      2     C=10.0, max_iter=1000, solver="lbfgs", class_weight="balanced"
      3 )
----> 4 model.fit(train_X, train_y)
      5 
      6 

NameError: name 'train_y' is not defined

## === cell 3
test_path_map = build_id_path_map(TEST_DIR)

test_ids = sample_sub["id"].values
test_paths = [test_path_map.get(id_str) for id_str in test_ids]

test_X = np.empty((len(test_paths), 39), dtype=np.float64)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i, feats in enumerate(
        executor.map(compute_features_from_path_safe, test_paths, chunksize=chunksize)
    ):
        test_X[i] = feats

test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
test_X = scaler.transform(test_X)

test_pred = model.predict_proba(test_X)[:, 1]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3067344687.py in <cell line: 0>()
      7 
      8 with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
----> 9     for i, feats in enumerate(
     10         executor.map(compute_features_from_path_safe, test_paths, chunksize=chunksize)
     11     ):

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/2198945111.py in compute_features_from_path_safe(path)
     80     if path is None:
     81         return np.zeros(39, dtype=np.float64)
---> 82     return compute_features_from_path(path)
     83 
     84 

/tmp/ipykernel_11/2198945111.py in compute_features_from_path(path)
     37     arr = np.load(path, mmap_mode="r")
     38     # Replace NaNs / Infs in‑place where possible.
---> 39     np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
     40 
     41     # Overall statistics (single pass per reduction).

/usr/local/lib/python3.11/dist-packages/numpy/lib/type_check.py in nan_to_num(x, copy, nan, posinf, neginf)
    515         idx_posinf = isposinf(d)
    516         idx_neginf = isneginf(d)
--> 517         _nx.copyto(d, nan, where=idx_nan)
    518         _nx.copyto(d, maxf, where=idx_posinf)
    519         _nx.copyto(d, minf, where=idx_neginf)

ValueError: assignment destination is read-only

## === cell 4
submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1102402944.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_ids, "target": test_pred})
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'test_pred' is not defined
