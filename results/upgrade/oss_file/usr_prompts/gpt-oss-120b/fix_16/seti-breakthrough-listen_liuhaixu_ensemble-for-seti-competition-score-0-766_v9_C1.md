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

0.7627048215298181

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The script now safely loads any available submission files (skipping missing ones), falls back to the official sample submission, builds the ensemble only from the files that exist, and writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'Implemented a uniform‑weight ensemble and added NaN handling so every ID receives a valid probability. This prevents missing predictions from turning the whole blend into NaNs (which caused the 0.5 AUC) and uses equal contribution from all available submissions, moving the score toward the target.'
- What this solution (achieved 0.48796) has done: 'I add a lightweight “mean‑intensity” model that is trained on a random subset of the training snippets and predicts probabilities for the test set. Its predictions are then merged into the existing ensemble (with a modest weight) so that, even when only the sample submission is available, the final CSV contains a non‑random signal that lifts the AUC toward the target score.'
- What this solution (achieved 0.48796) has done: 'I compute per‑submission AUC on the training labels and set the ensemble weights proportionally (giving zero weight to random‑looking submissions). This keeps the original pipeline while shifting the blend toward better models, which should raise the AUC toward the target.'
- What this solution (achieved 0.48796) has done: 'The fix adds missing imports, corrects path handling for the sample submission and dataset files, and streamlines the ensemble creation so it always produces a valid `submission.csv`. It also safely trains the lightweight mean‑intensity model when training data is available, computes sensible ensemble weights (boosting the model weight when present), and merges all predictions while keeping the required columns and value range.'
- What this solution (achieved 0.5) has done: 'I enrich the lightweight mean‑intensity model by adding a few simple statistical features (max and standard deviation) and increase the training sample size modestly. These extra features give the logistic regression more discriminative power while preserving the original model‑training flow, which should raise the validation AUC toward the target without altering the overall pipeline.'
- What this solution (achieved 0.5) has done: 'The fix adds safety when extracting features from the `.npy` snippets: any NaNs or infinities are replaced with zeros before computing statistics, and rows with non‑finite feature values are skipped. This prevents the logistic regression from receiving invalid inputs, allowing the model to train and produce predictions. The rest of the pipeline (ensemble weighting and CSV output) remains unchanged, so the solution now runs end‑to‑end and can achieve a higher AUC toward the target.'
- What this solution (achieved 0.5) has done: 'The fix adds a proper validation split so the logistic‑regression model’s AUC can be measured on held‑out training data. That AUC is then used to give the model a much larger weight in the final ensemble, while the dummy sample submission keeps only a tiny weight. The code now extracts features safely, trains on 80 % of the sampled training set, evaluates on 20 % to obtain `model_auc`, and builds weights from these scores before blending and writing the submission.'
- What this solution (achieved 0.5) has done: 'I enhance the feature extraction to include per‑cadence slice statistics (mean and std for each of the 6 positions) and increase the training sample size, giving the logistic‑regression model more discriminative power. Then I compute ensemble weights directly from the validation AUCs (without forcing a large weight on the model) so that a better‑performing model receives a proportionally higher contribution, moving the overall AUC toward the target.'
- What this solution (achieved 0.5) has done: 'I parallelize the costly .npy loading and feature computation using a process pool, keeping the exact feature formulas and model training unchanged. By extracting train and test features concurrently, we eliminate the sequential I/O bottleneck while preserving deterministic behavior (same random seed, same feature order). The rest of the pipeline (model fit, ensembling, and CSV output) remains identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import concurrent.futures

train_labels_path = os.path.join("kaggle", "data", "train_labels.csv")
train_dir = os.path.join("kaggle", "data", "train")
test_dir = os.path.join("kaggle", "data", "test")
sample_submission_path = os.path.join("kaggle", "data", "sample_submission.csv")

train_labels = pd.read_csv(train_labels_path)
base_df = pd.read_csv(sample_submission_path)  # contains the test ids
submissions = [base_df.copy()]  # start with the baseline sample submission




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3129435548.py in <cell line: 0>()
     14 
     15 # Load required tables
---> 16 train_labels = pd.read_csv(train_labels_path)
     17 base_df = pd.read_csv(sample_submission_path)  # contains the test ids
     18 submissions = [base_df.copy()]  # start with the baseline sample submission

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'kaggle/data/train_labels.csv'

## === cell 1
def build_id_path_map(root_dir):
    """Return a dict mapping id string to its .npy file path."""
    id_path = {}
    for sub in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        for fname in os.listdir(sub_path):
            if fname.endswith(".npy"):
                id_str = fname[:-4]  # strip .npy
                id_path[id_str] = os.path.join(sub_path, fname)
    return id_path


train_id_path = build_id_path_map(train_dir)
test_id_path = build_id_path_map(test_dir)


def extract_feats(id_str, id_path_map):
    """Load a .npy snippet and compute simple statistics."""
    path = id_path_map.get(id_str)
    if path is None or not os.path.exists(path):
        return None
    try:
        arr = np.load(path)  # shape (6, 273, 256)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        overall_mean = arr.mean()
        overall_max = arr.max()
        overall_min = arr.min()
        overall_std = arr.std()

        resh = arr.reshape(6, -1)
        pos_means = resh.mean(axis=1)
        pos_stds = resh.std(axis=1)
        pos_maxs = resh.max(axis=1)
        pos_mins = resh.min(axis=1)

        feats = (
            [overall_mean, overall_max, overall_min, overall_std]
            + pos_means.tolist()
            + pos_stds.tolist()
            + pos_maxs.tolist()
            + pos_mins.tolist()
        )
        if np.isfinite(feats).all():
            return feats
    except Exception:
        return None
    return None


def _worker(args):
    """Worker for ProcessPool: returns (id, feats) or (id, None)."""
    id_str, path_map = args
    return (id_str, extract_feats(id_str, path_map))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2459050075.py in <cell line: 0>()
     13 
     14 
---> 15 train_id_path = build_id_path_map(train_dir)
     16 test_id_path = build_id_path_map(test_dir)
     17 

/tmp/ipykernel_11/2459050075.py in build_id_path_map(root_dir)
      2     """Return a dict mapping id string to its .npy file path."""
      3     id_path = {}
----> 4     for sub in os.listdir(root_dir):
      5         sub_path = os.path.join(root_dir, sub)
      6         if not os.path.isdir(sub_path):

FileNotFoundError: [Errno 2] No such file or directory: 'kaggle/data/train'

## === cell 2
train_sample = train_labels.sample(frac=1.0, random_state=42)

train_ids = train_sample["id"].tolist()
train_args = [(i, train_id_path) for i in train_ids]
n_jobs = max(1, min(8, os.cpu_count() or 1))

feature_rows, targets = [], []
with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
    for id_str, feats in executor.map(_worker, train_args, chunksize=64):
        if feats is not None:
            feature_rows.append(feats)
            targets.append(
                train_sample.loc[train_sample["id"] == id_str, "target"].values[0]
            )

model_auc = 0.5  # fallback if model cannot be trained
if feature_rows:
    X = np.array(feature_rows)
    y = np.array(targets)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=2.0, solver="lbfgs"
    )
    model.fit(X_tr, y_tr)

    val_pred = model.predict_proba(X_val)[:, 1]
    model_auc = roc_auc_score(y_val, val_pred)

    test_ids = base_df["id"].tolist()
    test_args = [(i, test_id_path) for i in test_ids]

    test_ids_kept, test_features = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for id_str, feats in executor.map(_worker, test_args, chunksize=64):
            if feats is not None:
                test_ids_kept.append(id_str)
                test_features.append(feats)

    if test_features:
        X_test = np.array(test_features)
        probs = model.predict_proba(X_test)[:, 1]
        df_model = pd.DataFrame({"id": test_ids_kept, "target": probs})
        submissions.append(df_model)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/176217208.py in <cell line: 0>()
      1 # Shuffle training labels for reproducibility
----> 2 train_sample = train_labels.sample(frac=1.0, random_state=42)
      3 
      4 train_ids = train_sample["id"].tolist()
      5 train_args = [(i, train_id_path) for i in train_ids]

NameError: name 'train_labels' is not defined

## === cell 3
if len(submissions) > 1:
    weights = np.array([0.0, 1.0])
else:
    weights = np.ones(len(submissions)) / len(submissions)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2957127340.py in <cell line: 0>()
      1 # Determine ensemble weights
----> 2 if len(submissions) > 1:
      3     # Give full weight to the trained model, zero weight to the baseline sample submission
      4     weights = np.array([0.0, 1.0])
      5 else:

NameError: name 'submissions' is not defined

## === cell 4
merged = base_df[["id"]].copy()
merged["target"] = 0.0

for df, w in zip(submissions, weights):
    merged = merged.merge(df, on="id", how="left", suffixes=("", "_tmp"))
    if "target_tmp" in merged.columns:
        merged["target_tmp"].fillna(0.0, inplace=True)
        merged["target"] += w * merged["target_tmp"]
        merged.drop(columns=["target_tmp"], inplace=True)
    else:
        merged["target"] = w * merged["target"]

merged["target"] = merged["target"].clip(0.0, 1.0)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3085836385.py in <cell line: 0>()
----> 1 merged = base_df[["id"]].copy()
      2 merged["target"] = 0.0
      3 
      4 for df, w in zip(submissions, weights):
      5     merged = merged.merge(df, on="id", how="left", suffixes=("", "_tmp"))

NameError: name 'base_df' is not defined

## === cell 5
merged.to_csv("submission.csv", index=False)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/471988406.py in <cell line: 0>()
----> 1 merged.to_csv("submission.csv", index=False)

NameError: name 'merged' is not defined
