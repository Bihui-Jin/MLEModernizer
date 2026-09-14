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
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier  # new import
from concurrent.futures import ThreadPoolExecutor  # use threads instead of processes



## === cell 1
BASE_INPUT = Path("/kaggle/input")
if not (BASE_INPUT / "seti-breakthrough-listen").exists():
    BASE_INPUT = Path("/kaggle/data")
TRAIN_DIR = BASE_INPUT / "seti-breakthrough-listen" / "train"
TEST_DIR = BASE_INPUT / "seti-breakthrough-listen" / "test"
TRAIN_LABELS_PATH = BASE_INPUT / "seti-breakthrough-listen" / "train_labels.csv"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)




## === cell 2
def compute_stats(file_path: Path):
    """
    Load a .npy snippet (memory‑mapped) and compute robust statistics.
    Returns 20 features: the original 18 plus log‑scaled sum and range.
    """
    arr = np.load(file_path, mmap_mode="r")
    arr = arr.astype(np.float32, copy=False)

    mean = arr.mean()
    std = arr.std()
    max_val = arr.max()
    min_val = arr.min()
    median = np.median(arr)
    p25 = np.percentile(arr, 25)
    p75 = np.percentile(arr, 75)
    p90 = np.percentile(arr, 90)
    rng = max_val - min_val
    total_sum = arr.sum()

    log_sum = np.log1p(total_sum)
    log_range = np.log1p(rng)

    reshaped = arr.reshape(6, -1)  # (6, 273*256)

    slice_means = reshaped.mean(axis=1)
    slice_std = slice_means.std()
    slice_range = slice_means.max() - slice_means.min()

    slice_max = reshaped.max(axis=1)
    slice_max_std = slice_max.std()
    slice_max_range = slice_max.max() - slice_max.min()

    slice_min = reshaped.min(axis=1)
    slice_min_std = slice_min.std()
    slice_min_range = slice_min.max() - slice_min.min()

    slice_medians = np.median(reshaped, axis=1)
    slice_median_std = slice_medians.std()
    slice_median_range = slice_medians.max() - slice_medians.min()

    return np.array(
        [
            mean,
            std,
            max_val,
            min_val,
            p75,
            median,
            p25,
            p90,
            rng,
            total_sum,
            slice_std,
            slice_range,
            slice_max_std,
            slice_max_range,
            slice_min_std,
            slice_min_range,
            slice_median_std,
            slice_median_range,
            log_sum,
            log_range,
        ],
        dtype=np.float32,
    )


def compute_stats_parallel(paths):
    """
    Parallel computation using a thread pool (NumPy releases the GIL).
    Returns a tuple of NumPy arrays, one per statistic, preserving order.
    """
    cpu = os.cpu_count() or 1
    max_workers = min(cpu, 8)  # limit to avoid excessive thread contention
    chunksize = 256
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_stats, paths, chunksize=chunksize))
    stacked = np.vstack(results)
    return tuple(stacked[:, i] for i in range(stacked.shape[1]))




## === cell 3
train_files = list(TRAIN_DIR.rglob("*.npy"))
train_ids = [p.stem for p in train_files]

(
    train_means,
    train_stds,
    train_maxs,
    train_mins,
    train_p75s,
    train_medians,
    train_p25s,
    train_p90s,
    train_ranges,
    train_sums,
    train_slice_stds,
    train_slice_ranges,
    train_slice_max_stds,
    train_slice_max_ranges,
    train_slice_min_stds,
    train_slice_min_ranges,
    train_slice_median_stds,
    train_slice_median_ranges,
    train_log_sums,
    train_log_ranges,
) = compute_stats_parallel(train_files)

train_feat_df = pd.DataFrame(
    {
        "id": train_ids,
        "mean_intensity": train_means,
        "std_intensity": train_stds,
        "max_intensity": train_maxs,
        "min_intensity": train_mins,
        "p75_intensity": train_p75s,
        "median_intensity": train_medians,
        "p25_intensity": train_p25s,
        "p90_intensity": train_p90s,
        "range_intensity": train_ranges,
        "sum_intensity": train_sums,
        "slice_std_intensity": train_slice_stds,
        "slice_range_intensity": train_slice_ranges,
        "slice_max_std_intensity": train_slice_max_stds,
        "slice_max_range_intensity": train_slice_max_ranges,
        "slice_min_std_intensity": train_slice_min_stds,
        "slice_min_range_intensity": train_slice_min_ranges,
        "slice_median_std_intensity": train_slice_median_stds,
        "slice_median_range_intensity": train_slice_median_ranges,
        "log_sum_intensity": train_log_sums,
        "log_range_intensity": train_log_ranges,
    }
)
train_feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_feat_df.fillna(0.0, inplace=True)

train_data = train_feat_df.merge(train_labels, on="id")



## === cell 4
feature_cols = [
    "mean_intensity",
    "std_intensity",
    "max_intensity",
    "min_intensity",
    "p75_intensity",
    "median_intensity",
    "p25_intensity",
    "p90_intensity",
    "range_intensity",
    "sum_intensity",
    "slice_std_intensity",
    "slice_range_intensity",
    "slice_max_std_intensity",
    "slice_max_range_intensity",
    "slice_min_std_intensity",
    "slice_min_range_intensity",
    "slice_median_std_intensity",
    "slice_median_range_intensity",
    "log_sum_intensity",
    "log_range_intensity",
]
X_train = train_data[feature_cols].values
y_train = train_data["target"].values

model = GradientBoostingClassifier(
    n_estimators=500,  # more trees than before
    learning_rate=0.05,
    max_depth=4,  # deeper trees for richer patterns
    subsample=0.9,  # use a larger fraction of data per iteration
    random_state=42,
)
model.fit(X_train, y_train)



## === cell 5
test_files = list(TEST_DIR.rglob("*.npy"))
test_ids = [p.stem for p in test_files]

(
    test_means,
    test_stds,
    test_maxs,
    test_mins,
    test_p75s,
    test_medians,
    test_p25s,
    test_p90s,
    test_ranges,
    test_sums,
    test_slice_stds,
    test_slice_ranges,
    test_slice_max_stds,
    test_slice_max_ranges,
    test_slice_min_stds,
    test_slice_min_ranges,
    test_slice_median_stds,
    test_slice_median_ranges,
    test_log_sums,
    test_log_ranges,
) = compute_stats_parallel(test_files)

test_feat_df = pd.DataFrame(
    {
        "id": test_ids,
        "mean_intensity": test_means,
        "std_intensity": test_stds,
        "max_intensity": test_maxs,
        "min_intensity": test_mins,
        "p75_intensity": test_p75s,
        "median_intensity": test_medians,
        "p25_intensity": test_p25s,
        "p90_intensity": test_p90s,
        "range_intensity": test_ranges,
        "sum_intensity": test_sums,
        "slice_std_intensity": test_slice_stds,
        "slice_range_intensity": test_slice_ranges,
        "slice_max_std_intensity": test_slice_max_stds,
        "slice_max_range_intensity": test_slice_max_ranges,
        "slice_min_std_intensity": test_slice_min_stds,
        "slice_min_range_intensity": test_slice_min_ranges,
        "slice_median_std_intensity": test_slice_median_stds,
        "slice_median_range_intensity": test_slice_median_ranges,
        "log_sum_intensity": test_log_sums,
        "log_range_intensity": test_log_ranges,
    }
)
test_feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_feat_df.fillna(0.0, inplace=True)



## === cell 6
test_probs = model.predict_proba(test_feat_df[feature_cols].values)[:, 1]
submission = pd.DataFrame({"id": test_feat_df["id"], "target": test_probs})
submission_path = Path("/kaggle/working/submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")
