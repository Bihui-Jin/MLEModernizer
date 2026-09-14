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

0.7565577900934172

# 6. Current score

0.50196

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the failing imports of nonexistent submission files with a simple, self‑contained routine that loads the provided sample submission, computes a constant baseline prediction (the overall mean target from the training labels), fills the `target` column, and writes a valid `submission.csv`. This eliminates the FileNotFound errors and guarantees a correctly formatted output while keeping the core logic untouched.'
- What this solution (achieved 0.5) has done: 'Implemented parallel, memory‑mapped loading of .npy files and reduced per‑iteration overhead by pre‑computing ID sets. The core feature (mean intensity) and logistic‑regression model remain unchanged, guaranteeing identical predictions while cutting I/O and Python‑loop time dramatically.'
- What this solution (achieved 0.48796) has done: 'The fix adds a deduplication step for test features to prevent duplicate IDs from causing a length mismatch when assigning predictions to the submission DataFrame. This ensures the prediction array matches the submission rows, allowing the script to run end‑to‑end and produce a valid `submission.csv`. No core modeling logic is changed.'
- What this solution (achieved 0.5) has done: 'I added a simple scaling step and a second intensity feature (standard deviation) to improve the logistic‑regression model without changing its overall architecture. The helper now returns both mean and std, the training matrix includes these two columns, the data are standardized with StandardScaler, and the same transformation is applied to the test features. These tweaks are lightweight yet often raise AUC, moving the score closer to the target while preserving the original pipeline logic.'
- What this solution (achieved 0.48801) has done: 'The update narrows file searching to the exact `train` and `test` folders that correspond to the loaded `train_labels.csv`, removing the costly recursive glob over unrelated directories.  It also reuses a single thread‑pool executor and sets the worker count to the full CPU count (capped at 64) to better overlap I/O, while keeping all feature‑extraction logic identical.  All other steps—including scaling, model fitting, and prediction—remain unchanged, preserving exact model behavior.'
- What this solution (achieved 0.48833) has done: 'Implemented lightweight feature expansion and a balanced logistic regression to raise AUC toward the target. The `_load_features` function now returns mean, std, max, min, and range of each snippet, and these columns are incorporated into training and test matrices. A `class_weight='balanced'` setting is added to the existing LogisticRegression to better handle label imbalance, while preserving the original pipeline logic and all I/O handling. This modest enhancement is expected to improve the AUC without altering the core model architecture.'
- What this solution (achieved 0.48651) has done: 'The update switches the heavy per‑file feature extraction from a `ThreadPoolExecutor` that creates a separate future for every file (large scheduling overhead) to a `ProcessPoolExecutor` that streams the file list with a reasonable `chunksize`.  Processing in separate processes avoids the GIL and speeds up the NumPy‑heavy calculations, while limiting workers to the actual CPU count prevents oversubscription.  The rest of the pipeline, model training, scaling and submission generation remains unchanged, preserving exact results.'
- What this solution (achieved 0.50196) has done: 'I add several lightweight intensity‑based statistics (median, on/off medians, on/off max/min and their differences) to the feature extraction function and include them in the numeric column list used for scaling and modeling. These extra features are simple extensions of the existing logic, keep the logistic‑regression pipeline unchanged, and are expected to raise the AUC toward the target without altering the core model architecture.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler




## === cell 1
sample_paths = glob.glob(
    os.path.join("..", "input", "**", "sample_submission.csv"), recursive=True
)
if not sample_paths:
    raise FileNotFoundError("sample_submission.csv not found in any input directory.")
sample_path = sample_paths[0]
submission_df = pd.read_csv(sample_path)




## === cell 2
train_labels_path = os.path.join(
    "..", "input", "seti-breakthrough-listen", "train_labels.csv"
)
if not os.path.exists(train_labels_path):
    alt_paths = glob.glob(
        os.path.join("..", "input", "**", "train_labels.csv"), recursive=True
    )
    if not alt_paths:
        raise FileNotFoundError("train_labels.csv not found.")
    train_labels_path = alt_paths[0]
train_labels = pd.read_csv(train_labels_path)




## === cell 3
base_dir = os.path.dirname(train_labels_path)  # e.g., .../seti-breakthrough-listen
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_id_set = set(train_labels["id"])

train_files = glob.glob(os.path.join(train_dir, "*", "*.npy"))
train_files = [
    fp
    for fp in train_files
    if os.path.splitext(os.path.basename(fp))[0] in train_id_set
]


def _load_features(fp):
    """Load a .npy snippet and return an enriched set of intensity statistics."""
    try:
        arr = np.load(fp, mmap_mode="r")  # shape (6, 273, 256)

        mean_intensity = float(np.mean(arr))
        std_intensity = float(np.std(arr))
        max_intensity = float(np.max(arr))
        min_intensity = float(np.min(arr))
        range_intensity = max_intensity - min_intensity
        median_intensity = float(np.median(arr))

        slice_means = arr.mean(axis=(1, 2))  # (6,)
        slice_stds = arr.std(axis=(1, 2))  # (6,)
        slice_maxs = arr.max(axis=(1, 2))  # (6,)
        slice_mins = arr.min(axis=(1, 2))  # (6,)
        slice_medians = np.median(arr, axis=(1, 2))  # (6,)

        on_idx = [0, 2, 4]
        off_idx = [1, 3, 5]

        mean_on = float(slice_means[on_idx].mean())
        mean_off = float(slice_means[off_idx].mean())
        diff_mean = mean_on - mean_off

        std_on = float(slice_stds[on_idx].mean())
        std_off = float(slice_stds[off_idx].mean())
        diff_std = std_on - std_off

        max_on = float(slice_maxs[on_idx].mean())
        max_off = float(slice_maxs[off_idx].mean())
        diff_max = max_on - max_off

        min_on = float(slice_mins[on_idx].mean())
        min_off = float(slice_mins[off_idx].mean())
        diff_min = min_on - min_off

        median_on = float(slice_medians[on_idx].mean())
        median_off = float(slice_medians[off_idx].mean())
        diff_median = median_on - median_off

        id_ = os.path.splitext(os.path.basename(fp))[0]
        return (
            id_,
            mean_intensity,
            std_intensity,
            max_intensity,
            min_intensity,
            range_intensity,
            median_intensity,
            mean_on,
            mean_off,
            diff_mean,
            std_on,
            std_off,
            diff_std,
            max_on,
            max_off,
            diff_max,
            min_on,
            min_off,
            diff_min,
            median_on,
            median_off,
            diff_median,
        )
    except Exception:
        return None


max_workers = os.cpu_count() or 1
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    train_features_iter = executor.map(_load_features, train_files, chunksize=100)
    train_features = [r for r in train_features_iter if r is not None]

train_feat_df = pd.DataFrame(
    train_features,
    columns=[
        "id",
        "mean_intensity",
        "std_intensity",
        "max_intensity",
        "min_intensity",
        "range_intensity",
        "median_intensity",
        "mean_on",
        "mean_off",
        "diff_mean",
        "std_on",
        "std_off",
        "diff_std",
        "max_on",
        "max_off",
        "diff_max",
        "min_on",
        "min_off",
        "diff_min",
        "median_on",
        "median_off",
        "diff_median",
    ],
)

numeric_cols = [
    "mean_intensity",
    "std_intensity",
    "max_intensity",
    "min_intensity",
    "range_intensity",
    "median_intensity",
    "mean_on",
    "mean_off",
    "diff_mean",
    "std_on",
    "std_off",
    "diff_std",
    "max_on",
    "max_off",
    "diff_max",
    "min_on",
    "min_off",
    "diff_min",
    "median_on",
    "median_off",
    "diff_median",
]

train_feat_df[numeric_cols] = train_feat_df[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)
train_feat_df.fillna(0, inplace=True)

train_data = train_labels.merge(train_feat_df, on="id", how="inner")

if train_data.empty:
    baseline_pred = train_labels["target"].mean()
    submission_df["target"] = baseline_pred
else:
    scaler = StandardScaler()
    X_raw = train_data[numeric_cols].values
    X = scaler.fit_transform(X_raw)
    y = train_data["target"].values

    model = LogisticRegression(
        max_iter=1000, n_jobs=-1, solver="lbfgs", class_weight="balanced"
    )
    model.fit(X, y)

    test_id_set = set(submission_df["id"])
    test_files = glob.glob(os.path.join(test_dir, "*", "*.npy"))
    test_files = [
        fp
        for fp in test_files
        if os.path.splitext(os.path.basename(fp))[0] in test_id_set
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        test_features_iter = executor.map(_load_features, test_files, chunksize=100)
        test_features = [r for r in test_features_iter if r is not None]

    test_feat_df = pd.DataFrame(
        test_features,
        columns=[
            "id",
            "mean_intensity",
            "std_intensity",
            "max_intensity",
            "min_intensity",
            "range_intensity",
            "median_intensity",
            "mean_on",
            "mean_off",
            "diff_mean",
            "std_on",
            "std_off",
            "diff_std",
            "max_on",
            "max_off",
            "diff_max",
            "min_on",
            "min_off",
            "diff_min",
            "median_on",
            "median_off",
            "diff_median",
        ],
    )
    test_feat_df = test_feat_df.drop_duplicates(subset="id", keep="first")
    test_feat_df[numeric_cols] = test_feat_df[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    test_feat_df.fillna(0, inplace=True)

    merged = submission_df[["id"]].merge(test_feat_df, on="id", how="left")
    overall_means = train_data[numeric_cols].mean()
    merged[numeric_cols] = merged[numeric_cols].fillna(overall_means)

    X_test = scaler.transform(merged[numeric_cols].values)
    preds = model.predict_proba(X_test)[:, 1]
    submission_df["target"] = preds




## === cell 4
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
