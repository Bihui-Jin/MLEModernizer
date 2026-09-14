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

0.7507452893851788

# 6. Current score

0.49503

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix replaces the missing external submissions with the provided `sample_submission.csv`, computes a simple baseline prediction using the overall mean target from the training labels, and writes a valid `submission.csv`. This resolves the file‑not‑found errors and guarantees a correctly formatted submission.'
- What this solution (achieved 0.48188) has done: 'I add a second simple feature (the mean intensity of the three “A” observations) and combine its scaled value with the existing max‑intensity feature. This keeps the original logic but gives the model a bit more signal, and I use the average of the two scaled features as the final probability. Missing files still fall back to the overall mean probability so the script always writes a valid `submission.csv`.'
- What this solution (achieved 0.48909) has done: 'I add a third simple feature (overall mean intensity), scale it alongside the existing two features, and replace the naïve equal‑weight averaging with a tiny linear regression fitted on the training data. This keeps the original feature extraction logic but gives a slightly better calibrated probability, which should raise the ROC‑AUC toward the target while still falling back to the global mean when a file is missing.'
- What this solution (achieved 0.49106) has done: 'I enrich the feature set with two easy‑to‑compute statistics (standard deviation of the snippet and the maximum intensity among the three “A” observations) and add squared terms of the three original scaled features. The linear‑regression fitting stays the same (just on a larger feature matrix), and the test‑time prediction is updated to use the new coefficients. These minimal, mathematically‑sound extensions are expected to raise the ROC‑AUC toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.49148) has done: 'The changes introduce parallel processing for feature extraction on the training set and for probability prediction on the test set using ThreadPoolExecutor, which speeds up the many file‑load operations without altering any logic, scaling, or model fitting. Feature‑scaling and regression calculations remain identical; only the loops are replaced by concurrent maps that keep the original order. This keeps results deterministic while fitting comfortably inside the 600‑second limit.'
- What this solution (achieved 0.49503) has done: 'I add a few intuitive derived statistics (differences and ratios between A and non‑A observations), scale them with the same min‑max approach, and include their squared terms in the linear‑regression model. This keeps the overall pipeline unchanged while giving the model extra signal that should raise the ROC‑AUC toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

train_dir = "../input/train"
test_dir = "../input/test"
train_labels_path = "../input/train_labels.csv"
sample_submission_path = "../input/sample_submission.csv"

train_labels = pd.read_csv(train_labels_path)


def build_id_path_map(base_dir):
    id_path = {}
    for folder in map(str, range(16)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.endswith(".npy"):
                id_str = fname[:-4]  # strip .npy
                id_path[id_str] = os.path.join(folder_path, fname)
    return id_path


train_path_map = build_id_path_map(train_dir)
test_path_map = build_id_path_map(test_dir)


def load_snippet(id_str):
    """
    Load the numpy snippet for the given id from the pre‑scanned dictionaries.
    Returns the array with shape (6, 273, 256) or None if the file is missing.
    """
    path = train_path_map.get(id_str) or test_path_map.get(id_str)
    if path is None:
        return None
    return np.load(path, mmap_mode="r")


def extract_features(arr):
    """
    Compute basic and “A‑vs‑non‑A” statistics:
    - max_intensity : overall max pixel value
    - mean_A        : mean over A positions (0,2,4)
    - max_A         : max over A positions
    - mean_nonA     : mean over non‑A positions (1,3,5)
    - max_nonA      : max over non‑A positions
    - overall_mean  : mean over whole snippet
    - std_intensity : std over whole snippet
    """
    max_intensity = float(arr.max())
    a_positions = arr[[0, 2, 4], :, :]
    b_positions = arr[[1, 3, 5], :, :]  # non‑A observations
    mean_A = float(a_positions.mean())
    max_A = float(a_positions.max())
    mean_nonA = float(b_positions.mean())
    max_nonA = float(b_positions.max())
    overall_mean = float(arr.mean())
    std_intensity = float(arr.std())
    return (
        max_intensity,
        mean_A,
        max_A,
        mean_nonA,
        max_nonA,
        overall_mean,
        std_intensity,
    )


def _feat_worker(id_str):
    arr = load_snippet(id_str)
    if arr is None:
        return (np.nan,) * 7
    else:
        return extract_features(arr)


ids = train_labels["id"].tolist()
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(executor.map(_feat_worker, ids, chunksize=100))

feature_names = [
    "max_intensity",
    "mean_A",
    "max_A",
    "mean_nonA",
    "max_nonA",
    "overall_mean",
    "std_intensity",
]
features = {name: [] for name in feature_names}
for res in results:
    for name, val in zip(feature_names, res):
        features[name].append(val)

for col, vals in features.items():
    train_labels[col] = vals

train_labels = train_labels.dropna(
    subset=[
        "max_intensity",
        "mean_A",
        "max_A",
        "mean_nonA",
        "max_nonA",
        "overall_mean",
        "std_intensity",
    ]
).reset_index(drop=True)

feat_min_max = train_labels["max_intensity"].min()
feat_max_max = train_labels["max_intensity"].max()
feat_min_a = train_labels["mean_A"].min()
feat_max_a = train_labels["mean_A"].max()
feat_min_maxA = train_labels["max_A"].min()
feat_max_maxA = train_labels["max_A"].max()
feat_min_nonA = train_labels["mean_nonA"].min()
feat_max_nonA = train_labels["mean_nonA"].max()
feat_min_maxNonA = train_labels["max_nonA"].min()
feat_max_maxNonA = train_labels["max_nonA"].max()
feat_min_over = train_labels["overall_mean"].min()
feat_max_over = train_labels["overall_mean"].max()
feat_min_std = train_labels["std_intensity"].min()
feat_max_std = train_labels["std_intensity"].max()

train_labels["scaled_max"] = (train_labels["max_intensity"] - feat_min_max) / (
    feat_max_max - feat_min_max
)
train_labels["scaled_meanA"] = (train_labels["mean_A"] - feat_min_a) / (
    feat_max_a - feat_min_a
)
train_labels["scaled_maxA"] = (train_labels["max_A"] - feat_min_maxA) / (
    feat_max_maxA - feat_min_maxA
)
train_labels["scaled_meanNonA"] = (train_labels["mean_nonA"] - feat_min_nonA) / (
    feat_max_nonA - feat_min_nonA
)
train_labels["scaled_maxNonA"] = (train_labels["max_nonA"] - feat_min_maxNonA) / (
    feat_max_maxNonA - feat_min_maxNonA
)
train_labels["scaled_overall"] = (train_labels["overall_mean"] - feat_min_over) / (
    feat_max_over - feat_min_over
)
train_labels["scaled_std"] = (train_labels["std_intensity"] - feat_min_std) / (
    feat_max_std - feat_min_std
)

train_labels["diff_max"] = train_labels["max_A"] - train_labels["max_nonA"]
train_labels["diff_mean"] = train_labels["mean_A"] - train_labels["mean_nonA"]
eps = 1e-6
train_labels["ratio_max"] = train_labels["max_A"] / (train_labels["max_nonA"] + eps)
train_labels["ratio_mean"] = train_labels["mean_A"] / (train_labels["mean_nonA"] + eps)

feat_min_diff_max = train_labels["diff_max"].min()
feat_max_diff_max = train_labels["diff_max"].max()
feat_min_diff_mean = train_labels["diff_mean"].min()
feat_max_diff_mean = train_labels["diff_mean"].max()
feat_min_ratio_max = train_labels["ratio_max"].min()
feat_max_ratio_max = train_labels["ratio_max"].max()
feat_min_ratio_mean = train_labels["ratio_mean"].min()
feat_max_ratio_mean = train_labels["ratio_mean"].max()

train_labels["scaled_diff_max"] = (train_labels["diff_max"] - feat_min_diff_max) / (
    feat_max_diff_max - feat_min_diff_max
)
train_labels["scaled_diff_mean"] = (train_labels["diff_mean"] - feat_min_diff_mean) / (
    feat_max_diff_mean - feat_min_diff_mean
)
train_labels["scaled_ratio_max"] = (train_labels["ratio_max"] - feat_min_ratio_max) / (
    feat_max_ratio_max - feat_min_ratio_max
)
train_labels["scaled_ratio_mean"] = (
    train_labels["ratio_mean"] - feat_min_ratio_mean
) / (feat_max_ratio_mean - feat_min_ratio_mean)

for col in [
    "scaled_max",
    "scaled_meanA",
    "scaled_maxA",
    "scaled_meanNonA",
    "scaled_maxNonA",
    "scaled_overall",
    "scaled_std",
    "scaled_diff_max",
    "scaled_diff_mean",
    "scaled_ratio_max",
    "scaled_ratio_mean",
]:
    if train_labels[col].isnull().any():
        train_labels[col] = 0.5

train_labels["scaled_max_sq"] = train_labels["scaled_max"] ** 2
train_labels["scaled_meanA_sq"] = train_labels["scaled_meanA"] ** 2
train_labels["scaled_overall_sq"] = train_labels["scaled_overall"] ** 2
train_labels["scaled_std_sq"] = train_labels["scaled_std"] ** 2
train_labels["scaled_diff_max_sq"] = train_labels["scaled_diff_max"] ** 2
train_labels["scaled_diff_mean_sq"] = train_labels["scaled_diff_mean"] ** 2
train_labels["scaled_ratio_max_sq"] = train_labels["scaled_ratio_max"] ** 2
train_labels["scaled_ratio_mean_sq"] = train_labels["scaled_ratio_mean"] ** 2

feature_cols = [
    "scaled_max",
    "scaled_meanA",
    "scaled_maxA",
    "scaled_meanNonA",
    "scaled_maxNonA",
    "scaled_overall",
    "scaled_std",
    "scaled_diff_max",
    "scaled_diff_mean",
    "scaled_ratio_max",
    "scaled_ratio_mean",
    "scaled_max_sq",
    "scaled_meanA_sq",
    "scaled_overall_sq",
    "scaled_std_sq",
    "scaled_diff_max_sq",
    "scaled_diff_mean_sq",
    "scaled_ratio_max_sq",
    "scaled_ratio_mean_sq",
]
X = train_labels[feature_cols].values
X_int = np.column_stack([np.ones(X.shape[0]), X])  # intercept
y = train_labels["target"].values

coef = np.linalg.lstsq(X_int, y, rcond=None)[0]

train_pred = np.clip(X_int @ coef, 0.0, 1.0)
train_labels["pred_prob"] = train_pred




## === cell 1
sub_df = pd.read_csv(sample_submission_path)

global_mean_prob = train_labels["pred_prob"].mean()


def _predict_worker(fid):
    arr = load_snippet(fid)
    if arr is None:
        return global_mean_prob
    (
        m_max,
        m_a,
        m_max_a,
        m_nonA,
        m_max_nonA,
        m_overall,
        m_std,
    ) = extract_features(arr)

    s_max = (
        (m_max - feat_min_max) / (feat_max_max - feat_min_max)
        if feat_max_max > feat_min_max
        else 0.5
    )
    s_a = (
        (m_a - feat_min_a) / (feat_max_a - feat_min_a)
        if feat_max_a > feat_min_a
        else 0.5
    )
    s_maxA = (
        (m_max_a - feat_min_maxA) / (feat_max_maxA - feat_min_maxA)
        if feat_max_maxA > feat_min_maxA
        else 0.5
    )
    s_nonA = (
        (m_nonA - feat_min_nonA) / (feat_max_nonA - feat_min_nonA)
        if feat_max_nonA > feat_min_nonA
        else 0.5
    )
    s_maxNonA = (
        (m_max_nonA - feat_min_maxNonA) / (feat_max_maxNonA - feat_min_maxNonA)
        if feat_max_maxNonA > feat_min_maxNonA
        else 0.5
    )
    s_over = (
        (m_overall - feat_min_over) / (feat_max_over - feat_min_over)
        if feat_max_over > feat_min_over
        else 0.5
    )
    s_std = (
        (m_std - feat_min_std) / (feat_max_std - feat_min_std)
        if feat_max_std > feat_min_std
        else 0.5
    )

    diff_max = m_max_a - m_max_nonA
    diff_mean = m_a - m_nonA
    eps = 1e-6
    ratio_max = m_max_a / (m_max_nonA + eps)
    ratio_mean = m_a / (m_nonA + eps)

    s_diff_max = (
        (diff_max - feat_min_diff_max) / (feat_max_diff_max - feat_min_diff_max)
        if feat_max_diff_max > feat_min_diff_max
        else 0.5
    )
    s_diff_mean = (
        (diff_mean - feat_min_diff_mean) / (feat_max_diff_mean - feat_min_diff_mean)
        if feat_max_diff_mean > feat_min_diff_mean
        else 0.5
    )
    s_ratio_max = (
        (ratio_max - feat_min_ratio_max) / (feat_max_ratio_max - feat_min_ratio_max)
        if feat_max_ratio_max > feat_min_ratio_max
        else 0.5
    )
    s_ratio_mean = (
        (ratio_mean - feat_min_ratio_mean) / (feat_max_ratio_mean - feat_min_ratio_mean)
        if feat_max_ratio_mean > feat_min_ratio_mean
        else 0.5
    )

    vec = np.array(
        [
            1.0,
            s_max,
            s_a,
            s_maxA,
            s_nonA,
            s_maxNonA,
            s_over,
            s_std,
            s_diff_max,
            s_diff_mean,
            s_ratio_max,
            s_ratio_mean,
            s_max**2,
            s_a**2,
            s_over**2,
            s_std**2,
            s_diff_max**2,
            s_diff_mean**2,
            s_ratio_max**2,
            s_ratio_mean**2,
        ]
    )
    prob = float(np.clip(vec @ coef, 0.0, 1.0))
    return prob


ids_test = sub_df["id"].tolist()
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    test_probs = list(executor.map(_predict_worker, ids_test, chunksize=100))

sub_df["target"] = test_probs




## === cell 2
output_path = "submission.csv"
sub_df.to_csv(output_path, index=False)
