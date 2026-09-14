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

0.757186373534826

# 6. Current score

0.50492

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix replaces the missing‑file loads with a simple baseline that uses the average training target as the prediction for every test snippet, ensuring the script runs without errors and writes a correctly formatted `submission.csv`. It gathers all test IDs from the unpacked `test/` folders, builds the required DataFrame, and saves it.'
- What this solution (achieved 0.51911) has done: 'I replace the constant‑mean baseline with a tiny supervised model: for each snippet I compute simple statistics (mean and max intensity) and train a LogisticRegression on the training set. These features are cheap to compute, keep the original workflow, and are expected to raise the AUC from 0.5 toward the target 0.757 while still producing a correctly ordered `submission.csv`.'
- What this solution (achieved 0.49341) has done: 'I fix the feature extraction to guard against NaN/inf values, clean the training and test feature matrices before fitting/predicting, and ensure the submission DataFrame is always defined before saving. This resolves the ValueError and subsequent NotFittedError, allowing the pipeline to train and produce a correctly ordered `submission.csv` file.'
- What this solution (achieved 0.48905) has done: 'I added richer per‑position statistics (mean and max for each of the six cadence positions) to the feature set, which gives the model more signal‑specific information useful for distinguishing “needle” patterns. The training and test matrices now automatically use all extracted feature columns (except the id and target), so the new features are included without changing the overall pipeline. I also increased the LogisticRegression regularisation parameter C to reduce bias, which should raise the AUC toward the target while preserving the original model structure.'
- What this solution (achieved 0.49901) has done: 'I fix the infinities by converting the feature matrices to float64 before scaling, ensure the pipeline is fitted, and correctly create and save the submission DataFrame. This resolves the ValueError, the NotFittedError, and the NameError while keeping the original model architecture unchanged.'
- What this solution (achieved 0.50236) has done: 'The update adds a few inexpensive yet informative statistics (overall sum, range, per‑position standard deviation and range) to the feature set, which gives the logistic‑regression model more signal to separate needles from noise. At the same time the model’s class weighting is removed (the data is already fairly balanced), allowing the classifier to focus on maximizing AUC. These small, targeted changes keep the original pipeline intact while expectedly raising the score toward the target.'
- What this solution (achieved 0.52371) has done: 'We speed up the pipeline by (1) filtering the training file list to only ids that actually have labels, (2) loading each ``.npy`` file as a read‑only memory‑map and casting to ``float32`` to avoid costly ``float16`` → ``float64`` conversions, (3) switching the parallel extraction from a heavyweight ``ProcessPoolExecutor`` to a lightweight ``ThreadPoolExecutor`` (the work is I/O‑bound), (4) using a larger ``chunksize`` and all available CPU cores, and (5) keeping the feature matrices in ``float32`` until they are fed to scikit‑learn, which internally promotes them to ``float64``. These changes cut down disk‑I/O overhead and multiprocessing launch cost while preserving every computed statistic and the exact model‑training logic.'
- What this solution (achieved 0.5051) has done: 'I add a few extra informative statistics (standard‑deviation difference and mean/max ratios) to the feature set, and simplify the model pipeline by dropping the polynomial expansion (which was causing many noisy interaction terms). I also increase the logistic‑regression regularisation strength (C=10) to let the model fit the data a bit more. These limited changes keep the overall workflow intact while giving the classifier stronger, cleaner signals, which should raise the AUC toward the target.'
- What this solution (achieved 0.50492) has done: 'I add a few inexpensive but potentially informative derived statistics (overall mean/std ratio, std ratio between A‑ and B‑positions, and a max‑range feature) to give the model more signal, and I simplify the logistic‑regression settings by removing the balanced class‑weight (the data are roughly balanced) and using the default regularisation (C=1.0). These small adjustments keep the overall pipeline intact while aiming to raise the AUC toward the target.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import concurrent.futures
import multiprocessing



## === cell 1
train_labels_path = "/kaggle/input/train_labels.csv"
train_df = pd.read_csv(train_labels_path)
print(f"Training labels loaded: {train_df.shape[0]} rows")



## === cell 2
test_root = "/kaggle/input/test"
test_pattern = os.path.join(test_root, "*", "*.npy")
test_files = glob.glob(test_pattern)
test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_files]
print(f"Found {len(test_ids)} test files.")



## === cell 3
train_root = "/kaggle/input/train"
train_pattern = os.path.join(train_root, "*", "*.npy")
all_train_files = glob.glob(train_pattern)

train_ids_set = set(train_df["id"].astype(str))
train_files = [
    p
    for p in all_train_files
    if os.path.splitext(os.path.basename(p))[0] in train_ids_set
]


def _process_file(p):
    """Compute enriched features for a single .npy file."""
    fid = os.path.splitext(os.path.basename(p))[0]
    try:
        arr = np.load(p, mmap_mode="r").astype(np.float32, copy=False)  # (6,273,256)

        overall_mean = np.nan_to_num(arr.mean())
        overall_max = np.nan_to_num(arr.max())
        overall_min = np.nan_to_num(arr.min())
        overall_std = np.nan_to_num(arr.std())
        overall_med = np.nan_to_num(np.median(arr))
        overall_sum = np.nan_to_num(arr.sum())
        overall_range = overall_max - overall_min
        overall_mean_std_ratio = overall_mean / (overall_std + 1e-6)

        pos_mean = np.nan_to_num(arr.mean(axis=(1, 2)))  # (6,)
        pos_max = np.nan_to_num(arr.max(axis=(1, 2)))  # (6,)
        pos_min = np.nan_to_num(arr.min(axis=(1, 2)))  # (6,)
        pos_std = np.nan_to_num(arr.std(axis=(1, 2)))  # (6,)
        pos_range = pos_max - pos_min
        pos_median = np.nan_to_num(np.median(arr, axis=(1, 2)))  # (6,)

        mean_A = pos_mean[[0, 2, 4]].mean()
        mean_B = pos_mean[[1, 3, 5]].mean()
        diff_mean = mean_A - mean_B
        ratio_mean = mean_A / (mean_B + 1e-6)

        max_A = pos_max[[0, 2, 4]].mean()
        max_B = pos_max[[1, 3, 5]].mean()
        diff_max = max_A - max_B
        ratio_max = max_A / (max_B + 1e-6)

        std_A = pos_std[[0, 2, 4]].mean()
        std_B = pos_std[[1, 3, 5]].mean()
        diff_std = std_A - std_B
        ratio_std = std_A / (std_B + 1e-6)

        max_range = overall_max - overall_min
    except Exception:
        overall_mean = overall_max = overall_min = overall_std = overall_med = (
            overall_sum
        ) = overall_range = overall_mean_std_ratio = diff_mean = diff_max = (
            ratio_mean
        ) = ratio_max = diff_std = ratio_std = max_range = 0.0
        pos_mean = np.zeros(6, dtype=np.float32)
        pos_max = np.zeros(6, dtype=np.float32)
        pos_min = np.zeros(6, dtype=np.float32)
        pos_std = np.zeros(6, dtype=np.float32)
        pos_range = np.zeros(6, dtype=np.float32)
        pos_median = np.zeros(6, dtype=np.float32)

    features = {
        "id": fid,
        "mean": overall_mean,
        "max": overall_max,
        "min": overall_min,
        "range": overall_range,
        "std": overall_std,
        "median": overall_med,
        "sum": overall_sum,
        "diff_mean": diff_mean,
        "diff_max": diff_max,
        "diff_std": diff_std,
        "ratio_mean": ratio_mean,
        "ratio_max": ratio_max,
        "ratio_std": ratio_std,
        "overall_mean_std_ratio": overall_mean_std_ratio,
        "max_range": max_range,
    }
    for i in range(6):
        features[f"mean_{i}"] = pos_mean[i]
        features[f"max_{i}"] = pos_max[i]
        features[f"min_{i}"] = pos_min[i]
        features[f"std_{i}"] = pos_std[i]
        features[f"range_{i}"] = pos_range[i]
        features[f"median_{i}"] = pos_median[i]
    return features


def extract_features(file_list, n_jobs=None):
    """Parallel feature extraction using a thread pool (I/O bound)."""
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count())
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        rows = list(executor.map(_process_file, file_list, chunksize=1000))
    return pd.DataFrame(rows)


print("Extracting features from training data...")
train_feat_df = extract_features(train_files)
print(f"Training features shape: {train_feat_df.shape}")

print("Extracting features from test data...")
test_feat_df = extract_features(test_files)
print(f"Test features shape: {test_feat_df.shape}")



## === cell 4
train_merged = train_feat_df.merge(train_df, on="id", how="inner")
X_train = train_merged.drop(columns=["id", "target"]).values.astype(np.float32)
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
y_train = train_merged["target"].values

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                C=1.0,  # default regularisation
                solver="lbfgs",  # keep same solver
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)
print("Model training completed.")



## === cell 5
X_test = test_feat_df.drop(columns=["id"]).values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
test_probs = pipeline.predict_proba(X_test)[:, 1]
submission_df = pd.DataFrame({"id": test_feat_df["id"], "target": test_probs})



## === cell 6
sample_sub_path = "/kaggle/input/sample_submission.csv"
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    submission_df = submission_df.set_index("id").loc[sample_sub["id"]].reset_index()
print("Submission head after ordering:")
print(submission_df.head())



## === cell 7
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
