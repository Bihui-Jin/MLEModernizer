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

0.7571687076302562

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The script was failing because it tried to read non‑existent submission files, causing all subsequent variables to be undefined. I replaced those reads with a robust baseline: load the official `sample_submission.csv`, compute the overall positive rate from the training labels, and assign that constant probability to every test row. This guarantees a valid `submission.csv` is written while keeping the core logic minimal and reproducible.'
- What this solution (achieved 0.5) has done: 'I add a lightweight feature‑based model: compute simple statistics (mean and std) for each snippet, train a logistic regression on a random subset of the training data, and use it to generate probabilities for the test set. This replaces the constant‑baseline prediction with a data‑driven one, which should raise the ROC‑AUC above 0.5 and move the score toward the target while keeping the overall pipeline simple and preserving the original submission format.'
- What this solution (achieved 0.49838) has done: 'I add the missing test files to the ID‑to‑path map, extend the feature extractor with max/min statistics, and safely impute any remaining NaNs before scaling so the model can predict without errors. These minimal changes keep the original logic while allowing the classifier to use a richer feature set, which should improve the ROC‑AUC toward the target score and ensure a valid submission.csv is written.'
- What this solution (achieved 0.49977) has done: 'I added richer per‑cadence statistics (mean, std, max, min for each of the 6 positions) to the feature extractor, and I now train on the full set of training snippets instead of a random subset.  These extra, still‑lightweight features give the logistic regression more signal without altering the core model or training loop, moving the ROC‑AUC toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.49208) has done: 'I add two simple “on‑target vs off‑target” difference features (mean‑difference and std‑difference) to the existing statistics and switch the model from LogisticRegression to a balanced RandomForestClassifier, which is still lightweight but usually yields a higher ROC‑AUC. These changes keep the overall pipeline and feature extraction logic intact while providing a stronger learner to move the score closer to the target.'
- What this solution (achieved 0.48586) has done: 'I added two additional difference features (max‐difference and min‐difference between on‑target and off‑target positions) and updated the feature matrix size. I also fixed the imputation so the same SimpleImputer trained on the training data is used for the test set, preventing data‑leakage. Finally, I boosted the RandomForest by increasing the number of trees to 500 and using the default “sqrt” max_features, which should raise the ROC‑AUC closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.48472) has done: 'The main slowdown was the RandomForest training using a single CPU (`n_jobs=1`). Switching to full parallelism (`n_jobs=-1`) lets all available cores build trees simultaneously, cutting wall‑time dramatically while keeping every other step—including feature extraction and model hyper‑parameters—unchanged.'
- What this solution (achieved 0.4803) has done: 'The changes parallelize the heavy feature‑extraction loop using a thread pool and memory‑map the .npy files to avoid extra copies, which dramatically cuts I/O‑bound runtime while leaving every statistical computation, model, and preprocessing step unchanged.'
- What this solution (achieved 0.48047) has done: 'I add a few more informative statistics (overall variance and on/off ratios for mean, std, max, min) to the feature vector and adjust the parallel extraction routine to handle the new feature length automatically. These richer features give the ExtraTrees model more signal without altering its core training logic, which should raise the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier


def build_id_path_map(root_dir: str) -> dict:
    """
    Walk through sub‑folders of `root_dir` (e.g. train/ or test/) and
    create a dictionary mapping each snippet id (filename without .npy)
    to its absolute file path.
    """
    mapping = {}
    pattern = os.path.join(root_dir, "*", "*.npy")
    for file_path in glob.glob(pattern):
        fname = os.path.basename(file_path)
        iid = fname[:-4]  # strip .npy
        mapping[iid] = file_path
    return mapping


def _read_csv_candidates(fname):
    """
    Try a handful of typical Kaggle locations for the CSV file.
    Returns a pandas DataFrame if found, otherwise raises FileNotFoundError.
    """
    candidates = [
        fname,
        os.path.join("kaggle", "input", "seti-breakthrough-listen", fname),
        os.path.join("kaggle", "input", "seti-breakthrough-listen", "data", fname),
        os.path.join("kaggle", "input", fname),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Unable to locate {fname} in any known location.")


TRAIN_ROOT = os.path.join("train")
TEST_ROOT = os.path.join("test")

if not os.path.isdir(TRAIN_ROOT):
    TRAIN_ROOT = os.path.join("kaggle", "input", "seti-breakthrough-listen", "train")
if not os.path.isdir(TEST_ROOT):
    TEST_ROOT = os.path.join("kaggle", "input", "seti-breakthrough-listen", "test")

id_to_path = build_id_path_map(TRAIN_ROOT)
test_id_to_path = build_id_path_map(TEST_ROOT)

train_labels = _read_csv_candidates("train_labels.csv")
sample_submission = _read_csv_candidates("sample_submission.csv")
test_ids = sample_submission["id"].tolist()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1778692413.py in <cell line: 0>()
     55 
     56 # Load labels and sample submission using robust path helper
---> 57 train_labels = _read_csv_candidates("train_labels.csv")
     58 sample_submission = _read_csv_candidates("sample_submission.csv")
     59 test_ids = sample_submission["id"].tolist()

/tmp/ipykernel_11/1778692413.py in _read_csv_candidates(fname)
     38         if os.path.isfile(path):
     39             return pd.read_csv(path)
---> 40     raise FileNotFoundError(f"Unable to locate {fname} in any known location.")
     41 
     42 

FileNotFoundError: Unable to locate train_labels.csv in any known location.

## === cell 1
def _compute_features_for_id(iid, id_path_map):
    """Compute an extended feature vector for a single id."""
    path = id_path_map.get(iid)
    if path and os.path.exists(path):
        arr = np.load(path, mmap_mode="r").astype(np.float32)  # (6, 273, 256)

        overall_mean = arr.mean()
        overall_std = arr.std()
        overall_var = arr.var()
        overall_max = arr.max()
        overall_min = arr.min()
        overall_median = np.median(arr)

        q25, q75 = np.quantile(arr, [0.25, 0.75])
        overall_iqr = q75 - q25
        overall_range = overall_max - overall_min

        pos_means = arr.mean(axis=(1, 2))
        pos_stds = arr.std(axis=(1, 2))
        pos_max = arr.max(axis=(1, 2))
        pos_min = arr.min(axis=(1, 2))
        pos_medians = np.median(arr, axis=(1, 2))

        on_idx = [0, 2, 4]
        off_idx = [1, 3, 5]

        on_mean = pos_means[on_idx].mean()
        off_mean = pos_means[off_idx].mean()
        diff_mean = on_mean - off_mean
        ratio_mean = on_mean / (off_mean + 1e-6)

        on_std = pos_stds[on_idx].mean()
        off_std = pos_stds[off_idx].mean()
        diff_std = on_std - off_std
        ratio_std = on_std / (off_std + 1e-6)

        on_max = pos_max[on_idx].mean()
        off_max = pos_max[off_idx].mean()
        diff_max = on_max - off_max
        ratio_max = on_max / (off_max + 1e-6)

        on_min = pos_min[on_idx].mean()
        off_min = pos_min[off_idx].mean()
        diff_min = on_min - off_min
        ratio_min = on_min / (off_min + 1e-6)

        var_pos_means = pos_means.var()
        var_pos_stds = pos_stds.var()
        var_pos_max = pos_max.var()
        var_pos_min = pos_min.var()

        return np.concatenate(
            [
                [
                    overall_mean,
                    overall_std,
                    overall_var,
                    overall_max,
                    overall_min,
                    overall_median,
                    overall_range,
                    overall_iqr,
                ],
                pos_means,
                pos_stds,
                pos_max,
                pos_min,
                pos_medians,
                [
                    diff_mean,
                    diff_std,
                    diff_max,
                    diff_min,
                    ratio_mean,
                    ratio_std,
                    ratio_max,
                    ratio_min,
                ],
                [
                    var_pos_means,
                    var_pos_stds,
                    var_pos_max,
                    var_pos_min,
                ],
            ]
        ).astype(np.float32)
    else:
        return np.full(50, np.nan, dtype=np.float32)


def extract_features(ids, id_path_map):
    """Parallel feature extraction (process‑based) preserving order."""
    import concurrent.futures

    max_workers = os.cpu_count() or 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _compute_features_for_id, ids, [id_path_map] * len(ids), chunksize=64
        )
        feature_array = np.vstack(list(results))
    return feature_array




## === cell 2
train_ids = train_labels["id"].tolist()
raw_train_feats = extract_features(train_ids, id_to_path)

imputer = SimpleImputer(strategy="mean")
train_feats = imputer.fit_transform(raw_train_feats)

train_targets = train_labels.set_index("id").loc[train_ids, "target"].values

scaler = StandardScaler()
train_feats_scaled = scaler.fit_transform(train_feats)

model = ExtraTreesClassifier(
    n_estimators=1200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    min_samples_leaf=1,
    max_features="sqrt",
)
model.fit(train_feats_scaled, train_targets)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2608398104.py in <cell line: 0>()
      1 # ----- training pipeline -----
----> 2 train_ids = train_labels["id"].tolist()
      3 raw_train_feats = extract_features(train_ids, id_to_path)
      4 
      5 imputer = SimpleImputer(strategy="mean")

NameError: name 'train_labels' is not defined

## === cell 3
raw_test_feats = extract_features(test_ids, test_id_to_path)

test_feats = imputer.transform(raw_test_feats)
test_feats_scaled = scaler.transform(test_feats)

test_probs = model.predict_proba(test_feats_scaled)[:, 1]

submission = pd.DataFrame({"id": test_ids, "target": test_probs})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2678854335.py in <cell line: 0>()
      1 # ----- inference on test set -----
----> 2 raw_test_feats = extract_features(test_ids, test_id_to_path)
      3 
      4 test_feats = imputer.transform(raw_test_feats)
      5 test_feats_scaled = scaler.transform(test_feats)

NameError: name 'test_ids' is not defined
