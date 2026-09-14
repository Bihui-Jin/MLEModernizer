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

0.7554475030351673

# 6. Current score

0.4809

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the invalid reads of non‑existent submission files, compute a simple baseline probability (the mean target from the training set) and apply it to all test IDs taken from the provided sample submission. This guarantees a valid `submission.csv` with the correct columns and format, fixing the runtime errors while keeping the core logic unchanged.'
- What this solution (achieved 0.49477) has done: 'I add a lightweight feature‑extraction step and train a simple logistic‑regression model on those features. Computing per‑snippet statistics (mean, std, max, min) gives the model signal to distinguish needles from noise, which raises the ROC‑AUC from the constant‑baseline 0.5 toward the target 0.755. The new cells keep the original workflow intact, only extending it to produce better predictions while still writing a valid `submission.csv`.'
- What this solution (achieved 0.47873) has done: 'I extended the feature extraction to include per‑cadence‑position mean and standard deviation (adding 12 informative features) while keeping the original overall statistics. The feature matrix is now richer, so a StandardScaler is applied before the logistic regression via a small pipeline, which typically improves ROC‑AUC without altering the core modeling approach. All other workflow steps remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.48416) has done: 'I added richer cadence‑aware statistics to the feature extractor (per‑position max/min, target‑vs‑off‑target means/stds and their differences) and increased the logistic‑regression regularisation parameter C to give the model more capacity. These extra features give the classifier clearer signals about “needle” patterns without altering the overall pipeline, so the validation ROC‑AUC should move closer to the target while still writing a correct `submission.csv`.'
- What this solution (achieved 0.47814) has done: 'I replace the logistic‑regression model with a GradientBoostingClassifier, which usually captures non‑linear patterns in the extracted statistics and thus raises the ROC‑AUC toward the target. The feature extraction remains unchanged, and the rest of the pipeline (train/validation split, submission creation) is kept the same.'
- What this solution (achieved 0.48551) has done: 'I replace the GradientBoosting model with a balanced RandomForest classifier and increase its capacity (more trees, no depth limit). This generally improves ROC‑AUC on tabular features while keeping the same feature extraction logic, moving the validation score upward toward the target.'
- What this solution (achieved 0.5) has done: 'I add a few more informative statistics (per‑position median) to the feature set and adjust the RandomForest hyper‑parameters to increase model capacity (more trees, limited depth, entropy split, and a smaller max‑features fraction). These changes keep the overall pipeline unchanged while giving the classifier richer signals and slightly more regularisation, which should raise the validation ROC‑AUC toward the target.'
- What this solution (achieved 0.5) has done: 'Optimized the feature extraction by parallelizing the per‑file computation with a process pool, which removes the dominant sequential I/O and CPU bottleneck while keeping the exact same feature set and order. Added a lightweight worker that safely handles missing files and returns a zero‑filled vector, preserving the original fallback logic. The rest of the pipeline—including data loading, model training, and prediction—remains unchanged, ensuring identical results but well within the 600‑second limit.'
- What this solution (achieved 0.5) has done: 'The script now limits parallel workers to a reasonable number and uses `executor.map` for lower overhead while keeping the exact same feature computation and model training logic. This reduces disk‑thrashed process spawning and speeds up the overall run without altering any algorithmic behavior.'
- What this solution (achieved 0.5) has done: 'We speed up feature extraction by (1) using all CPU cores, (2) increasing the executor chunk size to cut inter‑process overhead, (3) avoiding an unnecessary reshape and computing per‑position statistics directly on the original 3‑D array, and (4) using a `functools.partial` to pass the constant root directory once. These changes keep the exact same 42‑dimensional feature vector, so model training and evaluation remain unchanged while I/O‑bound work finishes well within the 600 s limit.'
- What this solution (achieved 0.5) has done: 'The changes focus on speeding up feature extraction, which is the main bottleneck.  
* We limit the number of parallel processes to leave CPU resources for NumPy and the RandomForest training.  
* A larger `chunksize` (500) greatly reduces inter‑process communication overhead.  
* The worker is built with `functools.partial` once, avoiding repeated argument passing.  
* All other logic, model configuration, and evaluation remain unchanged, preserving exact results.'
- What this solution (achieved 0.5) has done: 'The changes add a simple disk‑cache for the extracted feature matrices so the expensive per‑file loading/computation is performed only once, and they limit the number of parallel workers to avoid excessive overhead and memory pressure. The core feature‑extraction logic and model training remain unchanged, guaranteeing identical predictions after the first run.'
- What this solution (achieved 0.48954) has done: 'The changes switch the feature extraction to a thread‑based pool, which avoids the heavy process‑creation and data‑pickling overhead when loading many small NumPy files (the task is I/O‑bound). Using `ThreadPoolExecutor` keeps the same feature logic while dramatically cutting runtime, and the cache mechanism remains unchanged for subsequent runs.'
- What this solution (achieved 0.4809) has done: 'I tune the RandomForest hyper‑parameters to give it more capacity (more trees, unlimited depth, and a more suitable feature‑sampling strategy). This small change keeps the overall pipeline and feature extraction untouched while expected to raise the validation ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np




## === cell 1
train_labels_path = os.path.join("..", "input", "train_labels.csv")
if not os.path.exists(train_labels_path):
    train_labels_path = "train_labels.csv"  # fallback to current dir
train_labels = pd.read_csv(train_labels_path)
mean_target = train_labels["target"].mean()
print(f"Mean target (baseline probability): {mean_target:.6f}")




## === cell 2
sample_submission_paths = [
    os.path.join("..", "input", "sample_submission.csv"),
    "sample_submission.csv",
]
for path in sample_submission_paths:
    if os.path.exists(path):
        sample_submission_path = path
        break
else:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")

submission_df = pd.read_csv(sample_submission_path)
assert (
    "id" in submission_df.columns
), "sample_submission.csv must contain an 'id' column."




## === cell 3
submission_df["target"] = mean_target
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with {len(submission_df)} rows.")




## === cell 4
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from concurrent.futures import ThreadPoolExecutor  # switched to threads
import multiprocessing
import itertools
import functools  # new import for partial

warnings.filterwarnings("ignore", category=UserWarning)


def get_npy_path(root_dir, id_str):
    """
    Construct the path to a .npy file given the root directory and snippet id.
    The data are stored in 16 sub‑folders named 0‑15 corresponding to the first
    hex digit of the id.
    """
    try:
        folder = str(int(id_str[0], 16))  # convert first hex char to decimal string
    except Exception:
        folder = "0"
    return os.path.join(root_dir, folder, f"{id_str}.npy")


def _compute_feature(id_str, root_dir):
    """
    Worker that loads a single .npy file and returns its feature vector.
    Returns a list of 42 float values (same as original implementation).
    """
    npy_path = get_npy_path(root_dir, id_str)
    if not os.path.exists(npy_path):
        return [0.0] * 42

    arr = np.load(npy_path)  # shape (6, 273, 256), dtype float16
    arr = arr.astype(np.float32)

    mean_val = arr.mean()
    std_val = arr.std()
    median_val = np.median(arr)
    max_val = arr.max()
    min_val = arr.min()
    sum_val = arr.sum()

    pos_means = arr.mean(axis=(1, 2))  # shape (6,)
    pos_stds = arr.std(axis=(1, 2))
    pos_maxs = arr.max(axis=(1, 2))
    pos_mins = arr.min(axis=(1, 2))
    pos_medians = np.median(arr, axis=(1, 2))

    on_idx = [0, 2, 4]
    off_idx = [1, 3, 5]
    on_mean = pos_means[on_idx].mean()
    off_mean = pos_means[off_idx].mean()
    on_std = pos_stds[on_idx].mean()
    off_std = pos_stds[off_idx].mean()
    diff_mean = on_mean - off_mean
    diff_std = on_std - off_std

    feature_vec = (
        [mean_val, std_val, median_val, max_val, min_val, sum_val]  # 6 global
        + pos_means.tolist()  # 6
        + pos_stds.tolist()  # 6
        + pos_maxs.tolist()  # 6
        + pos_mins.tolist()  # 6
        + pos_medians.tolist()  # 6
        + [on_mean, off_mean, diff_mean, on_std, off_std, diff_std]  # 6 cadence‑aware
    )
    return feature_vec


def extract_features(id_series, root_dir, cache_path=None):
    """
    Parallel (thread‑based) feature extraction with optional disk caching.
    If `cache_path` exists, the cached NumPy array is loaded directly.
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    max_workers = min(32, (multiprocessing.cpu_count() or 1) * 2)
    worker = functools.partial(_compute_feature, root_dir=root_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(worker, id_series, chunksize=1000)

    features = np.array(list(results), dtype=np.float32)

    if cache_path:
        np.save(cache_path, features)

    return features


train_root = os.path.join("..", "input", "train")
if not os.path.isdir(train_root):
    train_root = "train"

print("Extracting features from training data (this may take a few minutes)...")
train_cache = "train_features.npy"
X = extract_features(train_labels["id"], train_root, cache_path=train_cache)
y = train_labels["target"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=2000,  # increased from 1000
    max_depth=None,  # allow full depth
    criterion="entropy",
    max_features="sqrt",  # more appropriate for high‑dimensional data
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)
val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation ROC‑AUC: {val_auc:.5f}")




## === cell 5
test_root = os.path.join("..", "input", "test")
if not os.path.isdir(test_root):
    test_root = "test"

print("Extracting features from test data...")
test_cache = "test_features.npy"
test_ids = submission_df["id"]
X_test = extract_features(test_ids, test_root, cache_path=test_cache)

test_pred = model.predict_proba(X_test)[:, 1]

submission_df["target"] = test_pred
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Improved submission written to {output_path} with {len(submission_df)} rows.")
