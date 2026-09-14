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
import os
import glob
import pandas as pd
import numpy as np


def find_sample_submission():
    pattern = "/kaggle/input/**/sample_submission.csv"
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError("sample_submission.csv not found in any input folder.")
    return matches[0]




## === cell 1
sample_path = find_sample_submission()
submission_df = pd.read_csv(sample_path)

train_labels_path = "/kaggle/input/seti-breakthrough-listen/train_labels.csv"
if not os.path.exists(train_labels_path):
    train_labels_path = "/kaggle/input/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)

possible_train_dirs = [
    "/kaggle/input/seti-breakthrough-listen/train/",
    "/kaggle/input/train/",
    "/kaggle/input/seti-breakthrough-listen/data/train/",
    "/kaggle/input/data/train/",
    "/kaggle/input/working/seti-breakthrough-listen/train/",
]
train_dir = None
for d in possible_train_dirs:
    if os.path.isdir(d):
        train_dir = d
        break
if train_dir is None:
    raise FileNotFoundError("Train data directory not found.")

possible_test_dirs = [
    "/kaggle/input/seti-breakthrough-listen/test/",
    "/kaggle/input/test/",
    "/kaggle/input/seti-breakthrough-listen/data/test/",
    "/kaggle/input/data/test/",
    "/kaggle/input/working/seti-breakthrough-listen/test/",
]
test_dir = None
for d in possible_test_dirs:
    if os.path.isdir(d):
        test_dir = d
        break
if test_dir is None:
    raise FileNotFoundError("Test data directory not found.")




## === cell 2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import itertools


def compute_features(arr):
    """
    Extract an enriched lightweight feature vector from a (6, 273, 256) snippet.
    Returns a list of numeric features.
    """
    arr = arr.astype(np.float32)

    max_A = np.max(arr[[0, 2, 4]])
    max_nonA = np.max(arr[[1, 3, 5]])
    diff_max = max_A - max_nonA

    max_all = np.max(arr)

    mean_A = np.mean(arr[[0, 2, 4]])
    mean_nonA = np.mean(arr[[1, 3, 5]])
    diff_mean = mean_A - mean_nonA

    ratio = diff_max / (max_all + 1e-6)  # avoid division by zero

    pos_max = np.max(arr, axis=(1, 2))  # (6,)
    pos_mean = np.mean(arr, axis=(1, 2))  # (6,)

    sum_A = np.sum(arr[[0, 2, 4]])
    sum_nonA = np.sum(arr[[1, 3, 5]])
    diff_sum = sum_A - sum_nonA
    sum_all = np.sum(arr)

    min_all = np.min(arr)
    median_all = np.median(arr)
    std_all = np.std(arr)

    pos_min = np.min(arr, axis=(1, 2))  # (6,)
    pos_std = np.std(arr, axis=(1, 2))  # (6,)

    features = [
        diff_max,
        max_all,
        mean_A,
        mean_nonA,
        diff_mean,
        ratio,
        sum_A,
        sum_nonA,
        diff_sum,
        sum_all,
        min_all,
        median_all,
        std_all,
    ]
    features.extend(pos_max.tolist())
    features.extend(pos_mean.tolist())
    features.extend(pos_min.tolist())
    features.extend(pos_std.tolist())
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).tolist()


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


id_to_path = {}
for root, _, files in os.walk(train_dir):
    for fname in files:
        if fname.endswith(".npy"):
            id_str = os.path.splitext(fname)[0]
            id_to_path[id_str] = os.path.join(root, fname)

missing_ids = set(train_labels["id"]) - set(id_to_path.keys())
if missing_ids:
    print(
        f"Warning: {len(missing_ids)} training ids have no .npy file and will be ignored."
    )


def process_train_item(item):
    path, target = item
    feats = compute_features(np.load(path))
    return feats, target


def process_test_item(item):
    idx, path = item
    feats = compute_features(np.load(path))
    return idx, feats


train_items = [
    (id_to_path[row["id"]], row["target"])
    for _, row in train_labels.iterrows()
    if row["id"] in id_to_path
]

cpu_cnt = max(1, os.cpu_count() - 1)  # leave one core free

with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cnt) as executor:
    train_results = list(executor.map(process_train_item, train_items))

train_features, train_targets = zip(*train_results)

train_feat_arr = np.array(train_features)  # (n_samples, n_features)
train_feat_arr = np.nan_to_num(train_feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

train_target_arr = np.array(train_targets)

if train_feat_arr.size == 0:
    model = None
    scaler = None
    fallback_prob = 0.5
else:
    scaler = StandardScaler().fit(train_feat_arr)
    train_feat_scaled = scaler.transform(train_feat_arr)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
    )
    model.fit(train_feat_scaled, train_target_arr)
    fallback_prob = float(train_target_arr.mean())  # overall positive rate


def predict_prob(feat_vec):
    """Return calibrated probability; fall back to overall mean if model missing."""
    if model is None:
        return fallback_prob
    feat_scaled = scaler.transform([feat_vec])
    prob = model.predict_proba(feat_scaled)[0, 1]
    return float(prob)




## === cell 3
test_id_to_path = {}
for root, _, files in os.walk(test_dir):
    for fname in files:
        if fname.endswith(".npy"):
            id_str = os.path.splitext(fname)[0]
            test_id_to_path[id_str] = os.path.join(root, fname)

test_ids = list(submission_df["id"])
test_features = [None] * len(test_ids)  # preserve order

test_idx_path = [
    (idx, test_id_to_path[cid])
    for idx, cid in enumerate(test_ids)
    if cid in test_id_to_path
]

with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cnt) as executor:
    test_results = list(executor.map(process_test_item, test_idx_path))

for idx, feats in test_results:
    test_features[idx] = feats

test_predictions = []
if model is not None:
    computed_idxs = [i for i, ft in enumerate(test_features) if ft is not None]
    if computed_idxs:
        batch_feats = np.array([test_features[i] for i in computed_idxs])
        batch_scaled = scaler.transform(batch_feats)
        batch_probs = model.predict_proba(batch_scaled)[:, 1]
        for i, prob in zip(computed_idxs, batch_probs):
            test_features[i] = prob  # store probability in place

for idx, ft in enumerate(test_features):
    if ft is None:
        pred = fallback_prob
    else:
        pred = ft if isinstance(ft, float) else predict_prob(ft)
    test_predictions.append(pred)

submission_df["target"] = test_predictions

output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}.")
