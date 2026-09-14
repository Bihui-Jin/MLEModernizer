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
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # new import for scaling
import concurrent.futures  # parallel processing




## === cell 1
train_labels_path = os.path.join("..", "input", "train_labels.csv")
if not os.path.exists(train_labels_path):
    train_labels_path = os.path.join("input", "train_labels.csv")
if not os.path.exists(train_labels_path):
    train_labels_path = "/kaggle/input/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)
global_mean = train_labels["target"].mean()




## === cell 2
sample_paths = [
    os.path.join("..", "input", "sample_submission.csv"),
    os.path.join("input", "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "sample_submission.csv",
]
for sp in sample_paths:
    if os.path.exists(sp):
        sample_submission_path = sp
        break
else:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")
sample_sub = pd.read_csv(sample_submission_path)




## === cell 3
def find_dir(possible_roots, folder_name):
    for root in possible_roots:
        candidate = os.path.join(root, folder_name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"{folder_name} directory not found.")


possible_roots = [
    os.path.join("..", "input"),
    "input",
    "/kaggle/input",
    ".",
]
train_dir = find_dir(possible_roots, "train")
test_dir = find_dir(possible_roots, "test")




## === cell 4
def id_to_path(id_str, base_dir):
    try:
        subfolder = int(id_str[0], 16) % 16
    except Exception:
        subfolder = 0
    return os.path.join(base_dir, str(subfolder), f"{id_str}.npy")




## === cell 5
def extract_features(path):
    """Return a cleaned feature vector for a snippet, now with extra
    panel‑wise statistics (median, energy) and A/B group differences."""
    try:
        arr = np.load(path, mmap_mode="r")
        panel_means = arr.mean(axis=(1, 2))
        panel_std = arr.std(axis=(1, 2))
        panel_max = arr.max(axis=(1, 2))
        panel_min = arr.min(axis=(1, 2))
        panel_sum = arr.sum(axis=(1, 2))
        panel_range = panel_max - panel_min
        panel_median = np.median(arr, axis=(1, 2))
        panel_energy = np.mean(arr**2, axis=(1, 2))  # mean of squares

        overall_mean = arr.mean()
        overall_std = arr.std()
        overall_max = arr.max()
        overall_min = arr.min()
        overall_sum = arr.sum()
        overall_range = overall_max - overall_min
        overall_energy = np.mean(arr**2)

        a_idx = [0, 2, 4]
        b_idx = [1, 3, 5]

        a_mean = panel_means[a_idx].mean()
        b_mean = panel_means[b_idx].mean()
        mean_diff = a_mean - b_mean

        a_std = panel_std[a_idx].mean()
        b_std = panel_std[b_idx].mean()
        std_diff = a_std - b_std

        a_max = panel_max[a_idx].mean()
        b_max = panel_max[b_idx].mean()
        max_diff = a_max - b_max

        a_sum = panel_sum[a_idx].mean()
        b_sum = panel_sum[b_idx].mean()
        sum_diff = a_sum - b_sum

        pair_mean_diff = np.array(
            [
                panel_means[0] - panel_means[1],
                panel_means[2] - panel_means[3],
                panel_means[4] - panel_means[5],
            ]
        )
        pair_std_diff = np.array(
            [
                panel_std[0] - panel_std[1],
                panel_std[2] - panel_std[3],
                panel_std[4] - panel_std[5],
            ]
        )
        eps = 1e-6
        pair_mean_ratio = np.array(
            [
                panel_means[0] / (panel_means[1] + eps),
                panel_means[2] / (panel_means[3] + eps),
                panel_means[4] / (panel_means[5] + eps),
            ]
        )
        pair_std_ratio = np.array(
            [
                panel_std[0] / (panel_std[1] + eps),
                panel_std[2] / (panel_std[3] + eps),
                panel_std[4] / (panel_std[5] + eps),
            ]
        )

        panel_means_var = np.var(panel_means)

        feats = np.concatenate(
            [
                panel_means,
                panel_std,
                panel_max,
                panel_min,
                panel_sum,
                panel_range,
                panel_median,
                panel_energy,
                [
                    overall_mean,
                    overall_std,
                    overall_max,
                    overall_min,
                    overall_sum,
                    overall_range,
                    overall_energy,
                    mean_diff,
                    std_diff,
                    max_diff,
                    sum_diff,
                    panel_means_var,
                ],
                pair_mean_diff,
                pair_std_diff,
                pair_mean_ratio,
                pair_std_ratio,
            ]
        )
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(float)
    except Exception:
        return None




## === cell 6
label_dict = train_labels.set_index("id")["target"].to_dict()
train_ids = train_labels["id"].values


def _process_train(id_str):
    f_path = id_to_path(id_str, train_dir)
    try:
        feats = extract_features(f_path)
        if feats is None:
            return None
        return feats, label_dict[id_str]
    except FileNotFoundError:
        return None


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(executor.map(_process_train, train_ids))

features = [r[0] for r in results if r is not None]
labels = [r[1] for r in results if r is not None]

if len(features) == 0:
    X_train = np.full((len(train_labels), 1), global_mean)
    y_train = train_labels["target"].values
    scaler = None
else:
    X_train_raw = np.vstack(features)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    y_train = np.array(labels)




## === cell 7
logreg = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,  # more iterations for convergence
    C=10.0,  # weaker regularisation to use richer features
    class_weight="balanced",
    n_jobs=1,
)
logreg.fit(X_train, y_train)




## === cell 8
test_ids = sample_sub["id"].values


def _process_test(args):
    idx, id_str = args
    f_path = id_to_path(id_str, test_dir)
    try:
        feats = extract_features(f_path)
        if feats is None:
            return None
        return idx, feats
    except FileNotFoundError:
        return None


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    test_results = list(executor.map(_process_test, enumerate(test_ids)))

valid_idx = []
test_features = []
for res in test_results:
    if res is not None:
        idx, feats = res
        valid_idx.append(idx)
        test_features.append(feats)

preds = np.full(len(test_ids), global_mean, dtype=float)
if test_features:
    test_features_arr = np.vstack(test_features)
    if scaler is not None:
        test_features_arr = scaler.transform(test_features_arr)
    probas = logreg.predict_proba(test_features_arr)[:, 1]
    preds[valid_idx] = probas

sample_sub["target"] = preds
output_path = "submission.csv"
sample_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}. Mean prediction = {preds.mean():.6f}")
