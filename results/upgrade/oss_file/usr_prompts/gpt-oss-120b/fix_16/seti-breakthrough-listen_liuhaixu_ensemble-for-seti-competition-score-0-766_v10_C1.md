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

0.7627043135520268

# 6. Current score

0.50065

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the failing reads of non‑existent submission files with a safe load of the provided `sample_submission.csv`. If the file cannot be found, the code create a minimal dataframe with the required `id` column (by listing files in the test folder) and fill all predictions with a neutral probability 0.5. Finally, the script writes the dataframe to `submission.csv`, guaranteeing a valid CSV output for the competition.'
- What this solution (achieved 0.49427) has done: 'The changes preload all `.npy` file locations into dictionaries and compute means in parallel using a thread pool, eliminating costly per‑iteration filesystem searches and serial I/O.  Path look‑ups become O(1) and file reads are overlapped, while the original mean‑to‑probability mapping and output format stay unchanged.'
- What this solution (achieved 0.48832) has done: 'The update adds a data‑driven mapping from snippet intensity to probability (using the empirical cumulative distribution of the training targets) and fixes the async collection of test predictions so each ID receives its correct score. This monotonic, calibrated transformation is expected to raise the AUC toward the target while keeping the original mean‑intensity feature and overall workflow unchanged.'
- What this solution (achieved 0.4823) has done: 'The update adds a more informative feature – the difference between the average intensity of the “on‑target” cadences (positions 0, 2, 4) and the “off‑target” cadences (positions 1, 3, 5).  
We compute this contrast for every snippet, build an empirical CDF from the training data, and map each test snippet’s contrast to a calibrated probability.  
All other workflow steps (parallel loading, fallback handling, CSV output) remain unchanged, keeping the core logic intact while moving the AUC toward the target.'
- What this solution (achieved 0.49235) has done: 'We add a second intensity feature (the overall mean of the 6 cadences) and fit a tiny logistic‑regression model on the training snippets using the contrast and this mean. The model replaces the simple empirical CDF mapping, giving a calibrated probability that better separates positives from negatives, which should raise the AUC toward the target while keeping the original workflow and parallel loading logic. All other steps (fallback handling, CSV output) stay unchanged.'
- What this solution (achieved 0.51957) has done: 'I added several simple statistical features (standard deviations and maxima of the on‑target and off‑target cadences) to the existing contrast and overall mean, and fed all of these into the same logistic‑regression model (now with class‑weight balancing).  The feature extraction and prediction code were updated to handle the extra values, keeping the overall workflow unchanged while giving the model more information to raise the AUC toward the target.'
- What this solution (achieved 0.48565) has done: 'I added a small feature engineering and preprocessing step that stays within the original logistic‑regression pipeline: a new “diff_std” column (std_on – std_off) and standard‑scaling of all features before fitting and predicting. This improves model calibration without changing the core architecture, and the prediction flow still writes a correct `submission.csv`.'
- What this solution (achieved 0.48116) has done: 'I replace the simple logistic‑regression model with a GradientBoostingClassifier, which can capture nonlinear relationships in the engineered features and is expected to raise the AUC toward the target. The feature extraction, parallel I/O, and CSV‑writing logic remain unchanged, only the model training and prediction steps are updated.'
- What this solution (achieved 0.49045) has done: 'I add a simple but informative feature – the difference between the maximum values of the on‑target and off‑target cadences – and include it in the training and test feature sets. I also slightly strengthen the GradientBoostingClassifier by increasing the number of trees and lowering the learning rate, which together usually improve AUC without changing the overall pipeline. These minimal changes keep the core workflow intact while giving the model more discriminative power to move the score toward the target.'
- What this solution (achieved 0.48936) has done: 'I add class‑balanced sample weights and a slightly stronger GradientBoosting configuration (more trees and depth) while keeping all existing feature extraction and workflow unchanged. This minimal tuning is expected to raise the AUC toward the target without altering the core logic.'
- What this solution (achieved 0.50092) has done: 'I replace the plain GradientBoostingClassifier with a HistGradientBoostingClassifier (still a gradient‑boosting model) and increase its capacity (more trees, lower learning rate) to give the model stronger predictive power while keeping the overall pipeline unchanged. This change is expected to raise the AUC toward the target score.'
- What this solution (achieved 0.505) has done: 'I add a light logistic‑regression model and standard‑scale the features, then average its probabilities with those from the existing HistGradientBoosting model. This keeps the original pipeline intact while providing a complementary predictor that should improve the ranking (AUC) toward the target score.'
- What this solution (achieved 0.5) has done: 'I add two informative ratio features (mean‑on/off ratio and std‑on/off ratio) derived from the existing statistics, include them in the feature list, and give the stronger HistGradientBoosting model a slightly higher weight in the final averaged probability. These minimal changes keep the overall pipeline and models unchanged while giving the classifier more discriminative information to raise the AUC toward the target.'
- What this solution (achieved 0.50065) has done: 'The fix replaces infinite values (generated by division by zero) with NaNs and casts the feature matrix to float64 before scaling, preventing the scaler from encountering infinities or the float16 overflow error. This allows the model to predict probabilities correctly and write a valid `submission.csv` while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight



## === cell 1
sample_path = Path("../input/sample_submission.csv")
if sample_path.is_file():
    submission_df = pd.read_csv(sample_path)
else:
    test_root = Path("../input/test")
    ids = [p.stem for p in test_root.rglob("*.npy")]
    submission_df = pd.DataFrame({"id": ids})



## === cell 2
train_root = Path("../input/train")
train_labels_path = Path("../input/train_labels.csv")
train_labels = pd.read_csv(train_labels_path)

train_path_dict = {p.stem: p for p in train_root.rglob("*.npy")}


def compute_features(path):
    arr = np.load(path, mmap_mode="r")  # shape (6, 273, 256)
    on_idx = (0, 2, 4)
    off_idx = (1, 3, 5)

    on_cadences = arr[on_idx]
    off_cadences = arr[off_idx]

    mean_on = on_cadences.mean()
    mean_off = off_cadences.mean()
    contrast = mean_on - mean_off
    overall_mean = arr.mean()

    std_on = on_cadences.std()
    std_off = off_cadences.std()
    max_on = on_cadences.max()
    max_off = off_cadences.max()
    max_diff = max_on - max_off

    return (
        contrast,
        overall_mean,
        std_on,
        std_off,
        max_on,
        max_off,
        max_diff,
        mean_on,
        mean_off,
        std_on,
        std_off,
    )


max_workers = min(32, (os.cpu_count() or 1) + 4)

train_feat = {}
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_id = {
        executor.submit(compute_features, train_path_dict[nid]): nid
        for nid in train_labels["id"]
        if nid in train_path_dict
    }
    for future in as_completed(future_to_id):
        nid = future_to_id[future]
        try:
            train_feat[nid] = future.result()
        except Exception:
            train_feat[nid] = (np.nan,) * 11

missing_ids = set(train_labels["id"]) - set(train_feat.keys())
for nid in missing_ids:
    train_feat[nid] = (np.nan,) * 11

train_df = pd.DataFrame(
    {
        "id": list(train_feat.keys()),
        "contrast": [v[0] for v in train_feat.values()],
        "mean": [v[1] for v in train_feat.values()],
        "std_on": [v[2] for v in train_feat.values()],
        "std_off": [v[3] for v in train_feat.values()],
        "max_on": [v[4] for v in train_feat.values()],
        "max_off": [v[5] for v in train_feat.values()],
        "max_diff": [v[6] for v in train_feat.values()],
        "mean_on": [v[7] for v in train_feat.values()],
        "mean_off": [v[8] for v in train_feat.values()],
    }
)
train_df["target"] = train_labels.set_index("id").loc[train_df["id"], "target"].values
train_df = train_df.dropna()

train_df["mean_on_off_ratio"] = train_df["mean_on"] / (train_df["mean_off"] + 1e-6)
train_df["std_on_off_ratio"] = train_df["std_on"] / (train_df["std_off"] + 1e-6)
train_df["diff_std"] = train_df["std_on"] - train_df["std_off"]

feature_cols = [
    "contrast",
    "mean",
    "std_on",
    "std_off",
    "max_on",
    "max_off",
    "max_diff",
    "diff_std",
    "mean_on_off_ratio",
    "std_on_off_ratio",
]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["target"].astype(int).values,
)
weight_map = {0: class_weights[0], 1: class_weights[1]}
sample_weights = train_df["target"].map(weight_map)

gbc = HistGradientBoostingClassifier(
    max_iter=1500,
    learning_rate=0.03,
    max_depth=6,
    l2_regularization=0.0,
    random_state=42,
)
gbc.fit(train_df[feature_cols], train_df["target"], sample_weight=sample_weights)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_cols])
lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
lr.fit(X_train_scaled, train_df["target"])



## === cell 3
test_root = Path("../input/test")
test_path_dict = {p.stem: p for p in test_root.rglob("*.npy")}


def get_test_features(nid):
    if nid in test_path_dict:
        return compute_features(test_path_dict[nid])
    else:
        return (np.nan,) * 11


test_feat = {}
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_nid = {
        executor.submit(get_test_features, nid): nid for nid in submission_df["id"]
    }
    for future in as_completed(future_to_nid):
        nid = future_to_nid[future]
        try:
            test_feat[nid] = future.result()
        except Exception:
            test_feat[nid] = (np.nan,) * 11

test_feat_df = pd.DataFrame(
    {
        "id": list(test_feat.keys()),
        "contrast": [v[0] for v in test_feat.values()],
        "mean": [v[1] for v in test_feat.values()],
        "std_on": [v[2] for v in test_feat.values()],
        "std_off": [v[3] for v in test_feat.values()],
        "max_on": [v[4] for v in test_feat.values()],
        "max_off": [v[5] for v in test_feat.values()],
        "max_diff": [v[6] for v in test_feat.values()],
        "mean_on": [v[7] for v in test_feat.values()],
        "mean_off": [v[8] for v in test_feat.values()],
    }
)

test_feat_df["diff_std"] = test_feat_df["std_on"] - test_feat_df["std_off"]
test_feat_df["mean_on_off_ratio"] = test_feat_df["mean_on"] / (
    test_feat_df["mean_off"] + 1e-6
)
test_feat_df["std_on_off_ratio"] = test_feat_df["std_on"] / (
    test_feat_df["std_off"] + 1e-6
)

test_feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)

mask_valid = test_feat_df[feature_cols].notna().all(axis=1)
probs = np.full(len(test_feat_df), 0.5)

if mask_valid.any():
    prob_hist = gbc.predict_proba(test_feat_df.loc[mask_valid, feature_cols])[:, 1]
    X_test_scaled = scaler.transform(
        test_feat_df.loc[mask_valid, feature_cols].astype(np.float64)
    )
    prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    probs[mask_valid] = 0.6 * prob_hist + 0.4 * prob_lr

submission_df["target"] = probs



## === cell 4
submission_df.to_csv("submission.csv", index=False)
