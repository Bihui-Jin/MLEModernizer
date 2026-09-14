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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import os

train_labels_path = "../input/train_labels.csv"
sample_submission_path = "../input/sample_submission.csv"
train_dir = "../input/train"
test_dir = "../input/test"

train_df = pd.read_csv(train_labels_path)

overall_mean_full = train_df["target"].mean()


def smoothed_prefix_means(df, prefix_len, alpha, overall_mean):
    grp = df.groupby(df["id"].str[:prefix_len])["target"]
    sum_target = grp.sum()
    count = grp.count()
    return (sum_target + alpha * overall_mean) / (count + alpha)


def prefix_counts(df, prefix_len):
    """Return a Series mapping each prefix to the count of its occurrences."""
    return df.groupby(df["id"].str[:prefix_len]).size()


def prefix_feature_matrix(id_series, p2, p4, p6, p8, overall_mean):
    """Return a (n,4) array with the four prefix mean predictions."""
    f2 = id_series.str[:2].map(p2).fillna(overall_mean).values.reshape(-1, 1)
    f4 = id_series.str[:4].map(p4).fillna(overall_mean).values.reshape(-1, 1)
    f6 = id_series.str[:6].map(p6).fillna(overall_mean).values.reshape(-1, 1)
    f8 = id_series.str[:8].map(p8).fillna(overall_mean).values.reshape(-1, 1)
    return np.hstack([f2, f4, f6, f8])


def prefix_feature_matrix_extended(
    id_series, p2, p4, p6, p8, c2, c4, c6, c8, overall_mean
):
    """Return a (n,8) array with prefix means and their corresponding counts."""
    f2 = id_series.str[:2].map(p2).fillna(overall_mean).values.reshape(-1, 1)
    f4 = id_series.str[:4].map(p4).fillna(overall_mean).values.reshape(-1, 1)
    f6 = id_series.str[:6].map(p6).fillna(overall_mean).values.reshape(-1, 1)
    f8 = id_series.str[:8].map(p8).fillna(overall_mean).values.reshape(-1, 1)

    cnt2 = id_series.str[:2].map(c2).fillna(0).values.reshape(-1, 1)
    cnt4 = id_series.str[:4].map(c4).fillna(0).values.reshape(-1, 1)
    cnt6 = id_series.str[:6].map(c6).fillna(0).values.reshape(-1, 1)
    cnt8 = id_series.str[:8].map(c8).fillna(0).values.reshape(-1, 1)

    return np.hstack([f2, f4, f6, f8, cnt2, cnt4, cnt6, cnt8])


def compute_intensity_mean(id_series, base_dir):
    """Compute mean intensity of each .npy file identified by id."""
    means = []
    for iid in id_series:
        subfolder = iid[0]  # first hex digit corresponds to folder name
        file_path = os.path.join(base_dir, subfolder, f"{iid}.npy")
        try:
            arr = np.load(file_path)
            means.append(arr.mean())
        except Exception:
            means.append(np.nan)
    arr_means = np.array(means, dtype=float)
    overall = np.nanmean(arr_means)
    arr_means = np.where(np.isnan(arr_means), overall, arr_means)
    return arr_means.reshape(-1, 1)


train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["target"],
    random_state=42,
)

overall_mean_train = train_split["target"].mean()

candidate_alphas = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
best_alpha = None
best_auc_alpha = -1.0

for alpha in candidate_alphas:
    p2 = smoothed_prefix_means(train_split, 2, alpha, overall_mean_train)
    p4 = smoothed_prefix_means(train_split, 4, alpha, overall_mean_train)
    p6 = smoothed_prefix_means(train_split, 6, alpha, overall_mean_train)
    p8 = smoothed_prefix_means(train_split, 8, alpha, overall_mean_train)

    X_train_alpha = prefix_feature_matrix(
        train_split["id"], p2, p4, p6, p8, overall_mean_train
    )
    X_val_alpha = prefix_feature_matrix(
        val_split["id"], p2, p4, p6, p8, overall_mean_train
    )

    X_train_alpha = np.hstack(
        [X_train_alpha, compute_intensity_mean(train_split["id"], train_dir)]
    )
    X_val_alpha = np.hstack(
        [X_val_alpha, compute_intensity_mean(val_split["id"], train_dir)]
    )

    y_train_alpha = train_split["target"]
    y_val_alpha = val_split["target"]

    tmp_model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
    tmp_model.fit(X_train_alpha, y_train_alpha)
    val_pred = tmp_model.predict_proba(X_val_alpha)[:, 1]
    auc = roc_auc_score(y_val_alpha, val_pred)

    if auc > best_auc_alpha:
        best_auc_alpha = auc
        best_alpha = alpha

alpha = best_alpha if best_alpha is not None else 10.0
print(f"Selected alpha: {alpha:.3f} (validation‑only AUC: {best_auc_alpha:.5f})")

overall_mean = overall_mean_full
prefix2_means = smoothed_prefix_means(train_df, 2, alpha, overall_mean)
prefix4_means = smoothed_prefix_means(train_df, 4, alpha, overall_mean)
prefix6_means = smoothed_prefix_means(train_df, 6, alpha, overall_mean)
prefix8_means = smoothed_prefix_means(train_df, 8, alpha, overall_mean)

prefix2_counts = prefix_counts(train_df, 2)
prefix4_counts = prefix_counts(train_df, 4)
prefix6_counts = prefix_counts(train_df, 6)
prefix8_counts = prefix_counts(train_df, 8)

candidate_cs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_c = None
best_auc_c = -1.0

X_train_c = prefix_feature_matrix_extended(
    train_split["id"],
    prefix2_means,
    prefix4_means,
    prefix6_means,
    prefix8_means,
    prefix2_counts,
    prefix4_counts,
    prefix6_counts,
    prefix8_counts,
    overall_mean,
)
X_val_c = prefix_feature_matrix_extended(
    val_split["id"],
    prefix2_means,
    prefix4_means,
    prefix6_means,
    prefix8_means,
    prefix2_counts,
    prefix4_counts,
    prefix6_counts,
    prefix8_counts,
    overall_mean,
)

X_train_c = np.hstack([X_train_c, compute_intensity_mean(train_split["id"], train_dir)])
X_val_c = np.hstack([X_val_c, compute_intensity_mean(val_split["id"], train_dir)])

y_train_c = train_split["target"]
y_val_c = val_split["target"]

for c in candidate_cs:
    model = LogisticRegression(solver="lbfgs", max_iter=1000, C=c)
    model.fit(X_train_c, y_train_c)
    val_pred = model.predict_proba(X_val_c)[:, 1]
    auc = roc_auc_score(y_val_c, val_pred)
    if auc > best_auc_c:
        best_auc_c = auc
        best_c = c

print(f"Selected C: {best_c:.3f} (validation AUC: {best_auc_c:.5f})")

X_train_full = prefix_feature_matrix_extended(
    train_df["id"],
    prefix2_means,
    prefix4_means,
    prefix6_means,
    prefix8_means,
    prefix2_counts,
    prefix4_counts,
    prefix6_counts,
    prefix8_counts,
    overall_mean,
)
X_train_full = np.hstack(
    [X_train_full, compute_intensity_mean(train_df["id"], train_dir)]
)

y_train_full = train_df["target"]

calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, C=best_c)
calibrator.fit(X_train_full, y_train_full)




## === cell 1
submission_df = pd.read_csv(sample_submission_path)

if "target" not in submission_df.columns:
    submission_df["target"] = np.nan

X_test = prefix_feature_matrix_extended(
    submission_df["id"],
    prefix2_means,
    prefix4_means,
    prefix6_means,
    prefix8_means,
    prefix2_counts,
    prefix4_counts,
    prefix6_counts,
    prefix8_counts,
    overall_mean,
)

X_test = np.hstack([X_test, compute_intensity_mean(submission_df["id"], test_dir)])

calibrated_preds = calibrator.predict_proba(X_test)[:, 1]
submission_df["target"] = calibrated_preds




## === cell 2
submission_df.to_csv("submission.csv", index=False)
