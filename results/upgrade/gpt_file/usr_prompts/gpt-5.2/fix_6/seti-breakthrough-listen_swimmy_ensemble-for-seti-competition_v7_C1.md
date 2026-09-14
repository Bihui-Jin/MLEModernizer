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
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score




## === cell 1
def pick_existing_path(candidates):
    for p in candidates:
        if p is not None and os.path.exists(p):
            return p
    return None


BASE = pick_existing_path(
    [
        "/kaggle/input/seti-breakthrough-listen",
        "/kaggle/data/input/seti-breakthrough-listen",
        "/kaggle/data/seti-breakthrough-listen",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
    ]
)

if BASE is None:
    raise FileNotFoundError("Could not locate Kaggle input data directory.")

train_labels_path = pick_existing_path(
    [
        os.path.join(BASE, "train_labels.csv"),
        os.path.join(BASE, "seti-breakthrough-listen", "train_labels.csv"),
        "/kaggle/data/train_labels.csv",
        "/kaggle/data/input/train_labels.csv",
        "/kaggle/input/train_labels.csv",
    ]
)
sample_sub_path = pick_existing_path(
    [
        os.path.join(BASE, "sample_submission.csv"),
        os.path.join(BASE, "seti-breakthrough-listen", "sample_submission.csv"),
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)

train_dir = pick_existing_path(
    [
        os.path.join(BASE, "train"),
        os.path.join(BASE, "seti-breakthrough-listen", "train"),
        "/kaggle/data/train",
        "/kaggle/data/input/train",
        "/kaggle/input/train",
    ]
)
test_dir = pick_existing_path(
    [
        os.path.join(BASE, "test"),
        os.path.join(BASE, "seti-breakthrough-listen", "test"),
        "/kaggle/data/test",
        "/kaggle/data/input/test",
        "/kaggle/input/test",
    ]
)

if (
    train_labels_path is None
    or sample_sub_path is None
    or train_dir is None
    or test_dir is None
):
    raise FileNotFoundError(
        f"Missing required paths. "
        f"train_labels_path={train_labels_path}, sample_sub_path={sample_sub_path}, "
        f"train_dir={train_dir}, test_dir={test_dir}"
    )

train_labels = pd.read_csv(train_labels_path)
sample_sub = pd.read_csv(sample_sub_path)


def list_npy_files(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "*", "*.npy")))


train_files = list_npy_files(train_dir)
test_files = list_npy_files(test_dir)

if len(train_files) == 0 or len(test_files) == 0:
    raise FileNotFoundError(
        f"No .npy files found under train_dir={train_dir} or test_dir={test_dir}"
    )


def id_from_path(p):
    return os.path.splitext(os.path.basename(p))[0]


train_path_by_id = {id_from_path(p): p for p in train_files}
test_path_by_id = {id_from_path(p): p for p in test_files}

train_labels = train_labels[
    train_labels["id"].isin(train_path_by_id.keys())
].reset_index(drop=True)




## === cell 2
def extract_features_from_array(x):
    """
    Core logic preserved: handcrafted summary stats + A/O aggregates + A-vs-O.
    Minimal score-improving extension: add robust A-vs-O features (median/percentiles)
    and simple SNR-like normalization, which better matches the metric signal prior
    without changing model/training approach.
    """
    x = x.astype(np.float32, copy=False)  # stable stats, still fast

    means = x.mean(axis=(1, 2))
    stds = x.std(axis=(1, 2))
    maxs = x.max(axis=(1, 2))
    mins = x.min(axis=(1, 2))

    A = x[[0, 2, 4]]
    O = x[[1, 3, 5]]

    A_mean = A.mean(axis=0)  # (273,256)
    O_mean = O.mean(axis=0)  # (273,256)
    A_max = A.max(axis=0)
    O_max = O.max(axis=0)

    diff_mean = A_mean - O_mean
    diff_max = A_max - O_max

    d_mean = float(diff_mean.mean())
    d_std = float(diff_mean.std())
    d_max = float(diff_mean.max())
    d_min = float(diff_mean.min())

    A_time_std = float(A_mean.mean(axis=1).std())
    A_freq_std = float(A_mean.mean(axis=0).std())
    D_time_std = float(diff_mean.mean(axis=1).std())
    D_freq_std = float(diff_mean.mean(axis=0).std())

    dm_mean = float(diff_max.mean())
    dm_std = float(diff_max.std())
    dm_max = float(diff_max.max())
    dm_min = float(diff_max.min())

    absd = np.abs(diff_mean)
    absd_mean = float(absd.mean())
    absd_std = float(absd.std())

    d_med = float(np.median(diff_mean))
    d_p90 = float(np.percentile(diff_mean, 90.0))
    d_p10 = float(np.percentile(diff_mean, 10.0))
    absd_p95 = float(np.percentile(absd, 95.0))

    O_std = O.std(axis=0)
    snr = diff_mean / (O_std + 1e-3)
    snr_mean = float(snr.mean())
    snr_std = float(snr.std())
    snr_p95 = float(np.percentile(snr, 95.0))
    snr_p05 = float(np.percentile(snr, 5.0))

    feat = np.concatenate(
        [
            means,
            stds,
            maxs,
            mins,
            np.array(
                [
                    d_mean,
                    d_std,
                    d_max,
                    d_min,
                    A_time_std,
                    A_freq_std,
                    D_time_std,
                    D_freq_std,
                    dm_mean,
                    dm_std,
                    dm_max,
                    dm_min,
                    absd_mean,
                    absd_std,
                    d_med,
                    d_p90,
                    d_p10,
                    absd_p95,
                    snr_mean,
                    snr_std,
                    snr_p95,
                    snr_p05,
                ],
                dtype=np.float32,
            ),
        ]
    )

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    return feat


def load_and_extract(ids, path_by_id):
    if len(ids) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    first_arr = np.load(path_by_id[ids[0]])
    d = int(extract_features_from_array(first_arr).shape[0])

    X = np.zeros((len(ids), d), dtype=np.float32)
    X[0] = extract_features_from_array(first_arr)
    for i in range(1, len(ids)):
        _id = ids[i]
        arr = np.load(path_by_id[_id])  # expected (6,273,256) float16
        X[i] = extract_features_from_array(arr)
    return X




## === cell 3
train_ids = train_labels["id"].values
y = train_labels["target"].astype(np.int32).values
X = load_and_extract(train_ids, train_path_by_id)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                n_jobs=None,
                class_weight=None,
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)

va_pred = clf.predict_proba(X_va)[:, 1]
va_auc = roc_auc_score(y_va, va_pred)
print("Validation ROC AUC (sanity check, not Kaggle score):", float(va_auc))



## === cell 4
test_ids_present = sorted(test_path_by_id.keys())
X_test_present = load_and_extract(test_ids_present, test_path_by_id)
pred_present = clf.predict_proba(X_test_present)[:, 1].astype(np.float64)

pred_by_id = dict(zip(test_ids_present, pred_present))

sub_ids = sample_sub["id"].astype(str).values
missing = [i for i in sub_ids if i not in pred_by_id]
if len(missing) > 0:
    raise FileNotFoundError(
        f"{len(missing)} ids from sample_submission not found in test folders. Example: {missing[:5]}"
    )

pred_aligned = np.array([pred_by_id[i] for i in sub_ids], dtype=np.float64)

data6 = pd.DataFrame({"id": sub_ids, "target": pred_aligned})



## === cell 5
data6.head()



## === cell 6
data6.describe()



## === cell 7
data6["target"] = data6["target"].clip(0.0, 1.0)



## === cell 8
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print("Columns:", list(data6.columns))
print(data6.head())
