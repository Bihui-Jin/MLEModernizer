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
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",  # common Kaggle competition path
    "/kaggle/data/seti-breakthrough-listen",  # as provided in this environment
    "/kaggle/input",  # generic
    "/kaggle/data",  # generic
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE = _first_existing(BASE_INPUT_CANDIDATES)
if BASE is None:
    raise FileNotFoundError(
        "Could not find Kaggle input data directory in expected locations."
    )


def _find_file(filename):
    search_roots = [BASE, "/kaggle/input", "/kaggle/data"]
    for root in search_roots:
        cand = os.path.join(root, filename)
        if os.path.exists(cand):
            return cand
        pattern = os.path.join(root, "**", filename)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            matches = sorted(matches, key=lambda x: (x.count(os.sep), len(x)))
            return matches[0]
    return None


train_labels_path = _find_file("train_labels.csv")
sample_sub_path = _find_file("sample_submission.csv")

if train_labels_path is None or sample_sub_path is None:
    raise FileNotFoundError(
        "Required CSVs (train_labels.csv, sample_submission.csv) not found."
    )


def _find_dir(dirname):
    search_roots = [BASE, "/kaggle/input", "/kaggle/data"]
    for root in search_roots:
        cand = os.path.join(root, dirname)
        if os.path.isdir(cand):
            return cand
        matches = glob.glob(os.path.join(root, "**", dirname), recursive=True)
        matches = [m for m in matches if os.path.isdir(m)]
        if matches:
            matches = sorted(matches, key=lambda x: (x.count(os.sep), len(x)))
            return matches[0]
    return None


train_dir = _find_dir("train")
test_dir = _find_dir("test")
if train_dir is None or test_dir is None:
    raise FileNotFoundError("Could not locate train/ and test/ directories.")

print("Using:")
print(" train_labels:", train_labels_path)
print(" sample_sub  :", sample_sub_path)
print(" train_dir   :", train_dir)
print(" test_dir    :", test_dir)




## === cell 1
def try_read_csv(path):
    if path is None:
        return None
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


external_paths = [
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    "../input/seti-bl-tf-starter-tpu/submission.csv",
    "../input/seti-learned-image-resizing/submission.csv",
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
]

data1 = try_read_csv(external_paths[0])
data2 = try_read_csv(external_paths[1])
data3 = try_read_csv(external_paths[2])
data4 = try_read_csv(external_paths[3])
data5 = try_read_csv(external_paths[4])
data6_ext = try_read_csv(external_paths[5])

available = {
    f"data{i+1}": d is not None
    for i, d in enumerate([data1, data2, data3, data4, data5, data6_ext])
}
print("External submissions available:", available)



## === cell 2
if data1 is not None:
    print(data1.head(2))
else:
    print("data1 not available in this environment.")



## === cell 3
if data2 is not None:
    print(data2.head(2))
else:
    print("data2 not available in this environment.")



## === cell 4
import multiprocessing as mp

sample_sub = pd.read_csv(sample_sub_path)
sample_ids = sample_sub["id"].tolist()


def is_valid_submission_df(df, sample_ids_):
    if df is None:
        return False
    if not {"id", "target"}.issubset(df.columns):
        return False
    return set(df["id"]) == set(sample_ids_)


def align_to_sample(df, sample_df):
    """
    Ensure predictions are aligned to the exact sample_submission id order.
    """
    df = df[["id", "target"]].copy()
    out = sample_df[["id"]].merge(df, on="id", how="left")
    if out["target"].isna().any():
        missing = int(out["target"].isna().sum())
        raise ValueError(
            f"Aligned submission has {missing} missing targets after merge-on-id."
        )
    return out


use_external_ensemble = all(
    is_valid_submission_df(d, sample_ids) for d in [data4, data5]
) and (data1 is not None and is_valid_submission_df(data1, sample_ids))

if use_external_ensemble:
    d1 = align_to_sample(data1, sample_sub)
    d4 = align_to_sample(data4, sample_sub)
    d5 = align_to_sample(data5, sample_sub)

    if data2 is not None and is_valid_submission_df(data2, sample_ids):
        d2 = align_to_sample(data2, sample_sub)
        t2 = d2["target"].values
    else:
        t2 = 0.0

    if data3 is not None and is_valid_submission_df(data3, sample_ids):
        d3 = align_to_sample(data3, sample_sub)
        t3 = d3["target"].values
    else:
        t3 = 0.0

    data6 = sample_sub.copy()
    data6["target"] = (
        0.769 * d5["target"].values
        + 0.131 * d4["target"].values
        + 0.1 * d1["target"].values
        + 0.0 * t2
        + 0.0 * t3
    )
else:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    labels = pd.read_csv(train_labels_path)
    labels = labels[["id", "target"]].copy()

    def id_to_npy_path(root_dir, _id):
        return os.path.join(root_dir, _id[0], f"{_id}.npy")

    train_ids = labels["id"].values
    train_paths = [id_to_npy_path(train_dir, _id) for _id in train_ids]
    test_paths = [id_to_npy_path(test_dir, _id) for _id in sample_ids]

    for pth in (train_paths[0], test_paths[0]):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Expected npy file not found: {pth}")

    A_IDX = np.array([0, 2, 4], dtype=np.int64)
    B_IDX = np.array([1, 3, 5], dtype=np.int64)

    _R2_N = 273 * 256
    _Q_PROBS = (0.10, 0.50, 0.90)
    _Q_KS = np.asarray([int(p * (_R2_N - 1)) for p in _Q_PROBS], dtype=np.int64)

    def _panel_quantiles_exact_from_r2_partition(r2, ks=_Q_KS):
        p = np.partition(r2, ks, axis=1)
        return p[:, ks]  # (6,3), exact kth values

    _N_PANEL = 6
    _BASE_BLOCKS = 8 * _N_PANEL  # mean,std,max,min,time_std,q10,q50,q90 = 48
    _EXTRA = 25
    _N_FEATS = _BASE_BLOCKS + _EXTRA  # 73

    def extract_features_from_path(path):
        x = np.load(path, mmap_mode="r")  # (6,273,256) float16 on disk
        r2 = x.reshape(6, -1)  # view on mmap (no copy)

        panel_mean = r2.mean(axis=1, dtype=np.float32)
        panel_std = r2.std(axis=1, dtype=np.float32)
        panel_max = r2.max(axis=1)
        panel_min = r2.min(axis=1)

        a = r2[A_IDX]
        b = r2[B_IDX]

        a_mean = np.float32(a.mean(dtype=np.float32))
        b_mean = np.float32(b.mean(dtype=np.float32))
        a_std = np.float32(a.std(dtype=np.float32))
        b_std = np.float32(b.std(dtype=np.float32))
        a_max = np.float32(a.max())
        b_max = np.float32(b.max())

        mean_diff = np.float32(a_mean - b_mean)
        std_diff = np.float32(a_std - b_std)
        max_diff = np.float32(a_max - b_max)

        a_abs_mean = np.float32(np.abs(a).mean(dtype=np.float32))
        b_abs_mean = np.float32(np.abs(b).mean(dtype=np.float32))
        abs_mean_diff = np.float32(a_abs_mean - b_abs_mean)

        time_profile = x.mean(axis=2, dtype=np.float32)  # (6,273)
        time_std = time_profile.std(axis=1, dtype=np.float32)  # (6,)

        a_time_std_mean = np.float32(time_std[A_IDX].mean(dtype=np.float32))
        b_time_std_mean = np.float32(time_std[B_IDX].mean(dtype=np.float32))
        time_std_diff = np.float32(a_time_std_mean - b_time_std_mean)

        r2_f32 = np.asarray(r2, dtype=np.float32, order="C")
        q = _panel_quantiles_exact_from_r2_partition(r2_f32).astype(
            np.float32, copy=False
        )
        q10 = q[:, 0]
        q50 = q[:, 1]
        q90 = q[:, 2]

        q10_diff = np.float32(
            q10[A_IDX].mean(dtype=np.float32) - q10[B_IDX].mean(dtype=np.float32)
        )
        q50_diff = np.float32(
            q50[A_IDX].mean(dtype=np.float32) - q50[B_IDX].mean(dtype=np.float32)
        )
        q90_diff = np.float32(
            q90[A_IDX].mean(dtype=np.float32) - q90[B_IDX].mean(dtype=np.float32)
        )

        pair_mean_diff0 = np.float32(panel_mean[0] - panel_mean[1])
        pair_mean_diff1 = np.float32(panel_mean[2] - panel_mean[3])
        pair_mean_diff2 = np.float32(panel_mean[4] - panel_mean[5])

        pair_std_diff0 = np.float32(panel_std[0] - panel_std[1])
        pair_std_diff1 = np.float32(panel_std[2] - panel_std[3])
        pair_std_diff2 = np.float32(panel_std[4] - panel_std[5])

        pair_max_diff0 = np.float32(panel_max[0] - panel_max[1])
        pair_max_diff1 = np.float32(panel_max[2] - panel_max[3])
        pair_max_diff2 = np.float32(panel_max[4] - panel_max[5])

        feats = np.empty((_N_FEATS,), dtype=np.float32)
        k = 0
        feats[k : k + 6] = panel_mean.astype(np.float32, copy=False)
        k += 6
        feats[k : k + 6] = panel_std.astype(np.float32, copy=False)
        k += 6
        feats[k : k + 6] = panel_max.astype(np.float32, copy=False)
        k += 6
        feats[k : k + 6] = panel_min.astype(np.float32, copy=False)
        k += 6
        feats[k : k + 6] = time_std.astype(np.float32, copy=False)
        k += 6
        feats[k : k + 6] = q10.astype(np.float32, copy=False)
        k += 6
        feats[k : k + 6] = q50.astype(np.float32, copy=False)
        k += 6
        feats[k : k + 6] = q90.astype(np.float32, copy=False)
        k += 6

        feats[k : k + 25] = np.array(
            [
                a_mean,
                b_mean,
                a_std,
                b_std,
                a_max,
                b_max,
                mean_diff,
                std_diff,
                max_diff,
                a_abs_mean,
                b_abs_mean,
                abs_mean_diff,
                time_std_diff,
                q10_diff,
                q50_diff,
                q90_diff,
                pair_mean_diff0,
                pair_mean_diff1,
                pair_mean_diff2,
                pair_std_diff0,
                pair_std_diff1,
                pair_std_diff2,
                pair_max_diff0,
                pair_max_diff1,
                pair_max_diff2,
            ],
            dtype=np.float32,
        )
        return feats

    _G_PATHS = None

    def _init_pool_worker(paths):
        global _G_PATHS
        _G_PATHS = paths

    def _feats_worker(i):
        return i, extract_features_from_path(_G_PATHS[i])

    def _parallel_features(paths, n_workers=None, chunksize=None):
        n = len(paths)
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        if n_workers is None:
            n_workers = max(1, min(os.cpu_count() or 2, 8))
        if chunksize is None:
            chunksize = max(256, n // (n_workers * 2) if n else 256)

        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method)

        out = np.empty((n, _N_FEATS), dtype=np.float32)

        with ctx.Pool(
            processes=n_workers,
            initializer=_init_pool_worker,
            initargs=(paths,),
            maxtasksperchild=2000,
        ) as pool:
            for i, f in pool.imap_unordered(
                _feats_worker, range(n), chunksize=chunksize
            ):
                out[i] = f
        return out

    def _serial_features(paths):
        out = np.empty((len(paths), _N_FEATS), dtype=np.float32)
        for i in range(len(paths)):
            out[i] = extract_features_from_path(paths[i])
        return out

    try:
        X = _parallel_features(train_paths)
    except Exception as e:
        print(
            "Multiprocessing feature extraction failed; falling back to serial. Error:",
            repr(e),
        )
        X = _serial_features(train_paths)

    if X.shape[1] != _N_FEATS:
        raise RuntimeError("Inconsistent feature dimensions detected.")

    y = labels["target"].values.astype(np.int32)

    try:
        test_X = _parallel_features(test_paths)
    except Exception as e:
        print(
            "Multiprocessing feature extraction (test) failed; falling back to serial. Error:",
            repr(e),
        )
        test_X = _serial_features(test_paths)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    test_pred = np.zeros(len(sample_ids), dtype=np.float64)
    oof_pred = np.zeros(len(labels), dtype=np.float64)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        X_te = scaler.transform(test_X)

        model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_tr, y[tr])

        oof_pred[va] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / skf.get_n_splits()

        fold_auc = roc_auc_score(y[va], oof_pred[va])
        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    oof_auc = roc_auc_score(y, oof_pred)
    print(f"OOF AUC: {oof_auc:.6f}")

    def quantile_map_to_reference(x, ref):
        x = np.asarray(x, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.int64)
        ranks[order] = np.arange(len(x))
        ref_sorted = np.sort(ref)
        q = (ranks + 0.5) / len(x)
        idx = np.clip((q * (len(ref_sorted) - 1)).astype(int), 0, len(ref_sorted) - 1)
        return ref_sorted[idx]

    test_pred_mapped = quantile_map_to_reference(test_pred, oof_pred)

    data6 = sample_sub.copy()
    data6["target"] = test_pred_mapped

data6["target"] = data6["target"].astype(float).clip(0.0, 1.0)



## === cell 5
data6 = data6[["id", "target"]].copy()
data6 = sample_sub[["id"]].merge(data6, on="id", how="left")
if data6["target"].isna().any():
    raise ValueError("Final submission has missing target values after alignment.")
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print(data6.head())
