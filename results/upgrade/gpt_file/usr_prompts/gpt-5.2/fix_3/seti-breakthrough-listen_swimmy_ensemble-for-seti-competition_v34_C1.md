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

0.7571740335820211

# 6. Current score

0.5014

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49992) has done: 'Your notebook is trying to ensemble multiple external Kaggle Dataset submissions from `../input/...`, but those folders don’t exist in this environment, causing the `FileNotFoundError` and preventing any `submission.csv` from being written. I keep the “ensemble submissions” core logic, but make it robust: it load any available submission files if present, otherwise fall back to generating a baseline prediction directly from the competition’s own data (by computing simple per-snippet statistics and fitting a scikit-learn LogisticRegression with cross-validated out-of-fold predictions). This guarantees an end-to-end run and a valid `submission.csv` with `id,target`. The fallback is legitimate, uses only provided training labels/snippets, and should yield a reasonable AUC (moving toward your target) without changing evaluation semantics.'
- What this solution (achieved 0.5014) has done: 'Your current 0.49992 AUC indicates the fallback model is effectively near-random; the smallest likely cause is that the test predictions are not aligned to the correct `id` order (because `glob`-built `id->path` and `sample_submission` ordering can differ, and you only use `sample_ids` order when building `test_X`, but you never realign external submissions either). I keep your exact feature set and LogisticRegression training approach, but make the ensembling and fallback strictly `id`-aligned by merging on `id` and reindexing to `sample_submission` order. I also add `StandardScaler` (a minimal, conventional step for LogisticRegression) inside each fold to improve calibration/fit without changing the model family or objective, which should move AUC upward toward your target. All paths and outputs remain the same and it still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd



## === cell 1
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



## === cell 2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)




## === cell 3
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
data6 = try_read_csv(external_paths[5])

available = {
    f"data{i+1}": d is not None
    for i, d in enumerate([data1, data2, data3, data4, data5, data6])
}
print("External submissions available:", available)



## === cell 4
if data1 is not None:
    print(data1.head())
else:
    print("data1 not available in this environment.")



## === cell 5
if data2 is not None:
    print(data2.head())
else:
    print("data2 not available in this environment.")



## === cell 6
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
    Minimal but crucial for AUC: ensure predictions are aligned to the exact sample_submission id order.
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

    labels = pd.read_csv(train_labels_path)
    labels = labels[["id", "target"]].copy()

    def build_id_to_path(root_dir):
        npy_files = glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)
        id_to_path = {}
        for f in npy_files:
            base = os.path.basename(f)
            _id = os.path.splitext(base)[0]
            id_to_path[_id] = f
        return id_to_path

    train_id2path = build_id_to_path(train_dir)
    test_id2path = build_id_to_path(test_dir)

    missing_train = [i for i in labels["id"].values if i not in train_id2path]
    if missing_train:
        raise FileNotFoundError(
            f"Missing {len(missing_train)} train npy files; example: {missing_train[0]}"
        )

    missing_test = [i for i in sample_ids if i not in test_id2path]
    if missing_test:
        raise FileNotFoundError(
            f"Missing {len(missing_test)} test npy files; example: {missing_test[0]}"
        )

    def extract_features_from_path(path):
        x = np.load(path)  # shape (6,273,256)
        x = x.astype(np.float32)

        panel_mean = x.mean(axis=(1, 2))  # (6,)
        panel_std = x.std(axis=(1, 2))  # (6,)
        panel_max = x.max(axis=(1, 2))  # (6,)
        panel_min = x.min(axis=(1, 2))  # (6,)

        a = x[[0, 2, 4]]
        b = x[[1, 3, 5]]

        a_mean = a.mean()
        b_mean = b.mean()
        a_std = a.std()
        b_std = b.std()
        a_max = a.max()
        b_max = b.max()

        mean_diff = a_mean - b_mean
        std_diff = a_std - b_std
        max_diff = a_max - b_max

        a_abs_mean = np.abs(a).mean()
        b_abs_mean = np.abs(b).mean()
        abs_mean_diff = a_abs_mean - b_abs_mean

        time_profile = x.mean(axis=2)  # (6,273)
        time_std = time_profile.std(axis=1)  # (6,)

        feats = np.concatenate(
            [
                panel_mean,
                panel_std,
                panel_max,
                panel_min,
                time_std,
                np.array(
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
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return feats

    X = np.zeros((len(labels), 6 * 5 + 12), dtype=np.float32)
    for idx, _id in enumerate(labels["id"].values):
        X[idx] = extract_features_from_path(train_id2path[_id])

    y = labels["target"].values.astype(np.int32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    test_X = np.zeros((len(sample_ids), X.shape[1]), dtype=np.float32)
    for j, _id in enumerate(sample_ids):
        test_X[j] = extract_features_from_path(test_id2path[_id])

    test_pred = np.zeros(len(sample_ids), dtype=np.float64)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        X_te = scaler.transform(test_X)

        model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=-1,
        )
        model.fit(X_tr, y[tr])
        test_pred += model.predict_proba(X_te)[:, 1] / skf.get_n_splits()

    data6 = sample_sub.copy()
    data6["target"] = test_pred

data6["target"] = data6["target"].astype(float).clip(0.0, 1.0)



## === cell 7
data6 = data6[["id", "target"]].copy()
data6 = sample_sub[["id"]].merge(data6, on="id", how="left")
if data6["target"].isna().any():
    raise ValueError("Final submission has missing target values after alignment.")
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print(data6.head())
