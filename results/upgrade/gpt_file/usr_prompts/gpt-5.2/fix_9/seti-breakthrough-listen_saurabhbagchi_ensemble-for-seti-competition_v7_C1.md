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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

BASE_CANDIDATES = [
    "/kaggle/input",  # common Kaggle path
    "/kaggle/data",  # provided in this environment description
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input/seti-breakthrough-listen",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE = first_existing(BASE_CANDIDATES[:2])  # prefer /kaggle/input or /kaggle/data
if BASE is None:
    BASE = first_existing(BASE_CANDIDATES[2:])
if BASE is None:
    raise FileNotFoundError(
        "Could not find Kaggle dataset base directory among expected candidates."
    )

train_labels_path = os.path.join(BASE, "train_labels.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")
train_dir = os.path.join(BASE, "train")
test_dir = os.path.join(BASE, "test")

if not os.path.exists(train_labels_path):
    alt_base = os.path.join(BASE, "seti-breakthrough-listen")
    if os.path.exists(os.path.join(alt_base, "train_labels.csv")):
        BASE = alt_base
        train_labels_path = os.path.join(BASE, "train_labels.csv")
        sample_sub_path = os.path.join(BASE, "sample_submission.csv")
        train_dir = os.path.join(BASE, "train")
        test_dir = os.path.join(BASE, "test")

for p in [train_labels_path, sample_sub_path, train_dir, test_dir]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required path: {p}")

train_labels = pd.read_csv(train_labels_path)
sample_sub = pd.read_csv(sample_sub_path)

train_labels.head(), sample_sub.head()




## === cell 1
def list_npy_files(root_dir):
    out = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        out.append(f.path)
    out.sort()
    return out


train_files = list_npy_files(train_dir)
test_files = list_npy_files(test_dir)

len(train_files), len(test_files)




## === cell 2
def file_id_from_path(p):
    b = os.path.basename(p)
    return b[:-4] if b.endswith(".npy") else os.path.splitext(b)[0]


train_map = {file_id_from_path(p): p for p in train_files}
test_map = {file_id_from_path(p): p for p in test_files}

train_ids_in_files = sorted(train_map.keys())
labels_idx = train_labels.set_index("id")
missing_train_labels = [i for i in train_ids_in_files if i not in labels_idx.index]
if missing_train_labels:
    raise FileNotFoundError(
        f"{len(missing_train_labels)} train file ids have no label. Example: {missing_train_labels[:5]}"
    )

train_df = labels_idx.loc[train_ids_in_files].reset_index()  # columns: id, target

test_ids = sample_sub["id"].astype(str).tolist()
missing_test = [i for i in test_ids if i not in test_map]
if missing_test:
    raise FileNotFoundError(
        f"{len(missing_test)} sample_submission ids have no corresponding test .npy files. Example: {missing_test[:5]}"
    )

train_df.shape, train_df["target"].mean()




## === cell 3
def extract_features(arr):
    x = np.asarray(arr, dtype=np.float32)  # (6,H,W)
    on = x[[0, 2, 4]]  # (3,H,W)
    off = x[[1, 3, 5]]  # (3,H,W)

    on_mean = on.mean(axis=0)
    off_mean = off.mean(axis=0)
    diff = on_mean - off_mean

    f = []
    f_append = f.append

    f_append(float(diff.mean()))
    f_append(float(diff.std()))
    abs_diff = np.abs(diff)
    f_append(float(abs_diff.mean()))
    f_append(float(diff.max()))
    f_append(float(diff.min()))

    flat_abs = abs_diff.reshape(-1)
    k = 512
    if flat_abs.size >= k:
        idx = np.argpartition(flat_abs, -k)[-k:]
        topk = flat_abs[idx]
        f_append(float(topk.mean()))
        f_append(float(topk.max()))
    else:
        f_append(float(flat_abs.mean()))
        f_append(float(flat_abs.max() if flat_abs.size else 0.0))

    row_prof = diff.mean(axis=1)  # (H,)
    col_prof = diff.mean(axis=0)  # (W,)
    f_append(float(row_prof.std()))
    f_append(float(col_prof.std()))
    f_append(float(np.mean(np.abs(np.diff(row_prof)))))
    f_append(float(np.mean(np.abs(np.diff(col_prof)))))

    f_append(float(on.mean()))
    f_append(float(off.mean()))
    f_append(float(on.std()))
    f_append(float(off.std()))

    on_panel_means = on.mean(axis=(1, 2))
    off_panel_means = off.mean(axis=(1, 2))
    f_append(float(on_panel_means.mean()))
    f_append(float(off_panel_means.mean()))
    f_append(float(on_panel_means.std()))
    f_append(float(off_panel_means.std()))
    per_pair = on_panel_means - off_panel_means
    f_append(float(per_pair.mean()))
    f_append(float(per_pair.std()))
    f_append(float(per_pair.max()))
    f_append(float(per_pair.min()))

    eps = 1e-8
    on_flat = on.reshape(3, -1)
    on_mu = on_flat.mean(axis=1, keepdims=True)
    on_sd = on_flat.std(axis=1, keepdims=True) + eps
    on_z = (on_flat - on_mu) / on_sd
    c01 = float(np.mean(on_z[0] * on_z[1]))
    c02 = float(np.mean(on_z[0] * on_z[2]))
    c12 = float(np.mean(on_z[1] * on_z[2]))
    f_append((c01 + c02 + c12) / 3.0)

    on_row = on_mean.mean(axis=1)
    on_col = on_mean.mean(axis=0)
    off_row = off_mean.mean(axis=1)
    off_col = off_mean.mean(axis=0)

    f_append(float(on_row.max()))
    f_append(float(off_row.max()))
    f_append(float(on_col.max()))
    f_append(float(off_col.max()))
    f_append(float(on_row.max() - off_row.max()))
    f_append(float(on_col.max() - off_col.max()))
    f_append(float(np.abs(on_row).max()))
    f_append(float(np.abs(off_row).max()))
    f_append(float(np.abs(on_col).max()))
    f_append(float(np.abs(off_col).max()))
    f_append(float(np.abs(on_row).max() - np.abs(off_row).max()))
    f_append(float(np.abs(on_col).max() - np.abs(off_col).max()))

    a = on_mean.reshape(-1)
    b = off_mean.reshape(-1)
    a0 = a - a.mean()
    b0 = b - b.mean()
    cov = float(np.mean(a0 * b0))
    corr = float(cov / ((a0.std() + eps) * (b0.std() + eps)))
    f_append(cov)
    f_append(corr)

    on_abs_mean = np.abs(on).mean(axis=(1, 2))
    off_abs_mean = np.abs(off).mean(axis=(1, 2))
    f_append(float(on_abs_mean.mean()))
    f_append(float(off_abs_mean.mean()))
    f_append(float((on_abs_mean - off_abs_mean).mean()))
    f_append(float(on_abs_mean.std()))
    f_append(float(off_abs_mean.std()))

    kk = 256
    on_abs_flat = np.abs(on).reshape(3, -1)
    off_abs_flat = np.abs(off).reshape(3, -1)

    def tail_mean_max(v2d, kk):
        n = v2d.shape[1]
        if n <= kk:
            return v2d.mean(axis=1), v2d.max(axis=1)
        means = np.empty(3, dtype=np.float32)
        maxs = np.empty(3, dtype=np.float32)
        for j in range(3):
            v = v2d[j]
            idx2 = np.argpartition(v, -kk)[-kk:]
            top = v[idx2]
            means[j] = top.mean()
            maxs[j] = top.max()
        return means, maxs

    on_tail_mean, on_tail_max = tail_mean_max(on_abs_flat, kk)
    off_tail_mean, off_tail_max = tail_mean_max(off_abs_flat, kk)

    f_append(float(on_tail_mean.mean()))
    f_append(float(off_tail_mean.mean()))
    f_append(float((on_tail_mean - off_tail_mean).mean()))
    f_append(float(on_tail_max.mean()))
    f_append(float(off_tail_max.mean()))
    f_append(float((on_tail_max - off_tail_max).mean()))

    on_freq_prof = on.max(axis=1)  # (3,W)
    on_time_prof = on.max(axis=2)  # (3,H)
    off_freq_prof = off.max(axis=1)  # (3,W)
    off_time_prof = off.max(axis=2)  # (3,H)

    def prof_stats(freq_prof, time_prof):
        return np.stack(
            [
                freq_prof.max(axis=1),
                freq_prof.mean(axis=1),
                freq_prof.std(axis=1),
                time_prof.max(axis=1),
                time_prof.mean(axis=1),
                time_prof.std(axis=1),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    on_ps = prof_stats(on_freq_prof, on_time_prof)
    off_ps = prof_stats(off_freq_prof, off_time_prof)

    f.extend(on_ps.mean(axis=0).tolist())
    f.extend(off_ps.mean(axis=0).tolist())
    f.extend((on_ps.mean(axis=0) - off_ps.mean(axis=0)).tolist())
    f.extend(on_ps.std(axis=0).tolist())

    return np.asarray(f, dtype=np.float32)


tmp_id = train_df.loc[0, "id"]
tmp_arr = np.load(train_map[tmp_id])
feat_tmp = extract_features(tmp_arr)
feat_tmp, tmp_arr.shape, feat_tmp.shape



## === cell 4
train_df_fit = train_df

n_features = int(feat_tmp.shape[0])

X = np.zeros((len(train_df_fit), n_features), dtype=np.float32)
y = train_df_fit["target"].values.astype(np.int64)

extract = extract_features
tmap = train_map
ids = train_df_fit["id"].values

for i, id_ in enumerate(ids):
    p = tmap.get(id_)
    if p is None:
        raise FileNotFoundError(f"Train id from labels not found in train files: {id_}")
    X[i] = extract(np.load(p))

assert X.shape[0] == y.shape[0] == train_df_fit.shape[0]
assert train_df_fit[
    "id"
].is_monotonic_increasing, "Expected train ids to be in canonical sorted order."

X.shape, y.mean(), n_features



## === cell 5
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=4000,
                solver="saga",
                penalty="l2",
                n_jobs=None,
                class_weight="balanced",
                C=3.0,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

val_pred = model.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, val_pred)



## === cell 6
X_test = np.zeros((len(test_ids), n_features), dtype=np.float32)

extract = extract_features
tmap = test_map
for i, id_ in enumerate(test_ids):
    X_test[i] = extract(np.load(tmap[id_]))

test_pred = model.predict_proba(X_test)[:, 1].astype(np.float64)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission = sample_sub[["id"]].merge(submission, on="id", how="left")
if submission["target"].isna().any():
    raise RuntimeError(
        "Some test predictions are missing after merge; id alignment issue."
    )

submission.to_csv("submission.csv", index=False)
submission.head(), submission.shape
