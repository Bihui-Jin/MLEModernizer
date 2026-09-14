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

os.environ.setdefault("PYTHONHASHSEED", "42")



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_PATH = "/kaggle/data"  # as provided in the file tree
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def list_npy_files(root_dir: str):
    return sorted(glob.glob(os.path.join(root_dir, "*", "*.npy")))


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

train_id_to_target = dict(
    zip(train_labels["id"].astype(str), train_labels["target"].astype(int))
)

train_files = [
    fp
    for fp in train_files
    if os.path.splitext(os.path.basename(fp))[0] in train_id_to_target
]

len(train_files), len(test_files), train_labels.shape, sample_sub.shape




## === cell 2
def extract_features_from_path(npy_path: str) -> np.ndarray:
    x = np.load(npy_path)  # (6, 273, 256), stored float16
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]
    O = x[[1, 3, 5]]

    A_mean = A.mean()
    O_mean = O.mean()
    A_std = A.std()
    O_std = O.std()
    A_max = A.max()
    O_max = O.max()

    A_f = A.mean(axis=1)  # (3, 256)
    O_f = O.mean(axis=1)  # (3, 256)
    A_m = A_f.mean(axis=0)  # (256,)
    O_m = O_f.mean(axis=0)  # (256,)

    diff = A_m - O_m
    absdiff = np.abs(diff)

    denom = np.abs(O_m) + 1e-6
    ratio = A_m / denom
    ratio_c = np.clip(ratio, -10, 10)

    q = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float32)
    diff_q = np.quantile(diff, q).astype(np.float32, copy=False)
    absdiff_q = np.quantile(absdiff, q).astype(np.float32, copy=False)

    pos = np.maximum(diff, 0.0)
    neg = np.maximum(-diff, 0.0)

    A_panel_means = A_f.mean(axis=1)  # (3,)
    O_panel_means = O_f.mean(axis=1)  # (3,)
    A_panel_std = A_panel_means.std()
    O_panel_std = O_panel_means.std()

    A_t = A.mean(axis=2)  # (3, 273)
    O_t = O.mean(axis=2)  # (3, 273)
    A_t_std = A_t.std()
    O_t_std = O_t.std()

    D = A.mean(axis=0) - O.mean(axis=0)  # (273, 256)
    absD = np.abs(D)

    d_t = np.diff(D, axis=0)  # (272, 256)
    d_f = np.diff(D, axis=1)  # (273, 255)
    grad_t_mean = np.mean(np.abs(d_t))
    grad_f_mean = np.mean(np.abs(d_f))
    grad_t_std = np.std(d_t)
    grad_f_std = np.std(d_f)

    flat_absD = absD.reshape(-1)
    k = 200
    if flat_absD.size >= k:
        topk = np.partition(flat_absD, -k)[-k:]
        topk_mean = float(np.mean(topk))
        topk_sum = float(np.sum(topk))
    else:
        topk_mean = float(np.mean(flat_absD))
        topk_sum = float(np.sum(flat_absD))

    row_energy = absD.mean(axis=1)  # (273,)
    col_energy = absD.mean(axis=0)  # (256,)
    row_max = float(row_energy.max())
    col_max = float(col_energy.max())
    row_std = float(row_energy.std())
    col_std = float(col_energy.std())

    D_pos_sum = float(np.sum(np.maximum(D, 0.0)))
    D_neg_sum = float(np.sum(np.maximum(-D, 0.0)))
    D_pos_frac = float(np.mean(D > 0.0))

    diff_l2 = float(np.sqrt(np.mean(diff * diff)))
    diff_abs_topk = float(np.mean(np.partition(absdiff, -10)[-10:]))
    D_l2 = float(np.sqrt(np.mean(D * D)))
    D_abs_topk = (
        float(np.mean(np.partition(flat_absD, -500)[-500:]))
        if flat_absD.size >= 500
        else float(np.mean(flat_absD))
    )
    D_conc = float(topk_sum / (float(np.sum(flat_absD)) + 1e-6))

    panel_diff_f = A_f - O_f  # (3, 256)
    panel_absdiff_f = np.abs(panel_diff_f)

    panel_absdiff_mean = panel_absdiff_f.mean(axis=1)  # (3,)
    panel_absdiff_std_across_panels = float(panel_absdiff_mean.std())
    panel_absdiff_min = float(panel_absdiff_mean.min())
    panel_absdiff_max = float(panel_absdiff_mean.max())

    panel_pos_frac = np.mean(panel_diff_f > 0.0, axis=1)  # (3,)
    panel_pos_frac_mean = float(panel_pos_frac.mean())
    panel_pos_frac_std = float(panel_pos_frac.std())

    kf = 20
    panel_topk_means = np.empty(3, dtype=np.float32)
    panel_max_over_mean = np.empty(3, dtype=np.float32)
    for i in range(3):
        v = panel_absdiff_f[i]
        if v.size >= kf:
            tk = np.partition(v, -kf)[-kf:]
            panel_topk_means[i] = float(np.mean(tk))
        else:
            panel_topk_means[i] = float(np.mean(v))
        panel_max_over_mean[i] = float((v.max() + 1e-6) / (v.mean() + 1e-6))

    panel_topk_mean_mean = float(panel_topk_means.mean())
    panel_topk_mean_std = float(panel_topk_means.std())
    panel_max_over_mean_mean = float(panel_max_over_mean.mean())
    panel_max_over_mean_std = float(panel_max_over_mean.std())

    panel_diff_l2 = np.sqrt(np.mean(panel_diff_f * panel_diff_f, axis=1))  # (3,)
    panel_diff_l1 = np.mean(np.abs(panel_diff_f), axis=1)  # (3,)
    panel_diff_l2_mean = float(panel_diff_l2.mean())
    panel_diff_l2_std = float(panel_diff_l2.std())
    panel_diff_l1_mean = float(panel_diff_l1.mean())
    panel_diff_l1_std = float(panel_diff_l1.std())
    panel_diff_l2_min = float(panel_diff_l2.min())
    panel_diff_l2_max = float(panel_diff_l2.max())

    def _corr(a, b):
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom0 = np.sqrt(np.sum(a0 * a0)) * np.sqrt(np.sum(b0 * b0)) + 1e-6
        return float(np.sum(a0 * b0) / denom0)

    c01 = _corr(panel_diff_f[0], panel_diff_f[1])
    c02 = _corr(panel_diff_f[0], panel_diff_f[2])
    c12 = _corr(panel_diff_f[1], panel_diff_f[2])
    corr_mean = float((c01 + c02 + c12) / 3.0)
    corr_min = float(min(c01, c02, c12))
    corr_std = float(np.std([c01, c02, c12]))

    absD_sum = float(np.sum(flat_absD) + 1e-6)
    absD_top1000 = (
        float(np.sum(np.partition(flat_absD, -1000)[-1000:]))
        if flat_absD.size >= 1000
        else float(np.sum(flat_absD))
    )
    absD_conc_1000 = float(absD_top1000 / absD_sum)

    feats = np.empty(13 + 5 + 5 + 7 + 19 + 5 + 9 + 10, dtype=np.float32)
    j = 0

    feats[j : j + 13] = np.array(
        [
            diff.mean(),
            diff.std(),
            diff.max(),
            diff.min(),
            absdiff.mean(),
            absdiff.std(),
            ratio.mean(),
            ratio.std(),
            ratio_c.max(),
            ratio_c.min(),
            A_mean - O_mean,
            A_std - O_std,
            A_max - O_max,
        ],
        dtype=np.float32,
    )
    j += 13

    feats[j : j + 5] = diff_q
    j += 5
    feats[j : j + 5] = absdiff_q
    j += 5

    feats[j : j + 7] = np.array(
        [
            pos.mean(),
            pos.sum(),
            neg.mean(),
            neg.sum(),
            np.mean(diff > 0.0),
            A_panel_std - O_panel_std,
            A_t_std - O_t_std,
        ],
        dtype=np.float32,
    )
    j += 7

    feats[j : j + 19] = np.array(
        [
            D.mean(),
            D.std(),
            D.max(),
            D.min(),
            absD.mean(),
            absD.std(),
            grad_t_mean,
            grad_f_mean,
            grad_t_std,
            grad_f_std,
            topk_mean,
            topk_sum,
            row_max,
            col_max,
            row_std,
            col_std,
            D_pos_sum,
            D_neg_sum,
            D_pos_frac,
        ],
        dtype=np.float32,
    )
    j += 19

    feats[j : j + 5] = np.array(
        [diff_l2, diff_abs_topk, D_l2, D_abs_topk, D_conc], dtype=np.float32
    )
    j += 5

    feats[j : j + 9] = np.array(
        [
            panel_absdiff_std_across_panels,
            panel_absdiff_min,
            panel_absdiff_max,
            panel_pos_frac_mean,
            panel_pos_frac_std,
            panel_topk_mean_mean,
            panel_topk_mean_std,
            panel_max_over_mean_mean,
            panel_max_over_mean_std,
        ],
        dtype=np.float32,
    )
    j += 9

    feats[j : j + 10] = np.array(
        [
            panel_diff_l2_mean,
            panel_diff_l2_std,
            panel_diff_l1_mean,
            panel_diff_l1_std,
            panel_diff_l2_min,
            panel_diff_l2_max,
            corr_mean,
            corr_min,
            corr_std,
            absD_conc_1000,
        ],
        dtype=np.float32,
    )
    j += 10

    return feats


extract_features_from_path(train_files[0]).shape



## === cell 3
import multiprocessing as mp


def _init_worker():
    os.environ.setdefault("PYTHONHASHSEED", "42")


def _feat_worker(path):
    return extract_features_from_path(path)


def build_feature_matrix(paths, n_workers=None, chunksize=32):
    d = extract_features_from_path(paths[0]).shape[0]
    X_out = np.empty((len(paths), d), dtype=np.float32)

    if n_workers is None:
        n_workers = max(
            1, min(8, (os.cpu_count() or 2))
        )  # cap for Kaggle-like CPU quotas

    if n_workers == 1:
        for i, p in enumerate(paths):
            X_out[i] = extract_features_from_path(p)
        return X_out

    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for i, feat in enumerate(pool.imap(_feat_worker, paths, chunksize=chunksize)):
            X_out[i] = feat
    return X_out


train_ids = [os.path.splitext(os.path.basename(fp))[0] for fp in train_files]
y = np.fromiter(
    (train_id_to_target[i] for i in train_ids), dtype=np.int32, count=len(train_ids)
)

X = build_feature_matrix(train_files, n_workers=None, chunksize=32)

X.shape, y.shape, y.mean()



## === cell 4
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
                max_iter=400,
                n_jobs=None,
                C=3.0,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)

va_pred = clf.predict_proba(X_va)[:, 1]
va_auc = roc_auc_score(y_va, va_pred)
va_auc



## === cell 5
clf.fit(X, y)

test_ids = [os.path.splitext(os.path.basename(fp))[0] for fp in test_files]
X_test = build_feature_matrix(test_files, n_workers=None, chunksize=64)

test_pred = clf.predict_proba(X_test)[:, 1].astype(np.float32)

sub = pd.DataFrame({"id": np.array(test_ids, dtype=str), "target": test_pred})

assert sub.shape[0] == len(test_files) == 6000, "Unexpected test size / missing files."
assert sub["id"].isna().sum() == 0 and sub["target"].isna().sum() == 0
assert sub["id"].duplicated().sum() == 0, "Duplicate ids in submission."

sub.to_csv("submission.csv", index=False)
sub.head()
