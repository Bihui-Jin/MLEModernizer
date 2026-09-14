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

0.7571715757510159

# 6. Current score

0.51103

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48974) has done: 'The timeout is dominated by disk I/O and multiprocessing overhead: each snippet is loaded as a separate `.npy` file (tens of thousands of files), and the current `multiprocessing.Pool.imap` setup also pays heavy IPC cost by shipping numpy arrays back to the parent. I keep the exact same feature math and the same sklearn model/training, but speed up by (1) using `os.scandir`-based path construction, (2) switching to a `ThreadPool` (I/O-bound workload; avoids pickling/IPC and is typically faster for many small files), (3) batching work so each worker returns a block of features (much fewer Python/IPC transactions), and (4) ensuring pools are properly closed and not left dangling. These changes are operationally equivalent (same files, same features, same model) and should bring runtime under 600s on Kaggle’s filesystem.'
- What this solution (achieved 0.49833) has done: 'Your current AUC is far below target, so we should cautiously improve the model score while keeping the same overall approach (handcrafted features + StandardScaler + LogisticRegression). The biggest likely issue is that the features are too weakly aligned to the known cadence structure (signal should appear in A panels 0/2/4 but not in B/C/D 1/3/5), so I add a small set of additional *derived* features from the same data (differences/ratios and simple robust statistics per A vs BCD) without changing the model type or training loop. I also set `class_weight="balanced"` to better handle class imbalance, which typically improves ROC-AUC for linear models without changing evaluation semantics. Everything else (I/O paths, featurization flow, submission writing) remains the same.'
- What this solution (achieved 0.49967) has done: 'We keep the exact same pipeline (handcrafted stats → StandardScaler → LogisticRegression) and only make two small, score-relevant adjustments: (1) switch LogisticRegression to `solver="saga"` with a light elastic-net penalty (still logistic regression; same training semantics) to better handle correlated/partially redundant handcrafted features, and (2) slightly strengthen regularization (tune `C`) to improve generalization toward your target AUC. We also add a deterministic stratified train/validation AUC printout so you can verify that changes move AUC in the right direction before submitting, without changing how the final model is trained for submission. All I/O paths and submission formatting remain unchanged, and the script still runs end-to-end under the same constraints.'
- What this solution (achieved 0.50197) has done: 'Your current AUC (0.49967) is far below the target (0.75717), so we should improve generalization while keeping your exact feature math and the same LogisticRegression pipeline. The most likely score-killer is the combination of `class_weight="balanced"` (which can distort probability ranking for AUC) plus relatively strong regularization (`C=0.5`) on a small handcrafted feature set; I revert class weighting to `None` and relax regularization slightly to let the linear model use the engineered A-vs-BCD features more effectively. I also ensure deterministic, stratified CV reporting remains unchanged and keep I/O and submission formatting identical. These are minimal parameter-only adjustments that typically move ROC-AUC upward for this competition without changing core logic.'
- What this solution (achieved 0.49834) has done: 'Your current score (0.50197) is far below the target (0.75717), so we should improve rank-separation while keeping your exact “handcrafted stats → StandardScaler → LogisticRegression” core approach. The most impactful minimal change is to fit the logistic regression with weaker regularization and without elastic-net shrinkage, because your current elastic-net + relatively strong regularization can collapse predictions toward ~0.5 and kill AUC. Concretely, I keep the exact same feature extraction and training flow, but switch LogisticRegression to standard L2 with `lbfgs` and a modestly larger `C`, which is still logistic regression and preserves evaluation semantics while typically improving ROC-AUC on this competition. I also set `n_jobs=1` for determinism/compatibility with `lbfgs` and keep the same holdout reporting and submission writing.'
- What this solution (achieved 0.51103) has done: 'We keep your exact feature extraction and the same StandardScaler → LogisticRegression pipeline, but make two minimal, score-relevant fixes that should move AUC up from ~0.50 toward your target. First, we remove `random_state` from `LogisticRegression` because it is ignored by `lbfgs` and can mask the real issue: the model is likely saturating due to too-weak regularization (`C=10`) on a small feature set, producing near-constant ranks; we nudge regularization stronger (smaller `C`) to increase usable ranking signal. Second, we switch the scaler to `with_mean=False` (still StandardScaler) to be more numerically stable/consistent with float32 feature matrices and reduce potential mean-centering artifacts on heavy-tailed handcrafted stats, without changing the overall approach or semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input"
TRAIN_LABELS_PATH = os.path.join(BASE_INPUT, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing {TRAIN_LABELS_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 2
def _quantiles_linear_from_flat(x_flat: np.ndarray, qs) -> np.ndarray:
    n = x_flat.size
    if n == 0:
        return np.array([np.nan] * len(qs), dtype=np.float32)

    qs = np.asarray(qs, dtype=np.float64)
    pos = (n - 1) * qs
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)

    idx = np.unique(np.concatenate((lo, hi)))
    part = np.partition(x_flat, idx)

    v_lo = part[lo]
    v_hi = part[hi]
    frac = (pos - lo).astype(np.float64)
    out = (v_lo + (v_hi - v_lo) * frac).astype(np.float32)
    return out


def extract_features(x: np.ndarray) -> np.ndarray:
    """
    Keep the same handcrafted-statistics approach; features unchanged from your current run.
    """
    x32 = x.astype(np.float32, copy=False)

    sums = x32.sum(axis=(1, 2), dtype=np.float32)
    sums2 = (x32 * x32).sum(axis=(1, 2), dtype=np.float32)
    n_panel = np.float32(x32.shape[1] * x32.shape[2])
    panel_mean = sums / n_panel
    panel_var = sums2 / n_panel - panel_mean * panel_mean
    panel_var = np.maximum(panel_var, 0.0, dtype=np.float32)
    panel_std = np.sqrt(panel_var, dtype=np.float32)

    a = x32[[0, 2, 4]]
    bcd = x32[[1, 3, 5]]

    a_sum = np.float32(a.sum(dtype=np.float32))
    bcd_sum = np.float32(bcd.sum(dtype=np.float32))
    a_n = np.float32(a.size)
    bcd_n = np.float32(bcd.size)
    a_mean = a_sum / a_n
    bcd_mean = bcd_sum / bcd_n

    a_sums2 = np.float32((a * a).sum(dtype=np.float32))
    bcd_sums2 = np.float32((bcd * bcd).sum(dtype=np.float32))
    a_var = a_sums2 / a_n - a_mean * a_mean
    bcd_var = bcd_sums2 / bcd_n - bcd_mean * bcd_mean
    a_std = np.float32(np.sqrt(max(a_var, 0.0)))
    bcd_std = np.float32(np.sqrt(max(bcd_var, 0.0)))

    xf = x32.reshape(-1)
    q95, q99 = _quantiles_linear_from_flat(xf, (0.95, 0.99))
    xmax = np.float32(xf.max())
    l2 = np.float32(np.sqrt(np.dot(xf, xf) / xf.size))
    abs_mean = np.float32(np.abs(xf).mean(dtype=np.float32))

    af = a.reshape(-1)
    bf = bcd.reshape(-1)

    a_q50, a_q90, a_q99 = _quantiles_linear_from_flat(af, (0.50, 0.90, 0.99))
    b_q50, b_q90, b_q99 = _quantiles_linear_from_flat(bf, (0.50, 0.90, 0.99))

    eps = np.float32(1e-6)
    mean_diff = np.float32(a_mean - bcd_mean)
    std_diff = np.float32(a_std - bcd_std)
    mean_ratio = np.float32(a_mean / (bcd_mean + eps))
    std_ratio = np.float32(a_std / (bcd_std + eps))

    thr = np.float32(b_q99)
    a_excess_frac = np.float32((af > thr).mean(dtype=np.float32))
    b_excess_frac = np.float32((bf > thr).mean(dtype=np.float32))
    excess_frac_diff = np.float32(a_excess_frac - b_excess_frac)

    a_max = np.float32(af.max())
    b_max = np.float32(bf.max())
    max_diff = np.float32(a_max - b_max)
    max_ratio = np.float32(a_max / (b_max + eps))

    feats = np.concatenate(
        [
            panel_mean.astype(np.float32, copy=False),  # 6
            panel_std.astype(np.float32, copy=False),  # 6
            np.array(
                [
                    mean_diff,
                    std_diff,
                    q95,
                    q99,
                    xmax,
                    l2,
                    abs_mean,
                    a_q50 - b_q50,
                    a_q90 - b_q90,
                    a_q99 - b_q99,
                    mean_ratio,
                    std_ratio,
                    excess_frac_diff,
                    max_diff,
                    max_ratio,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return feats


tmp_id = train_labels["id"].iloc[0]
tmp_path = os.path.join(TRAIN_DIR, tmp_id[0], f"{tmp_id}.npy")
tmp_x = np.load(tmp_path, mmap_mode="r")
print(tmp_x.shape, tmp_x.dtype, extract_features(tmp_x).shape)



## === cell 3
import multiprocessing as mp
from multiprocessing.pool import ThreadPool


def ids_to_paths(root_dir: str, ids: np.ndarray):
    join = os.path.join
    out = [None] * len(ids)
    for i, _id in enumerate(ids):
        out[i] = join(root_dir, _id[0], f"{_id}.npy")
    return out


train_ids = train_labels["id"].to_numpy()
y = train_labels["target"].to_numpy(dtype=np.int64, copy=False)

train_paths = ids_to_paths(TRAIN_DIR, train_ids)

_first_path = train_paths[0]
_tmp = np.load(_first_path, mmap_mode="r")
n_feats = extract_features(_tmp).shape[0]


def _featurize_paths_block(args):
    start_i, paths = args
    m = len(paths)
    block = np.empty((m, n_feats), dtype=np.float32)
    for k, p in enumerate(paths):
        x = np.load(p, mmap_mode="r")
        block[k] = extract_features(x)
    return start_i, block


cpu = os.cpu_count() or 2
n_workers = min(8, cpu)

block_size = 256

X = np.empty((len(train_paths), n_feats), dtype=np.float32)

_pool = ThreadPool(processes=n_workers)
try:
    tasks = []
    for start in range(0, len(train_paths), block_size):
        tasks.append((start, train_paths[start : start + block_size]))

    for start_i, block in _pool.imap_unordered(
        _featurize_paths_block, tasks, chunksize=1
    ):
        X[start_i : start_i + block.shape[0]] = block
finally:
    _pool.close()
    _pool.join()

print("X shape:", X.shape, "y shape:", y.shape, "positive rate:", float(y.mean()))



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

LR_C = 1.0

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                max_iter=4000,
                n_jobs=1,
                C=LR_C,
                class_weight=None,
            ),
        ),
    ]
)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, va_idx = next(sss.split(X, y))
clf.fit(X[tr_idx], y[tr_idx])
va_pred = clf.predict_proba(X[va_idx])[:, 1]
print("Holdout ROC-AUC:", roc_auc_score(y[va_idx], va_pred))

clf.fit(X, y)



## === cell 5
test_ids = sample_sub["id"].to_numpy()
test_paths = ids_to_paths(TEST_DIR, test_ids)

X_test = np.empty((len(test_paths), n_feats), dtype=np.float32)

_pool = ThreadPool(processes=n_workers)
try:
    tasks = []
    for start in range(0, len(test_paths), block_size):
        tasks.append((start, test_paths[start : start + block_size]))

    for start_i, block in _pool.imap_unordered(
        _featurize_paths_block, tasks, chunksize=1
    ):
        X_test[start_i : start_i + block.shape[0]] = block
finally:
    _pool.close()
    _pool.join()

preds = clf.predict_proba(X_test)[:, 1].astype(np.float32, copy=False)
preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": preds})
submission.head(), submission["target"].describe()



## === cell 6
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert list(check.columns) == ["id", "target"]
assert len(check) == len(sample_sub)
print(f"Wrote {out_path} with shape {check.shape}")
print(check.head())
