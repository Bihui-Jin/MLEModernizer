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

0.7571658004342819

# 6. Current score

0.51613

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51009) has done: 'Your notebook fails because it tries to read six external submissions from `../input/...` that do not exist in this environment, so nothing downstream is defined and no `submission.csv` is written. I replace those missing ensemble inputs with a minimal, local baseline that reads the provided `sample_submission.csv` and creates deterministic predictions from the available `test/` `.npy` files, ensuring the submission has the required `id,target` columns and 6000 rows. This keeps the “inference-only” nature of the original (no training) while producing a valid file end-to-end. I also make the path handling robust to either `/kaggle/input/...` or `/kaggle/data/...` layouts and ensure IDs align exactly to the sample submission order.'
- What this solution (achieved 0.51391) has done: 'Your current inference-only heuristic is too weak (AUC ~0.51), so the smallest safe way to move toward the 0.757 target is to keep the same “A vs (B,C,D)” contrast core idea but extract a slightly more discriminative signal-energy feature. I replace the single global mean(|A−B|) with a robust high-percentile contrast on time-collapsed spectra (which emphasizes narrowband/drifting lines) and combine it with the original global contrast, then apply the same sigmoid mapping. This preserves the no-training pipeline and keeps runtime within limits while typically providing a meaningful AUC lift for this competition. Output format/path stays identical and still writes `submission.csv` with `id,target` and 6000 rows.'
- What this solution (achieved 0.50047) has done: 'Your current inference-only contrast feature is still too weak for this competition (AUC ~0.51), so we should keep the same “A vs (B,C,D)” core idea but make the feature slightly more signal-aware in a way that typically lifts AUC without introducing training. I add one additional component that targets drifting/narrowband lines by taking a high-percentile of the frequency-derivative of the time-collapsed spectrum contrast, then combine it with your existing global and peak features. I also calibrate the sigmoid mapping using robust per-file normalization (median/IQR computed from the same file) so the probabilities spread more sensibly across test without requiring labels. Submission writing, paths, and the end-to-end inference-only flow remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.5155) has done: 'Your current heuristic is producing probabilities that are too tightly centered (per-file median/IQR normalization collapses between-snippet separability), so AUC stays near random. I keep the same inference-only “A vs (B,C,D)” contrast core, but switch to a dataset-level robust calibration: compute the same features for all test snippets, then center/scale once using global median/IQR so scores become comparable across snippets (this typically lifts AUC without any training). I also add one more very cheap, signal-relevant feature that still fits the same core idea: a “max-over-time then percentile-over-frequency” contrast (helps when the signal is present only part of the time). Output format and the submission writing remain unchanged and it still produce `submission.csv` with 6000 rows.'
- What this solution (achieved 0.51613) has done: 'We keep your inference-only “A vs (B,C,D)” contrast logic and the same four features, but make the score mapping more AUC-friendly by switching from a fixed linear weight-sum to a robust rank-based aggregation across features (still unsupervised, no training). This reduces sensitivity to feature scale/outliers and usually improves separability, which should move AUC upward toward your 0.757 target from ~0.515. We also apply a monotonic, globally-calibrated sigmoid on the aggregated score so the submission remains a probability while preserving ranking (AUC-relevant). Output paths and the submission format stay identical and it still writes `submission.csv` with 6000 rows.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd




## === cell 1
def _find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


BASE = _find_existing_path(
    [
        "/kaggle/input/seti-breakthrough-listen",
        "/kaggle/data/seti-breakthrough-listen",
        "/kaggle/input",
        "/kaggle/data",
    ]
)

if BASE is None:
    raise FileNotFoundError(
        "Could not locate Kaggle data directory under /kaggle/input or /kaggle/data."
    )

COMP_ROOT = _find_existing_path(
    [
        os.path.join(BASE, "seti-breakthrough-listen"),
        BASE,
    ]
)

sample_path = _find_existing_path(
    [
        os.path.join(COMP_ROOT, "sample_submission.csv"),
        os.path.join(BASE, "sample_submission.csv"),
    ]
)
test_dir = _find_existing_path(
    [
        os.path.join(COMP_ROOT, "test"),
        os.path.join(BASE, "test"),
    ]
)

if sample_path is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")
if test_dir is None:
    raise FileNotFoundError("test/ directory not found in expected locations.")

sample = pd.read_csv(sample_path)
if list(sample.columns) != ["id", "target"]:
    sample = sample[["id", "target"]].copy()

test_files = glob.glob(os.path.join(test_dir, "*", "*.npy"))
id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in test_files}

missing = [i for i in sample["id"].tolist() if i not in id_to_path]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test .npy files referenced by sample_submission.csv. "
        f"First few missing: {missing[:5]}"
    )


def compute_features_from_npy(path):
    """
    Same inference-only core logic: compare on-target A panels vs off-target (B,C,D) panels.

    We keep the exact same four features as before to preserve semantics and runtime.
    """
    x = np.load(path)  # (6, 273, 256), float16
    x = x.astype(np.float32, copy=False)

    a = x[[0, 2, 4]]
    b = x[[1, 3, 5]]

    global_feat = float(np.mean(np.abs(a - b)))

    a_spec = np.mean(a, axis=1)  # (3, 256)
    b_spec = np.mean(b, axis=1)  # (3, 256)
    diff_spec = np.abs(a_spec - b_spec)  # (3, 256)
    peak_feat = float(np.percentile(diff_spec, 99.5))

    d_diff = np.abs(np.diff(diff_spec, axis=1))  # (3, 255)
    grad_peak_feat = float(np.percentile(d_diff, 99.5))

    a_tmax = np.max(a, axis=1)  # (3, 256)
    b_tmax = np.max(b, axis=1)  # (3, 256)
    tmax_diff = np.abs(a_tmax - b_tmax)  # (3, 256)
    tmax_peak_feat = float(np.percentile(tmax_diff, 99.5))

    return np.array(
        [global_feat, peak_feat, grad_peak_feat, tmax_peak_feat], dtype=np.float64
    )




## === cell 2
sample.head()



## === cell 3
pd.DataFrame({"n_sample_rows": [len(sample)], "n_test_files_found": [len(test_files)]})



## === cell 4
feat_mat = np.zeros((len(sample), 4), dtype=np.float64)
for i, _id in enumerate(sample["id"].tolist()):
    feat_mat[i] = compute_features_from_npy(id_to_path[_id])

weights = np.array([0.54, 0.22, 0.10, 0.14], dtype=np.float64)
weights = weights / weights.sum()

ranks = np.empty_like(feat_mat, dtype=np.float64)
n = feat_mat.shape[0]
for j in range(feat_mat.shape[1]):
    order = np.argsort(feat_mat[:, j], kind="mergesort")
    r = np.empty(n, dtype=np.float64)
    r[order] = np.arange(n, dtype=np.float64)
    ranks[:, j] = r / max(n - 1, 1)

raw = ranks @ weights  # in [0,1]

med = float(np.median(raw))
q75, q25 = np.percentile(raw, [75.0, 25.0])
iqr = float(q75 - q25)
if iqr < 1e-12:
    iqr = 1e-12

z = (raw - med) / (1.35 * iqr)
z = np.clip(z, -8.0, 8.0)
preds = 1.0 / (1.0 + np.exp(-z))

submission = pd.DataFrame(
    {"id": sample["id"].values, "target": preds.astype(np.float64)}
)
submission["target"] = submission["target"].clip(0.0, 1.0)

data6 = submission



## === cell 5
data6.to_csv("submission.csv", index=False)

print(data6.shape)
print(data6.head())
print("Wrote submission.csv")
