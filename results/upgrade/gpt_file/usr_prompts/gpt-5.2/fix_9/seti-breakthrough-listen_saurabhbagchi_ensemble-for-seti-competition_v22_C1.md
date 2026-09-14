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

0.7571313712626022

# 6. Current score

0.49876

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50899) has done: 'Your notebook fails because it tries to read multiple external Kaggle Dataset/Notebook submissions that don’t exist in this environment (`../input/.../submission.csv`). To keep the core “ensemble blend” logic intact, I replaced those missing inputs with locally generated “base submissions” computed directly from the provided `.npy` snippets using simple, deterministic heuristics, then blend them with the same weights and write `submission.csv`. I also ensure IDs align exactly with `sample_submission.csv` to avoid row-order mismatches and guarantee a valid submission file is produced end-to-end within time limits. This should yield a meaningful AUC (better than random) and move toward your target given you currently have no valid submission.'
- What this solution (achieved 0.50898) has done: 'Most of the timeout comes from repeatedly loading and converting the same 6000 `.npy` files four separate times (once per heuristic), plus extra Python-level loops inside the “drift” feature. To preserve identical logic/semantics while cutting runtime, I load each snippet exactly once, compute all four heuristic scores in that single pass, and keep the same sigmoid/threshold formulas. I also vectorize the drift computation across rows/panels (still exactly “argmax per row then median abs diff”) and avoid building intermediate Python lists/DataFrames until the end. These changes reduce disk I/O by ~4× and remove per-panel Python loops, while keeping results deterministically equivalent up to negligible float rounding.'
- What this solution (achieved 0.48807) has done: 'Your current 0.50898 AUC is close to random, so to move toward the 0.757 target we need a stronger signal while keeping your “single-pass heuristic features + weighted blend” core logic unchanged. The smallest high-impact change is to replace the fixed, untuned sigmoid centers/scales with parameters calibrated on the provided `train_labels.csv` by fitting a 1D logistic regression per heuristic (no architecture/training-loop changes; it’s still a sigmoid of the same feature, just with data-driven shift/scale). We then keep your same 0.45/0.15/0.30/0.10 blend weights and generate the test predictions with those calibrated sigmoids. This typically lifts AUC substantially versus hand-picked constants while staying deterministic and within time.'
- What this solution (achieved 0.50517) has done: 'We keep your exact feature extraction and weighted blending logic, but fix the calibration so it matches the AUC objective better: your current setup fits each 1D logistic on all 54k points, which can overfit the sigmoid scale/shift and collapse test ranking (hurting AUC). The minimal change is to fit each 1D calibration with a small, stratified holdout and use out-of-fold (OOF) predictions on train to learn the final 4-weight blend via a single logistic regression (still just a sigmoid of your 4 heuristic probabilities, preserving evaluation semantics). This typically improves ranking stability substantially while keeping the same four base features and producing a valid `submission.csv`. All I/O paths remain unchanged and runtime stays within limits (features are computed once, then small sklearn fits).'
- What this solution (achieved 0.50156) has done: 'We keep your same four heuristic features and the same “1D sigmoid per feature + blender sigmoid” structure, but make the calibration more AUC-friendly by standardizing each feature using robust (median/IQR) scaling computed on the training fold only. This prevents any one feature with extreme scale/outliers from dominating the logistic fits and typically improves ranking stability (and thus ROC-AUC) with minimal semantic change. We apply the same robust scaling to test using train-derived parameters, then proceed with the same OOF 1D logistic calibrations and final logistic blender. All paths and the submission writing remain unchanged and it still produce a valid `submission.csv`.'
- What this solution (achieved 0.49876) has done: 'Main timeout is from repeatedly loading ~60k `.npy` files (train+test) with single-threaded `np.load` and doing per-sample Python work; the ML part is tiny. I keep the exact same feature definitions and calibration/blending logic, but speed up data access by (1) precomputing id→path without per-id `os.path.exists` loops, (2) using `np.load(..., mmap_mode="r")` to reduce read/copy overhead, and (3) extracting features in parallel with a thread pool (I/O bound) while writing results into preallocated arrays. These changes are equivalent (same arrays read, same computations in `extract_features_single`) and deterministic via ordered assignment. Everything else (LogReg folds, parameters, blend) stays the same.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

DATA_ROOTS = [
    "/kaggle/input/seti-breakthrough-listen",  # typical Kaggle competition mount
    "/kaggle/data",  # user provided alt mirror
    "/kaggle/input",  # fallback
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


base_root = find_first_existing(DATA_ROOTS)
if base_root is None:
    raise FileNotFoundError(
        "Could not find expected Kaggle data root in known locations."
    )

sample_path = None
for cand in [
    os.path.join(base_root, "sample_submission.csv"),
    os.path.join(base_root, "seti-breakthrough-listen", "sample_submission.csv"),
]:
    if os.path.exists(cand):
        sample_path = cand
        break
if sample_path is None:
    raise FileNotFoundError(
        "sample_submission.csv not found under expected data roots."
    )

test_dir = None
for cand in [
    os.path.join(base_root, "test"),
    os.path.join(base_root, "seti-breakthrough-listen", "test"),
]:
    if os.path.isdir(cand):
        test_dir = cand
        break
if test_dir is None:
    raise FileNotFoundError("test/ directory not found under expected data roots.")

train_dir = None
for cand in [
    os.path.join(base_root, "train"),
    os.path.join(base_root, "seti-breakthrough-listen", "train"),
]:
    if os.path.isdir(cand):
        train_dir = cand
        break
if train_dir is None:
    raise FileNotFoundError("train/ directory not found under expected data roots.")

train_labels_path = None
for cand in [
    os.path.join(base_root, "train_labels.csv"),
    os.path.join(base_root, "seti-breakthrough-listen", "train_labels.csv"),
]:
    if os.path.exists(cand):
        train_labels_path = cand
        break
if train_labels_path is None:
    raise FileNotFoundError("train_labels.csv not found under expected data roots.")

sample = pd.read_csv(sample_path)
if list(sample.columns) != ["id", "target"]:
    sample = sample.rename(
        columns={sample.columns[0]: "id", sample.columns[1]: "target"}
    )
sample["id"] = sample["id"].astype(str)
ids = sample["id"].tolist()

train_labels = pd.read_csv(train_labels_path)
train_labels["id"] = train_labels["id"].astype(str)
train_ids = train_labels["id"].tolist()
y_train = train_labels["target"].to_numpy().astype(np.int32)


def build_id_to_path(root_dir):
    all_npy = glob.glob(os.path.join(root_dir, "*", "*.npy"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in all_npy}


id_to_path_test = build_id_to_path(test_dir)
missing = [i for i in ids if i not in id_to_path_test]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} test ids from sample_submission.csv were not found as .npy files. "
        f"Example: {missing[:3]}"
    )

id_to_path_train = build_id_to_path(train_dir)
missing_train = [i for i in train_ids if i not in id_to_path_train]
if missing_train:
    raise FileNotFoundError(
        f"{len(missing_train)} train ids from train_labels.csv were not found as .npy files. "
        f"Example: {missing_train[:3]}"
    )



## === cell 1
from concurrent.futures import ThreadPoolExecutor


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def extract_features_single(x):
    """
    Core feature extraction is unchanged: energy/std/rowmax/drift from A vs O panels.
    Returns raw feature scores (pre-sigmoid), which we will calibrate on train.
    """
    x = x.astype(np.float32)  # (6,273,256)
    A = x[[0, 2, 4]]
    O = x[[1, 3, 5]]
    diff = A - O

    s_energy = float(np.mean(np.abs(diff)))

    s_std = float(np.std(A) - np.std(O))

    pos = diff[diff > 0]
    if pos.size == 0:
        s_rowmax = 0.0
    else:
        s_rowmax = float(np.quantile(pos, 0.995))

    idx_A = np.argmax(A, axis=2).astype(np.float32)  # (3,273)
    idx_O = np.argmax(O, axis=2).astype(np.float32)  # (3,273)
    dA = np.diff(idx_A, axis=1)
    dO = np.diff(idx_O, axis=1)
    drift_A = float(np.mean(np.median(np.abs(dA), axis=1)))
    drift_O = float(np.mean(np.median(np.abs(dO), axis=1)))
    s_drift = float(drift_A - drift_O)

    return s_energy, s_std, s_rowmax, s_drift


def make_all_raw_features(
    ids, id_to_path, batch_size=512, num_workers=None, mmap_mode="r"
):
    n = len(ids)
    f_energy = np.empty(n, dtype=np.float32)
    f_std = np.empty(n, dtype=np.float32)
    f_rowmax = np.empty(n, dtype=np.float32)
    f_drift = np.empty(n, dtype=np.float32)

    if num_workers is None:
        cpu = os.cpu_count() or 4
        num_workers = min(16, max(4, cpu))

    def _one(idx, _id):
        x = np.load(id_to_path[_id], mmap_mode=mmap_mode)
        return idx, extract_features_single(x)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        chunk_ids = ids[start:end]
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for idx, (a, b, c, d) in ex.map(
                _one, range(start, end), chunk_ids, chunksize=32
            ):
                f_energy[idx] = a
                f_std[idx] = b
                f_rowmax[idx] = c
                f_drift[idx] = d

    return f_energy, f_std, f_rowmax, f_drift




## === cell 2
fE_tr, fS_tr, fR_tr, fD_tr = make_all_raw_features(
    train_ids, id_to_path_train, batch_size=512
)



## === cell 3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def robust_params(f):
    f = f.astype(np.float32)
    med = float(np.median(f))
    q25 = float(np.quantile(f, 0.25))
    q75 = float(np.quantile(f, 0.75))
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 1e-8:
        iqr = 1.0
    return med, iqr


def robust_transform(f, med, iqr):
    return ((f.astype(np.float32) - med) / iqr).astype(np.float32)


def oof_1d_logit_probs_robust(f, y, n_splits=5, seed=42):
    """
    Same OOF calibration structure as before, but with fold-wise robust scaling.
    Minimal AUC-improving change: use class_weight='balanced' to handle imbalance,
    which typically improves ranking signal for ROC-AUC without changing core logic.
    """
    f = f.astype(np.float32)
    y = y.astype(np.int32)

    oof = np.empty_like(f, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, va_idx in skf.split(f, y):
        f_tr = f[tr_idx]
        f_va = f[va_idx]

        med, iqr = robust_params(f_tr)
        X_tr = robust_transform(f_tr, med, iqr).reshape(-1, 1)
        X_va = robust_transform(f_va, med, iqr).reshape(-1, 1)
        y_tr = y[tr_idx]

        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=800,
            C=1.0,
            class_weight="balanced",
        )
        lr.fit(X_tr, y_tr)
        oof[va_idx] = lr.predict_proba(X_va)[:, 1].astype(np.float32)

    med_full, iqr_full = robust_params(f)
    X_full = robust_transform(f, med_full, iqr_full).reshape(-1, 1)
    lr_full = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        C=1.0,
        class_weight="balanced",
    )
    lr_full.fit(X_full, y)

    return oof, (med_full, iqr_full), lr_full


pE_oof, (medE, iqrE), lrE = oof_1d_logit_probs_robust(
    fE_tr, y_train, n_splits=5, seed=42
)
pS_oof, (medS, iqrS), lrS = oof_1d_logit_probs_robust(
    fS_tr, y_train, n_splits=5, seed=42
)
pR_oof, (medR, iqrR), lrR = oof_1d_logit_probs_robust(
    fR_tr, y_train, n_splits=5, seed=42
)
pD_oof, (medD, iqrD), lrD = oof_1d_logit_probs_robust(
    fD_tr, y_train, n_splits=5, seed=42
)

X_blend = np.vstack([pE_oof, pS_oof, pR_oof, pD_oof]).T.astype(np.float32)

blend = LogisticRegression(
    solver="lbfgs",
    max_iter=800,
    C=1.0,
    class_weight="balanced",
)
blend.fit(X_blend, y_train)

print("Blend coef/intercept:", blend.coef_.ravel().tolist(), float(blend.intercept_[0]))



## === cell 4
fE_te, fS_te, fR_te, fD_te = make_all_raw_features(ids, id_to_path_test, batch_size=512)

pE_te = lrE.predict_proba(robust_transform(fE_te, medE, iqrE).reshape(-1, 1))[
    :, 1
].astype(np.float32)
pS_te = lrS.predict_proba(robust_transform(fS_te, medS, iqrS).reshape(-1, 1))[
    :, 1
].astype(np.float32)
pR_te = lrR.predict_proba(robust_transform(fR_te, medR, iqrR).reshape(-1, 1))[
    :, 1
].astype(np.float32)
pD_te = lrD.predict_proba(robust_transform(fD_te, medD, iqrD).reshape(-1, 1))[
    :, 1
].astype(np.float32)

X_te_blend = np.vstack([pE_te, pS_te, pR_te, pD_te]).T.astype(np.float32)
pred = blend.predict_proba(X_te_blend)[:, 1].astype(np.float32)

data9 = pd.DataFrame({"id": ids, "target": pred})
data9["target"] = data9["target"].clip(0.0, 1.0)



## === cell 5
data9 = data9[["id", "target"]]
data9.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data9.shape)
print(data9.head())
