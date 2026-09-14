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

0.75718

# 6. Current score

0.49662

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49661) has done: 'The timeout is dominated by the fallback path that computes features by loading tens of thousands of `.npy` files and running expensive `std/max/min/percentile` operations, plus repeated full test scaling inside each CV fold. I keep the exact same features, CV, and model, but speed up by (1) replacing the percentile implementation with `np.partition` on a reusable copy buffer (provably equivalent), (2) using a thread pool (I/O bound `np.load`) and larger chunks to cut overhead, (3) vectorizing `X_test` scaling once (same result as scaling each fold), and (4) avoiding repeated scans/list allocations and ensuring faster directory traversal. These changes preserve evaluation semantics and only reduce overhead and repeated work.'
- What this solution (achieved 0.49661) has done: 'Your current score (0.49661 AUC) is far below the target (0.75718), so we should improve performance (not degrade it). The biggest issue is that your “blend” branch (when external submissions are found) is likely producing near-random predictions in this environment because those `/kaggle/input/...` external files aren’t actually available here, and even if they were, the fixed weights are arbitrary; meanwhile the fallback model is weak but at least legitimate. I make the script always run the local CV LogisticRegression feature model (same features, CV, model) and then (only if external submissions truly exist) blend it with them using an out-of-fold optimized weight chosen to maximize AUC on the training labels (this aligns directly with the evaluation metric without changing model architecture/loops). This is a minimal semantic change: it keeps your exact feature extraction and LR training, but fixes the blending logic so it can only help (or do nothing if externals aren’t present), pushing the score upward toward the target.'
- What this solution (achieved 0.49661) has done: 'Your current AUC (0.49661) is far below the target (0.75718), so we should safely improve it without changing the model/features. The biggest likely issue is that the external-submission blending is not reliable here: external files may be missing or misaligned, and even when present, NaNs/constant predictions can drag performance down. I keep your exact feature extraction + 5-fold LR CV, but (1) pre-scale `X_test` once per fold-equivalent (same math, less variability risk), and (2) make blending “do no harm” by only blending if the external mean has non-trivial signal on train (finite + non-constant) and if its best blended OOF AUC is at least as good as the local model. This preserves core semantics and should move the score upward toward the target by preventing harmful blends while still allowing helpful ones.'
- What this solution (achieved 0.49661) has done: 'Your current AUC (0.49661) is far below the target (0.75718), so we should improve it with the smallest changes that preserve your feature set and LogisticRegression CV core logic. The biggest likely quality issue is that your OOF AUC is being measured on (and trained from) a potentially misaligned `labels` order vs. `train_paths`/`X` due to filesystem listing order, and any misalignment collapse AUC toward ~0.5 even if the model is reasonable. I enforce deterministic, correct ID→path alignment by building `X` strictly in the same order as `labels['id']` (with a hard check for missing IDs), and also ensure `sample` and test features align by reindexing from a single `id->path` map. These are “correctness” fixes (not model changes) and are the most plausible minimal way to move AUC upward toward your target without touching architecture/training semantics.'
- What this solution (achieved 0.49661) has done: 'The current AUC being ~0.5 strongly suggests a correctness issue rather than “model weakness”; the most likely culprit is that the `train/` files you featurize are not the same dataset/version as the `train_labels.csv` you read (your `find_existing_path` prefers `/kaggle/data/seti-breakthrough-listen/...` which can be a different copy than `/kaggle/data/...`), causing near-random label↔feature pairing even though IDs “exist”. I make a minimal fix that forces *all* inputs (labels, sample, train dir, test dir) to come from the same resolved root folder, and I add a hard sanity check that detects ID-set mismatch early (instead of silently filtering). This preserves your exact feature extraction + 5-fold LogisticRegression CV and only fixes dataset path consistency/alignment, which should move AUC upward toward the target. I also make external blending align to the same `sample` IDs (already done) and keep the “do no harm” OOF gating unchanged.'
- What this solution (achieved 0.49661) has done: 'Your current AUC (~0.50) is far below target, which strongly suggests a correctness/data-alignment problem rather than “weak model”. I keep your exact feature set and 5-fold LogisticRegression core logic, but (1) force train/test `.npy` discovery to be strictly keyed by the `id` lists from `train_labels.csv` and `sample_submission.csv` (no silent missing→zeros), and (2) add a hard check that the on-disk IDs exactly match the label/sample IDs for the chosen `COMP_ROOT` to prevent accidental root mixing. I also make the test-time scaling mathematically identical to your current per-fold scaling but avoid any possibility of `None` paths or partial featurization corrupting predictions. This should move performance upward toward the target without changing the model/training semantics.'
- What this solution (achieved 0.49662) has done: 'Your current AUC (~0.50) is far below the target, which is most consistent with a remaining correctness issue rather than “weak features”. I keep your exact feature set and 5-fold LogisticRegression CV, but make two minimal, score-relevant fixes: (1) enforce deterministic, stable ID ordering by sorting `train_labels` to match the known directory structure (ids starting with `0`/`1`/...) so the learned signal isn’t accidentally blurred by inconsistent ordering; and (2) align each fold’s test-time scaling to the fold’s scaler (as you do now) while ensuring the test set is featurized once and never re-ordered by any intermediate operations. These changes preserve the same evaluation semantics and should move AUC upward toward the target by removing subtle ordering/alignment failure modes.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
SUB_PATH = "submission.csv"


def _percentile_linear_partition_inplace(work: np.ndarray, q: float) -> float:
    n = work.size
    if n == 0:
        return float("nan")
    pos = (q / 100.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(np.partition(work, lo)[lo])
    np.partition(work, (lo, hi))
    v_lo = float(work[lo])
    v_hi = float(work[hi])
    return v_lo + (pos - lo) * (v_hi - v_lo)


def extract_features(npy_path: str) -> np.ndarray:
    x = np.load(npy_path)  # (6,273,256), float16
    x = x.astype(np.float32, copy=False)

    mean = float(x.mean())
    std = float(x.std())
    mx = float(x.max())
    mn = float(x.min())
    a = float(x[[0, 2, 4]].mean())
    b = float(x[[1, 3, 5]].mean())
    diff = a - b

    flat = x.reshape(-1)
    work = flat.copy()
    p95 = _percentile_linear_partition_inplace(work, 95.0)
    work[:] = flat
    p99 = _percentile_linear_partition_inplace(work, 99.0)

    return np.array([mean, std, mx, mn, diff, p95, p99], dtype=np.float32)


def _job_extract_features(args):
    i, p = args
    return i, extract_features(p)


def _featurize_paths_in_order(paths, max_workers=None, chunksize=256):
    import concurrent.futures

    n = len(paths)
    out = np.zeros((n, 7), dtype=np.float32)
    if n == 0:
        return out

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = min(16, max(4, cpu))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, feats in ex.map(
                _job_extract_features, enumerate(paths), chunksize=chunksize
            ):
                out[i] = feats
        return out
    except Exception as e:
        print(
            f"Warning: ThreadPoolExecutor failed ({type(e).__name__}: {e}); falling back to sequential."
        )
        for i, p in enumerate(paths):
            out[i] = extract_features(p)
        return out


external_paths = [
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    "../input/seti-bl-tf-starter-tpu/submission.csv",
    "../input/seti-learned-image-resizing/submission.csv",
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
]


def try_read_csv(path):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "id" in df.columns and "target" in df.columns:
                return df[["id", "target"]].copy()
    except Exception:
        return None
    return None


data_list = [try_read_csv(p) for p in external_paths]
available = [d for d in data_list if d is not None]
available_paths = [p for p, d in zip(external_paths, data_list) if d is not None]
print(f"Found {len(available)} external submission files.")
if available_paths:
    for p in available_paths:
        print(" -", p)

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

INPUT_ROOTS = [
    "/kaggle/input/seti-breakthrough-listen/seti-breakthrough-listen",
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/data",
]


def resolve_competition_root():
    required = ["train_labels.csv", "sample_submission.csv", "train", "test"]
    for root in INPUT_ROOTS:
        ok = True
        for rel in required:
            if not os.path.exists(os.path.join(root, rel)):
                ok = False
                break
        if ok:
            return root
    return None


COMP_ROOT = resolve_competition_root()
if COMP_ROOT is None:
    raise FileNotFoundError(
        "Could not locate a single root containing train_labels.csv, sample_submission.csv, train/, test/ "
        f"within {INPUT_ROOTS}"
    )

train_labels_path = os.path.join(COMP_ROOT, "train_labels.csv")
sample_sub_path = os.path.join(COMP_ROOT, "sample_submission.csv")
train_dir = os.path.join(COMP_ROOT, "train")
test_dir = os.path.join(COMP_ROOT, "test")
print(f"Using COMP_ROOT={COMP_ROOT}")

labels = pd.read_csv(train_labels_path)
sample = pd.read_csv(sample_sub_path)

labels["id"] = labels["id"].astype(str)
labels = labels.sort_values("id", kind="mergesort").reset_index(drop=True)

sample["id"] = sample["id"].astype(str)
sample = sample.sort_values("id", kind="mergesort").reset_index(drop=True)


def build_id_to_path_map(folder):
    files = glob.glob(os.path.join(folder, "*", "*.npy"))
    id_to_path = {}
    for f in files:
        _id = os.path.splitext(os.path.basename(f))[0]
        if _id not in id_to_path:
            id_to_path[_id] = f
    return id_to_path


train_map = build_id_to_path_map(train_dir)
test_map = build_id_to_path_map(test_dir)

label_ids = labels["id"].values
test_ids = sample["id"].values

train_ids_on_disk = set(train_map.keys())
label_id_set = set(label_ids.tolist())
missing_train_ids = sorted(label_id_set - train_ids_on_disk)
extra_train_ids = sorted(train_ids_on_disk - label_id_set)
if missing_train_ids:
    raise FileNotFoundError(
        f"{len(missing_train_ids)} train label ids are missing from {train_dir} "
        f"(e.g. {missing_train_ids[:5]}). This indicates a dataset/root mismatch."
    )
if extra_train_ids:
    print(
        f"Warning: {len(extra_train_ids)} extra train npy ids exist on disk but not in train_labels.csv "
        f"(e.g. {extra_train_ids[:5]})."
    )

test_ids_on_disk = set(test_map.keys())
test_id_set = set(test_ids.tolist())
missing_test_ids = sorted(test_id_set - test_ids_on_disk)
extra_test_ids = sorted(test_ids_on_disk - test_id_set)
if missing_test_ids:
    raise FileNotFoundError(
        f"{len(missing_test_ids)} sample_submission ids are missing from {test_dir} "
        f"(e.g. {missing_test_ids[:5]}). This indicates a dataset/root mismatch."
    )
if extra_test_ids:
    print(
        f"Warning: {len(extra_test_ids)} extra test npy ids exist on disk but not in sample_submission.csv "
        f"(e.g. {extra_test_ids[:5]})."
    )

train_paths = [train_map[_id] for _id in label_ids]
test_paths = [test_map[_id] for _id in test_ids]

X = _featurize_paths_in_order(train_paths)
y = labels["target"].astype(int).values
X_test = _featurize_paths_in_order(test_paths)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_local = np.zeros(len(sample), dtype=np.float64)
oof_local = np.zeros(len(labels), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_va = scaler.transform(X[va_idx])

    X_te = scaler.transform(X_test)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=400,
        n_jobs=None,
        class_weight=None,
        random_state=42,
    )
    model.fit(X_tr, y[tr_idx])

    oof_local[va_idx] = model.predict_proba(X_va)[:, 1]
    test_preds_local += model.predict_proba(X_te)[:, 1] / skf.get_n_splits()
    print(f"Fold {fold} done.")

try:
    auc_local = roc_auc_score(y, oof_local)
    print(f"OOF AUC (local model): {auc_local:.6f}")
except Exception as e:
    print(f"Warning: could not compute OOF AUC ({type(e).__name__}: {e})")
    auc_local = None

final_test_preds = np.clip(test_preds_local, 0.0, 1.0)

if len(available) >= 1 and auc_local is not None:
    ext_frames = []
    for df in available[:6]:
        df2 = df.copy()
        df2["id"] = df2["id"].astype(str)
        df2["target"] = pd.to_numeric(df2["target"], errors="coerce")
        df2 = df2.dropna(subset=["id"]).drop_duplicates("id")
        ext_frames.append(df2)

    ext_train_mat = []
    for df2 in ext_frames:
        s = df2.set_index("id")["target"].reindex(label_ids)
        ext_train_mat.append(s.values.astype(np.float64))
    ext_train_mat = np.vstack(ext_train_mat)  # (n_ext, n_train)
    ext_train_mean = np.nanmean(ext_train_mat, axis=0)

    m = np.nanmean(ext_train_mean)
    if not np.isfinite(m):
        m = 0.5
    ext_train_mean = np.where(np.isfinite(ext_train_mean), ext_train_mean, m)

    ext_test_mat = []
    for df2 in ext_frames:
        s = df2.set_index("id")["target"].reindex(test_ids)
        ext_test_mat.append(s.values.astype(np.float64))
    ext_test_mat = np.vstack(ext_test_mat)
    ext_test_mean = np.nanmean(ext_test_mat, axis=0)
    mt = np.nanmean(ext_test_mean)
    if not np.isfinite(mt):
        mt = 0.5
    ext_test_mean = np.where(np.isfinite(ext_test_mean), ext_test_mean, mt)

    ext_std = float(np.std(ext_train_mean))
    can_blend = np.isfinite(ext_std) and (ext_std > 1e-6)

    if can_blend:
        best_alpha = 1.0
        best_auc = auc_local
        for alpha in np.linspace(0.0, 1.0, 51):
            blended_oof = alpha * oof_local + (1.0 - alpha) * ext_train_mean
            auc = roc_auc_score(y, blended_oof)
            if auc > best_auc:
                best_auc = auc
                best_alpha = float(alpha)

        print(
            f"Blend search done. best_alpha={best_alpha:.2f}, blended OOF AUC={best_auc:.6f} (local={auc_local:.6f})"
        )

        if best_auc >= auc_local:
            final_test_preds = np.clip(
                best_alpha * test_preds_local + (1.0 - best_alpha) * ext_test_mean,
                0.0,
                1.0,
            )
        else:
            print("Blending rejected (would reduce OOF AUC); using local predictions.")
    else:
        print(
            "Blending skipped (external predictions degenerate/constant); using local predictions."
        )

sub = pd.DataFrame({"id": test_ids, "target": final_test_preds})
sub.to_csv(SUB_PATH, index=False)
print(f"Wrote submission to {SUB_PATH} with {len(sub)} rows.")



## === cell 2
sub_check = pd.read_csv("submission.csv")
print(sub_check.head())
print(sub_check.shape)
assert list(sub_check.columns) == ["id", "target"]
assert sub_check["target"].between(0, 1).all()
assert sub_check["id"].nunique() == len(sub_check)
assert len(sub_check) == 6000
