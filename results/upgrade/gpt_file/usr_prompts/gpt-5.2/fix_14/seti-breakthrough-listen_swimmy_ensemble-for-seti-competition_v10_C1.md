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

0.7570391264032398

# 6. Current score

0.51904

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The notebook is failing because it tries to read six external `../input/.../submission.csv` files that are not present in your Kaggle environment, so none of the `data1..data6` DataFrames exist and the later ensemble/write steps crash. To keep the core “blend multiple submissions” logic intact while making it runnable end-to-end, I add a small loader that (1) searches for each submission file under known local dataset roots and (2) gracefully falls back to the competition `sample_submission.csv` with a neutral constant prediction when a source file is missing. I also make the blending robust by aligning on `id` (avoids silent row-order mismatches) and normalizing weights only across available sources, so you always get a valid `submission.csv` with the correct columns. This should yield a reasonable AUC vs. a constant submission and run within the constraints of the provided packages and paths.'
- What this solution (achieved 0.51394) has done: 'Your current 0.5 score happens because all six external “../input/.../submission.csv” sources are missing here, so you always fall back to a constant 0.5 prediction (AUC=0.5). To move toward the 0.757 target without changing the core “blend submissions” approach, I keep your ensemble structure but replace the missing sources with a lightweight, legitimate model that generates non-constant predictions from the provided train/test snippet .npy files. Specifically, I add a small feature extractor (simple per-panel summary stats emphasizing A vs B/C/D differences), train a logistic regression (sklearn) on train_labels.csv, then use its probabilities as one ensemble source with a high weight while leaving your blending logic intact. This should lift AUC substantially above 0.5 and produce a valid `submission.csv` end-to-end within the 600s limit by using a capped number of training samples.'
- What this solution (achieved 0.50461) has done: 'To move your AUC up toward the 0.757 target without changing the core “simple feature extractor + logistic regression + blend” approach, I make the smallest change that materially improves signal: train on the full `train_labels.csv` instead of the first 12k rows (your current cap is likely underfitting and leaving AUC near ~0.51). To keep runtime within 600s on CPU, I speed up feature extraction by replacing expensive `np.percentile` with a much cheaper “top-k mean” proxy for high-intensity structure and by streaming features into preallocated numpy arrays. Everything else (model type, loss/solver, submission format, and blending logic) stays the same, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.50349) has done: 'The timeout is dominated by per-file feature extraction over ~54k train + 6k test `.npy` files (disk I/O + many NumPy passes per file), plus an extra per-id path existence check. I keep the exact same feature definitions and logistic-regression training, but eliminate redundant work by (1) precomputing the full list of existing `.npy` paths once and joining labels to those paths, (2) batching feature computation with a faster, single-load-per-file routine that reuses shared intermediates and avoids repeated stack/allocations, and (3) using `np.load(..., mmap_mode="r")` and `os.scandir`-based file listing to reduce I/O overhead. These changes are mathematically equivalent to the original features (negligible FP diffs only) and preserve the same model, training loop, and blending semantics. The rest of the script remains unchanged.'
- What this solution (achieved 0.50327) has done: 'Your current AUC (~0.503) indicates the logistic-regression source is producing nearly constant/weakly-informative predictions, so the blend remains close to random. To move toward the 0.757 target with minimal changes and identical core approach (same feature extractor + LogisticRegression + blending), I (1) fix a silent bug where missing test files are treated as all-zero features (not neutral), (2) add `class_weight="balanced"` to make LR learn from the imbalanced target distribution, and (3) apply a tiny, validation-free probability sharpening (power transform) to increase ranking separation, which typically improves ROC-AUC without changing semantics. These changes keep runtime similar, don’t alter the feature set or training loop structure, and still write a valid `submission.csv`.'
- What this solution (achieved 0.50396) has done: 'The timeout is dominated by feature extraction over ~54k train + 6k test `.npy` files, with heavy per-file NumPy reductions and unnecessary duplicate work (test features are computed twice) plus high process-pool overhead. I keep the exact same features and logistic-regression pipeline, but make extraction cheaper by (1) reading each `.npy` once and computing all “panel” stats in a single pass over the stacked `(6,273,256)` array, (2) computing top-k means for all 6 panels via one `argpartition` call, and (3) removing the duplicated test extraction loop. I also switch to a thread pool (NumPy reductions release the GIL and this avoids large inter-process serialization) and use chunked `map` to reduce scheduling overhead, while keeping deterministic behavior and identical semantics. These changes reduce constant factors and eliminate wasted computation without changing the algorithm or outputs beyond negligible float-order effects.'
- What this solution (achieved 0.5078) has done: 'Your current ~0.504 AUC strongly suggests the logistic-regression source is not learning a meaningful ranking (likely because the snippet values are normalized and your raw mean/std-type features are close to uninformative), so the blend stays near-random. To move toward the 0.757 target while preserving the exact same “handcrafted features → LogisticRegression → blend” core logic, I add a tiny set of drift/line-sensitive features that are cheap to compute and well-matched to the competition (needles often look like thin diagonal/ridge structures in A panels). Specifically, I add (1) a frequency-gradient energy feature, (2) a time-gradient energy feature, and (3) a “diagonal consistency” feature (mean abs diff between adjacent time rows shifted by ±1 frequency bin), all computed for A vs off-target panels and appended to your existing 33 features. Everything else (training, model, weights, submission writing, paths) remains unchanged, and it still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.51799) has done: 'Your current score is far below the target, so we should improve ranking quality with the smallest changes that keep your “handcrafted features → LogisticRegression → blend” core intact. The main issue is that the strongest line-like evidence is in the *A vs off-target difference*, but the current ridge/gradient features are only using plain means (which can wash out needle-only-in-A structure). I keep your feature extractor and LR, but add a tiny set of *contrast ridge* features computed directly on A−O paired differences (A0−B1, A2−C3, A4−D5), which is closely aligned with the cadence logic and usually boosts ROC-AUC. I also set the LR regularization strength via a very small fixed grid (still LogisticRegression, no CV) by fitting a few Cs and picking the one with the best training AUC (cheap and deterministic), then generate the submission exactly as before.'
- What this solution (achieved 0.51904) has done: 'I keep your existing “handcrafted features → LogisticRegression → blend” core intact and make only small, score-relevant adjustments that typically improve ROC-AUC ranking. First, I fix a subtle but impactful bug in your variance/contrast normalization (you’re mixing per-panel means with a global mean), replacing it with a mathematically consistent per-panel std computation for A and off-target panels. Second, I add a tiny amount of robustness to the feature scale by using a slightly larger fixed C grid (still no CV) and selecting by training AUC as you already do, which often moves performance upward without changing the approach. Everything else (data paths, feature set structure, LR pipeline, blending, and submission writing) stays the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

BASE_CANDIDATES = [
    Path("/kaggle/input"),
    Path("/kaggle/data"),
    Path("/kaggle/working"),
]



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)




## === cell 2
def _resolve_under_bases(*relative_candidates: str) -> str | None:
    for rel_path in relative_candidates:
        rel = Path(rel_path)
        if rel.is_absolute() and rel.exists():
            return str(rel)
        if rel.exists():
            return str(rel.resolve())
        for base in BASE_CANDIDATES:
            cand = (base / rel).resolve()
            if cand.exists():
                return str(cand)
    return None


SAMPLE_PATH = _resolve_under_bases(
    "sample_submission.csv",
    "seti-breakthrough-listen/sample_submission.csv",
)
if SAMPLE_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under /kaggle/input, /kaggle/data, or /kaggle/working"
    )

sample = pd.read_csv(SAMPLE_PATH)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv must have columns id,target but has: {list(sample.columns)}"
    )


def load_submission_or_fallback(
    rel_path: str, fallback_target: float = 0.5
) -> pd.DataFrame:
    p = _resolve_under_bases(rel_path)
    if p is None:
        df = sample.copy()
        df["target"] = float(fallback_target)
        return df
    df = pd.read_csv(p)
    if "id" not in df.columns:
        if df.index.name == "id":
            df = df.reset_index()
        else:
            raise ValueError(f"{p} missing 'id' column.")
    if "target" not in df.columns:
        for alt in ["prediction", "pred", "y", "label"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "target"})
                break
    if "target" not in df.columns:
        raise ValueError(f"{p} missing 'target' column.")
    df = df[["id", "target"]].copy()
    df["target"] = (
        pd.to_numeric(df["target"], errors="coerce")
        .fillna(fallback_target)
        .astype(float)
    )
    df["target"] = df["target"].clip(0.0, 1.0)
    return df


data1 = load_submission_or_fallback(
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv"
)
data2 = load_submission_or_fallback(
    "../input/seti-bl-spatial-info-tf-tpu/submission.csv"
)
data3 = load_submission_or_fallback("../input/seti-bl-tf-starter-tpu/submission.csv")
data4 = load_submission_or_fallback(
    "../input/seti-learned-image-resizing/submission.csv"
)
data5 = load_submission_or_fallback(
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv"
)
data6 = load_submission_or_fallback(
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv"
)



## === cell 3
data1.head()



## === cell 4
data2.head()



## === cell 5
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

TRAIN_LABELS_PATH = _resolve_under_bases(
    "train_labels.csv",
    "seti-breakthrough-listen/train_labels.csv",
)
if TRAIN_LABELS_PATH is None:
    raise FileNotFoundError("Could not locate train_labels.csv under known roots.")

TRAIN_DIR = _resolve_under_bases("train", "seti-breakthrough-listen/train")
TEST_DIR = _resolve_under_bases("test", "seti-breakthrough-listen/test")
if TRAIN_DIR is None or TEST_DIR is None:
    raise FileNotFoundError(
        "Could not locate train/ and test/ directories under known roots."
    )

TRAIN_DIR = Path(TRAIN_DIR)
TEST_DIR = Path(TEST_DIR)

labels = pd.read_csv(TRAIN_LABELS_PATH)
labels["id"] = labels["id"].astype(str)

use_ids = labels["id"].values


def _list_npy_id_to_path(root: Path) -> dict:
    out = {}
    root_str = str(root)
    for d in "0123456789abcdef":
        sub = os.path.join(root_str, d)
        if not os.path.isdir(sub):
            continue
        with os.scandir(sub) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".npy"):
                    out[e.name[:-4]] = e.path
    return out


def _topk_mean_panels(x6: np.ndarray, k: int = 256) -> np.ndarray:
    flat = x6.reshape(x6.shape[0], -1)
    k = int(min(k, flat.shape[1]))
    if k <= 0:
        return flat.mean(axis=1, dtype=np.float32)
    idx = np.argpartition(flat, -k, axis=1)[:, -k:]
    rows = np.arange(flat.shape[0])[:, None]
    return flat[rows, idx].mean(axis=1, dtype=np.float32)


def _central_moments_from_raw(x: np.ndarray) -> tuple[float, float]:
    xf = x.reshape(-1).astype(np.float32, copy=False)
    m1 = float(xf.mean())
    m2 = float(np.mean(xf * xf))
    m3 = float(np.mean(xf * xf * xf))
    m4 = float(np.mean(xf * xf * xf * xf))
    v = m2 - m1 * m1
    if v <= 1e-12:
        return 0.0, 0.0
    s = float(np.sqrt(v))
    inv_s3 = 1.0 / (s * s * s + 1e-12)
    inv_s4 = 1.0 / (s * s * s * s + 1e-12)
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 * m1 * m1 * m1
    skew = float(mu3 * inv_s3)
    kurt_ex = float(mu4 * inv_s4 - 3.0)
    return skew, kurt_ex


def _mean_std_axis(a: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    m = a.mean(axis=axis, dtype=np.float32)
    m2 = np.mean(a * a, axis=axis, dtype=np.float32)
    v = np.maximum(m2 - m * m, 0.0)
    return m, np.sqrt(v, dtype=np.float32)


def _panel_ridge_features(x2d: np.ndarray) -> tuple[float, float, float, float]:
    """
    Returns (grad_f, grad_t, diag_p1, diag_m1) for one (273,256) panel.
    These capture line-like/diagonal structure via local differences.
    """
    x = x2d.astype(np.float32, copy=False)

    df = x[:, 1:] - x[:, :-1]
    grad_f = float(np.mean(np.abs(df), dtype=np.float32))

    dt = x[1:, :] - x[:-1, :]
    grad_t = float(np.mean(np.abs(dt), dtype=np.float32))

    d1 = x[:-1, :-1] - x[1:, 1:]
    diag_p1 = float(np.mean(np.abs(d1), dtype=np.float32))

    d2 = x[:-1, 1:] - x[1:, :-1]
    diag_m1 = float(np.mean(np.abs(d2), dtype=np.float32))

    return grad_f, grad_t, diag_p1, diag_m1


def extract_features_from_path(p: str) -> np.ndarray:
    x = np.load(p, mmap_mode="r")  # (6, 273, 256) float16
    x = x.astype(np.float32, copy=False)

    panel_means = x.mean(axis=(1, 2), dtype=np.float32)  # (6,)
    panel_absmeans = np.abs(x).mean(axis=(1, 2), dtype=np.float32)
    panel_max = x.max(axis=(1, 2))  # (6,)
    panel_topk = _topk_mean_panels(x, k=256)  # (6,)

    A_idx = np.array([0, 2, 4])
    O_idx = np.array([1, 3, 5])

    A_mean = float(panel_means[A_idx].mean())
    O_mean = float(panel_means[O_idx].mean())

    A_var = float(
        ((x[A_idx] - panel_means[A_idx, None, None]) ** 2)
        .mean(axis=(1, 2), dtype=np.float32)
        .mean()
    )
    O_var = float(
        ((x[O_idx] - panel_means[O_idx, None, None]) ** 2)
        .mean(axis=(1, 2), dtype=np.float32)
        .mean()
    )
    A_std = float(np.sqrt(A_var))
    O_std = float(np.sqrt(O_var))

    A_absmean = float(panel_absmeans[A_idx].mean())
    O_absmean = float(panel_absmeans[O_idx].mean())

    A_hi = float(panel_topk[A_idx].mean())
    O_hi = float(panel_topk[O_idx].mean())

    A_max = float(panel_max[A_idx].max())
    O_max = float(panel_max[O_idx].max())

    d_mean = A_mean - O_mean
    d_absmean = A_absmean - O_absmean
    d_hi = float(A_hi - O_hi)
    d_max = A_max - O_max

    _, tstd = _mean_std_axis(x, axis=2)  # (6,273)
    _, fstd = _mean_std_axis(x, axis=1)  # (6,256)
    A_tstd = float(tstd[A_idx].mean())
    O_tstd = float(tstd[O_idx].mean())
    A_fstd = float(fstd[A_idx].mean())
    O_fstd = float(fstd[O_idx].mean())

    A_panel_max_mean = float(panel_max[A_idx].mean())
    O_panel_max_mean = float(panel_max[O_idx].mean())
    d_panel_max_mean = A_panel_max_mean - O_panel_max_mean

    A_panel_mean_std = float(panel_means[A_idx].astype(np.float32).std())
    O_panel_mean_std = float(panel_means[O_idx].astype(np.float32).std())

    eps = 1e-6
    r_mean = float((A_mean + eps) / (O_mean + eps))
    r_hi = float((A_hi + eps) / (O_hi + eps))

    A_sk0, A_ku0 = _central_moments_from_raw(x[0])
    A_sk2, A_ku2 = _central_moments_from_raw(x[2])
    A_sk4, A_ku4 = _central_moments_from_raw(x[4])
    O_sk1, O_ku1 = _central_moments_from_raw(x[1])
    O_sk3, O_ku3 = _central_moments_from_raw(x[3])
    O_sk5, O_ku5 = _central_moments_from_raw(x[5])

    A_sk = (A_sk0 + A_sk2 + A_sk4) / 3.0
    O_sk = (O_sk1 + O_sk3 + O_sk5) / 3.0
    A_ku = (A_ku0 + A_ku2 + A_ku4) / 3.0
    O_ku = (O_ku1 + O_ku3 + O_ku5) / 3.0
    d_sk = float(A_sk - O_sk)
    d_ku = float(A_ku - O_ku)

    A_contrast = float(
        (
            np.abs(x[0] - x[1]).mean()
            + np.abs(x[2] - x[3]).mean()
            + np.abs(x[4] - x[5]).mean()
        )
        / 3.0
    )
    norm = float(A_std + O_std + 1e-6)
    A_contrast_n = float(A_contrast / norm)

    ridge = np.array([_panel_ridge_features(x[i]) for i in range(6)], dtype=np.float32)
    A_r = ridge[A_idx].mean(axis=0)
    O_r = ridge[O_idx].mean(axis=0)
    d_r = A_r - O_r
    A_grad_f, A_grad_t, A_diag_p1, A_diag_m1 = [float(v) for v in A_r]
    O_grad_f, O_grad_t, O_diag_p1, O_diag_m1 = [float(v) for v in O_r]
    d_grad_f, d_grad_t, d_diag_p1, d_diag_m1 = [float(v) for v in d_r]

    dif01 = x[0] - x[1]
    dif23 = x[2] - x[3]
    dif45 = x[4] - x[5]
    dif_ridge = np.array(
        [
            _panel_ridge_features(dif01),
            _panel_ridge_features(dif23),
            _panel_ridge_features(dif45),
        ],
        dtype=np.float32,
    ).mean(axis=0)
    dif_grad_f, dif_grad_t, dif_diag_p1, dif_diag_m1 = [float(v) for v in dif_ridge]

    feats = np.array(
        [
            A_mean,
            O_mean,
            d_mean,
            A_std,
            O_std,
            A_absmean,
            O_absmean,
            d_absmean,
            A_hi,
            O_hi,
            d_hi,
            A_max,
            O_max,
            d_max,
            A_tstd,
            O_tstd,
            A_fstd,
            O_fstd,
            A_panel_max_mean,
            O_panel_max_mean,
            d_panel_max_mean,
            A_panel_mean_std,
            O_panel_mean_std,
            r_mean,
            r_hi,
            A_sk,
            O_sk,
            d_sk,
            A_ku,
            O_ku,
            d_ku,
            A_contrast,
            A_contrast_n,
            A_grad_f,
            O_grad_f,
            d_grad_f,
            A_grad_t,
            O_grad_t,
            d_grad_t,
            A_diag_p1,
            O_diag_p1,
            d_diag_p1,
            A_diag_m1,
            O_diag_m1,
            d_diag_m1,
            dif_grad_f,
            dif_grad_t,
            dif_diag_p1,
            dif_diag_m1,
        ],
        dtype=np.float32,
    )
    return feats


train_id2path = _list_npy_id_to_path(TRAIN_DIR)
test_id2path = _list_npy_id_to_path(TEST_DIR)

label_map = labels.set_index("id")["target"].to_dict()
n_feats = 49

from concurrent.futures import ThreadPoolExecutor


def _feat_job_train(_id: str):
    p = train_id2path.get(_id)
    if p is None:
        return None
    return _id, extract_features_from_path(p), int(label_map[_id])


kept_rows = []
missing_train = 0
use_ids_str = [str(i) for i in use_ids]

max_workers = min(8, (os.cpu_count() or 2))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for res in ex.map(_feat_job_train, use_ids_str, chunksize=256):
        if res is None:
            missing_train += 1
            continue
        kept_rows.append(res)

kept = len(kept_rows)
X_train = np.zeros((kept, n_feats), dtype=np.float32)
y_train = np.zeros((kept,), dtype=np.int64)
for i, (_id, feats, y) in enumerate(kept_rows):
    X_train[i] = feats
    y_train[i] = y

if X_train.shape[0] < 100:
    model_pred_test = pd.Series(0.5, index=sample["id"].astype(str))
    missing_test = 0
else:
    Cs = [0.05, 0.2, 1.0, 5.0, 20.0]
    best_auc = -1.0
    best_clf = None

    for C in Cs:
        clf_try = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lr",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=600,
                        n_jobs=1,
                        C=float(C),
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        clf_try.fit(X_train, y_train)
        p_tr = clf_try.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, p_tr)
        if auc > best_auc:
            best_auc = auc
            best_clf = clf_try

    clf = best_clf

    test_ids = sample["id"].astype(str).values

    def _feat_job_test(_id: str):
        p = test_id2path.get(_id)
        if p is None:
            return None
        return _id, extract_features_from_path(p)

    X_test = np.full((len(test_ids), n_feats), np.nan, dtype=np.float32)
    id2pos = {str(_id): i for i, _id in enumerate(test_ids)}
    missing_test = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_feat_job_test, list(test_ids), chunksize=256):
            if res is None:
                missing_test += 1
                continue
            _id, feats = res
            X_test[id2pos[_id]] = feats

    ok = np.isfinite(X_test).all(axis=1)
    proba = np.full((len(test_ids),), 0.5, dtype=np.float64)
    if ok.any():
        proba_ok = clf.predict_proba(X_test[ok])[:, 1].astype(np.float64)
        proba[ok] = proba_ok

    gamma = 1.0
    if gamma != 1.0:
        p_clip = np.clip(proba, 1e-6, 1 - 1e-6)
        logit = np.log(p_clip / (1.0 - p_clip))
        proba = 1.0 / (1.0 + np.exp(-gamma * logit))

    model_pred_test = pd.Series(proba, index=test_ids)

data_model = sample.copy()
data_model["id"] = data_model["id"].astype(str)
data_model["target"] = model_pred_test.reindex(data_model["id"]).fillna(0.5).values
data_model["target"] = data_model["target"].astype(float).clip(0.0, 1.0)

print(
    "Trained logistic model on rows:",
    int(X_train.shape[0]),
    "missing_train_files:",
    int(missing_train),
    "missing_test_files:",
    int(missing_test) if "missing_test" in locals() else -1,
    "test_rows:",
    int(len(data_model)),
    "n_features:",
    int(n_feats),
)
if "best_auc" in locals():
    print("Selected C via training AUC:", float(best_auc))



## === cell 6
base = sample[["id"]].copy()
base["id"] = base["id"].astype(str)


def align_targets(df: pd.DataFrame, base_ids: pd.Series) -> pd.Series:
    s = df.copy()
    s["id"] = s["id"].astype(str)
    s = s.set_index("id")["target"]
    return s.reindex(base_ids).fillna(0.5).astype(float)


t6_orig = align_targets(data6, base["id"])
t5 = align_targets(data5, base["id"])
t4 = align_targets(data4, base["id"])
t2 = align_targets(data2, base["id"])
t3 = align_targets(data3, base["id"])
t_model = align_targets(data_model, base["id"])


def is_constant(s: pd.Series, tol: float = 1e-12) -> bool:
    return float(s.std()) < tol


w_model, w5, w4, w6, w2, w3 = 0.90, 0.05, 0.03, 0.02, 0.0, 0.0

sources = [
    ("t_model", t_model, w_model),
    ("t5", t5, w5),
    ("t4", t4, w4),
    ("t6", t6_orig, w6),
    ("t2", t2, w2),
    ("t3", t3, w3),
]

active = [(name, s, w) for (name, s, w) in sources if (w != 0.0 and not is_constant(s))]
if len(active) == 0:
    blended = pd.Series(0.5, index=base["id"])
else:
    wsum = sum(w for _, _, w in active)
    blended = sum((w / wsum) * s for _, s, w in active)

submission = base.copy()
submission["target"] = blended.values
submission["target"] = submission["target"].clip(0.0, 1.0)



## === cell 7
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("target summary:", submission["target"].describe())
