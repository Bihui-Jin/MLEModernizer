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

0.7571323872181847

# 6. Current score

0.50586

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to ensemble external Kaggle dataset submissions that do not exist in your current environment (`../input/...`). To keep the core “weighted-ensemble of submissions” logic intact while making it run end-to-end, I add a safe loader that uses those files if present, otherwise falls back to the provided `sample_submission.csv` (or a constant prediction) so a valid `submission.csv` is always produced. I also fix the aliasing/overwrite bug where `data6` was reused incorrectly, and I align/merge on `id` to prevent row-order mismatches. This should run reliably and generate a correctly formatted submission file.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by the fallback baseline builder: it scans directories repeatedly (`os.walk` inside `npy_path_for_id` for every id) and loads/featurizes tens of thousands of `.npy` files in pure Python loops. I make the baseline path resolution O(1) by precomputing an `{id: path}` index once (using `os.scandir`), and I compute features faster by avoiding extra copies/casts, using `np.partition` (exact quantiles for fixed indices) instead of `np.percentile`, and parallelizing feature extraction with `ProcessPoolExecutor` while keeping results deterministic. These changes preserve the exact feature definitions and the logistic-regression CV logic, but remove the worst constant factors and redundant filesystem work so it can fit under 600 seconds.'
- What this solution (achieved 0.5) has done: 'I fix the crash by making the multiprocessing worker function picklable (move it to top-level) and by using a safer multiprocessing start method that works in Kaggle notebooks. This keeps the exact same feature definitions and LogisticRegression CV training logic, but allows the baseline builder to run end-to-end instead of failing and producing the constant 0.5 submission. I also ensure the baseline is actually used when the external ensemble files are missing (which they are here), so the score should move up toward the target range. Finally, I keep the submission format aligned to `sample_submission.csv` and always write `submission.csv`.'
- What this solution (achieved 0.5) has done: 'The timeout is almost certainly coming from the fallback “baseline builder”: it scans the entire train/test directories and then loads tens of thousands of `.npy` files, computing multiple `np.percentile` calls per file—this is very expensive and can easily exceed 600s. The fastest safe fix is to make the baseline builder effectively impossible to trigger when external submissions are missing, by treating constant-fallback external submissions (all 0.5) as “not available” and then skipping baseline training anyway (keeping the ensemble behavior exactly as-is: default 0.5 when externals are absent). Additionally, when baseline is enabled (e.g., in other environments), we reduce overhead without changing semantics by (1) building the file index via deterministic globbing (fewer Python-level calls than nested scandir), (2) computing all percentiles for a file in one `np.percentile` call, and (3) using `ThreadPoolExecutor` for IO-bound loading instead of process spawning overhead.'
- What this solution (achieved 0.51088) has done: 'Your current 0.5 score is coming from ensembling six missing external submission files, so every input becomes the constant fallback 0.5; the ensemble therefore cannot beat random. To move the score upward toward the 0.757 target with minimal change, I enable the existing “baseline builder” (LogisticRegression on simple handcrafted features) only when all externals are effectively constant/missing. I also keep the ensemble logic intact when real external files exist, so this remains stable in other environments. Finally, I keep the submission aligned to `sample_submission.csv` ids and always write a valid `submission.csv`.'
- What this solution (achieved 0.50265) has done: 'Your score is stuck near random because the ensemble inputs are missing and therefore constant 0.5, while the fallback baseline is a very weak 10-feature logistic regression; to move toward the 0.757 target with minimal disruption, we keep the same baseline model and CV loop but strengthen the features slightly without changing the overall training approach. Specifically, we (1) add a small, cheap set of additional handcrafted features that capture “on-target vs off-target” contrast and time/frequency structure (still just deterministic numpy stats), (2) standardize features using `StandardScaler` inside each CV fold via a `Pipeline` (LogisticRegression is sensitive to feature scale), and (3) ensure we actually use the baseline whenever externals are constant/missing (as intended). These are minimal, metric-aligned changes that should improve AUC materially while keeping the core logic (feature extraction + 5-fold LR averaging) intact and producing the same `submission.csv` format.'
- What this solution (achieved 0.50772) has done: 'The timeout is almost certainly coming from the fallback “baseline builder”: it recursively globs every `.npy` file and then computes many `np.percentile` calls per file over tens of thousands of files, which is very slow in pure NumPy/Python and disk I/O heavy. I keep the exact same feature definitions and CV LogisticRegression ensemble, but make it fast by (1) building the id→path index without recursive globbing (directly probing the known 16 subfolders), (2) avoiding repeated percentile passes by sorting each flattened array once and reading all required quantiles from that single sorted copy (exactly equivalent to NumPy’s default percentile method), and (3) reducing Python overhead and thread scheduling overhead with tuned batching and fewer conversions. This preserves the algorithm’s core logic and produces numerically equivalent (or negligibly different) floats while dramatically cutting runtime.'
- What this solution (achieved 0.50778) has done: 'Your score is still near-random because the final ensemble stays effectively constant (external submissions are missing, and the code only switches to the baseline when the ensemble is exactly 0.5 everywhere). I keep your current “external weighted ensemble + baseline fallback” logic, but make the fallback actually trigger when externals are missing/constant by routing the final prediction to the baseline whenever `external_available` is false (the same condition you already use to build it). I also strengthen the baseline slightly without changing its training approach (still 5-fold LogisticRegression with StandardScaler): add `class_weight="balanced"` to handle class imbalance, which usually improves AUC with minimal risk. These are small, metric-aligned changes intended to move AUC up toward the 0.757 target without changing the overall solution structure or producing invalid submissions.'
- What this solution (achieved 0.50586) has done: 'Your current score is near-random because the “baseline fallback” is still too weak; the external ensemble inputs are missing in this environment, so performance depends entirely on that baseline. To move AUC up toward the 0.757 target without changing the overall approach (handcrafted features → 5-fold LogisticRegression averaging), I make two minimal, metric-aligned upgrades: (1) add a few very cheap but informative features that capture “A vs B” cadence structure (correlations and robust energy/contrast stats), and (2) slightly tune LogisticRegression regularization (still liblinear) to better fit the expanded standardized feature space. The ensemble logic, training loop structure, and submission formatting remain the same, and it still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_ROOTS = [
    "/kaggle/input",  # Kaggle standard
    "/kaggle/data",  # provided environment mirror
    "/kaggle/data/seti-breakthrough-listen",
]


def find_first_existing(rel_path_candidates):
    for base in [""] + DATA_ROOTS:
        for rel in rel_path_candidates:
            p = os.path.join(base, rel) if base else rel
            if os.path.exists(p):
                return p
    return None


SAMPLE_PATH = find_first_existing(
    [
        "sample_submission.csv",
        "seti-breakthrough-listen/sample_submission.csv",
    ]
)
if SAMPLE_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected locations."
    )

sample = pd.read_csv(SAMPLE_PATH)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv must contain columns id,target; got {sample.columns.tolist()}"
    )
sample = sample[["id", "target"]].copy()


def load_submission_or_fallback(path, fallback_df, default_target=0.5):
    """
    Try to load a submission CSV from `path`. If it doesn't exist, fallback to `fallback_df`
    with target filled to `default_target`.
    Returns dataframe with columns: id, target
    """
    if path is not None and os.path.exists(path):
        df = pd.read_csv(path)
        if "id" not in df.columns:
            raise ValueError(f"{path} missing 'id' column.")
        if "target" not in df.columns:
            raise ValueError(f"{path} missing 'target' column.")
        df = df[["id", "target"]].copy()
        return df
    fb = fallback_df[["id"]].copy()
    fb["target"] = float(default_target)
    return fb


def align_on_id(dfs):
    """
    Inner-merge all dfs on 'id' and return merged frame.
    Assumes each df has 'id' and 'target' columns.
    """
    merged = None
    for i, df in enumerate(dfs):
        df2 = df.rename(columns={"target": f"target_{i}"})
        merged = (
            df2
            if merged is None
            else merged.merge(df2, on="id", how="inner", validate="one_to_one")
        )
    return merged




## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 2
paths = {
    "data1": "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "data2": "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    "data3": "../input/seti-bl-tf-starter-tpu/submission.csv",
    "data4": "../input/seti-learned-image-resizing/submission.csv",
    "data5": "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "data6": "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
}

data1 = load_submission_or_fallback(paths["data1"], sample, default_target=0.5)
data2 = load_submission_or_fallback(paths["data2"], sample, default_target=0.5)
data3 = load_submission_or_fallback(paths["data3"], sample, default_target=0.5)
data4 = load_submission_or_fallback(paths["data4"], sample, default_target=0.5)
data5 = load_submission_or_fallback(paths["data5"], sample, default_target=0.5)
data6 = load_submission_or_fallback(paths["data6"], sample, default_target=0.5)



## === cell 3
data1.head()



## === cell 4
data2.head()



## === cell 5
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_LABELS_PATH = find_first_existing(
    ["train_labels.csv", "seti-breakthrough-listen/train_labels.csv"]
)
TRAIN_DIR = find_first_existing(["train", "seti-breakthrough-listen/train"])
TEST_DIR = find_first_existing(["test", "seti-breakthrough-listen/test"])


def is_constant_pred(df, val=0.5):
    t = df["target"].to_numpy()
    return np.allclose(t, float(val), atol=0.0, rtol=0.0)


def _is_real_external(df, expected_len):
    return (len(df) == expected_len) and (not is_constant_pred(df, 0.5))


external_available = any(
    os.path.exists(p) for p in paths.values() if p is not None
) and any(
    _is_real_external(d, len(sample))
    for d in [data1, data2, data3, data4, data5, data6]
)


def build_id_to_path_index(root_dir: str):
    idx = {}
    if not root_dir or not os.path.isdir(root_dir):
        return idx

    for sub in range(16):
        d = os.path.join(root_dir, str(sub))
        if not os.path.isdir(d):
            continue
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".npy"):
                        idx[e.name[:-4]] = e.path
        except FileNotFoundError:
            continue
    return idx


def _percentile_from_sorted(sorted_1d: np.ndarray, q: float) -> float:
    n = sorted_1d.size
    if n == 0:
        return float("nan")
    if q <= 0.0:
        return float(sorted_1d[0])
    if q >= 100.0:
        return float(sorted_1d[-1])
    pos = (q / 100.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_1d[lo])
    w = pos - lo
    return float(sorted_1d[lo] * (1.0 - w) + sorted_1d[hi] * w)


N_FEATURES = 38


def extract_features_from_npy(path):
    try:
        x = np.load(path)  # float16 on disk
        xf = x.astype(np.float32, copy=False)

        if xf.ndim != 3 or xf.shape[0] != 6:
            return np.zeros(N_FEATURES, dtype=np.float32)

        A = xf[[0, 2, 4]]  # on-target
        B = xf[[1, 3, 5]]  # off-target

        x_flat = xf.reshape(-1)
        A_flat = A.reshape(-1)
        B_flat = B.reshape(-1)

        x_sorted = np.sort(x_flat, kind="quicksort")
        A_sorted = np.sort(A_flat, kind="quicksort")
        B_sorted = np.sort(B_flat, kind="quicksort")

        p_full_995 = _percentile_from_sorted(x_sorted, 99.5)
        p_full_005 = _percentile_from_sorted(x_sorted, 0.5)
        p_full_99 = _percentile_from_sorted(x_sorted, 99.0)
        p_full_01 = _percentile_from_sorted(x_sorted, 1.0)
        p_full_95 = _percentile_from_sorted(x_sorted, 95.0)
        p_full_05 = _percentile_from_sorted(x_sorted, 5.0)

        pA995 = _percentile_from_sorted(A_sorted, 99.5)
        pB995 = _percentile_from_sorted(B_sorted, 99.5)

        meanA, meanB = float(A.mean()), float(B.mean())
        stdA, stdB = float(A.std()), float(B.std())
        maxA, maxB = float(A.max()), float(B.max())

        A_time = A.mean(axis=(0, 2))
        B_time = B.mean(axis=(0, 2))
        A_freq = A.mean(axis=(0, 1))
        B_freq = B.mean(axis=(0, 1))

        time_diff = A_time - B_time
        freq_diff = A_freq - B_freq

        time_contrast_mean = float(time_diff.mean())
        time_contrast_std = float(time_diff.std())
        freq_contrast_mean = float(freq_diff.mean())
        freq_contrast_std = float(freq_diff.std())

        A_img = A.mean(axis=0)  # (273,256)
        dt = np.diff(A_img, axis=0)
        df = np.diff(A_img, axis=1)
        grad_time = float(np.mean(np.abs(dt)))
        grad_freq = float(np.mean(np.abs(df)))

        time_abs_mean = float(np.mean(np.abs(time_diff)))
        freq_abs_mean = float(np.mean(np.abs(freq_diff)))

        pA95 = _percentile_from_sorted(A_sorted, 95.0)
        pB95 = _percentile_from_sorted(B_sorted, 95.0)
        pA05 = _percentile_from_sorted(A_sorted, 5.0)
        pB05 = _percentile_from_sorted(B_sorted, 5.0)
        tail95_diff = float(pA95 - pB95)
        tail05_diff = float(pA05 - pB05)

        A_max_over_time = A.max(axis=1).mean(axis=0)  # (256,)
        B_max_over_time = B.max(axis=1).mean(axis=0)  # (256,)
        narrowA = float(A_max_over_time.max() - np.median(A_max_over_time))
        narrowB = float(B_max_over_time.max() - np.median(B_max_over_time))
        narrow_diff = float(narrowA - narrowB)

        A_abs = float(np.mean(np.abs(A_flat)))
        B_abs = float(np.mean(np.abs(B_flat)))
        abs_diff = A_abs - B_abs

        def _safe_corr(u, v):
            u = np.asarray(u, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32)
            su = float(u.std())
            sv = float(v.std())
            if su == 0.0 or sv == 0.0:
                return 0.0
            return float(np.mean((u - u.mean()) * (v - v.mean())) / (su * sv))

        time_corr = _safe_corr(A_time, B_time)
        freq_corr = _safe_corr(A_freq, B_freq)

        frac_A_gt_B995 = float(np.mean(A_flat > pB995))

        pA75 = _percentile_from_sorted(A_sorted, 75.0)
        pA25 = _percentile_from_sorted(A_sorted, 25.0)
        pB75 = _percentile_from_sorted(B_sorted, 75.0)
        pB25 = _percentile_from_sorted(B_sorted, 25.0)
        iqr_diff = float((pA75 - pA25) - (pB75 - pB25))

        contrast_balance = float(time_abs_mean - freq_abs_mean)

        feat = []
        feat += [float(xf.mean()), float(xf.std()), float(np.median(x_sorted))]
        feat += [meanA - meanB, stdA - stdB]
        feat += [float(xf.max()), float(p_full_995), float(p_full_005)]
        feat += [maxA - maxB, float(pA995 - pB995)]
        feat += [float(p_full_99), float(p_full_01)]
        feat += [maxA, maxB, stdA, stdB]
        feat += [
            time_contrast_mean,
            time_contrast_std,
            freq_contrast_mean,
            freq_contrast_std,
        ]
        feat += [grad_time, grad_freq]
        feat += [
            float(p_full_95),
            float(p_full_05),
            time_abs_mean,
            freq_abs_mean,
            tail95_diff,
            tail05_diff,
            narrowA,
            narrow_diff,
        ]

        feat += [
            abs_diff,
            time_corr,
            freq_corr,
            frac_A_gt_B995,
            iqr_diff,
            contrast_balance,
            narrowB,
            float(narrowB - narrowA),
        ]

        return np.asarray(feat, dtype=np.float32)
    except Exception:
        return np.zeros(N_FEATURES, dtype=np.float32)


def _feature_worker(arg):
    i, p = arg
    return i, extract_features_from_npy(p)


def build_feature_matrix(ids, id_to_path, max_workers=None, chunksize=1024):
    X = np.zeros((len(ids), N_FEATURES), dtype=np.float32)
    missing = 0

    tasks = []
    for i, _id in enumerate(ids):
        p = id_to_path.get(_id)
        if p is None:
            missing += 1
        else:
            tasks.append((i, p))

    if not tasks:
        if missing:
            print(f"Warning: {missing} npy files not found while building features.")
        return X

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = min(8, max(2, cpu // 2))

    try:
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, feat in ex.map(_feature_worker, tasks, chunksize=chunksize):
                X[i] = feat
    except Exception as e:
        print(
            f"Warning: parallel feature extraction failed ({type(e).__name__}: {e}). Falling back to single-process."
        )
        for i, p in tasks:
            X[i] = extract_features_from_npy(p)

    if missing:
        print(f"Warning: {missing} npy files not found while building features.")
    return X


baseline_pred = None

if (not external_available) and TRAIN_LABELS_PATH and TRAIN_DIR and TEST_DIR:
    train_labels = pd.read_csv(TRAIN_LABELS_PATH, usecols=["id", "target"])
    train_labels["target"] = train_labels["target"].astype(int)

    test_ids = sample["id"].tolist()
    train_ids = train_labels["id"].tolist()
    y = train_labels["target"].to_numpy()

    train_index = build_id_to_path_index(TRAIN_DIR)
    test_index = build_id_to_path_index(TEST_DIR)

    X_train = build_feature_matrix(train_ids, train_index)
    X_test = build_feature_matrix(test_ids, test_index)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_pred_accum = np.zeros(len(test_ids), dtype=np.float64)

    for tr_idx, va_idx in skf.split(X_train, y):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        solver="liblinear",
                        C=2.0,
                        max_iter=400,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        clf.fit(X_train[tr_idx], y[tr_idx])
        test_pred_accum += clf.predict_proba(X_test)[:, 1] / skf.get_n_splits()

    baseline_pred = pd.DataFrame(
        {"id": test_ids, "target": test_pred_accum.astype(float)}
    )
    print(
        "Built baseline predictions because external submission files were not found (or are constant)."
    )
else:
    print(
        "External submission files found (or required data missing); skipping baseline builder."
    )



## === cell 6
merged = align_on_id([data1, data2, data3, data4, data5, data6])

ens_target = (
    0.75 * merged["target_4"]  # data5
    + 0.125 * merged["target_3"]  # data4
    + 0.125 * merged["target_5"]  # data6
    + 0.00 * merged["target_1"]  # data2 (kept as in original)
    + 0.00 * merged["target_2"]  # data3 (kept as in original)
)

submission = merged[["id"]].copy()
submission["target"] = ens_target.astype(float).clip(0.0, 1.0)

submission = sample[["id"]].merge(submission, on="id", how="left")

if baseline_pred is not None:
    submission = submission.merge(
        baseline_pred.rename(columns={"target": "baseline_target"}), on="id", how="left"
    )

    if not external_available:
        submission["target"] = submission["baseline_target"]

    submission = submission.drop(columns=["baseline_target"])

submission["target"] = submission["target"].fillna(0.5).astype(float).clip(0.0, 1.0)



## === cell 7
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Prediction summary:", submission["target"].describe())
