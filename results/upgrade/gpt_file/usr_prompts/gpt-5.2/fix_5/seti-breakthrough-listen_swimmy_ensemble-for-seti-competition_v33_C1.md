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

0.7569879613171004

# 6. Current score

0.5047

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to ensemble submissions from other Kaggle notebooks that are not available in this environment (`../input/...` paths don’t exist), so nothing ever gets loaded or written. To make it run end-to-end and still preserve the “submission blending” core intent, I switch to a robust fallback that reads the provided `sample_submission.csv` from the competition data and outputs a valid `submission.csv` with constant probabilities. This guarantees a correctly formatted CSV is produced without relying on unavailable external files; since no current score exists, this is the minimal safe fix to yield a valid submission (score be ~0.5 AUC). If you later provide the allowed ensemble submission files in the environment, the same code automatically blend them.'
- What this solution (achieved 0.50344) has done: 'The timeout is dominated by per-file Python overhead and repeated expensive `np.quantile` calls inside `extract_features_from_array` for ~60k arrays (54k train + 6k test). I keep the exact same feature definitions and model, but (1) speed up loading by precomputing file paths and using a small thread pool to overlap disk I/O, and (2) replace `np.quantile` with equivalent `np.partition`-based selection for the fixed quantiles (0.99 and 0.999), which is mathematically identical to NumPy’s default linear interpolation for these specific array lengths (they land on exact order statistics). These changes preserve the algorithm and outputs (up to negligible float rounding) while cutting feature-extraction wall time substantially.'
- What this solution (achieved 0.5047) has done: 'Your current model is likely stuck near random because the engineered features don’t capture the main discriminative pattern of this competition: signals that appear in the on-target A panels (0/2/4) but not in off-target panels (1/3/5), often as narrow bright tracks in frequency-time. Keeping the same core approach (feature extraction + LogisticRegression), I add a few minimal, cheap features that measure (1) positive excess in A over B, (2) how much of A is “brighter than B” at the pixel level, and (3) high-intensity tail differences using max/mean-of-top-k, plus a simple A-minus-B difference-image summary. These are still simple statistics over the same arrays (no new model, no new training loop), but they align much better with the ROC-AUC objective by increasing separability. The submission writing and paths remain unchanged and the pipeline still runs end-to-end within the time limit.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input",  # standard Kaggle
    "/kaggle/data",  # provided in this environment description
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input/seti-breakthrough-listen",
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


data_root = find_first_existing(DATA_ROOT_CANDIDATES)
if data_root is None:
    raise FileNotFoundError(
        "Could not find Kaggle data root. Tried: " + ", ".join(DATA_ROOT_CANDIDATES)
    )

sample_path_candidates = [
    os.path.join(data_root, "sample_submission.csv"),
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = find_first_existing(sample_path_candidates)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv. Tried: "
        + ", ".join(sample_path_candidates)
    )

train_labels_path_candidates = [
    os.path.join(data_root, "train_labels.csv"),
    "/kaggle/data/train_labels.csv",
    "/kaggle/input/train_labels.csv",
]
train_labels_path = find_first_existing(train_labels_path_candidates)
if train_labels_path is None:
    raise FileNotFoundError(
        "Could not find train_labels.csv. Tried: "
        + ", ".join(train_labels_path_candidates)
    )

train_dir_candidates = [
    os.path.join(data_root, "train"),
    "/kaggle/data/train",
    "/kaggle/input/train",
]
test_dir_candidates = [
    os.path.join(data_root, "test"),
    "/kaggle/data/test",
    "/kaggle/input/test",
]
train_dir = find_first_existing(train_dir_candidates)
test_dir = find_first_existing(test_dir_candidates)
if train_dir is None or test_dir is None:
    raise FileNotFoundError(
        f"Could not find train/test directories. Tried train={train_dir_candidates}, test={test_dir_candidates}"
    )

sample_sub = pd.read_csv(sample_path)
assert list(sample_sub.columns)[:2] == [
    "id",
    "target",
], "Unexpected sample submission columns"
sample_sub["id"] = sample_sub["id"].astype(str)

train_labels = pd.read_csv(train_labels_path)
train_labels["id"] = train_labels["id"].astype(str)
assert set(train_labels.columns) >= {"id", "target"}

print("data_root:", data_root)
print("train_dir:", train_dir)
print("test_dir:", test_dir)
print("sample_sub shape:", sample_sub.shape)
print("train_labels shape:", train_labels.shape)



## === cell 2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from concurrent.futures import ThreadPoolExecutor


def _quantile_linear_partition(a: np.ndarray, q: float) -> float:
    a = a.reshape(-1)
    n = a.size
    if n == 0:
        return float("nan")
    h = (n - 1) * q
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi:
        return float(np.partition(a, lo)[lo])
    part = np.partition(a, (lo, hi))
    a_lo = part[lo]
    a_hi = part[hi]
    return float(a_lo + (h - lo) * (a_hi - a_lo))


def _mean_of_topk(a: np.ndarray, k: int) -> float:
    a = a.reshape(-1)
    n = a.size
    if n == 0:
        return float("nan")
    k = int(k)
    if k <= 1:
        return float(a.max())
    if k >= n:
        return float(a.mean())
    part = np.partition(a, n - k)
    return float(part[n - k :].mean())


def id_to_path(base_dir: str, sample_id: str) -> str:
    shard = sample_id[0]
    return os.path.join(base_dir, shard, f"{sample_id}.npy")


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    """
    x: (6, 273, 256) float16/float32

    Core logic preserved: simple engineered stats + LogisticRegression.
    Change (score-improving): add a few A-vs-B pixel-level excess/tail features to better align
    with the competition structure (needle appears in A panels but not in B panels).
    """
    x = x.astype(np.float32, copy=False)
    A = x[[0, 2, 4]]
    B = x[[1, 3, 5]]

    mean_A = float(A.mean())
    mean_B = float(B.mean())
    std_all = float(x.std())
    std_A = float(A.std())
    std_B = float(B.std())

    q99_all = _quantile_linear_partition(x, 0.99)
    q999_all = _quantile_linear_partition(x, 0.999)
    q99_A = _quantile_linear_partition(A, 0.99)
    q99_B = _quantile_linear_partition(B, 0.99)

    panel_means = x.mean(axis=(1, 2))
    panel_stds = x.std(axis=(1, 2))
    var_panel_means = float(panel_means.var())
    var_panel_stds = float(panel_stds.var())

    denom = abs(mean_A) + abs(mean_B) + 1e-6
    contrast = (mean_A - mean_B) / denom
    tail_contrast = (q99_A - q99_B) / (abs(q99_A) + abs(q99_B) + 1e-6)

    D = A.mean(axis=0) - B.mean(axis=0)  # (273,256)
    mean_D = float(D.mean())
    std_D = float(D.std())
    pos_mean_D = float(np.maximum(D, 0.0).mean())
    pos_frac_D = float((D > 0.0).mean())

    pix_excess = A - B  # broadcast over the 3 paired panels (3,273,256)
    pix_excess_pos_mean = float(np.maximum(pix_excess, 0.0).mean())
    pix_excess_pos_frac = float((pix_excess > 0.0).mean())

    max_A = float(A.max())
    max_B = float(B.max())
    topk_A = _mean_of_topk(A, k=256)  # small tail average
    topk_B = _mean_of_topk(B, k=256)
    topk_diff = float(topk_A - topk_B)

    return np.array(
        [
            mean_A,
            mean_B,
            mean_A - mean_B,
            contrast,
            std_all,
            std_A,
            std_B,
            q99_all,
            q999_all,
            q99_A,
            q99_B,
            q99_A - q99_B,
            tail_contrast,
            var_panel_means,
            var_panel_stds,
            mean_D,
            std_D,
            pos_mean_D,
            pos_frac_D,
            pix_excess_pos_mean,
            pix_excess_pos_frac,
            max_A,
            max_B,
            max_A - max_B,
            topk_A,
            topk_B,
            topk_diff,
        ],
        dtype=np.float32,
    )


def build_feature_matrix(ids, base_dir: str, max_workers: int = 8) -> np.ndarray:
    paths = [id_to_path(base_dir, sid) for sid in ids]

    feats = np.zeros((len(ids), 27), dtype=np.float32)

    def _load_and_featurize(i_p):
        i, p = i_p
        arr = np.load(p, allow_pickle=False)
        return i, extract_features_from_array(arr)

    mw = max(1, min(int(max_workers), (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=mw) as ex:
        for i, f in ex.map(_load_and_featurize, enumerate(paths), chunksize=64):
            feats[i] = f
    return feats


train_ids = train_labels["id"].to_list()
y = train_labels["target"].to_numpy(dtype=np.int64)

X_train = build_feature_matrix(train_ids, train_dir, max_workers=8)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, solver="lbfgs")),
    ]
)

clf.fit(X_train, y)
print("Trained feature model on:", X_train.shape)



## === cell 3
test_ids = sample_sub["id"].to_list()
X_test = build_feature_matrix(test_ids, test_dir, max_workers=8)

pred = clf.predict_proba(X_test)[:, 1].astype(np.float64)
pred = np.clip(pred, 0.0, 1.0)

sub = sample_sub.copy()
sub["target"] = pred
sub.head()



## === cell 4
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape={sub.shape} columns={list(sub.columns)}")
print(sub.describe(include="all"))
