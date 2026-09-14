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

0.7571609277550078

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code fails because it tries to read multiple other notebooks’ `submission.csv` files from `../input/...`, but those datasets are not available in this environment, so nothing gets defined and the ensemble step crashes. I keep the “ensemble submissions” core idea, but make it robust: automatically discover any available `submission*.csv` files under `/kaggle/input`, validate/align them by `id`, and then average them (or fall back safely to `sample_submission.csv` with 0.5 if none exist). This guarantees an end-to-end run that always writes a valid `submission.csv` with the correct columns and 6000 rows. The averaging ensemble is a reasonable, minimal replacement for missing inputs and should move the score upward from “no submission” toward the target.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score is coming from the fallback path that outputs a constant 0.5 because there are no external `submission*.csv` files to ensemble in this environment. To move the ROC-AUC upward toward your target with minimal change and without altering “core logic” (still producing probabilities per `id`), I keep the same “ensemble if available, otherwise fallback” structure but add a simple, legitimate model-based fallback using the provided training snippets: extract lightweight summary features from each `.npy` (no deep learning), train a logistic regression, and predict on test. This should materially improve over 0.5 while staying within the 600s budget by subsampling the training set deterministically and using fast per-file feature extraction. The script still writes a valid `submission.csv` with `id,target` and 6000 rows.'
- What this solution (achieved 0.5) has done: 'We keep your “ensemble if available, otherwise fallback” core logic unchanged, but make the fallback model stronger and more stable so the ROC-AUC moves up from 0.5 toward your ~0.757 target. The smallest high-impact fix is to (1) standardize features and use a slightly more appropriate logistic setup (regularization + class balancing) and (2) add a few additional cheap, domain-relevant summary features that emphasize the A-vs-O cadence difference without changing the overall approach. We also make training selection deterministic and mildly increase `max_train` (still fast) to reduce variance and improve generalization. The output remains a valid `submission.csv` with `id,target` and 6000 rows.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
INPUT_ROOTS = ["/kaggle/input", "/kaggle/data"]  # support either layout if present


def find_submission_csvs():
    paths = []
    for root in INPUT_ROOTS:
        if os.path.isdir(root):
            paths.extend(
                glob.glob(os.path.join(root, "**", "submission.csv"), recursive=True)
            )
            paths.extend(
                glob.glob(os.path.join(root, "**", "*submission*.csv"), recursive=True)
            )
    seen = set()
    uniq = []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


sub_paths = find_submission_csvs()
sub_paths[:10], len(sub_paths)



## === cell 2
sample_path_candidates = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
]
sample_path = next((p for p in sample_path_candidates if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError("sample_submission.csv not found in expected Kaggle paths.")

sample = pd.read_csv(sample_path)
if list(sample.columns) != ["id", "target"]:
    sample = sample.rename(
        columns={sample.columns[0]: "id", sample.columns[1]: "target"}
    )
sample["id"] = sample["id"].astype(str)

sample.head(), sample.shape




## === cell 3
def load_and_validate_submission(path, id_template):
    df = pd.read_csv(path)
    if "id" not in df.columns or "target" not in df.columns:
        return None
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = id_template[["id"]].merge(df, on="id", how="left")
    if df["target"].isna().mean() > 0.05:
        return None
    df["target"] = df["target"].fillna(0.5).clip(0.0, 1.0)
    return df


subs = []
used_paths = []
for p in sub_paths:
    if os.path.basename(p).lower().endswith(".csv"):
        df = load_and_validate_submission(p, sample)
        if df is not None and df.shape[0] == sample.shape[0]:
            subs.append(df)
            used_paths.append(p)

len(subs), used_paths[:5]



## === cell 4
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def resolve_data_root():
    candidates = [
        "/kaggle/input/seti-breakthrough-listen",
        "/kaggle/data/seti-breakthrough-listen",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for root in candidates:
        if os.path.isdir(root):
            return root
    raise FileNotFoundError("Could not resolve Kaggle data root.")


DATA_ROOT = resolve_data_root()


def resolve_labels_path():
    candidates = [
        os.path.join(DATA_ROOT, "train_labels.csv"),
        "/kaggle/input/train_labels.csv",
        "/kaggle/data/train_labels.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("train_labels.csv not found in expected Kaggle paths.")


def resolve_train_dir():
    candidates = [
        os.path.join(DATA_ROOT, "train"),
        "/kaggle/input/train",
        "/kaggle/data/train",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("train directory not found in expected Kaggle paths.")


def resolve_test_dir():
    candidates = [
        os.path.join(DATA_ROOT, "test"),
        "/kaggle/input/test",
        "/kaggle/data/test",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("test directory not found in expected Kaggle paths.")


LABELS_PATH = resolve_labels_path()
TRAIN_DIR = resolve_train_dir()
TEST_DIR = resolve_test_dir()

LABELS_PATH, TRAIN_DIR, TEST_DIR




## === cell 5
def snippet_path(base_dir, snippet_id):
    return os.path.join(base_dir, snippet_id[0], f"{snippet_id}.npy")


def extract_features_from_array(x):
    """
    x: (6, 273, 256) float16/float32
    Returns a small numeric feature vector.

    Change rationale (score-up, minimal): add a few extra cheap summary features that
    emphasize A vs O differences and robustness (percentiles, per-panel contrasts),
    while keeping the same "handcrafted features + logistic regression" core logic.
    """
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]
    O = x[[1, 3, 5]]

    mean_all = x.mean()
    std_all = x.std() + 1e-6

    mean_A = A.mean()
    mean_O = O.mean()
    std_A = A.std() + 1e-6
    std_O = O.std() + 1e-6

    abs_mean = np.mean(np.abs(x))
    abs_mean_A = np.mean(np.abs(A))
    abs_mean_O = np.mean(np.abs(O))

    dtime_A = np.mean(np.abs(np.diff(A, axis=1)))
    dfreq_A = np.mean(np.abs(np.diff(A, axis=2)))
    dtime_O = np.mean(np.abs(np.diff(O, axis=1)))
    dfreq_O = np.mean(np.abs(np.diff(O, axis=2)))

    contrast_mean = (mean_A - mean_O) / std_all
    contrast_abs = (abs_mean_A - abs_mean_O) / (abs_mean + 1e-6)
    contrast_grad_time = (dtime_A - dtime_O) / (dtime_O + 1e-6)
    contrast_grad_freq = (dfreq_A - dfreq_O) / (dfreq_O + 1e-6)

    ax = np.abs(x).reshape(-1)
    p50 = np.percentile(ax, 50)
    p90 = np.percentile(ax, 90)
    p99 = np.percentile(ax, 99)

    aax = np.abs(A).reshape(-1)
    oax = np.abs(O).reshape(-1)
    p90_A = np.percentile(aax, 90)
    p90_O = np.percentile(oax, 90)
    p99_A = np.percentile(aax, 99)
    p99_O = np.percentile(oax, 99)

    panel_max = x.reshape(6, -1).max(axis=1)
    panel_mean = x.reshape(6, -1).mean(axis=1)
    mm_ratio = (panel_max - panel_mean) / (x.reshape(6, -1).std(axis=1) + 1e-6)
    mm_A = mm_ratio[[0, 2, 4]].mean()
    mm_O = mm_ratio[[1, 3, 5]].mean()
    mm_contrast = mm_A - mm_O

    feats = np.array(
        [
            mean_all,
            std_all,
            mean_A,
            mean_O,
            std_A,
            std_O,
            abs_mean,
            abs_mean_A,
            abs_mean_O,
            dtime_A,
            dfreq_A,
            dtime_O,
            dfreq_O,
            contrast_mean,
            contrast_abs,
            contrast_grad_time,
            contrast_grad_freq,
            p50,
            p90,
            p99,
            p90_A,
            p90_O,
            p99_A,
            p99_O,
            (p90_A - p90_O) / (p90_O + 1e-6),
            (p99_A - p99_O) / (p99_O + 1e-6),
            mm_A,
            mm_O,
            mm_contrast,
        ],
        dtype=np.float32,
    )
    return feats


def build_feature_matrix(ids, base_dir, n_features):
    X = np.zeros((len(ids), n_features), dtype=np.float32)
    missing = 0
    for i, sid in enumerate(ids):
        p = snippet_path(base_dir, sid)
        try:
            arr = np.load(p)
            X[i] = extract_features_from_array(arr)
        except Exception:
            missing += 1
            X[i] = 0.0
    return X, missing




## === cell 6
def model_fallback_predict(
    sample_df, labels_path, train_dir, test_dir, max_train=20000, seed=123
):
    """
    Change rationale (score-up, minimal):
    - StandardScaler + LogisticRegression improves conditioning because features are on
      very different scales; this usually boosts AUC with the same linear model.
    - class_weight='balanced' handles any class imbalance without changing semantics.
    - Slightly larger max_train reduces variance while staying within time budget.
    """
    labels = pd.read_csv(labels_path)
    labels["id"] = labels["id"].astype(str)

    rng = np.random.RandomState(seed)
    if max_train is not None and max_train < len(labels):
        pos = labels[labels["target"] == 1]
        neg = labels[labels["target"] == 0]

        pos_n = int(round(max_train * (len(pos) / len(labels))))
        neg_n = max_train - pos_n

        pos_ids = pos["id"].to_numpy()
        neg_ids = neg["id"].to_numpy()
        pos_sel = rng.choice(pos_ids, size=min(pos_n, len(pos_ids)), replace=False)
        neg_sel = rng.choice(neg_ids, size=min(neg_n, len(neg_ids)), replace=False)

        sub_labels = (
            pd.concat(
                [
                    labels[labels["id"].isin(pos_sel)],
                    labels[labels["id"].isin(neg_sel)],
                ],
                axis=0,
                ignore_index=True,
            )
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    else:
        sub_labels = labels

    train_ids = sub_labels["id"].tolist()
    y = sub_labels["target"].to_numpy(dtype=np.int32)

    n_features = extract_features_from_array(
        np.zeros((6, 273, 256), dtype=np.float32)
    ).shape[0]

    X_train, miss_tr = build_feature_matrix(train_ids, train_dir, n_features=n_features)
    X_test, miss_te = build_feature_matrix(
        sample_df["id"].tolist(), test_dir, n_features=n_features
    )

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            solver="lbfgs",
            max_iter=600,
            n_jobs=None,
            class_weight="balanced",
            C=1.0,
        ),
    )
    clf.fit(X_train, y)
    proba = clf.predict_proba(X_test)[:, 1].astype(np.float64)

    info = {
        "train_rows": len(train_ids),
        "train_missing": miss_tr,
        "test_missing": miss_te,
        "n_features": n_features,
    }
    return proba, info


if len(subs) == 0:
    pred, fb_info = model_fallback_predict(
        sample_df=sample,
        labels_path=LABELS_PATH,
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        max_train=20000,
        seed=123,
    )
else:
    pred_stack = np.stack(
        [df["target"].to_numpy(dtype=np.float64) for df in subs], axis=0
    )
    pred = pred_stack.mean(axis=0)
    fb_info = None

submission = sample.copy()
submission["target"] = np.clip(pred, 0.0, 1.0)

submission.head(), submission["target"].describe()



## === cell 7
submission.to_csv("submission.csv", index=False)

print(f"Wrote submission.csv with shape {submission.shape}")
print(f"Used {len(subs)} external submission file(s).")
for p in used_paths[:20]:
    print(" -", p)
if fb_info is not None:
    print("Fallback model info:", fb_info)
