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

0.7507452893851788

# 6. Current score

0.50702

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48924) has done: 'I fix the immediate runtime blockers while keeping the same overall approach (feature extraction → standardize → logistic regression → probability submission). First, I replace the invalid `DataFrame.sample(..., stratify=...)` call with a stratified downsampling via `train_test_split`, which preserves class balance without changing the model/training logic. Second, I correct the feature vector length mismatch by allocating arrays based on the actual feature length returned by `extract_features` (15), eliminating the broadcast error for both train and test. Finally, I renumber cells to start at 1 (your current script starts at cell 0) and ensure the pipeline always writes a valid `submission.csv` with `id,target`.'
- What this solution (achieved 0.50035) has done: 'We keep your exact pipeline (hand-crafted features → StandardScaler → LogisticRegression → probability submission) and make only small, directly score-relevant adjustments to reduce underfitting that’s holding AUC down. Specifically, we (1) train on all available training rows instead of downsampling to 20k (this is not a logic change and should materially lift AUC), (2) switch LogisticRegression to `class_weight="balanced"` to counter class imbalance (often improves ROC-AUC for this dataset with linear models), and (3) set a slightly stronger regularization search point (`C=3.0`) while keeping the same solver/approach. Everything else (feature extraction, split, evaluation, submission formatting/paths) remains the same and it still write a valid `submission.csv`.'
- What this solution (achieved 0.50035) has done: 'You’re already generating a valid submission, but the AUC is stuck near random (0.50), which strongly suggests a label/feature mismatch from sorting/filtering `train_df` independently of the filesystem (i.e., you may be pairing features from one id with the target from another). I make the smallest fix that preserves your exact approach (same handcrafted features → StandardScaler → LogisticRegression) by reindexing `train_labels` to the exact file-id order and building `X,y` in that same order. I also add a lightweight sanity check to confirm that each `id` used for `X` matches the row used for `y`, which should move the score upward toward your target without changing the model logic. Everything else (feature function, model, submission format/path) remains the same.'
- What this solution (achieved 0.50702) has done: 'Your pipeline is already valid but the score near 0.50 suggests the model is not seeing signal because the handcrafted features are too weakly aligned with the cadence structure. To move the ROC-AUC upward toward your 0.7507 target without changing the core approach (handcrafted features → StandardScaler → LogisticRegression), I make a minimal, metric-aligned enhancement inside `extract_features`: add a few additional summary statistics that capture “on-only” structure (A panels) vs “off” structure (B/C/D panels), including per-panel variability and consistency across the three ON observations. I also keep everything deterministic and keep the same model/training loop, paths, and submission formatting so it still runs end-to-end and writes `submission.csv`. These changes should improve separability while preserving the same overall logic and semantics.'

# 9. Code solution

## === cell 0
import os
import glob
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
    files = glob.glob(os.path.join(root_dir, "*", "*.npy"))
    return sorted(files)


train_files = list_npy_files(train_dir)
test_files = list_npy_files(test_dir)

len(train_files), len(test_files)




## === cell 2
def file_id_from_path(p):
    return os.path.splitext(os.path.basename(p))[0]


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
    """
    Minimal score-relevant enhancement:
    - Keep the same core idea (handcrafted stats from ON vs OFF),
      but add a few cadence-consistency features that help AUC:
        * ON panel-to-panel variability (needle tends to repeat in ON panels)
        * OFF panel-to-panel variability (often different behavior for RFI/noise)
        * ON-OFF per-panel mean diffs distribution (robust separability)
    """
    x = arr.astype(np.float32)  # (6,H,W)
    on = x[[0, 2, 4]]  # (3,H,W)
    off = x[[1, 3, 5]]  # (3,H,W)

    on_mean = on.mean(axis=0)
    off_mean = off.mean(axis=0)
    diff = on_mean - off_mean

    f = []
    f.append(diff.mean())
    f.append(diff.std())
    f.append(np.mean(np.abs(diff)))
    f.append(np.max(diff))
    f.append(np.min(diff))

    flat = diff.ravel()
    absflat = np.abs(flat)
    k = 512  # small, fixed
    if absflat.size >= k:
        idx = np.argpartition(absflat, -k)[-k:]
        f.append(absflat[idx].mean())
        f.append(absflat[idx].max())
    else:
        f.append(absflat.mean())
        f.append(absflat.max() if absflat.size else 0.0)

    row_prof = diff.mean(axis=1)  # (H,)
    col_prof = diff.mean(axis=0)  # (W,)
    f.append(row_prof.std())
    f.append(col_prof.std())
    f.append(np.mean(np.abs(np.diff(row_prof))))
    f.append(np.mean(np.abs(np.diff(col_prof))))

    f.append(on.mean())
    f.append(off.mean())
    f.append(on.std())
    f.append(off.std())

    on_panel_means = on.mean(axis=(1, 2))
    off_panel_means = off.mean(axis=(1, 2))
    f.append(on_panel_means.mean())
    f.append(off_panel_means.mean())
    f.append(on_panel_means.std())
    f.append(off_panel_means.std())
    per_pair = on_panel_means - off_panel_means
    f.append(per_pair.mean())
    f.append(per_pair.std())
    f.append(np.max(per_pair))
    f.append(np.min(per_pair))

    on0 = on[0].ravel()
    on1 = on[1].ravel()
    on2 = on[2].ravel()
    eps = 1e-8
    on0 = (on0 - on0.mean()) / (on0.std() + eps)
    on1 = (on1 - on1.mean()) / (on1.std() + eps)
    on2 = (on2 - on2.mean()) / (on2.std() + eps)
    c01 = float(np.mean(on0 * on1))
    c02 = float(np.mean(on0 * on2))
    c12 = float(np.mean(on1 * on2))
    f.append((c01 + c02 + c12) / 3.0)

    return np.array(f, dtype=np.float32)


tmp_id = train_df.loc[0, "id"]
tmp_arr = np.load(train_map[tmp_id])
feat_tmp = extract_features(tmp_arr)
feat_tmp, tmp_arr.shape, feat_tmp.shape



## === cell 4
train_df_fit = train_df

n_features = int(feat_tmp.shape[0])

X = np.zeros((len(train_df_fit), n_features), dtype=np.float32)
y = train_df_fit["target"].values.astype(np.int64)

for i, id_ in enumerate(train_df_fit["id"].values):
    if id_ not in train_map:
        raise FileNotFoundError(f"Train id from labels not found in train files: {id_}")
    arr = np.load(train_map[id_])
    X[i] = extract_features(arr)

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
                max_iter=2000,
                solver="lbfgs",
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
for i, id_ in enumerate(test_ids):
    arr = np.load(test_map[id_])
    X_test[i] = extract_features(arr)

test_pred = model.predict_proba(X_test)[:, 1].astype(np.float64)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission = sample_sub[["id"]].merge(submission, on="id", how="left")
if submission["target"].isna().any():
    raise RuntimeError(
        "Some test predictions are missing after merge; id alignment issue."
    )

submission.to_csv("submission.csv", index=False)
submission.head(), submission.shape
