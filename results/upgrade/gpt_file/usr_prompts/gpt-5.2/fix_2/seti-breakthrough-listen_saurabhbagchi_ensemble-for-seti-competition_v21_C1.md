# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7569740310030568

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression



## === cell 1
BASE_INPUT = "/kaggle/input"
DATA_ROOT = os.path.join(BASE_INPUT, "seti-breakthrough-listen")

TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

assert os.path.exists(TRAIN_LABELS_CSV), f"Missing: {TRAIN_LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

train_labels.head(), sample_sub.head()




## === cell 2
def list_npy_files(root_dir: str):
    files = glob.glob(os.path.join(root_dir, "*", "*.npy"))
    files.sort()
    return files


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

train_id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in train_files}
test_id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in test_files}

missing_train = [i for i in train_labels["id"].tolist() if i not in train_id_to_path]
if len(missing_train) > 0:
    raise FileNotFoundError(
        f"Some labeled train ids are missing .npy files. Example: {missing_train[:5]}"
    )

missing_test = [i for i in sample_sub["id"].tolist() if i not in test_id_to_path]
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Some sample_submission test ids are missing .npy files. Example: {missing_test[:5]}"
    )

len(train_files), len(test_files), len(train_id_to_path), len(test_id_to_path)




## === cell 3
def extract_features_from_snippet(x: np.ndarray) -> np.ndarray:
    """
    x shape: (6, 273, 256) float16/float32
    Cadence order: A, B, A, C, A, D
    Minimal, deterministic features:
      - global stats per panel
      - contrasts: A_panels vs off-target (B/C/D) panels
      - simple gradient magnitude stats to capture line-like structures
    """
    x = x.astype(np.float32, copy=False)

    A1, B, A2, C, A3, D = x[0], x[1], x[2], x[3], x[4], x[5]
    A = (A1 + A2 + A3) / 3.0
    Off = (B + C + D) / 3.0
    Diff = A - Off
    AbsDiff = np.abs(Diff)

    def stats_2d(z):
        m = z.mean()
        s = z.std()
        mn = z.min()
        mx = z.max()
        q10, q50, q90 = np.quantile(z, [0.10, 0.50, 0.90])
        return np.array([m, s, mn, mx, q10, q50, q90], dtype=np.float32)

    def grad_stats(z):
        gx = np.diff(z, axis=1)
        gy = np.diff(z, axis=0)
        g = np.sqrt(gx[:, :-1] ** 2 + gy[:-1, :] ** 2)
        return stats_2d(g)

    feats = []
    for i in range(6):
        feats.append(stats_2d(x[i]))
    feats.append(stats_2d(A))
    feats.append(stats_2d(Off))
    feats.append(stats_2d(Diff))
    feats.append(stats_2d(AbsDiff))

    feats.append(grad_stats(A))
    feats.append(grad_stats(Off))
    feats.append(grad_stats(Diff))
    feats.append(grad_stats(AbsDiff))

    def stats_1d(v):
        m = v.mean()
        s = v.std()
        mn = v.min()
        mx = v.max()
        q10, q50, q90 = np.quantile(v, [0.10, 0.50, 0.90])
        return np.array([m, s, mn, mx, q10, q50, q90], dtype=np.float32)

    A_time = A.mean(axis=1)
    A_freq = A.mean(axis=0)
    Off_time = Off.mean(axis=1)
    Off_freq = Off.mean(axis=0)
    Diff_time = Diff.mean(axis=1)
    Diff_freq = Diff.mean(axis=0)

    feats.append(stats_1d(A_time))
    feats.append(stats_1d(A_freq))
    feats.append(stats_1d(Off_time))
    feats.append(stats_1d(Off_freq))
    feats.append(stats_1d(Diff_time))
    feats.append(stats_1d(Diff_freq))

    return np.concatenate(feats, axis=0)


p0 = train_id_to_path[train_labels["id"].iloc[0]]
x0 = np.load(p0)
f0 = extract_features_from_snippet(x0)
x0.shape, f0.shape, f0[:10]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2590836080.py in <cell line: 0>()
     80 p0 = train_id_to_path[train_labels["id"].iloc[0]]
     81 x0 = np.load(p0)
---> 82 f0 = extract_features_from_snippet(x0)
     83 x0.shape, f0.shape, f0[:10]
     84 

/tmp/ipykernel_11/2590836080.py in extract_features_from_snippet(x)
     45 
     46     # Gradient-based stats to emphasize line structures
---> 47     feats.append(grad_stats(A))
     48     feats.append(grad_stats(Off))
     49     feats.append(grad_stats(Diff))

/tmp/ipykernel_11/2590836080.py in grad_stats(z)
     31         gx = np.diff(z, axis=1)
     32         gy = np.diff(z, axis=0)
---> 33         g = np.sqrt(gx[:, :-1] ** 2 + gy[:-1, :] ** 2)
     34         return stats_2d(g)
     35 

ValueError: operands could not be broadcast together with shapes (273,254) (271,256) 

## === cell 4
def make_feature_matrix(ids, id_to_path, batch_size=256):
    X_list = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        for _id in batch_ids:
            arr = np.load(id_to_path[_id])
            X_list.append(extract_features_from_snippet(arr))
    return np.vstack(X_list).astype(np.float32, copy=False)


train_ids = train_labels["id"].tolist()
y = train_labels["target"].values.astype(np.int32)

X = make_feature_matrix(train_ids, train_id_to_path, batch_size=256)
X.shape



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1851129190.py in <cell line: 0>()
     13 
     14 # Extract train features (kept simple; should fit within time budget)
---> 15 X = make_feature_matrix(train_ids, train_id_to_path, batch_size=256)
     16 X.shape
     17 

/tmp/ipykernel_11/1851129190.py in make_feature_matrix(ids, id_to_path, batch_size)
      5         for _id in batch_ids:
      6             arr = np.load(id_to_path[_id])
----> 7             X_list.append(extract_features_from_snippet(arr))
      8     return np.vstack(X_list).astype(np.float32, copy=False)
      9 

/tmp/ipykernel_11/2590836080.py in extract_features_from_snippet(x)
     45 
     46     # Gradient-based stats to emphasize line structures
---> 47     feats.append(grad_stats(A))
     48     feats.append(grad_stats(Off))
     49     feats.append(grad_stats(Diff))

/tmp/ipykernel_11/2590836080.py in grad_stats(z)
     31         gx = np.diff(z, axis=1)
     32         gy = np.diff(z, axis=0)
---> 33         g = np.sqrt(gx[:, :-1] ** 2 + gy[:-1, :] ** 2)
     34         return stats_2d(g)
     35 

ValueError: operands could not be broadcast together with shapes (273,254) (271,256) 

## === cell 5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(train_ids), dtype=np.float32)

pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                C=1.0,
                class_weight=None,
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr = y[tr_idx]
    pipe.fit(Xtr, ytr)
    oof[va_idx] = pipe.predict_proba(Xva)[:, 1].astype(np.float32)

pipe.fit(X, y)

oof[:5], oof.mean(), oof.min(), oof.max()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/845579349.py in <cell line: 0>()
     22 )
     23 
---> 24 for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
     25     Xtr, Xva = X[tr_idx], X[va_idx]
     26     ytr = y[tr_idx]

NameError: name 'X' is not defined

## === cell 6
test_ids = sample_sub["id"].tolist()
X_test = make_feature_matrix(test_ids, test_id_to_path, batch_size=256)

test_pred = pipe.predict_proba(X_test)[:, 1].astype(np.float32)

sub = pd.DataFrame({"id": test_ids, "target": test_pred})
sub.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["id", "target"]

sub.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2781029503.py in <cell line: 0>()
      1 # Build test feature matrix in the exact order of sample_submission ids
      2 test_ids = sample_sub["id"].tolist()
----> 3 X_test = make_feature_matrix(test_ids, test_id_to_path, batch_size=256)
      4 
      5 test_pred = pipe.predict_proba(X_test)[:, 1].astype(np.float32)

/tmp/ipykernel_11/1851129190.py in make_feature_matrix(ids, id_to_path, batch_size)
      5         for _id in batch_ids:
      6             arr = np.load(id_to_path[_id])
----> 7             X_list.append(extract_features_from_snippet(arr))
      8     return np.vstack(X_list).astype(np.float32, copy=False)
      9 

/tmp/ipykernel_11/2590836080.py in extract_features_from_snippet(x)
     45 
     46     # Gradient-based stats to emphasize line structures
---> 47     feats.append(grad_stats(A))
     48     feats.append(grad_stats(Off))
     49     feats.append(grad_stats(Diff))

/tmp/ipykernel_11/2590836080.py in grad_stats(z)
     31         gx = np.diff(z, axis=1)
     32         gy = np.diff(z, axis=0)
---> 33         g = np.sqrt(gx[:, :-1] ** 2 + gy[:-1, :] ** 2)
     34         return stats_2d(g)
     35 

ValueError: operands could not be broadcast together with shapes (273,254) (271,256)
