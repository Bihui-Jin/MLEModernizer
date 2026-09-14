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

0.7480686684378159

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the file‑path errors by searching for the required CSV files under the Kaggle input directory, then load them, compute the overall mean target, fill the sample submission with this constant prediction, and finally write a valid `submission.csv`. This ensures the notebook runs end‑to‑end and produces a correctly formatted submission file.'
- What this solution (achieved 0.48088) has done: 'The script is accelerated by indexing all ``.npy`` files once at start, so each lookup avoids repeated recursive scans. The thread pool size is limited to a modest number to reduce I/O contention, while the core logic of loading files, computing max intensity, scaling, and writing the submission remains unchanged.'
- What this solution (achieved 0.51912) has done: 'I replace the simple min‑max scaling with a tiny learned mapping: after computing the max‑intensity feature for the training set, I fit a logistic‑regression model (using only this feature) to predict the binary target. Then, for the test set, I apply the same model to obtain probability predictions, handling missing values by filling them with the training‑set mean intensity. This keeps the overall pipeline unchanged while giving a calibrated, data‑driven prediction that should raise the AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I add a few simple intensity‑based features (mean and standard deviation) to the existing max‑intensity feature, keep the same logistic‑regression model, and use these richer inputs when fitting and predicting. This small change stays within the original pipeline while giving the model more signal to improve the AUC and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression


def find_file(filename):
    """
    Search for `filename` anywhere under the current directory or common Kaggle input locations.
    Returns the absolute path if found, otherwise raises FileNotFoundError.
    """
    search_paths = [
        os.path.join(".", filename),
        os.path.join("kaggle", "input", filename),
        os.path.join("kaggle", "working", filename),
        os.path.join("kaggle", "data", filename),
    ]
    for root, _, files in os.walk("."):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    for p in search_paths:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"{filename} not found in any searched location.")


def _load_npy_by_id(data_root, id_str):
    """
    Locate and load a .npy file given its `id` under `data_root` (train or test).
    The files are stored in a two‑level folder hierarchy (e.g., train/0/xxxx.npy).
    """
    for subdir in os.listdir(data_root):
        candidate = os.path.join(data_root, subdir, f"{id_str}.npy")
        if os.path.isfile(candidate):
            return np.load(candidate)
    raise FileNotFoundError(f"Numpy file for id {id_str} not found in {data_root}.")


def compute_features(id_str):
    """
    Compute intensity‑based features for a snippet identified by `id_str`.
    Tries the train directory first, then the test directory.
    Returns a list: [max_intensity, mean_intensity, std_intensity, diff_A_nonA].
    """
    try:
        arr = _load_npy_by_id(TRAIN_ROOT, id_str)
    except FileNotFoundError:
        arr = _load_npy_by_id(TEST_ROOT, id_str)

    max_int = np.nanmax(arr)
    mean_int = np.nanmean(arr)
    std_int = np.nanstd(arr)

    max_per_pos = np.nanmax(arr, axis=(1, 2))  # shape (6,)
    max_A = np.nanmax(max_per_pos[[0, 2, 4]])
    max_nonA = np.nanmax(max_per_pos[[1, 3, 5]])
    diff_A_nonA = max_A - max_nonA

    return [max_int, mean_int, std_int, diff_A_nonA]


train_labels_path = find_file("train_labels.csv")
sample_submission_path = find_file("sample_submission.csv")

TRAIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(train_labels_path), "train"))
TEST_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(sample_submission_path), "test")
)

if not os.path.isdir(TRAIN_ROOT):
    for root, dirs, _ in os.walk("."):
        if "train" in dirs:
            TRAIN_ROOT = os.path.abspath(os.path.join(root, "train"))
            break
if not os.path.isdir(TEST_ROOT):
    for root, dirs, _ in os.walk("."):
        if "test" in dirs:
            TEST_ROOT = os.path.abspath(os.path.join(root, "test"))
            break

max_workers = max(1, min(8, os.cpu_count() or 1))




## === cell 1
train_df = pd.read_csv(train_labels_path)

print("Computing intensity features for training snippets (parallel...)")
train_ids = train_df["id"].tolist()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    train_feat = list(executor.map(compute_features, train_ids))

train_feat_arr = np.array(train_feat, dtype=float)  # shape (n_samples, 4)
train_feat_arr[np.isinf(train_feat_arr)] = np.nan

train_df["max_intensity"] = train_feat_arr[:, 0]
train_df["mean_intensity"] = train_feat_arr[:, 1]
train_df["std_intensity"] = train_feat_arr[:, 2]
train_df["diff_A_nonA"] = train_feat_arr[:, 3]

for col in ["max_intensity", "mean_intensity", "std_intensity", "diff_A_nonA"]:
    train_df[col].fillna(train_df[col].mean(), inplace=True)

X_train = train_df[
    ["max_intensity", "mean_intensity", "std_intensity", "diff_A_nonA"]
].values
y_train = train_df["target"].values

col_means = np.nanmean(X_train, axis=0)
nan_idx = np.where(np.isnan(X_train))
if nan_idx[0].size > 0:
    X_train[nan_idx] = np.take(col_means, nan_idx[1])

print("Training LogisticRegression model...")
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
logreg.fit(X_train, y_train)

model = logreg  # keep the name used later for clarity




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1132077090.py in <cell line: 0>()
     32 print("Training LogisticRegression model...")
     33 logreg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
---> 34 logreg.fit(X_train, y_train)
     35 
     36 model = logreg  # keep the name used later for clarity

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 2
submission_df = pd.read_csv(sample_submission_path)

print("Computing intensity features for test snippets (parallel...)")
test_ids = submission_df["id"].tolist()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    test_feat = list(executor.map(compute_features, test_ids))

test_feat_arr = np.array(test_feat, dtype=float)
test_feat_arr[np.isinf(test_feat_arr)] = np.nan

col_means = [
    train_df["max_intensity"].mean(),
    train_df["mean_intensity"].mean(),
    train_df["std_intensity"].mean(),
    train_df["diff_A_nonA"].mean(),
]
for i in range(4):
    test_feat_arr[:, i] = np.where(
        np.isnan(test_feat_arr[:, i]), col_means[i], test_feat_arr[:, i]
    )

submission_df["max_intensity"] = test_feat_arr[:, 0]
submission_df["mean_intensity"] = test_feat_arr[:, 1]
submission_df["std_intensity"] = test_feat_arr[:, 2]
submission_df["diff_A_nonA"] = test_feat_arr[:, 3]

X_test = submission_df[
    ["max_intensity", "mean_intensity", "std_intensity", "diff_A_nonA"]
].values

test_probs = model.predict_proba(X_test)[:, 1]
submission_df["target"] = test_probs

output_path = "submission.csv"
submission_df[["id", "target"]].to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submission_df.shape}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/819269584.py in <cell line: 0>()
     30 ].values
     31 
---> 32 test_probs = model.predict_proba(X_test)[:, 1]
     33 submission_df["target"] = test_probs
     34 

NameError: name 'model' is not defined
