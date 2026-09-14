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

0.75697

# 6. Current score

0.48616

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48796) has done: 'I replace the missing external submission loads with a self‑contained baseline that creates a valid `submission.csv`. The script reads the provided `sample_submission.csv`, loads each test snippet (`.npy`), uses its overall mean intensity as a raw score, rescales those scores to the [0, 1] range, and writes the results. This fixes the FileNotFoundError and ensures a correctly formatted CSV is produced, moving the solution from “no submission” toward the target metric.'
- What this solution (achieved 0.50636) has done: 'I replace the naïve overall‑mean heuristic with a simple “on‑target vs off‑target” difference: for each snippet I compute the mean intensity of the three A‑position observations (indices 0, 2, 4) and subtract the mean of the three B‑positions (indices 1, 3, 5). This captures the expected signal‑only pattern and should raise the AUC toward the target while keeping the rest of the pipeline unchanged. Missing files are still imputed with the global mean of this new score.'
- What this solution (achieved 0.50636) has done: 'I refine the `snippet_score` function to capture not only the mean intensity difference between “A” and “B” positions but also the variability (standard deviation) between them. Adding a modest weight to the standard‑deviation difference provides a richer signal indicator while preserving the original pipeline and scaling steps, which should raise the AUC toward the target without altering the core logic.'
- What this solution (achieved 0.50636) has done: 'I add a quick validation step that evaluates a few choices for the std‑weight on a sampled subset of the training data and picks the best‑performing weight. This keeps the core scoring logic unchanged while tuning the weighting factor to improve the AUC, moving the score closer to the target.'
- What this solution (achieved 0.50636) has done: 'I extend the heuristic scoring to also consider the median intensity difference between the “A” and “B” observations, and tune both the standard‑deviation weight and a new median‑difference weight on a validation subset. This keeps the original simple scoring approach while adding a modest, data‑driven improvement that should raise the AUC toward the target.'
- What this solution (achieved 0.49014) has done: 'I keep the overall pipeline but replace the hand‑tuned weighted differences with a tiny linear model learned from a validation sample. By extracting four simple features (mean A‑B difference, std A‑B difference, median A‑B difference, and overall mean intensity) and fitting a least‑squares linear regressor, the predictions become better calibrated for AUC while still using the same simple heuristic core. Missing .npy files are imputed with the average feature values, and the final scores are min‑max scaled to the [0, 1] range as before.'
- What this solution (achieved 0.48802) has done: 'We replace the per‑ID `glob` searches with a single directory walk that builds an id→filepath dictionary for train and test, then use a thread pool to load the .npy files in parallel while preserving order. This removes the O(N) filesystem scans and speeds up I/O dramatically, keeping the exact same feature extraction, linear‑model fitting, and scaling logic unchanged.'
- What this solution (achieved 0.5) has done: 'I fixed the linear‑model fitting crash by handling possible NaNs and falling back to a pseudo‑inverse solution, then switched to a logistic‑regression model (still a simple linear approach) which better matches the AUC metric. I also ensured the coefficient/model object is always defined so the later cells can compute scores, and I added robust imputation for any remaining missing features. The script now runs end‑to‑end and writes a valid `submission.csv` with correctly scaled probabilities.'
- What this solution (achieved 0.5) has done: 'I added a quick train/validation split and a small grid‑search over the LogisticRegression regularisation strength C to choose the model that gives the highest validation AUC, then refit that best model on the full training data. This keeps the original feature extraction and linear‑model pipeline unchanged while providing a modest performance boost that moves the score toward the target.'
- What this solution (achieved 0.48616) has done: 'Implemented feature scaling and quadratic expansion to enhance the logistic‑regression model while keeping the original pipeline intact.  
1. Added `PolynomialFeatures` (degree 2) and `StandardScaler` to normalize the enriched feature set.  
2. Applied the same transformations to both training and test data, then refit the logistic regression using the scaled features.  
3. Simplified prediction by using the trained model’s `predict_proba` directly, removing manual coefficient handling. These changes are expected to raise the validation AUC toward the target without altering the core feature‑extraction logic.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import concurrent.futures

BASE_INPUT = "/kaggle/input/seti-breakthrough-listen"

SAMPLE_SUBMISSION_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_ROOT = os.path.join(BASE_INPUT, "test")
TRAIN_ROOT = os.path.join(BASE_INPUT, "train")
TRAIN_LABELS_PATH = os.path.join(BASE_INPUT, "train_labels.csv")


def build_id_path_map(root_dir: str) -> dict:
    """
    Walk the entire root once and create a mapping from file id (without .npy)
    to its full path.  This avoids repeated glob calls for every id.
    """
    pattern = os.path.join(root_dir, "**", "*.npy")
    paths = glob.glob(pattern, recursive=True)
    id_path = {}
    for p in paths:
        fname = os.path.basename(p)
        if fname.endswith(".npy"):
            fid = fname[:-4]  # strip extension
            id_path[fid] = p
    return id_path


TRAIN_PATH_MAP = build_id_path_map(TRAIN_ROOT)
TEST_PATH_MAP = build_id_path_map(TEST_ROOT)


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    Return an enriched feature vector:
    [mean diff (A‑B), std diff (A‑B), median diff (A‑B),
     overall mean intensity,
     max diff (A‑B), min diff (A‑B), overall std intensity]
    """
    a_idx = (0, 2, 4)
    b_idx = (1, 3, 5)
    a_slice = arr[a_idx]
    b_slice = arr[b_idx]

    mean_diff = a_slice.mean() - b_slice.mean()
    std_diff = a_slice.std() - b_slice.std()
    median_diff = np.median(a_slice) - np.median(b_slice)

    overall_mean = arr.mean()
    max_diff = a_slice.max() - b_slice.max()
    min_diff = a_slice.min() - b_slice.min()
    overall_std = arr.std()

    return np.array(
        [
            mean_diff,
            std_diff,
            median_diff,
            overall_mean,
            max_diff,
            min_diff,
            overall_std,
        ],
        dtype=np.float64,
    )


def load_feature(tid_path):
    """
    Helper for parallel execution: given (id, path) returns the feature vector
    or a NaN‑filled array if loading fails or the file is missing.
    """
    tid, path = tid_path
    if path is None or not os.path.isfile(path):
        return np.full(7, np.nan, dtype=np.float64)
    try:
        arr = np.load(path)
        return extract_features(arr)
    except Exception:
        return np.full(7, np.nan, dtype=np.float64)


train_labels = pd.read_csv(TRAIN_LABELS_PATH)
train_ids = train_labels["id"].tolist()

train_id_path_list = [(tid, TRAIN_PATH_MAP.get(tid)) for tid in train_ids]

with concurrent.futures.ThreadPoolExecutor() as executor:
    feature_list = list(executor.map(load_feature, train_id_path_list))

X = np.vstack(feature_list)  # shape (n_samples, 7)

col_means = np.nanmean(X, axis=0)
inds_nan = np.where(np.isnan(X))
X[inds_nan] = np.take(col_means, inds_nan[1])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

y = train_labels.set_index("id").loc[train_ids]["target"].values.astype(np.float64)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)  # shape (n_samples, 28)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

best_auc = -1.0
best_C = 1.0
for C in [0.01, 0.1, 1, 10, 100]:
    try:
        mdl = LogisticRegression(max_iter=1000, n_jobs=5, C=C, solver="lbfgs")
        mdl.fit(X_train, y_train)
        val_pred = mdl.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        if auc > best_auc:
            best_auc = auc
            best_C = C
    except Exception:
        continue

print(f"Best validation AUC during C‑search: {best_auc:.5f} (C={best_C})")

model = LogisticRegression(max_iter=1000, n_jobs=5, C=best_C, solver="lbfgs")
model.fit(X_scaled, y)

val_auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
print(f"Training AUC (on full train data): {val_auc:.5f}")



## === cell 1
submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
test_ids = submission_df["id"].tolist()

test_id_path_list = [(tid, TEST_PATH_MAP.get(tid)) for tid in test_ids]

with concurrent.futures.ThreadPoolExecutor() as executor:
    test_features = list(executor.map(load_feature, test_id_path_list))

test_feat_arr = np.vstack(test_features)  # shape (n_test, 7)

inds_nan_test = np.where(np.isnan(test_feat_arr))
test_feat_arr[inds_nan_test] = np.take(col_means, inds_nan_test[1])
test_feat_arr = np.nan_to_num(test_feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

test_feat_poly = poly.transform(test_feat_arr)
test_feat_scaled = scaler.transform(test_feat_poly)

test_probs = model.predict_proba(test_feat_scaled)[:, 1]

score_min, score_max = test_probs.min(), test_probs.max()
if score_max > score_min:
    scaled_scores = (test_probs - score_min) / (score_max - score_min)
else:
    scaled_scores = np.full_like(test_probs, 0.5)



## === cell 2
submission_df["target"] = scaled_scores
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)

print(f"Submission file written to {output_path}")
print(f"Number of test IDs processed: {len(test_ids)}")
