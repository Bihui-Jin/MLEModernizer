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

0.7571609277550078

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The script failed because it tried to read several non‑existent submission files, which caused `FileNotFoundError` and subsequent `NameError`s. I replaced those loads with a single read of the provided `sample_submission.csv`, kept the column name unchanged, and wrote the dataframe out as `submission.csv`. This restores a working end‑to‑end pipeline that produces a valid Kaggle‑format submission file.'
- What this solution (achieved 0.4964) has done: 'I make the feature extraction robust by sanitizing NaN/inf values and clean the training‑ and test‑feature DataFrames before fitting the model. This prevents the “infinity” error that stopped training and, once the model fits, eliminates the subsequent “no classes_” error. No other logic is changed, so the pipeline still trains a logistic regression and writes a proper `submission.csv` file.'
- What this solution (achieved 0.4964) has done: 'I enhance the feature extraction by adding per‑cadence‑position statistics (mean and standard deviation for each of the 6 slices) alongside the existing global statistics. This provides the logistic regression model with richer information while keeping the overall pipeline, model, and training logic unchanged, helping to raise the validation AUC toward the target score.'
- What this solution (achieved 0.50043) has done: 'I added a few extra, inexpensive features (per‑position max/min) to give the model more signal, and introduced standard‑scaling before fitting the logistic regression so the coefficients are learned on a normalized space. These changes keep the overall pipeline and model type unchanged while providing a modest boost in validation AUC, moving the score closer to the target.'
- What this solution (achieved 0.4993) has done: 'I add a few inexpensive but informative statistics to the feature set – per‑position median and energy (sum of squares) as well as a global energy feature – and keep the rest of the pipeline unchanged. These extra signals give the logistic regression more discriminative power, which should raise the validation AUC toward the target without altering the core model or training procedure.'
- What this solution (achieved 0.49778) has done: 'I added a richer set of statistical descriptors (global and per‑position sum, variance, skewness, and kurtosis) to the feature extraction, which gives the logistic regression more discriminative information while keeping the original model unchanged. I also increased the regularisation strength (`C=2.0`) to better fit the expanded feature set. The rest of the pipeline, scaling, train/validation split and CSV output remain identical, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.50224) has done: 'I replace the logistic‑regression model with a GradientBoostingClassifier, which works better with the handcrafted statistical features and does not require scaling. The feature extraction and data handling remain unchanged, and the prediction pipeline is updated to use the raw feature matrix directly, bringing the validation AUC closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

BASE_PATH = "../input"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)


def get_npy_path(root_dir, id_str):
    folder = str(int(id_str[:2], 16) % 16)
    return os.path.join(root_dir, folder, f"{id_str}.npy")


def extract_features(npy_path):
    """
    Extract richer statistical features, plus high‑level A‑vs‑non‑A cadence summaries.
    """
    arr = np.load(npy_path)  # shape (6, 273, 256)

    flat = arr.ravel()
    mean = float(np.nanmean(flat))
    std = float(np.nanstd(flat))
    max_ = float(np.nanmax(flat))
    min_ = float(np.nanmin(flat))
    median = float(np.nanmedian(flat))
    rng = max_ - min_
    energy = float(np.nansum(flat**2))
    total_sum = float(np.nansum(flat))
    var = float(np.nanvar(flat))

    if std > 0:
        diff = flat - mean
        m3 = float(np.nanmean(diff**3))
        skew = m3 / (std**3)
        m4 = float(np.nanmean(diff**4))
        kurt = m4 / (std**4) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    feats = {
        "mean": mean if np.isfinite(mean) else 0.0,
        "std": std if np.isfinite(std) else 0.0,
        "max": max_ if np.isfinite(max_) else 0.0,
        "min": min_ if np.isfinite(min_) else 0.0,
        "median": median if np.isfinite(median) else 0.0,
        "range": rng if np.isfinite(rng) else 0.0,
        "energy": energy if np.isfinite(energy) else 0.0,
        "sum": total_sum if np.isfinite(total_sum) else 0.0,
        "var": var if np.isfinite(var) else 0.0,
        "skewness": skew if np.isfinite(skew) else 0.0,
        "kurtosis": kurt if np.isfinite(kurt) else 0.0,
    }

    pos_means, pos_stds = [], []
    for i in range(arr.shape[0]):  # 6 positions
        slice_i = arr[i].ravel()
        pos_mean = float(np.nanmean(slice_i))
        pos_std = float(np.nanstd(slice_i))
        pos_max = float(np.nanmax(slice_i))
        pos_min = float(np.nanmin(slice_i))
        pos_median = float(np.nanmedian(slice_i))
        pos_range = pos_max - pos_min
        pos_energy = float(np.nansum(slice_i**2))
        pos_sum = float(np.nansum(slice_i))
        pos_var = float(np.nanvar(slice_i))

        if pos_std > 0:
            diff = slice_i - pos_mean
            m3 = float(np.nanmean(diff**3))
            pos_skew = m3 / (pos_std**3)
            m4 = float(np.nanmean(diff**4))
            pos_kurt = m4 / (pos_std**4) - 3.0
        else:
            pos_skew = 0.0
            pos_kurt = 0.0

        prefix = f"pos{i+1}_"
        feats.update(
            {
                f"{prefix}mean": pos_mean if np.isfinite(pos_mean) else 0.0,
                f"{prefix}std": pos_std if np.isfinite(pos_std) else 0.0,
                f"{prefix}max": pos_max if np.isfinite(pos_max) else 0.0,
                f"{prefix}min": pos_min if np.isfinite(pos_min) else 0.0,
                f"{prefix}median": pos_median if np.isfinite(pos_median) else 0.0,
                f"{prefix}range": pos_range if np.isfinite(pos_range) else 0.0,
                f"{prefix}energy": pos_energy if np.isfinite(pos_energy) else 0.0,
                f"{prefix}sum": pos_sum if np.isfinite(pos_sum) else 0.0,
                f"{prefix}var": pos_var if np.isfinite(pos_var) else 0.0,
                f"{prefix}skewness": pos_skew if np.isfinite(pos_skew) else 0.0,
                f"{prefix}kurtosis": pos_kurt if np.isfinite(pos_kurt) else 0.0,
            }
        )
        pos_means.append(pos_mean)
        pos_stds.append(pos_std)

    a_indices = [0, 2, 4]
    non_a_indices = [1, 3, 5]

    a_means = [pos_means[i] for i in a_indices]
    non_a_means = [pos_means[i] for i in non_a_indices]
    a_stds = [pos_stds[i] for i in a_indices]
    non_a_stds = [pos_stds[i] for i in non_a_indices]

    feats["A_mean_avg"] = float(np.mean(a_means))
    feats["nonA_mean_avg"] = float(np.mean(non_a_means))
    feats["A_minus_nonA_mean"] = feats["A_mean_avg"] - feats["nonA_mean_avg"]
    feats["A_std_avg"] = float(np.mean(a_stds))
    feats["nonA_std_avg"] = float(np.mean(non_a_stds))
    feats["A_minus_nonA_std"] = feats["A_std_avg"] - feats["nonA_std_avg"]

    return feats




## === cell 1
train_feat_list = []
train_ids = []
for _, row in train_labels.iterrows():
    id_str = row["id"]
    npy_path = get_npy_path(TRAIN_DIR, id_str)
    if os.path.exists(npy_path):
        feats = extract_features(npy_path)
        train_feat_list.append(feats)
        train_ids.append(id_str)

X_train = pd.DataFrame(train_feat_list)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(X_train.mean(), inplace=True)

y_train = train_labels.set_index("id").loc[train_ids, "target"].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
model.fit(X_tr, y_tr)

val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.5f}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1664352946.py in <cell line: 0>()
     26     random_state=42,
     27 )
---> 28 model.fit(X_tr, y_tr)
     29 
     30 val_pred = model.predict_proba(X_val)[:, 1]

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in fit(self, X, y, sample_weight, monitor)
    427         # trees use different types for X and y, checking them separately.
    428 
--> 429         X, y = self._validate_data(
    430             X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
    431         )

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
GradientBoostingClassifier does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
test_feat_list = []
test_ids = []
for _, row in sample_sub.iterrows():
    id_str = row["id"]
    npy_path = get_npy_path(TEST_DIR, id_str)
    if os.path.exists(npy_path):
        feats = extract_features(npy_path)
        test_feat_list.append(feats)
        test_ids.append(id_str)
    else:
        neutral = X_train.mean().to_dict()
        test_feat_list.append(neutral)
        test_ids.append(id_str)

X_test = pd.DataFrame(test_feat_list)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(X_train.mean(), inplace=True)

test_pred = model.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({"id": test_ids, "target": test_pred})
submission_df.to_csv("submission.csv", index=False)
print("submission.csv written with", len(submission_df), "rows")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1735611513.py in <cell line: 0>()
     19 X_test.fillna(X_train.mean(), inplace=True)
     20 
---> 21 test_pred = model.predict_proba(X_test)[:, 1]
     22 
     23 submission_df = pd.DataFrame({"id": test_ids, "target": test_pred})

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in predict_proba(self, X)
   1353             If the ``loss`` does not support probabilities.
   1354         """
-> 1355         raw_predictions = self.decision_function(X)
   1356         try:
   1357             return self._loss._raw_prediction_to_proba(raw_predictions)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in decision_function(self, X)
   1259             array of shape (n_samples,).
   1260         """
-> 1261         X = self._validate_data(
   1262             X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
   1263         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

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
GradientBoostingClassifier does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
