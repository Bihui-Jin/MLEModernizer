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

0.7529302425359502

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the missing external submission reads with a safe baseline that loads the provided sample_submission.csv, computes the overall average target from the training labels, and writes that constant value for every test id. This fixes the FileNotFoundError and NameError and guarantees a valid submission.csv file; using the global mean moves the prediction toward a reasonable score without altering any core modeling logic.'
- What this solution (achieved 0.48241) has done: 'I replace the constant‑mean baseline with a lightweight model that uses a single informative feature (the mean intensity of each snippet). This keeps the core logic simple, adds only minimal computation, and is expected to raise the ROC‑AUC from 0.5 toward the target 0.7529. The script now loads training labels, extracts the mean‑intensity feature for every training snippet, fits a logistic‑regression model, applies the same feature extraction to the test snippets, predicts probabilities, and writes a valid `submission.csv`.'
- What this solution (achieved 0.48432) has done: 'I add a second, complementary feature (the intensity standard deviation) to the existing mean‑intensity feature, and train the same LogisticRegression model on both features. Missing values are now filled with the column‑wise overall means, preserving the original workflow while giving the model more discriminatory power, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.48123) has done: 'The changes pre‑scan the train and test directories once to build a fast id‑to‑file lookup, then use a thread pool to load and extract features in parallel while preserving order. This removes the repeated 16‑folder existence checks and speeds up I/O‑bound work without altering any modeling logic or feature definitions.'
- What this solution (achieved 0.48123) has done: 'I add richer statistical features (overall max/min/median and per‑cadence‑position max/min) to give the model more discriminative power, and switch to a tree‑based GradientBoostingClassifier when available (which usually improves ROC‑AUC over plain logistic regression). The changes keep the overall workflow unchanged while providing a higher‑scoring model that moves the metric toward the target.'
- What this solution (achieved 0.48123) has done: 'I extend the feature extractor to add per‑position medians and simple “on‑target vs off‑target” statistics (mean of the three A positions vs the three others, plus their difference and variance). These extra but inexpensive features usually help distinguish needle signals without changing the overall modeling approach. I also increase the GradientBoosting trees to 500 estimators for a modest boost in performance while keeping the same classifier class.'
- What this solution (achieved 0.48123) has done: 'We boost the model’s capacity by increasing the number of trees, using a smaller learning rate and a deeper tree depth for the GradientBoosting classifier, and add a balanced class_weight to the LogisticRegression fallback. These tweaks keep the overall pipeline unchanged while giving the learner more expressive power and better handling of any class imbalance, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I make the file‑path lookups robust by checking both the “data” sub‑folder and the top‑level input folder, and I fall back to a safe default directory if a folder is missing. I also adjust the submission path to write inside the kernel’s working directory. These fixes remove the FileNotFoundError while keeping the original feature extraction, model fitting, and fallback‑to‑global‑mean logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    Return an enriched feature vector for a snippet array.
    Features:
      - overall mean, std, max, min, median, 10th, 25th, 75th, 90th percentiles (9)
      - per‑position mean, std, max, min, median for each of 6 positions (5*6 = 30)
      - aggregate statistics on the three “A” positions (0,2,4):
          * mean_A, mean_nonA, diff_A_nonA, var_A                         (4)
    Total length = 43.
    """
    overall_mean = arr.mean()
    overall_std = arr.std()
    overall_max = arr.max()
    overall_min = arr.min()
    overall_median = np.median(arr)
    overall_p10 = np.percentile(arr, 10)
    overall_q1 = np.percentile(arr, 25)
    overall_q3 = np.percentile(arr, 75)
    overall_p90 = np.percentile(arr, 90)

    pos_means = arr.mean(axis=(1, 2))  # (6,)
    pos_stds = arr.std(axis=(1, 2))  # (6,)
    pos_maxs = arr.max(axis=(1, 2))  # (6,)
    pos_mins = arr.min(axis=(1, 2))  # (6,)
    pos_medians = np.median(arr, axis=(1, 2))  # (6,)

    a_idx = np.array([0, 2, 4])
    non_a_idx = np.array([1, 3, 5])

    mean_A = pos_means[a_idx].mean()
    mean_nonA = pos_means[non_a_idx].mean()
    diff_A_nonA = mean_A - mean_nonA
    var_A = pos_means[a_idx].var()

    return np.concatenate(
        (
            [
                overall_mean,
                overall_std,
                overall_max,
                overall_min,
                overall_median,
                overall_p10,
                overall_q1,
                overall_q3,
                overall_p90,
            ],
            pos_means,
            pos_stds,
            pos_maxs,
            pos_mins,
            pos_medians,
            [mean_A, mean_nonA, diff_A_nonA, var_A],
        ),
        dtype=np.float32,
    )




## === cell 1
def find_data_dir(root_candidates):
    for cand in root_candidates:
        p = Path(cand)
        if p.is_dir():
            return p
    raise FileNotFoundError("Could not locate data directory among candidates.")


train_root = find_data_dir(
    [
        "./data/train",
        "./kaggle/data/train",
        "/kaggle/input/seti-breakthrough-listen/train",
        "/kaggle/input/train",
        "./input/train",
    ]
)
test_root = find_data_dir(
    [
        "./data/test",
        "./kaggle/data/test",
        "/kaggle/input/seti-breakthrough-listen/test",
        "/kaggle/input/test",
        "./input/test",
    ]
)

train_labels_path = Path("./data/train_labels.csv")
if not train_labels_path.exists():
    train_labels_path = Path("./kaggle/data/train_labels.csv")
train_labels = pd.read_csv(train_labels_path)

print("Extracting training features...")
train_ids = train_labels["id"].values
train_feat_list = []
missing_ids = []

for fid in tqdm(train_ids, desc="train"):
    sub_dir = train_root / fid[0]  # first character as folder (as in dataset)
    npy_path = sub_dir / f"{fid}.npy"
    if not npy_path.is_file():
        npy_path = train_root / f"{fid}.npy"
    if npy_path.is_file():
        arr = np.load(npy_path)  # shape (6,273,256), dtype float16
        train_feat_list.append(extract_features(arr))
    else:
        missing_ids.append(fid)

if missing_ids:
    print(f"Warning: {len(missing_ids)} training files not found and will be ignored.")

train_features = np.vstack(train_feat_list)

train_labels = train_labels[~train_labels["id"].isin(missing_ids)].reset_index(
    drop=True
)

use_model = False
model = None
try:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train_features, train_labels["target"])
    use_model = True
except Exception as e_rf:
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
        model.fit(train_features, train_labels["target"])
        use_model = True
    except Exception as e_gb:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=300,
                    n_jobs=1,
                    C=2.0,
                    class_weight="balanced",
                ),
            )
            model.fit(train_features, train_labels["target"])
            use_model = True
        except Exception as e_lr:
            print("All model attempts failed.", e_rf, e_gb, e_lr)
            use_model = False

print("Extracting test features and generating predictions...")
test_ids_path = Path("./data/sample_submission.csv")
if not test_ids_path.exists():
    test_ids_path = Path("./kaggle/data/sample_submission.csv")
test_ids_df = pd.read_csv(test_ids_path)[["id"]]
test_ids = test_ids_df["id"].values

test_feat_list = []
missing_test = []

for fid in tqdm(test_ids, desc="test"):
    sub_dir = test_root / fid[0]
    npy_path = sub_dir / f"{fid}.npy"
    if not npy_path.is_file():
        npy_path = test_root / f"{fid}.npy"
    if npy_path.is_file():
        arr = np.load(npy_path)
        test_feat_list.append(extract_features(arr))
    else:
        missing_test.append(fid)
        test_feat_list.append(np.zeros(43, dtype=np.float32))

if missing_test:
    print(
        f"Warning: {len(missing_test)} test files not found; predictions will be set to global mean."
    )

test_features = np.vstack(test_feat_list)

if use_model:
    if hasattr(model, "predict_proba"):
        test_pred = model.predict_proba(test_features)[:, 1]
    else:
        test_pred = model.decision_function(test_features)
else:
    global_mean = train_labels["target"].mean()
    test_pred = np.full(len(test_ids), global_mean, dtype=np.float32)

if missing_test:
    test_pred[[i for i, fid in enumerate(test_ids) if fid in missing_test]] = (
        train_labels["target"].mean()
    )

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/264981779.py in <cell line: 0>()
     33     # fallback to alternative location
     34     train_labels_path = Path("./kaggle/data/train_labels.csv")
---> 35 train_labels = pd.read_csv(train_labels_path)
     36 
     37 # ----------------------------------------------------------------------

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'kaggle/data/train_labels.csv'
