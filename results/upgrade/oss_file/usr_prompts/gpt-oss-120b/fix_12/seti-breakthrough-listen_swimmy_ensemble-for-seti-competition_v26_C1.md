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

0.7571566294813898

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the missing‑file reads with safe fall‑backs: load the provided `sample_submission.csv`, compute a simple baseline probability (the overall mean of the training targets), and use that for every test row. This guarantees a valid `submission.csv` is written and avoids the FileNotFoundErrors. The logic of creating a final submission DataFrame is kept, but now it works even when the external submissions are unavailable.'
- What this solution (achieved 0.48796) has done: 'I add a lightweight model that extracts a single mean‑intensity feature from each snippet, trains a logistic‑regression‑style classifier on the training set, and blends its predictions with the existing baseline and external ensemble. This extra signal‑based component should raise the ROC‑AUC toward the target while keeping the original workflow intact. If the model cannot be trained (e.g., missing sklearn), the code safely falls back to the original baseline only.'
- What this solution (achieved 0.5) has done: 'I speed up file look‑ups by scanning the train and test directories once to build a dictionary that maps each snippet id to its full path, eliminating the repeated `os.listdir` checks. Then I load the snippets in parallel using a thread pool, which keeps the exact same feature extraction and model training logic while dramatically reducing I/O‑bound latency. All other logic (ensemble blending, baseline handling, submission creation) remains unchanged.'
- What this solution (achieved 0.5) has done: 'The update adds richer features by including the mean intensity of each of the six cadence positions, giving the model more discriminative information while keeping the original simple statistics. The model’s contribution in the final blend is increased (weight 0.7) so the trained classifier has a larger impact on the predictions, which should raise the ROC‑AUC toward the target. All other logic, file handling and external‑ensemble blending remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.5192) has done: 'I load the provided sample_submission.csv so sample_submission is defined, replace the sklearn‑based model with a lightweight linear scoring rule built directly from the training features (difference between positive‑ and negative‑class means passed through a sigmoid), and keep the existing blending logic. This fixes the NameError, removes the unavailable sklearn dependency, and adds a modest learned component that should improve the ROC‑AUC toward the target while preserving the original workflow.'
- What this solution (achieved 0.5192) has done: 'I lower the influence of the weak linear‐discriminant model by reducing `model_weight` from 0.9 to 0.5. This gives more weight to the global baseline probability, which is more stable, and is expected to raise the ROC‑AUC toward the target while keeping all existing logic unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings
import concurrent.futures
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

train_labels_path = os.path.join("train_labels.csv")
sample_submission_path = os.path.join("sample_submission.csv")
train_labels = pd.read_csv(train_labels_path)
sample_submission = pd.read_csv(sample_submission_path)

baseline_prob = float(train_labels["target"].mean())


def compute_features(path):
    """Return richer statistical features for a snippet."""
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    overall_mean = float(arr.mean())
    overall_std = float(arr.std())
    overall_max = float(arr.max())
    overall_min = float(arr.min())
    per_pos_means = [float(arr[i].mean()) for i in range(arr.shape[0])]
    per_pos_stds = [float(arr[i].std()) for i in range(arr.shape[0])]
    on_target_mean = np.mean([per_pos_means[i] for i in [0, 2, 4]])
    off_target_mean = np.mean([per_pos_means[i] for i in [1, 3, 5]])
    diff_on_off = float(on_target_mean - off_target_mean)
    return (
        [overall_mean, overall_std, overall_max, overall_min]
        + per_pos_means
        + per_pos_stds
        + [diff_on_off]
    )


train_root = os.path.join("..", "input", "train")
if not os.path.isdir(train_root):
    train_root = "train"

test_root = os.path.join("..", "input", "test")
if not os.path.isdir(test_root):
    test_root = "test"


def build_id_path_map(root_dir):
    id_path = {}
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".npy"):
                id_str = f[:-4]
                id_path[id_str] = os.path.join(subdir, f)
    return id_path


train_id_path = build_id_path_map(train_root)
test_id_path = build_id_path_map(test_root)

train_ids = train_labels["id"].values
train_feats = []
train_y = []


def extract_train(id_str):
    p = train_id_path.get(id_str)
    if p is None:
        return None
    feats = compute_features(p)
    target = train_labels.loc[train_labels["id"] == id_str, "target"].values[0]
    return feats, target


with concurrent.futures.ThreadPoolExecutor() as executor:
    for result in executor.map(extract_train, train_ids):
        if result is not None:
            feats, target = result
            train_feats.append(feats)
            train_y.append(target)

final_pred = pd.Series(baseline_prob, index=sample_submission.index, dtype=np.float32)

if train_feats:
    X_train = np.array(train_feats, dtype=np.float32)
    y_train = np.array(train_y, dtype=np.int32)

    pos_mean = X_train[y_train == 1].mean(axis=0)
    neg_mean = X_train[y_train == 0].mean(axis=0)
    weight_vec = pos_mean - neg_mean

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y_train))
    split_idx = int(0.8 * len(y_train))
    train_idx, val_idx = perm[:split_idx], perm[split_idx:]

    X_sub = X_train[train_idx]
    y_sub = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    pos_mean_sub = X_sub[y_sub == 1].mean(axis=0)
    neg_mean_sub = X_sub[y_sub == 0].mean(axis=0)
    weight_vec_sub = pos_mean_sub - neg_mean_sub

    raw_val = X_val.dot(weight_vec_sub)
    prob_val = sigmoid(raw_val)

    def compute_auc(y_true, y_score):
        order = np.argsort(-y_score)
        y = y_true[order]
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        P = y.sum()
        N = len(y) - P
        if P == 0 or N == 0:
            return 0.0
        tpr = tp / P
        fpr = fp / N
        return np.trapz(tpr, fpr)

    best_weight = 0.5
    best_auc = -1.0
    for w in np.linspace(0, 1, 11):
        blended = (1 - w) * baseline_prob + w * prob_val
        auc = compute_auc(y_val, blended)
        if auc > best_auc:
            best_auc = auc
            best_weight = w
    model_weight = best_weight

    test_feats = []
    valid_test_idx = []

    def extract_test(pair):
        i, id_str = pair
        p = test_id_path.get(id_str)
        if p is None:
            return None
        return i, compute_features(p)

    indexed_ids = list(enumerate(sample_submission["id"]))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in executor.map(extract_test, indexed_ids):
            if res is not None:
                i, feats = res
                test_feats.append(feats)
                valid_test_idx.append(i)

    if test_feats:
        X_test = np.array(test_feats, dtype=np.float32)
        raw_scores = X_test.dot(weight_vec)
        test_probs = sigmoid(raw_scores).astype(np.float32)

        blend_series = pd.Series(
            baseline_prob, index=sample_submission.index, dtype=np.float32
        )
        blend_series.iloc[valid_test_idx] = test_probs
        final_pred = (1 - model_weight) * final_pred + model_weight * blend_series

final_pred = np.clip(final_pred, 0.0, 1.0)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3949870948.py in <cell line: 0>()
     10 train_labels_path = os.path.join("train_labels.csv")
     11 sample_submission_path = os.path.join("sample_submission.csv")
---> 12 train_labels = pd.read_csv(train_labels_path)
     13 sample_submission = pd.read_csv(sample_submission_path)
     14 

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

FileNotFoundError: [Errno 2] No such file or directory: 'train_labels.csv'

## === cell 1
submission = pd.DataFrame({"id": sample_submission["id"], "target": final_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2770703419.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": sample_submission["id"], "target": final_pred})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path} with {len(submission)} rows.")

NameError: name 'sample_submission' is not defined
