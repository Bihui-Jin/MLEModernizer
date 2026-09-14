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

0.7571671719743182

# 6. Current score

0.50692

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48693) has done: 'I remove the dependency on missing external Kaggle Dataset “../input/.../submission.csv” files, which is the direct cause of the FileNotFoundError and prevents any submission from being created. To keep the core “blend submissions with fixed weights” logic intact, I implement a safe loader that uses those external files if they exist, otherwise falls back to a deterministic baseline prediction derived from the provided local data (simple energy/contrast features from each `.npy` snippet) and calibrates it to a probability. I also enforce correct `id` alignment to `sample_submission.csv` so the output rows match exactly what Kaggle expects. Finally, the script always write a valid `submission.csv` with columns `id,target` in the working directory.'
- What this solution (achieved 0.49046) has done: 'Your current score is far below the target, and the biggest limiter is that the intended ensemble submissions are missing so you’re effectively submitting a weak handcrafted baseline. To move the AUC up toward the target while preserving the “blend submissions with fixed weights” core logic, I keep the blend unchanged but make the fallback predictions (used when external submissions aren’t present) much stronger by adding a small logistic-regression calibrator trained on the provided `train_labels.csv` using the same simple per-snippet features. This stays within the existing approach (still a deterministic feature-based probability feeding the same blender), but should materially increase separability vs. the current mean/percentile heuristic. I also keep strict `id` alignment to `sample_submission.csv` and ensure `submission.csv` is always written validly.'
- What this solution (achieved 0.50692) has done: 'I keep your “fixed-weight blend of external submissions” logic exactly as-is, but strengthen the fallback predictions that are used when those external `../input/.../submission.csv` files aren’t available (your current score indicates you’re mostly relying on the fallback). The smallest high-impact change is to make the fallback feature extractor more discriminative while preserving the same overall approach: still compute simple per-snippet statistics, then calibrate with the same in-notebook logistic regression. I also change the calibrator training subset selection from “first N rows” to a deterministic stratified sample (same size) so it better represents the full training distribution without changing the training approach. These changes should increase AUC toward your target without touching the ensemble weights, file paths, or submission formatting.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input"
DATA_ROOT = "/kaggle/data"

SAMPLE_SUB_PATHS = [
    os.path.join(BASE_INPUT, "sample_submission.csv"),
    os.path.join(DATA_ROOT, "sample_submission.csv"),
    os.path.join(BASE_INPUT, "seti-breakthrough-listen", "sample_submission.csv"),
    os.path.join(DATA_ROOT, "seti-breakthrough-listen", "sample_submission.csv"),
]

TEST_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "test"),
    os.path.join(DATA_ROOT, "test"),
    os.path.join(BASE_INPUT, "seti-breakthrough-listen", "test"),
    os.path.join(DATA_ROOT, "seti-breakthrough-listen", "test"),
]

TRAIN_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "train"),
    os.path.join(DATA_ROOT, "train"),
    os.path.join(BASE_INPUT, "seti-breakthrough-listen", "train"),
    os.path.join(DATA_ROOT, "seti-breakthrough-listen", "train"),
]

TRAIN_LABELS_CANDIDATES = [
    os.path.join(BASE_INPUT, "train_labels.csv"),
    os.path.join(DATA_ROOT, "train_labels.csv"),
    os.path.join(BASE_INPUT, "seti-breakthrough-listen", "train_labels.csv"),
    os.path.join(DATA_ROOT, "seti-breakthrough-listen", "train_labels.csv"),
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_path = first_existing(SAMPLE_SUB_PATHS)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )

sample_sub = pd.read_csv(sample_path)
if not {"id", "target"}.issubset(sample_sub.columns):
    raise ValueError(
        f"sample_submission.csv must contain columns id,target. Found: {sample_sub.columns.tolist()}"
    )

test_dir = first_existing(TEST_DIR_CANDIDATES)
if test_dir is None:
    raise FileNotFoundError("Could not find test/ directory in expected locations.")

train_dir = first_existing(TRAIN_DIR_CANDIDATES)
train_labels_path = first_existing(TRAIN_LABELS_CANDIDATES)




## === cell 2
def try_load_submission(path):
    if path is None or (not os.path.exists(path)):
        return None
    df = pd.read_csv(path)
    if not {"id", "target"}.issubset(df.columns):
        return None
    df = df[["id", "target"]].copy()
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"])
    return df


ext_paths = {
    "data1": "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "data2": "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    "data3": "../input/seti-bl-tf-starter-tpu/submission.csv",
    "data4": "../input/seti-learned-image-resizing/submission.csv",
    "data5": "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "data6": "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
}

data1 = try_load_submission(ext_paths["data1"])
data2 = try_load_submission(ext_paths["data2"])
data3 = try_load_submission(ext_paths["data3"])
data4 = try_load_submission(ext_paths["data4"])
data5 = try_load_submission(ext_paths["data5"])
data6_ext = try_load_submission(ext_paths["data6"])




## === cell 3
def iter_npy_files(root_dir):
    pattern = os.path.join(root_dir, "*", "*.npy")
    return glob.glob(pattern)


def compute_features_from_array(x):
    x = x.astype(np.float32)  # (6,273,256)
    on = x[[0, 2, 4]]
    off = x[[1, 3, 5]]

    std_all = float(x.std() + 1e-6)

    mu_on = float(on.mean())
    mu_off = float(off.mean())

    p_on = float(np.percentile(on, 99.5))
    p_off = float(np.percentile(off, 99.5))

    s_on = float(on.std())
    s_off = float(off.std())

    mx_on = float(on.max())
    mx_off = float(off.max())

    med_on = float(np.median(on))
    med_off = float(np.median(off))
    p995_on = float(np.percentile(on, 99.9))
    p995_off = float(np.percentile(off, 99.9))

    z_on = (on - on.mean()) / (on.std() + 1e-6)
    z_off = (off - off.mean()) / (off.std() + 1e-6)
    m3_on = float(np.mean(np.abs(z_on) ** 3))
    m3_off = float(np.mean(np.abs(z_off) ** 3))

    f1 = (mu_on - mu_off) / std_all
    f2 = (p_on - p_off) / std_all
    f3 = (s_on - s_off) / (std_all + 1e-6)
    f4 = (mx_on - mx_off) / std_all
    f5 = (med_on - med_off) / std_all
    f6 = (p995_on - p995_off) / std_all
    f7 = m3_on - m3_off  # already normalized inside z

    return np.array([f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)


def build_id_to_path(root_dir):
    files = iter_npy_files(root_dir)
    id2path = {}
    for fp in files:
        base = os.path.basename(fp)
        if base.endswith(".npy"):
            _id = base[:-4]
            id2path[_id] = fp
    return id2path


def extract_features_for_ids(root_dir, ids, id2path=None):
    if id2path is None:
        id2path = build_id_to_path(root_dir)
    X = np.zeros((len(ids), 7), dtype=np.float32)
    missing = 0
    for i, _id in enumerate(ids):
        fp = id2path.get(_id)
        if fp is None:
            missing += 1
            continue
        x = np.load(fp)
        X[i] = compute_features_from_array(x)
    return X, missing


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))




## === cell 4
def fit_logreg_lbfgs(X, y, reg=1.0, max_iter=80):
    """
    Minimal deterministic logistic regression using Newton steps (small feature dim).
    This avoids needing sklearn while producing calibrated probabilities.
    """
    n, d = X.shape
    Xb = np.concatenate([np.ones((n, 1), dtype=np.float32), X], axis=1)  # bias
    w = np.zeros(d + 1, dtype=np.float64)

    y = y.astype(np.float64)

    for _ in range(max_iter):
        z = Xb @ w
        p = sigmoid(z)
        g = Xb.T @ (p - y)
        g[1:] += reg * w[1:]

        r = p * (1.0 - p)
        XR = Xb * r[:, None]
        H = XR.T @ Xb
        for j in range(1, d + 1):
            H[j, j] += reg

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]

        w_new = w - step

        if np.max(np.abs(w_new - w)) < 1e-6:
            w = w_new
            break
        w = w_new

    return w


def predict_logreg(X, w):
    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
    return sigmoid(Xb @ w)


use_trained_calibrator = (
    (train_dir is not None)
    and (train_labels_path is not None)
    and os.path.exists(train_labels_path)
    and os.path.isdir(train_dir)
)

w_cal = None
train_id2path = None
if use_trained_calibrator:
    train_labels = pd.read_csv(train_labels_path)
    train_labels["id"] = train_labels["id"].astype(str)
    train_labels["target"] = train_labels["target"].astype(int)

    TRAIN_MAX = 12000
    if TRAIN_MAX is not None and len(train_labels) > TRAIN_MAX:
        rng = np.random.default_rng(123)
        pos = train_labels[train_labels["target"] == 1]
        neg = train_labels[train_labels["target"] == 0]
        n_pos = int(round(TRAIN_MAX * (len(pos) / len(train_labels))))
        n_pos = max(1, min(len(pos), n_pos))
        n_neg = TRAIN_MAX - n_pos
        n_neg = max(1, min(len(neg), n_neg))
        pos_idx = rng.choice(pos.index.values, size=n_pos, replace=False)
        neg_idx = rng.choice(neg.index.values, size=n_neg, replace=False)
        train_labels = train_labels.loc[np.concatenate([pos_idx, neg_idx])].copy()
        train_labels = train_labels.sort_values("id").reset_index(drop=True)

    train_ids = train_labels["id"].tolist()
    y_train = train_labels["target"].values

    train_id2path = build_id_to_path(train_dir)
    X_train, missing_train = extract_features_for_ids(
        train_dir, train_ids, id2path=train_id2path
    )

    if missing_train < 0.2 * len(train_ids):
        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True) + 1e-6
        Xs = (X_train - mu) / sd

        w_cal = fit_logreg_lbfgs(Xs, y_train, reg=2.0, max_iter=60)
        cal_norm = (mu.astype(np.float32), sd.astype(np.float32))
    else:
        w_cal = None
        cal_norm = None
else:
    cal_norm = None




## === cell 5
def baseline_predict_from_npy(test_root, ids_needed, w_cal=None, cal_norm=None):
    id2path = build_id_to_path(test_root)

    ids_list = list(ids_needed)
    X, _ = extract_features_for_ids(test_root, ids_list, id2path=id2path)

    if w_cal is not None and cal_norm is not None:
        mu, sd = cal_norm
        Xs = (X - mu) / sd
        prob = predict_logreg(Xs, w_cal).astype(np.float64)
    else:
        f1 = X[:, 0]
        f2 = X[:, 1]
        score = 0.7 * f1 + 0.3 * f2
        prob = sigmoid(2.0 * score).astype(np.float64)

    out = pd.DataFrame({"id": ids_list, "target": prob})
    return out


ids_needed = sample_sub["id"].astype(str).tolist()
baseline_df = baseline_predict_from_npy(
    test_dir, set(ids_needed), w_cal=w_cal, cal_norm=cal_norm
)

baseline_df = sample_sub[["id"]].merge(baseline_df, on="id", how="left")
baseline_df["target"] = baseline_df["target"].fillna(0.5).astype(float)




## === cell 6
def align_to_sample(df, sample_ids, default_targets):
    if df is None:
        return default_targets.copy()
    tmp = pd.DataFrame({"id": sample_ids}).merge(
        df[["id", "target"]], on="id", how="left"
    )
    tmp["target"] = tmp["target"].fillna(default_targets["target"]).astype(float)
    return tmp


sample_ids = sample_sub["id"].astype(str).tolist()

data1 = align_to_sample(data1, sample_ids, baseline_df)
data2 = align_to_sample(data2, sample_ids, baseline_df)
data3 = align_to_sample(data3, sample_ids, baseline_df)
data4 = align_to_sample(data4, sample_ids, baseline_df)
data5 = align_to_sample(data5, sample_ids, baseline_df)
data6_ext = align_to_sample(data6_ext, sample_ids, baseline_df)



## === cell 7
data6 = data1.copy()
data6["target"] = (
    0.765 * data5["target"].values
    + 0.135 * data4["target"].values
    + 0.1 * data6["target"].values
    + 0.0 * data2["target"].values
    + 0.0 * data3["target"].values
)

data6["target"] = data6["target"].clip(0.0, 1.0)

submission = data6[["id", "target"]].copy()
submission = pd.DataFrame({"id": sample_ids}).merge(submission, on="id", how="left")
submission["target"] = submission["target"].fillna(0.5).astype(float)



## === cell 8
submission.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert chk.shape[0] == sample_sub.shape[0]
assert list(chk.columns) == ["id", "target"]
print(chk.head())
print("Wrote submission.csv with", chk.shape[0], "rows")
if w_cal is not None:
    print("Used trained logistic calibrator for fallback predictions.")
else:
    print("Used heuristic fallback predictions (no calibrator trained).")
