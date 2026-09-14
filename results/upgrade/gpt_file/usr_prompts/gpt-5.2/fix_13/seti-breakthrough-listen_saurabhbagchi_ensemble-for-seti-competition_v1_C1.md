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

0.7277666482882906

# 6. Current score

0.48876

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50081) has done: 'Your notebook fails because it tries to read two external Kaggle Dataset submission files that don’t exist in this environment, so nothing downstream gets defined and no `submission.csv` is written. I replace that dependency with a minimal, self-contained baseline that reads the provided `sample_submission.csv` (for correct `id` order) and generates predictions from the available `test/` `.npy` snippets. To keep changes minimal and stable, the “model” be a simple deterministic heuristic based on the expected ABACAD structure (compare energy differences between A panels vs B/C/D), then normalize to valid probabilities and write `submission.csv` with the required columns.'
- What this solution (achieved 0.48436) has done: 'You’re currently getting ~0.50 AUC because the heuristic is essentially uncalibrated and likely near-constant across samples; to move toward the 0.727 target without changing the overall “simple deterministic heuristic” approach, I (1) strengthen the cadence-aligned signal by using robust, per-panel normalization and a contrast feature that looks for structure present in A but absent in B/C/D, and (2) calibrate the final probabilities using a monotonic rank-based mapping (still deterministic, no labels used) so predictions aren’t overly squashed around 0.5. These are minimal changes that preserve your core logic (ABACAD energy comparisons + sigmoid-to-probabilities) while typically yielding a materially better AUC. The script still run end-to-end within the time limit and write a valid `submission.csv` with correct `id` order.'
- What this solution (achieved 0.48437) has done: 'To move your AUC upward toward the 0.7278 target while keeping the same “deterministic ABACAD heuristic + monotonic calibration” core, I strengthen the snippet scoring in a minimal way: add one additional cadence-aligned feature that measures how much *peak structure* in A exceeds off-target panels (a robust top-quantile contrast), and slightly rebalance the existing components so scores have more useful spread. I also standardize raw scores (unsupervised, using test distribution only) before the sigmoid so it doesn’t saturate and waste ranking information. Finally, I keep the same rank-to-uniform blending (still monotonic and label-free) but adjust weights modestly to rely a bit more on the improved raw score ranking.'
- What this solution (achieved 0.48673) has done: 'Your current score (0.484) is far below the target (0.728), so we should improve ranking quality while keeping the same ABACAD deterministic heuristic + monotonic calibration structure. The smallest high-impact fix is to make the snippet score more sensitive to “A-only” diagonal/track-like structure by adding one more cadence-aligned feature: a robust row-wise (time-axis) maximum profile contrast between A panels and off panels, computed on the already robust-normalized data. This preserves the exact core approach (no learning, no labels, still just contrasts + monotonic mapping), but typically increases score spread in a way that better matches the known signal morphology. I keep your existing calibration, only adding this one feature with a modest weight so behavior remains stable.'
- What this solution (achieved 0.48753) has done: 'Your current AUC (0.4867) is far below the target (0.7278), so we should improve the *ranking* of predictions while keeping the same deterministic ABACAD-contrast heuristic and label-free monotonic calibration. The most likely issue is that the score is dominated by global energy/similarity statistics that don’t specifically reward the thin “track-like” morphology of many injected signals, so we add one minimal additional cadence-aligned feature: a robust per-row “peakiness” (top-quantile of row max) computed on a lightly smoothed frequency axis to emphasize narrowband tracks. This preserves the core approach (no training, no labels, same pipeline) and only changes the snippet scoring by adding a small extra contrast term, keeping weights modest for stability. Submission writing, id alignment, and probability mapping remain unchanged.'
- What this solution (achieved 0.48769) has done: 'Your current AUC is far below the target, and the biggest issue is that the heuristic is likely extracting a lot of “global energy” but not enough “A-only thin track” signal that drives true ranking. I keep your exact pipeline structure (deterministic ABACAD contrast scoring → robust standardize → sigmoid + rank-to-uniform blend) and only make a minimal, targeted change inside `score_snippet`: add a lightweight diagonal/trackness contrast based on a Sobel-like gradient magnitude (emphasizes thin sloped lines) computed per panel, then compare A vs off panels. I also make one minimal calibration tweak to use the standardized scores for the rank mapping (still monotonic, label-free) so ranking reflects the same robust scaling you already trust, which typically improves stability without changing semantics. Everything else (paths, reading, submission writing) remains unchanged.'
- What this solution (achieved 0.48934) has done: 'Your current AUC (0.4877) is far below the target (0.7278), so we should improve ranking signal while keeping your same deterministic ABACAD-contrast heuristic and label-free monotonic calibration. The most impactful minimal fix here is to correct and strengthen the “trackness” feature: your current gradient block has inconsistent intermediate shapes and is effectively just a simple central-difference magnitude; we replace it with a shape-safe Sobel-like gradient computed on the same smoothed magnitude and then compare robust high-quantiles in A vs off panels. To avoid destabilizing the rest of the heuristic, we keep all other features and weights unchanged and only swap the track_contrast computation plus a tiny epsilon/shape safety. Submission writing and probability mapping remain identical.'
- What this solution (achieved 0.48869) has done: 'Your current AUC (0.489) is far below the 0.728 target, so we need a small but meaningful ranking improvement without changing the overall “deterministic ABACAD heuristic → monotonic mapping → submission” structure. The biggest low-risk gain is to make the contrast features focus on the *expected A-only persistence*: needles should appear consistently across A panels (0,2,4) but not in off panels, so we replace A aggregation from mean to median (more robust) and add a minimal “A persistence” term using the minimum across A panels (penalizes one-off spikes). To keep calibration semantics the same and stable, we keep the same robust standardization and the same blend of rank-uniform and sigmoid, only slightly rebalancing the blend to rely more on ranking (AUC is rank-based). All paths and submission writing remain unchanged.'
- What this solution (achieved 0.48876) has done: 'To move your AUC upward toward the 0.7278 target without changing the overall “deterministic ABACAD heuristic → monotonic mapping → submission” core, I make one targeted improvement: add a minimal A-only “excess trackness” feature that uses the difference image (median(A) − median(OFF)) and measures diagonal/line-like structure on that residual instead of on raw panels. This stays fully label-free, deterministic, and cadence-aligned, but better matches the known injection morphology (signals appear in A and vanish off-target). I keep all existing features and calibration intact, only adding this one extra contrast term with a modest weight so behavior remains stable. The submission format, id alignment, and output writing remain unchanged.'
- What this solution (achieved 0.48979) has done: 'Your current AUC (~0.489) is far below the target (~0.728), so we need a small but meaningful ranking improvement while keeping the exact same “deterministic ABACAD heuristic → monotonic mapping → submission” structure. The lowest-risk change is to add one more cadence-aligned feature that better matches common injections: an “A-only residual line strength” computed on the signed residual (median(A) − median(OFF)) using a lightweight diagonal integral (main + anti-diagonal) to emphasize thin sloped tracks that appear only in A. This does not change your architecture/training (still none), keeps everything label-free and deterministic, and only adds one extra term to `score_snippet` with a modest weight. Everything else (paths, id alignment, standardization, rank blending, and CSV writing) remains unchanged.'
- What this solution (achieved 0.48892) has done: 'Your current AUC is far below the target, so we should increase ranking quality while keeping your exact “deterministic ABACAD heuristic → robust standardize → sigmoid + rank-uniform blend” structure. The smallest high-impact change is to stop forcing the residual line-strength feature to be unsigned: using an absolute residual makes both “A has extra signal” and “OFF has extra RFI” look the same, which hurts AUC. I keep all existing features and weights, but compute a signed residual and derive a signed “A-only line strength” contrast (A minus OFF) so OFF-only structure is penalized rather than rewarded. Everything else (paths, reading, calibration, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.48876) has done: 'Your AUC is far below the target, so we should improve ranking signal with the smallest possible change inside your existing deterministic ABACAD heuristic and keep the same calibration/post-processing. The most suspicious part is that `diff_track` is currently computed from an absolute A-vs-OFF residual (`diff = abs(...)`), which (like the issue you already fixed for `resid_line`) can reward OFF-only structure and hurt ranking. I make `diff_track` use the signed residual (A−OFF) and compute a signed, high-quantile “trackness” on that residual so OFF-dominant edges are penalized instead of helped, while keeping all weights, calibration, paths, and output format unchanged. This is a minimal, semantics-preserving adjustment (still deterministic, label-free, same overall pipeline) that should move AUC upward toward your target.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

np.random.seed(0)



## === cell 1
BASE_INPUT = "/kaggle/input"
TEST_DIR = os.path.join(BASE_INPUT, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/data/sample_submission.csv"

if not os.path.exists(TEST_DIR):
    alt = "/kaggle/data/test"
    if os.path.exists(alt):
        TEST_DIR = alt

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found at {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_DIR), f"test directory not found at {TEST_DIR}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub.head()



## === cell 2
test_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*", "*.npy")))
assert len(test_paths) > 0, f"No .npy files found under {TEST_DIR}"

id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in test_paths}

missing = [i for i in sample_sub["id"].values if i not in id_to_path]
if missing:
    print(
        f"Warning: {len(missing)} ids missing .npy files; they will get 0.5 predictions. Example: {missing[:5]}"
    )

len(test_paths), test_paths[0]




## === cell 3
def score_snippet(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[0] != 6:
        return 0.0

    x = arr.astype(np.float32)

    med = np.median(x, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(x - med), axis=(1, 2), keepdims=True) + 1e-6
    xn = (x - med) / mad  # (6, H, W)

    a_idx = np.array([0, 2, 4])
    o_idx = np.array([1, 3, 5])

    e = np.mean(np.abs(xn), axis=(1, 2))  # (6,)
    a_med = float(np.median(e[a_idx]))
    off_med = float(np.median(e[o_idx]))
    off_mad = float(np.median(np.abs(e[o_idx] - off_med)) + 1e-6)
    z = (a_med - off_med) / off_mad

    a_std = float(np.std(e[a_idx]) + 1e-6)
    consistency = -a_std

    flat = xn.reshape(6, -1)
    norms = np.linalg.norm(flat, axis=1) + 1e-6
    flatn = flat / norms[:, None]

    def cos(i, j):
        return float(np.dot(flatn[i], flatn[j]))

    a_sim = (cos(0, 2) + cos(0, 4) + cos(2, 4)) / 3.0
    ao_sim = (
        cos(0, 1)
        + cos(0, 3)
        + cos(0, 5)
        + cos(2, 1)
        + cos(2, 3)
        + cos(2, 5)
        + cos(4, 1)
        + cos(4, 3)
        + cos(4, 5)
    ) / 9.0
    sim_contrast = a_sim - ao_sim

    absxn = np.abs(xn).reshape(6, -1)
    q = np.quantile(absxn, 0.995, axis=1)  # (6,)

    q_a_mean = float(np.mean(q[a_idx]))
    q_a_min = float(np.min(q[a_idx]))
    q_o = float(np.mean(q[o_idx]))
    q_os = float(np.std(q[o_idx]) + 1e-6)

    peak_contrast = (q_a_mean - q_o) / q_os
    persist_contrast = (q_a_min - q_o) / q_os

    absxn2 = np.abs(xn)  # (6, H, W)
    row_max = np.max(absxn2, axis=2)  # (6, H)
    row_q = np.quantile(row_max, 0.98, axis=1)  # (6,)
    row_a = float(np.mean(row_q[a_idx]))
    row_o = float(np.mean(row_q[o_idx]))
    row_os = float(np.std(row_q[o_idx]) + 1e-6)
    rowmax_contrast = (row_a - row_o) / row_os

    absx = np.abs(xn)
    smooth = (absx[:, :, :-2] + 2.0 * absx[:, :, 1:-1] + absx[:, :, 2:]) / 4.0
    row_max_s = np.max(smooth, axis=2)  # (6, H)
    row_p = np.quantile(row_max_s, 0.995, axis=1)  # (6,)
    rp_a = float(np.mean(row_p[a_idx]))
    rp_o = float(np.mean(row_p[o_idx]))
    rp_os = float(np.std(row_p[o_idx]) + 1e-6)
    rowpeak_contrast = (rp_a - rp_o) / rp_os

    v = smooth  # (6, H, W-2)
    gx = (v[:, :-2, 2:] + 2.0 * v[:, 1:-1, 2:] + v[:, 2:, 2:]) - (
        v[:, :-2, :-2] + 2.0 * v[:, 1:-1, :-2] + v[:, 2:, :-2]
    )
    gy = (v[:, 2:, :-2] + 2.0 * v[:, 2:, 1:-1] + v[:, 2:, 2:]) - (
        v[:, :-2, :-2] + 2.0 * v[:, :-2, 1:-1] + v[:, :-2, 2:]
    )
    grad = np.sqrt(gx * gx + gy * gy + 1e-6)  # (6, H-2, W-4)

    grad_q = np.quantile(grad.reshape(6, -1), 0.995, axis=1)  # (6,)
    g_a = float(np.mean(grad_q[a_idx]))
    g_o = float(np.mean(grad_q[o_idx]))
    g_os = float(np.std(grad_q[o_idx]) + 1e-6)
    track_contrast = (g_a - g_o) / g_os

    resid = (np.median(xn[a_idx], axis=0) - np.median(xn[o_idx], axis=0)).astype(
        np.float32
    )  # (H, W), signed
    resid_s = (resid[:, :-2] + 2.0 * resid[:, 1:-1] + resid[:, 2:]) / 4.0  # (H, W-2)

    v2 = resid_s[None, :, :]  # (1, H, W-2) to reuse same Sobel-like code
    gx2 = (v2[:, :-2, 2:] + 2.0 * v2[:, 1:-1, 2:] + v2[:, 2:, 2:]) - (
        v2[:, :-2, :-2] + 2.0 * v2[:, 1:-1, :-2] + v2[:, 2:, :-2]
    )
    gy2 = (v2[:, 2:, :-2] + 2.0 * v2[:, 2:, 1:-1] + v2[:, 2:, 2:]) - (
        v2[:, :-2, :-2] + 2.0 * v2[:, :-2, 1:-1] + v2[:, :-2, 2:]
    )

    diff_track = float(
        max(
            np.quantile(np.maximum(gx2, 0.0).reshape(-1), 0.995),
            np.quantile(np.maximum(gy2, 0.0).reshape(-1), 0.995),
        )
    )

    H, W2 = resid_s.shape
    slopes = (-3, -2, -1, 0, 1, 2, 3)
    diag_vals = []
    for s in slopes:
        shifted = np.zeros_like(resid_s, dtype=np.float32)
        if s >= 0:
            shifted[:, s:] = resid_s[:, : W2 - s]
        else:
            ss = -s
            shifted[:, : W2 - ss] = resid_s[:, ss:]
        col_idx = np.arange(H, dtype=np.int64) % W2
        diag1 = shifted[np.arange(H, dtype=np.int64), col_idx]
        diag2 = shifted[np.arange(H, dtype=np.int64), (W2 - 1 - col_idx)]
        diag_vals.append(np.quantile(diag1, 0.98))
        diag_vals.append(np.quantile(diag2, 0.98))
    resid_line = float(np.max(diag_vals))

    return (
        (1.00 * z)
        + (0.20 * consistency)
        + (0.65 * sim_contrast)
        + (0.45 * peak_contrast)
        + (0.18 * persist_contrast)
        + (0.35 * rowmax_contrast)
        + (0.22 * rowpeak_contrast)
        + (0.28 * track_contrast)
        + (0.20 * diff_track)
        + (0.18 * resid_line)
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def rank_to_uniform(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(scores.size, dtype=np.int64)
    return (ranks.astype(np.float32) + 0.5) / float(scores.size)




## === cell 4
raw_scores = np.empty(len(sample_sub), dtype=np.float32)

for idx, _id in enumerate(sample_sub["id"].values):
    p = id_to_path.get(_id, None)
    if p is None:
        raw_scores[idx] = 0.0
        continue
    arr = np.load(p)  # float16 saved, loads as float16
    raw_scores[idx] = score_snippet(arr)

rs_med = np.median(raw_scores)
rs_mad = np.median(np.abs(raw_scores - rs_med)) + 1e-6
raw_scores_std = (raw_scores - rs_med) / rs_mad

sig_probs = sigmoid(raw_scores_std / 2.0).astype(np.float32)

rank_probs = rank_to_uniform(raw_scores_std)

probs = (0.85 * rank_probs + 0.15 * sig_probs).astype(np.float32)
probs = np.clip(probs, 0.0, 1.0)

submission = pd.DataFrame({"id": sample_sub["id"].values, "target": probs})
submission.head()



## === cell 5
submission.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv"), "submission.csv was not created"
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission.columns) == [
    "id",
    "target",
], "Submission columns must be ['id','target']"

print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe(include="all"))
