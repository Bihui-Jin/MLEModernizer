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

0.7569148672204666

# 6. Current score

0.48824

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50618) has done: 'The current notebook fails because it tries to ensemble submission files from other Kaggle datasets that are not available in this environment, so `pd.read_csv("../input/...")` raises `FileNotFoundError` and downstream variables are undefined. To make it run end-to-end and still keep the “ensemble of predictions” core idea, I replace those external reads with a simple, deterministic model that generates probabilities from the provided `.npy` snippet files in `/kaggle/data/test/`. I also ensure the produced `submission.csv` matches the required `id,target` format and row order from `sample_submission.csv`. This yield a valid submission; since no current score exists, the aim is a reasonable baseline without changing the overall “compute predictions then write submission” semantics.'
- What this solution (achieved 0.50896) has done: 'Your current score (0.50618) is far below the target (0.7569), so we should improve discriminative power while keeping the same “deterministic feature → probability → submission.csv” core logic. The biggest likely issue is the hard-coded sigmoid centering/scaling `(score - 0.8) * 2.0`, which can badly miscalibrate ranks and compress outputs; AUC only cares about ranking, so we instead use a monotonic mapping based on the test-score distribution itself (rank → (0,1)), which typically improves AUC for simple heuristics without changing the underlying feature logic. We also make the feature slightly more rank-informative by emphasizing the top tail of `|A-B|` (still the same A-vs-B contrast, just a different aggregation). Finally, we remove the redundant “fake ensemble” copies (they currently do nothing) and write the properly ordered submission from `sample_submission.csv`.'
- What this solution (achieved 0.50421) has done: 'Your current AUC (0.50896) is far below the target (0.7569), so we should improve ranking while keeping the same deterministic “A vs B/C/D contrast → scalar score → rank-to-probability submission” core logic. The most likely weakness is that `abs(A-B)` throws away drift-direction information; using a signed contrast (A−B) and aggregating its positive tail should better capture “excess energy in A only” without changing the overall approach. I keep the same normalization and rank-based probability mapping (AUC-safe), but update the score to combine positive and negative tails plus a small amount of the original absolute-tail to stay robust. This is a minimal change confined to `predict_score_one` and should move the score upward toward the target without altering I/O or submission format.'
- What this solution (achieved 0.50368) has done: 'I fix the runtime error in `_max_time_of_freq_boxsum` by correcting the sliding-window cumulative-sum math so the left/right slices have matching shapes. This unblocks score computation and ensures `submission` is created so the CSV write in the next cell succeeds. I keep the feature logic and rank-to-probability mapping unchanged (AUC-safe) to preserve evaluation semantics while making the pipeline run end-to-end. I also add a small guard for tiny `kfreq`/width edge cases to avoid future shape issues.'
- What this solution (achieved 0.50288) has done: 'Your current AUC (0.50368) is far below the target (0.7569), so we should improve ranking while keeping your core “deterministic A-vs-(B/C/D) contrast → scalar score → rank-to-probability → submission.csv” pipeline unchanged. The most impactful minimal change is to make the contrast feature more invariant to global normalization and better aligned to the known “needle only in A panels” structure by using a robust A/B ratio on strictly-positive “power” (ReLU) rather than only a signed z-scored difference. To stay within your existing semantics, we keep all the same I/O, keep rank-mapping (AUC-safe), and only adjust `predict_score_one` to combine your existing score with one additional ratio-based statistic and a simple “A-consistency vs BCD-consistency” term. This should raise separability without changing model class, training, or submission format.'
- What this solution (achieved 0.5031) has done: 'We need to move AUC up from ~0.503 toward 0.7569, so we should improve ranking while keeping your deterministic “feature → score → rank-to-probability” pipeline unchanged. The smallest high-impact fix is to correct the score directionality by using the ABACAD structure: compare each A panel against the average of B/C/D (not the mean of all three B/C/D panels), then aggregate per-A evidence so a needle present in only some A panels still scores high. I keep your existing statistics (tail percentiles, line enhancer, consistency) but compute them on per-A contrasts and combine them with a max/mean blend to better match the “needle appears in A panels only” physics. I also keep the rank mapping and submission formatting identical to preserve evaluation semantics and stability.'
- What this solution (achieved 0.49948) has done: 'Your current AUC is far below the target, so we should improve ranking while keeping your same deterministic “feature → scalar score → rank-to-probability → submission.csv” pipeline. The smallest likely high-impact change is to make the contrast explicitly “A-only” by subtracting the per-pixel maximum of (B,C,D) instead of their mean, which better suppresses RFI that appears in any off-target panel while keeping your per-A scoring and aggregation intact. To avoid destabilizing the feature scale, we also switch the tail-ratio reference to the same BCD-maximum baseline (still a monotonic term for ranking). Everything else (normalization, feature recipe structure, rank mapping, and submission formatting) stays the same.'
- What this solution (achieved 0.50015) has done: 'We need to increase AUC from 0.49948 toward 0.7569 (higher is better), so we should improve ranking while keeping your deterministic “A-vs-(B/C/D) contrast → scalar score → rank-to-probability → submission.csv” pipeline intact. The smallest likely improvement is to make the per-A evidence focus more on thinband/drifting “line” structure (common for needles) by adding one extra monotonic feature: a line-enhancement computed on the *mean A panel* contrasted against the B/C/D per-pixel max, reusing your existing `_max_time_of_freq_boxsum` helper. This does not change the core approach, doesn’t add training, and only adjusts `predict_score_one` by a small additive term that should improve separability. Everything else (I/O, normalization, rank mapping, submission format) remains unchanged.'
- What this solution (achieved 0.48566) has done: 'Your current AUC (0.50015) is far below the target (0.7569), so we should improve ranking while keeping your exact “deterministic feature → scalar score → rank-to-probability → submission.csv” pipeline unchanged. The smallest high-impact fix is to remove the per-file z-scoring, which likely destroys the very “excess only in A” amplitude relationships your contrast features depend on (especially since the snippets are already normalized by the dataset process). Instead we apply a strictly monotonic, robust per-file affine scaling based on percentiles (winsorized scaling), which preserves ordering-relevant amplitude differences while preventing outliers from dominating. Everything else (your contrast definitions, feature recipe, rank mapping, and submission writing) stays the same, so evaluation semantics are preserved while improving separability.'
- What this solution (achieved 0.48246) has done: 'To move AUC up from 0.48566 toward 0.7569 (higher is better) without changing your overall “deterministic feature → score → rank-to-probability” pipeline, I make the off-target reference more faithful to the ABACAD cadence by using a per-A nearest-neighbor off-target panel (A0 vs B, A1 vs C, A2 vs D) rather than a shared B/C/D max reference for all A’s. This is a minimal change confined to `predict_score_one` and keeps the same feature recipe and rank mapping, but typically improves ranking because it reduces mismatch/noise when RFI differs across B/C/D. I also blend the old BCD-max reference with the per-A references (small weight) for robustness, aiming for a stable improvement rather than an aggressive rewrite. The submission writing, ordering via `sample_submission.csv`, and all helper functions remain unchanged.'
- What this solution (achieved 0.48595) has done: 'Your current AUC (0.48246) is far below the target (0.75691), so we should improve ranking while keeping your exact “deterministic contrast features → scalar score → rank-to-probability → submission.csv” pipeline intact. The minimal high-impact change is to make the A-vs-off-target contrast invariant to per-cadence gain differences by normalizing each panel with a robust per-panel scale (MAD), because this preserves relative structure while reducing noise from differing panel intensity distributions. To keep the core recipe unchanged, we only adjust the inputs to `_panel_contrast_score` (whitened panels) and keep all the same aggregations, line enhancer, and rank mapping. We also keep your existing winsorized scaling and your cadence-aligned off-target references, so I/O, runtime, and submission format remain identical.'
- What this solution (achieved 0.48824) has done: 'We need to increase AUC from 0.48595 toward 0.75691, so we should improve ranking while keeping your deterministic “feature → scalar score → rank-to-probability → submission.csv” pipeline unchanged. The most likely ranking killer is whitening each panel independently (median/MAD), which can erase the very “A has signal, B/C/D don’t” amplitude differences that the cadence structure provides; instead we whiten all 6 panels with a single robust cadence-level scale so relative panel amplitudes are preserved. This is a minimal change: replace `_robust_whiten_panel` with `_robust_whiten_cadence` and apply it once per file, leaving the contrast features, aggregations, and rank mapping intact. Everything else (paths, file loading, submission order/format, runtime) remains the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from glob import glob

import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data"
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found at {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_DIR), f"test directory not found at {TEST_DIR}"



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 2
sample = pd.read_csv(SAMPLE_SUB_PATH)

test_paths = glob(os.path.join(TEST_DIR, "*", "*.npy"))
id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in test_paths}

missing = [i for i in sample["id"].values if i not in id_to_path]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test .npy files for ids (showing up to 5): {missing[:5]}"
    )


def _max_time_of_freq_boxsum(mat2d: np.ndarray, kfreq: int = 5) -> float:
    """
    Correct sliding-window sum along frequency so left/right slices align in shape.
    For each time row, compute max sum over width-kfreq windows, then take max over time.
    """
    if mat2d.ndim != 2:
        raise ValueError(f"Expected 2D array (time,freq); got shape {mat2d.shape}")

    t, w = mat2d.shape
    k = int(kfreq)
    if k <= 1:
        return float(np.max(mat2d))
    if k > w:
        return float(np.max(np.sum(mat2d, axis=1, dtype=np.float64)))

    cs = np.cumsum(mat2d, axis=1, dtype=np.float64)  # (t, w)
    first = cs[:, k - 1 : k]  # (t,1)
    rest = cs[:, k:] - cs[:, :-k]  # (t, w-k)
    win = np.concatenate([first, rest], axis=1)  # (t, w-k+1)
    return float(np.max(win))


def _panel_contrast_score(Ai: np.ndarray, BCD_ref: np.ndarray) -> float:
    """
    Helper that scores one A panel vs an off-target reference panel (same shape).
    Kept intentionally close to your existing feature recipe; only the contrast input differs.
    """
    d = Ai - BCD_ref

    pos = np.maximum(d, 0.0)
    neg = np.maximum(-d, 0.0)
    absd = np.abs(d)

    ppos90 = float(np.percentile(pos, 90))
    ppos99 = float(np.percentile(pos, 99))
    pneg90 = float(np.percentile(neg, 90))
    pneg99 = float(np.percentile(neg, 99))
    pabs99 = float(np.percentile(absd, 99))
    mx_abs = float(np.max(absd))

    row_max = float(np.max(pos.max(axis=1)))
    col_max = float(np.max(pos.max(axis=0)))
    pos_mass = float(np.mean(pos))
    line_enh = _max_time_of_freq_boxsum(pos, kfreq=5)

    base_score = (
        0.44 * ppos99
        + 0.20 * ppos90
        - 0.28 * pneg99
        - 0.08 * pneg90
        + 0.08 * pabs99
        + 0.04 * mx_abs
        + 0.10 * row_max
        + 0.06 * col_max
        + 0.04 * pos_mass
        + 0.10 * line_enh
    )
    return float(base_score)


def _robust_whiten_cadence(x6: np.ndarray) -> np.ndarray:
    """
    Change (score-improving, minimal): whiten using a single robust (median/MAD) scale
    computed across the whole cadence (all 6 panels), not per-panel.

    Why this should move AUC upward:
    - Per-panel whitening can remove relative amplitude differences between A and B/C/D,
      which are informative for "signal appears only in A" structure.
    - Cadence-level whitening still stabilizes scale across examples while preserving
      inter-panel amplitude relationships needed by A-vs-off-target contrasts.
    """
    med = float(np.median(x6))
    mad = float(np.median(np.abs(x6 - med))) + 1e-6
    return (x6 - med) / mad


def predict_score_one(path: str) -> float:
    """
    Deterministic feature -> scalar score.
    Core logic preserved: A panels (0,2,4) vs off-target panels (1,3,5) contrast,
    produce a scalar, then convert to probabilities by rank.

    Minimal change vs previous version:
    - Replace per-panel whitening with cadence-level whitening to preserve A vs B/C/D
      amplitude relationships (often important for ranking/AUC).
    """
    x = np.load(path)  # (6,273,256), float16
    x = x.astype(np.float32)

    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    scale = (hi - lo) + 1e-6
    x = (x - lo) / scale

    xw = _robust_whiten_cadence(x)

    A0, A1, A2 = xw[0], xw[2], xw[4]
    B, C, D = xw[1], xw[3], xw[5]

    BCD_max = np.maximum(B, np.maximum(C, D))

    blend = 0.20  # small, conservative
    ref0 = (1.0 - blend) * B + blend * BCD_max
    ref1 = (1.0 - blend) * C + blend * BCD_max
    ref2 = (1.0 - blend) * D + blend * BCD_max

    s0 = _panel_contrast_score(A0, ref0)
    s1 = _panel_contrast_score(A1, ref1)
    s2 = _panel_contrast_score(A2, ref2)

    base_score = 0.60 * max(s0, s1, s2) + 0.40 * ((s0 + s1 + s2) / 3.0)

    Ap = np.maximum(np.stack([A0, A1, A2], axis=0), 0.0)
    BCDp = np.maximum(BCD_max, 0.0)

    A_tail = float(np.percentile(Ap, 99.5))
    B_tail = float(np.percentile(BCDp, 99.5))
    tail_log_ratio = np.log(A_tail + 1e-6) - np.log(B_tail + 1e-6)

    A_mean = (A0 + A1 + A2) / 3.0
    BCD_mean = (B + C + D) / 3.0
    BCD_dev = float(np.mean(np.abs(np.stack([B, C, D], axis=0) - BCD_mean[None, :, :])))
    A_dev = float(np.mean(np.abs(np.stack([A0, A1, A2], axis=0) - A_mean[None, :, :])))
    consistency = BCD_dev - A_dev  # higher => A is more self-consistent than B/C/D

    d0 = A0 - ref0
    d1 = A1 - ref1
    d2 = A2 - ref2
    d_mean = (d0 + d1 + d2) / 3.0
    pos_mean = np.maximum(d_mean, 0.0)
    line_mean = _max_time_of_freq_boxsum(pos_mean, kfreq=7)

    score = (
        base_score
        + 0.35 * tail_log_ratio
        + 0.15 * consistency
        + 0.12 * float(line_mean)
    )
    return float(score)


scores = np.empty(len(sample), dtype=np.float64)
for i, _id in enumerate(sample["id"].values):
    scores[i] = predict_score_one(id_to_path[_id])

order = np.argsort(scores, kind="mergesort")
ranks = np.empty_like(order, dtype=np.float64)
ranks[order] = np.arange(len(scores), dtype=np.float64)

preds = (ranks + 0.5) / float(len(scores))
preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

submission = sample.copy()
submission["target"] = preds.astype(np.float64)



## === cell 3
submission.to_csv("submission.csv", index=False)

check = pd.read_csv("submission.csv")
assert list(check.columns) == ["id", "target"]
assert len(check) == len(sample)
assert check["target"].between(0.0, 1.0).all()
print("Wrote submission.csv with shape:", check.shape)
print(check.head())
