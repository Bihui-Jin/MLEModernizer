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

0.50158

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes with `FileNotFoundError` because it tries to read several Kaggle “../input/<notebook>/submission.csv” files that are not present in this environment. The only guaranteed submission-like file available from the provided paths is `/kaggle/data/sample_submission.csv` (also mirrored under `/kaggle/input/`). To keep downstream code working (cell 2 expects `data1` to be a DataFrame with `id`/`target` columns), we should load the available sample submission and create placeholder `data2..data8` with identical structure.

Patch summary: In cell 1, replace the missing external reads with a robust local read of the existing `sample_submission.csv`, and copy it to `data1..data8` so later cells can run unchanged. This fixes the crash without changing later interfaces.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: `data1` remains a pandas DataFrame and `data1.head()` in cell 2 work; `data2..data8` also exist with the same columns, preserving any later ensemble code that expects them.

Assumptions: The environment contains either `../input/sample_submission.csv` (Kaggle-style) or `/kaggle/data/sample_submission.csv` (as listed), and the submission schema is `id, target`.'
- What this solution (achieved 0.49857) has done: 'Your current pipeline writes essentially-constant predictions from `sample_submission.csv`, which yields an AUC near 0.5. To move the score toward the target with minimal changes and without introducing a new model, I replace the placeholder `target` with a simple, legitimate signal-derived heuristic computed directly from each test `.npy` snippet (using the ABACAD structure: compare A frames vs B/C/D frames). This keeps the “single-pass, no-training” approach intact, but makes predictions data-dependent and therefore meaningfully better than random. I also ensure the submission aligns exactly to the sample submission `id` order and always writes a valid `submission.csv`.'
- What this solution (achieved 0.50369) has done: 'We need to raise AUC from ~0.499 toward 0.75697, so we keep your “no-training, single-pass heuristic” core approach but make the score more discriminative by using a slightly richer, still-legitimate ABACAD-derived signal statistic. Specifically, we compute both a “peakiness” feature (high quantile of A−OFF after robust scaling) and a “structured energy” feature (mean positive tail), then calibrate with a fixed sigmoid scale so predictions aren’t overly saturated near 0/1. We also remove the dilution from constant-weight ensembling with unchanged sample-submission placeholders (data1/data4/data6), because that drags the heuristic back toward 0.5 and hurts AUC. Output format/path stays identical and still writes `submission.csv`.'
- What this solution (achieved 0.50464) has done: 'To move AUC up from ~0.50 toward your 0.75697 target while keeping the same “no-training ABACAD heuristic” core logic, I (1) make the per-snippet score more discriminative by combining a few simple, robust A-vs-OFF statistics (peakiness, positive-mass, and temporal persistence) computed from the same `diff = mean(A)-mean(OFF)` map. I (2) rank-normalize the resulting raw scores to a smooth 0–1 probability via an empirical CDF, which preserves ordering (thus AUC) while avoiding overly-saturated sigmoid outputs. I (3) keep the same input paths and submission schema, and still write `submission.csv` with `id,target` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.52069) has done: 'Your current score is far below the target (0.50464 vs 0.75697), so we should increase AUC while preserving the same “no-training ABACAD heuristic” approach. The smallest high-impact change is to make the raw score more aligned with needle morphology by using an SNR-like aggregation: compare A vs OFF, robust-normalize by OFF variability per frequency channel, then compute a “diagonal-line” matched-filter response via a small bank of drift (slope) sums. We keep ECDF rank-normalization (AUC-preserving) but feed it a more discriminative raw score. All paths and the submission schema remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.52296) has done: 'To move AUC upward toward 0.75697 while keeping your no-training ABACAD heuristic intact, I make the per-snippet raw score slightly more signal-morphology aligned without changing the overall approach. Concretely, I compute the A−OFF z-map using per-frequency robust centering/scaling derived from OFF, then add one extra lightweight “multi-column drift track” matched feature (uses a small neighborhood around the drift column rather than a single pixel) and a simple left-right max-pair feature to catch vertical/narrowband needles. I keep the ECDF rank-normalization (AUC-preserving) and keep all paths and submission formatting unchanged. This is a minimal extension of your existing scoring function and should improve separability beyond 0.52069 without introducing training or new dependencies.'
- What this solution (achieved 0.52615) has done: 'To move AUC upward toward your 0.75697 target while preserving the same no-training ABACAD heuristic, I keep your existing feature set but make one minimal, high-leverage adjustment: compute the OFF robust center/scale per-frequency from the same statistic being standardized (the OFF mean-map), rather than mixing a per-frequency 1D median with a 2D diff-map. This makes the z-map better calibrated and typically improves ranking quality without changing the overall approach. I also add a tiny second drift-bank computed on an A-only z-map (A normalized by OFF) to better capture needles that are strong in A but not necessarily well-expressed in A−OFF, then still apply the same ECDF rank normalization (AUC-preserving). All paths remain unchanged and the script still writes a valid `submission.csv` with `id,target` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.50323) has done: 'We need to raise AUC from 0.52615 toward 0.75697, so we keep your no-training ABACAD heuristic but make the raw ranking more aligned with typical needle morphology with minimal extra computation. The smallest high-impact tweak is to add a lightweight “multi-A consistency” feature (needles should be present in A0/A2/A4 similarly) and a “narrowbandness” feature (needles are typically thin in frequency), both computed from the same already-standardized maps so core semantics stay identical. We also slightly reduce reliance on the left-right difference term (which can be unstable) and keep the ECDF mapping unchanged (AUC-preserving). Output paths/format remain the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.50323) has done: 'You’re far below the target AUC (0.503 vs 0.757), so we should increase separability while keeping your same no-training ABACAD heuristic and ECDF (rank) calibration. The minimal, high-leverage fix is to correct the OFF robust normalization: right now you compute `off_mu/off_scale` from `off_mean_map` but apply it to `diff = Amean - off_mean_map`, which mixes centered and uncentered quantities and can collapse useful ranking signal. I compute a properly centered z-map as `(Amean - off_mean_map) / off_scale` where `off_scale` is derived from the OFF mean-map dispersion, and keep all your existing features/weights intact otherwise (only negligible floating-point differences). This should move AUC upward without changing architecture/training (still none) and still produces a valid `submission.csv`.'
- What this solution (achieved 0.50323) has done: 'Your current AUC (~0.503) is far below the target (~0.757), so we should increase separability while keeping your no-training ABACAD heuristic and ECDF calibration unchanged. The smallest high-impact correction is to fix the OFF robust normalization bug: you compute `off_mu/off_scale` from `off_mean_map` but then standardize `diff = Amean - off_mean_map` in a way that cancels the centering and can collapse useful ranking signal. I compute a consistent, properly centered/scaled z-map as `z = (Amean - off_mean_map) / off_scale`, where `off_scale` comes from robust dispersion of `off_mean_map` per frequency column, and keep all existing downstream features/weights intact. This keeps the same core approach (no training, same features) but should move AUC upward toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 0.49188) has done: 'We need to increase AUC from 0.50323 toward 0.75697 (higher-is-better), and you’re far outside the ±10% tolerance band, so we should improve separability while keeping the same no-training ABACAD heuristic and ECDF (rank) calibration. The minimal high-impact issue is that your robust normalization uses per-frequency MAD from `off_mean_map` but ignores that OFF variability differs by time-row as well, which can bury real needles; we can add a tiny row-wise robust scale term (still computed only from OFF) and use it to standardize `Amean - off_mean_map` without changing the overall approach. Then we keep all your existing features, but compute drift/peak/tail on this better-calibrated z-map, which should improve ranking without introducing training or new dependencies. Paths and submission formatting stay identical, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.50307) has done: 'To move AUC upward from 0.49188 toward 0.75697 without changing your core “no-training ABACAD heuristic + ECDF rank calibration”, I make two minimal scoring fixes that improve ranking signal while keeping the same overall semantics. First, I compute the OFF robust scale from the full OFF cube (not just the OFF mean-map), which gives a more reliable per-(time,freq) noise estimate and improves z-score calibration. Second, I slightly correct the drift matched-filter to actually model drift across time using a normalized time index (your current implementation inadvertently anchors drift to the row number itself, which over-penalizes larger row indices); this preserves the same drift-bank idea but makes it function as intended. All paths stay the same and the script still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5033) has done: 'We need to increase AUC from 0.50307 toward 0.75697 (higher-is-better), so we keep your same no-training ABACAD heuristic and ECDF rank calibration, but fix two small issues that can flatten ranking signal. First, we make the OFF robust scaling consistent with the statistic being standardized by computing scale directly from OFF (not mixing OFF-mean-map dispersion and cube dispersion inconsistently), then standardize both `Amean` and `diff` using that same per-(t,f) scale. Second, we make the drift features more sensitive to real thin tracks by adding a tiny “top-k along track” aggregation (still the same drift-bank idea, no training) which is often more stable than a single high quantile on sparse samples. These are minimal, local changes inside `score_snippet_raw`/drift scoring and preserve all I/O paths and the required `submission.csv` format.'
- What this solution (achieved 0.50158) has done: 'We need to raise AUC from 0.5033 toward 0.75697, so we should increase separability while keeping your same no-training ABACAD heuristic and ECDF rank calibration. The smallest high-leverage change is to fix score directionality: your drift features currently only score tracks passing through the image center, but real needles can occur at any frequency, so this collapses ranking and keeps AUC near random. I generalize the drift matched-filter to search over multiple starting frequency anchors (a small grid across columns) while keeping the same slope bank, sampling step, and “top‑k + quantile” aggregation—this preserves the core logic and keeps runtime bounded. Everything else (paths, reading sample_submission, ECDF mapping, and writing `submission.csv`) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
import os
from pathlib import Path

_candidates = [
    "../input/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
]

_sample_path = next((p for p in _candidates if os.path.exists(p)), None)
if _sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected paths: "
        + ", ".join(_candidates)
    )

_base = pd.read_csv(_sample_path)

data1 = _base.copy()
data2 = _base.copy()
data3 = _base.copy()
data4 = _base.copy()
data5 = _base.copy()
data6 = _base.copy()
data7 = _base.copy()
data8 = _base.copy()



## === cell 2
data1.head()



## === cell 3
data2.head()




## === cell 4
def _find_test_root():
    roots = [
        Path("../input/seti-breakthrough-listen/test"),
        Path("/kaggle/input/seti-breakthrough-listen/test"),
        Path("/kaggle/data/seti-breakthrough-listen/test"),
        Path("../input/test"),
        Path("/kaggle/input/test"),
        Path("/kaggle/data/test"),
    ]
    for r in roots:
        if r.exists():
            return r
    raise FileNotFoundError("Could not locate test directory in expected paths.")


TEST_ROOT = _find_test_root()


def _load_test_npy_by_id(sample_id: str) -> np.ndarray:
    p = TEST_ROOT / sample_id[0] / f"{sample_id}.npy"
    if not p.exists():
        p2 = TEST_ROOT / f"{sample_id}.npy"
        if not p2.exists():
            raise FileNotFoundError(
                f"Missing test file for id={sample_id}: tried {p} and {p2}"
            )
        p = p2
    return np.load(p)


def _ecdf_to_unit_interval(raw_scores: np.ndarray) -> np.ndarray:
    """
    AUC depends only on ranking. Mapping raw scores to empirical CDF yields well-spread
    probabilities while preserving ordering.
    """
    raw_scores = raw_scores.astype(np.float64, copy=False)
    order = np.argsort(raw_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(raw_scores.size, dtype=np.int64)
    p = (ranks + 0.5) / raw_scores.size
    return p.astype(np.float32)


def _drift_track_vals(
    z: np.ndarray, s: int, step: int = 13, halfwidth: int = 0, c0: int | None = None
):
    """
    Helper: sample values along a drift track from a chosen starting column c0
    (optionally taking max over a small band).
    """
    H, W = z.shape
    rows = np.arange(0, H, step, dtype=np.int32)
    if rows.size < 4:
        rows = np.arange(H, dtype=np.int32)

    if c0 is None:
        c0 = W // 2  # default: center (kept for backward-compat)
    base_r = int(rows[0])

    vals = []
    for r in rows:
        dt = int(r) - base_r
        c = c0 + s * (dt / float(step))
        c = int(round(c))
        if c < 0 or c >= W:
            continue
        if halfwidth <= 0:
            vals.append(float(z[int(r), c]))
        else:
            lo = max(0, c - halfwidth)
            hi = min(W, c + halfwidth + 1)
            vals.append(float(np.max(z[int(r), lo:hi])))
    return np.asarray(vals, dtype=np.float32), rows.size


def _drift_max_score(
    z: np.ndarray, slopes=range(-6, 7), step=13, q=0.995, anchors: int = 9
) -> float:
    """
    Base drift matched feature (kept): needles often look like drifting narrowband lines.

    Change (score-improving, minimal): search over multiple starting frequency anchors
    (columns) instead of only the center column. This fixes the main ranking failure mode
    where true needles are off-center, while keeping the same drift-bank core logic.
    """
    H, W = z.shape
    if anchors <= 1:
        c0_list = [W // 2]
    else:
        c0_list = np.linspace(0.1 * (W - 1), 0.9 * (W - 1), anchors)
        c0_list = [int(round(c)) for c in c0_list]

    best = -1e9
    for c0 in c0_list:
        for s in slopes:
            vals, nrows = _drift_track_vals(z, s=s, step=step, halfwidth=0, c0=c0)
            if vals.size < max(4, nrows // 3):
                continue

            vq = float(np.quantile(vals, q))
            k = max(3, int(np.ceil(0.25 * vals.size)))
            vtop = float(np.mean(np.partition(vals, -k)[-k:]))
            v = 0.55 * vq + 0.45 * vtop

            if v > best:
                best = v

    if best == -1e9:
        best = float(np.max(z))
    return float(best)


def _drift_band_score(
    z: np.ndarray,
    slopes=range(-7, 8),
    step=13,
    halfwidth=2,
    q=0.99,
    anchors: int = 9,
) -> float:
    """
    Drift feature over a small horizontal band around the drift column.

    Change (score-improving, minimal): same multi-anchor search as _drift_max_score
    to avoid assuming the signal is centered in frequency.
    """
    H, W = z.shape
    if anchors <= 1:
        c0_list = [W // 2]
    else:
        c0_list = np.linspace(0.1 * (W - 1), 0.9 * (W - 1), anchors)
        c0_list = [int(round(c)) for c in c0_list]

    best = -1e9
    for c0 in c0_list:
        for s in slopes:
            vals, nrows = _drift_track_vals(
                z, s=s, step=step, halfwidth=halfwidth, c0=c0
            )
            if vals.size < max(4, nrows // 3):
                continue

            vq = float(np.quantile(vals, q))
            k = max(3, int(np.ceil(0.25 * vals.size)))
            vtop = float(np.mean(np.partition(vals, -k)[-k:]))
            v = 0.55 * vq + 0.45 * vtop

            if v > best:
                best = v

    if best == -1e9:
        best = float(np.max(z))
    return float(best)


def score_snippet_raw(x: np.ndarray) -> float:
    """
    Keep the same no-training ABACAD A-vs-OFF approach and the same downstream feature family.

    Change (score-improving, minimal): drift features now scan multiple starting columns
    (anchors) so we don't assume the needle lies at the center frequency.
    """
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # (3, 273, 256)
    OFF = x[[1, 3, 5]]  # (3, 273, 256)

    off_mean_map = OFF.mean(axis=0)  # (273, 256)
    Amean = A.mean(axis=0)  # (273, 256)

    off_mu_col = np.median(off_mean_map, axis=0)  # (256,)

    off_abs_dev = np.abs(OFF - off_mean_map[None, :, :])  # (3,273,256)
    off_mad_map = np.median(off_abs_dev, axis=0) + 1e-6  # (273,256)
    off_scale_map = 1.4826 * off_mad_map + 1e-6  # (273,256)

    off_mad_col = np.median(np.abs(off_mean_map - off_mu_col[None, :]), axis=0) + 1e-6
    off_scale_col = 1.4826 * off_mad_col + 1e-6  # (256,)

    z = (Amean - off_mean_map) / off_scale_map
    zA = (Amean - off_mu_col[None, :]) / (
        np.median(off_scale_map, axis=0, keepdims=True) + 1e-6
    )

    z = np.clip(z, -8.0, 12.0)
    zA = np.clip(zA, -8.0, 12.0)

    f_peak = float(np.quantile(z, 0.9995))

    pos = z[z > 1.5]
    f_tailmean = float(pos.mean()) if pos.size else 0.0

    f_drift = _drift_max_score(z, slopes=range(-7, 8), step=13, q=0.99, anchors=9)

    row_max = z.max(axis=1)
    f_persist = float((row_max > 3.0).mean())

    f_drift_band = _drift_band_score(
        z, slopes=range(-7, 8), step=13, halfwidth=2, q=0.99, anchors=9
    )

    f_driftA_band = _drift_band_score(
        zA, slopes=range(-7, 8), step=13, halfwidth=2, q=0.99, anchors=9
    )

    A0 = (A[0] - off_mu_col[None, :]) / off_scale_col[None, :]
    A2 = (A[1] - off_mu_col[None, :]) / off_scale_col[None, :]
    A4 = (A[2] - off_mu_col[None, :]) / off_scale_col[None, :]
    A0 = np.clip(A0, -8.0, 12.0)
    A2 = np.clip(A2, -8.0, 12.0)
    A4 = np.clip(A4, -8.0, 12.0)
    qa0 = float(np.quantile(A0, 0.999))
    qa2 = float(np.quantile(A2, 0.999))
    qa4 = float(np.quantile(A4, 0.999))
    f_Acons = (qa0 + qa2 + qa4) / 3.0 - (
        abs(qa0 - qa2) + abs(qa2 - qa4) + abs(qa0 - qa4)
    ) / 6.0

    row_mean = np.mean(np.maximum(z, 0.0), axis=1) + 1e-6
    row_maxp = np.max(np.maximum(z, 0.0), axis=1)
    f_narrow = float(np.quantile(row_maxp / row_mean, 0.98))

    col_max = np.max(z, axis=0)  # (256,)
    k = 3
    padded = np.pad(col_max, (k, k), mode="edge")
    band_means = np.empty_like(col_max, dtype=np.float32)
    for i in range(col_max.size):
        band_means[i] = float(np.mean(padded[i : i + 2 * k + 1]))
    half = band_means.size // 2
    f_lr = float(np.max(band_means[:half]) - np.max(band_means[half:]))

    raw = (
        1.05 * f_peak
        + 0.65 * f_tailmean
        + 1.20 * f_drift
        + 0.85 * f_persist
        + 1.05 * f_drift_band
        + 0.45 * f_driftA_band
        + 0.55 * f_Acons
        + 0.35 * f_narrow
        + 0.10 * f_lr
    )
    return float(raw)


ids = data1["id"].astype(str).tolist()
raws = np.empty(len(ids), dtype=np.float32)

for i, sid in enumerate(ids):
    arr = _load_test_npy_by_id(sid)
    raws[i] = score_snippet_raw(arr)

preds = _ecdf_to_unit_interval(raws).clip(1e-6, 1 - 1e-6)

data5 = data5.copy()
data5["target"] = preds



## === cell 5
data9 = data1.copy()
data9["target"] = data5["target"].astype(np.float32).clip(1e-6, 1 - 1e-6)



## === cell 6
data9.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data9.shape)
print(data9.head())
