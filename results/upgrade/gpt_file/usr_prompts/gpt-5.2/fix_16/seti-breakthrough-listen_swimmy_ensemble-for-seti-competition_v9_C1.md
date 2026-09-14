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

0.7571658004342819

# 6. Current score

0.51768

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51009) has done: 'Your notebook fails because it tries to read six external submissions from `../input/...` that do not exist in this environment, so nothing downstream is defined and no `submission.csv` is written. I replace those missing ensemble inputs with a minimal, local baseline that reads the provided `sample_submission.csv` and creates deterministic predictions from the available `test/` `.npy` files, ensuring the submission has the required `id,target` columns and 6000 rows. This keeps the “inference-only” nature of the original (no training) while producing a valid file end-to-end. I also make the path handling robust to either `/kaggle/input/...` or `/kaggle/data/...` layouts and ensure IDs align exactly to the sample submission order.'
- What this solution (achieved 0.51391) has done: 'Your current inference-only heuristic is too weak (AUC ~0.51), so the smallest safe way to move toward the 0.757 target is to keep the same “A vs (B,C,D)” contrast core idea but extract a slightly more discriminative signal-energy feature. I replace the single global mean(|A−B|) with a robust high-percentile contrast on time-collapsed spectra (which emphasizes narrowband/drifting lines) and combine it with the original global contrast, then apply the same sigmoid mapping. This preserves the no-training pipeline and keeps runtime within limits while typically providing a meaningful AUC lift for this competition. Output format/path stays identical and still writes `submission.csv` with `id,target` and 6000 rows.'
- What this solution (achieved 0.50047) has done: 'Your current inference-only contrast feature is still too weak for this competition (AUC ~0.51), so we should keep the same “A vs (B,C,D)” core idea but make the feature slightly more signal-aware in a way that typically lifts AUC without introducing training. I add one additional component that targets drifting/narrowband lines by taking a high-percentile of the frequency-derivative of the time-collapsed spectrum contrast, then combine it with your existing global and peak features. I also calibrate the sigmoid mapping using robust per-file normalization (median/IQR computed from the same file) so the probabilities spread more sensibly across test without requiring labels. Submission writing, paths, and the end-to-end inference-only flow remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.5155) has done: 'Your current heuristic is producing probabilities that are too tightly centered (per-file median/IQR normalization collapses between-snippet separability), so AUC stays near random. I keep the same inference-only “A vs (B,C,D)” contrast core, but switch to a dataset-level robust calibration: compute the same features for all test snippets, then center/scale once using global median/IQR so scores become comparable across snippets (this typically lifts AUC without any training). I also add one more very cheap, signal-relevant feature that still fits the same core idea: a “max-over-time then percentile-over-frequency” contrast (helps when the signal is present only part of the time). Output format and the submission writing remain unchanged and it still produce `submission.csv` with 6000 rows.'
- What this solution (achieved 0.51613) has done: 'We keep your inference-only “A vs (B,C,D)” contrast logic and the same four features, but make the score mapping more AUC-friendly by switching from a fixed linear weight-sum to a robust rank-based aggregation across features (still unsupervised, no training). This reduces sensitivity to feature scale/outliers and usually improves separability, which should move AUC upward toward your 0.757 target from ~0.515. We also apply a monotonic, globally-calibrated sigmoid on the aggregated score so the submission remains a probability while preserving ranking (AUC-relevant). Output paths and the submission format stay identical and it still writes `submission.csv` with 6000 rows.'
- What this solution (achieved 0.51733) has done: 'Your current heuristic is still essentially ranking near-random, so to move AUC toward the 0.757 target with minimal risk we should keep the same inference-only “A vs (B,C,D)” contrast core but compute a slightly more discriminative contrast feature. The smallest reliable lift for SETI often comes from emphasizing *max-over-A minus max-over-off* (captures intermittent narrowband lines) and *A-only consistency across the 3 A panels* (needles tend to repeat in A panels). We add two such features, keep your existing four, and keep the same rank-based aggregation + global sigmoid calibration (so semantics stay monotonic and AUC-friendly). This remains fully unsupervised, runs within the same constraints, and still writes a valid `submission.csv` with `id,target`.'
- What this solution (achieved 0.51501) has done: 'Your current score (0.51733) is far below the target (0.75717), so we should improve ranking quality with the smallest change that keeps your inference-only “A vs (B,C,D)” contrast logic intact. The biggest issue is that most of your features take `abs(A−off)`, which discards the directionality that is actually signal-relevant here (needles are stronger in A than off), hurting AUC. I keep the exact same pipeline (same files, no training, same rank-aggregation + sigmoid), but switch the relevant contrast features from `abs` to a rectified directional form `max(A−off, 0)` to better align with the task while remaining monotonic. I also apply the same directional treatment to the gradient/tmax-contrast features so all features consistently reward “A-only” energy rather than “any difference”.'
- What this solution (achieved 0.51924) has done: 'We keep your inference-only “A vs (B,C,D)” contrast pipeline and the same 6-feature structure, but fix a key consistency issue: you currently compare `a` (shape 3×273×256) against `b` (shape 3×273×256) panelwise, which mismatches the cadence semantics (each A should be compared to the *average off-target*). I change the contrasts to use `off_mean = mean(B,C,D)` so each A panel is contrasted against the same off reference, which typically improves ranking stability and should move AUC upward toward the 0.757 target. Then, I make the “A consistency” feature consistent with the objective by measuring how much *directional* A−off differs across the three A panels (needles tend to repeat in A), instead of raw A dispersion which can be dominated by background. All I/O paths, submission format, and the rank-aggregation + global sigmoid calibration remain unchanged.'
- What this solution (achieved 0.51896) has done: 'Your current AUC (0.51924) is far below the target (0.75717), so we should make a small but meaningful improvement to ranking while keeping your inference-only “A vs mean(off)” contrast core intact. The biggest low-risk lift here is to aggregate information across the three A panels and the three off panels more explicitly: compute the directional contrast on an A-mean vs off-mean basis (reduces noise) and add one additional “A-mean peak” feature that emphasizes a persistent needle across A observations. To keep changes minimal, I keep your 6 existing features exactly as-is, append 2 new features, and keep the same rank-aggregation + global robust sigmoid mapping (AUC-preserving monotonic transform). This should improve separability without introducing any training or changing I/O paths, and it still write a valid `submission.csv` with `id,target` and 6000 rows.'
- What this solution (achieved 0.51896) has done: 'Your current unsupervised rank-ensemble is likely underperforming because the feature set doesn’t explicitly capture the key cadence property: “needle present in A panels but absent in off panels,” and especially not the *A-only repetition across the three A observations*. I keep the same inference-only pipeline (no training), same A-vs-off contrast core, same rank aggregation + global robust sigmoid, and just append two very cheap cadence-consistency features: (1) how similar the three A spectra are to each other relative to off (needles repeat across A), and (2) how much more self-consistent A is than off (off panels vary less for RFI-like artifacts). This is a minimal extension (2 more features + small weight update) that should improve ranking separability and move AUC upward toward your 0.757 target while staying within the 600s runtime and producing the same valid `submission.csv`.'
- What this solution (achieved 0.51776) has done: 'Your current score (0.51896) is far below the target (0.75717), so we should improve separability with the smallest change that preserves your inference-only “A vs off” contrast core and rank-aggregation semantics. The weakest spot is that all features are based on time/frequency collapsed summaries; a very cheap but often more discriminative cue for SETI is the presence of bright *line-like* pixels (high quantiles) that appear in A but not in off. I add two minimal features that compute high-percentile directional pixel contrast for (1) the A-mean image vs off-mean image and (2) the max-over-A image vs max-over-off image (captures intermittent signals), then give them small weights while keeping your existing 10 features unchanged. Everything else (paths, ranking, sigmoid calibration, output CSV format) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.51776) has done: 'We keep your inference-only A-vs-off contrast approach, your 12 existing features, and the same rank-based aggregation + global sigmoid calibration (so evaluation semantics stay identical and AUC-friendly). The smallest likely lift toward the 0.757 target is to add one more very cheap, signal-aligned feature that captures the *cadence-specific property* “signal repeats across A panels”: a high-quantile of the **minimum** directional A−off across the three A panels (persistent-only-in-A pixels survive the min, while transient noise/RFI is suppressed). We then give this new feature a small weight and renormalize weights, leaving the rest of the pipeline unchanged. This should improve ranking separability slightly without training, new data, or any heavy computation, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.51742) has done: 'Your current unsupervised rank-ensemble is still near-random AUC, so the smallest legitimate step toward the 0.757 target is to keep the exact same inference-only “A vs off_mean” core and rank-aggregation, but add one more cadence-consistency feature that better matches the competition’s signal property (“present across A panels, absent in off”). Specifically, we add a spectrum-level persistence feature: take the minimum (over the 3 A panels) of the directional A−off contrast in the time-collapsed spectrum, then use a high percentile; this is very cheap and complements your existing pixel-level min-over-A feature. We keep all existing 13 features unchanged, append this as feature 14 with a small weight, and leave the same global robust sigmoid calibration and submission writing intact to preserve evaluation semantics while nudging ranking quality upward.'
- What this solution (achieved 0.51768) has done: 'Your current AUC (0.51742) is far below the target (0.75717), so we should improve separability while preserving the same inference-only “A vs off_mean” core and the same rank-aggregation + sigmoid mapping. The smallest high-impact bug-like issue here is that `off_mean3` is created by repeating the same array and then you take `np.mean(off_mean3, axis=1)`, which is redundant and can subtly skew intent; we compute `off_spec` directly from `off_mean` and keep everything else identical. Then we add one minimal, signal-aligned feature that often helps in SETI without training: a *time-summed per-frequency SNR-like contrast* using robust (median/MAD) normalization **within each cadence**, still purely based on A−off directional contrast. Finally, we give this new feature a small weight and renormalize, keeping the rest of the pipeline unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved 0.51768) has done: 'We keep your inference-only “A vs off_mean” contrast pipeline, rank-aggregation, and global sigmoid calibration exactly the same, but make two minimal, signal-aligned feature tweaks that should improve ranking (and thus AUC) toward your 0.757 target. First, we remove an unnecessary `repeat()` for `off_mean3` in the per-panel contrasts by using broadcasting, which keeps semantics identical but avoids subtle shape/intent issues. Second, we replace the weakest feature (`grad_peak_feat`, currently based on `abs(diff)` which rewards any difference) with a directional version `diff(positive_contrast)` so it specifically rewards line-like structure that is stronger in A than off (more aligned with the metric and cadence property) without changing the overall approach. Everything still runs end-to-end within the same constraints and writes a valid `submission.csv` with `id,target` and 6000 rows.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd




## === cell 1
def _find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


BASE = _find_existing_path(
    [
        "/kaggle/input/seti-breakthrough-listen",
        "/kaggle/data/seti-breakthrough-listen",
        "/kaggle/input",
        "/kaggle/data",
    ]
)

if BASE is None:
    raise FileNotFoundError(
        "Could not locate Kaggle data directory under /kaggle/input or /kaggle/data."
    )

COMP_ROOT = _find_existing_path(
    [
        os.path.join(BASE, "seti-breakthrough-listen"),
        BASE,
    ]
)

sample_path = _find_existing_path(
    [
        os.path.join(COMP_ROOT, "sample_submission.csv"),
        os.path.join(BASE, "sample_submission.csv"),
    ]
)
test_dir = _find_existing_path(
    [
        os.path.join(COMP_ROOT, "test"),
        os.path.join(BASE, "test"),
    ]
)

if sample_path is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")
if test_dir is None:
    raise FileNotFoundError("test/ directory not found in expected locations.")

sample = pd.read_csv(sample_path)
if list(sample.columns) != ["id", "target"]:
    sample = sample[["id", "target"]].copy()

test_files = glob.glob(os.path.join(test_dir, "*", "*.npy"))
id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in test_files}

missing = [i for i in sample["id"].tolist() if i not in id_to_path]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test .npy files referenced by sample_submission.csv. "
        f"First few missing: {missing[:5]}"
    )


def compute_features_from_npy(path):
    """
    Inference-only core logic remains: compare on-target A panels (0,2,4) vs off-target (1,3,5).

    Changes (aimed to improve AUC toward target, while preserving core inference-only logic):
    1) Avoid explicit np.repeat(off_mean, 3, axis=0) and use broadcasting instead. This keeps the
       exact same semantics (A compared to the same off_mean reference) but removes unnecessary
       duplication and any subtle intent confusion.
    2) Make the gradient/edge feature directional: replace abs(diff(diff_spec)) with
       abs(diff(positive_contrast_spectrum)). Needles are expected to be stronger in A than off,
       so this avoids rewarding "any difference" and should improve ranking for AUC.
    """
    x = np.load(path)  # (6, 273, 256), float16
    x = x.astype(np.float32, copy=False)

    a = x[[0, 2, 4]]  # (3,273,256)
    off = x[[1, 3, 5]]  # (3,273,256)

    off_mean = np.mean(off, axis=0, keepdims=True)  # (1,273,256), shared reference

    global_feat = float(np.mean(np.maximum(a - off_mean, 0.0)))

    a_spec = np.mean(a, axis=1)  # (3,256)
    off_spec_1x256 = np.mean(off_mean[0], axis=0, keepdims=True)  # (1,256)
    off_spec = np.repeat(off_spec_1x256, 3, axis=0)  # (3,256)

    diff_spec = np.maximum(a_spec - off_spec, 0.0)  # (3,256)
    peak_feat = float(np.percentile(diff_spec, 99.5))

    d_diff = np.abs(np.diff(diff_spec, axis=1))  # (3,255)
    grad_peak_feat = float(np.percentile(d_diff, 99.5))

    a_tmax = np.max(a, axis=1)  # (3,256)
    off_tmax = np.max(off_mean, axis=1)  # (1,256)
    tmax_diff = np.maximum(a_tmax - off_tmax, 0.0)  # (3,256)
    tmax_peak_feat = float(np.percentile(tmax_diff, 99.5))

    a_max = float(np.max(a_tmax))
    off_max = float(np.max(off_tmax))
    max_contrast_feat = float(max(a_max - off_max, 0.0))

    diff_mean = np.mean(diff_spec, axis=0)  # (256,)
    diff_disp = float(np.mean(np.abs(diff_spec - diff_mean[None, :])))
    a_consistency_feat = float(-diff_disp)

    a_mean = np.mean(a, axis=0, keepdims=False)  # (273,256)
    off_m = off_mean[0]  # (273,256)

    global_meanA_feat = float(np.mean(np.maximum(a_mean - off_m, 0.0)))

    a_mean_spec = np.mean(a_mean, axis=0)  # (256,)
    off_mean_spec = np.mean(off_m, axis=0)  # (256,)
    diff_mean_spec = np.maximum(a_mean_spec - off_mean_spec, 0.0)  # (256,)
    peak_meanA_feat = float(np.percentile(diff_mean_spec, 99.5))

    eps = 1e-6

    def _mean_pairwise_cosine_similarity(spec_3x256):
        s = spec_3x256.astype(np.float32, copy=False)
        s = s - np.mean(s, axis=1, keepdims=True)
        norms = np.sqrt(np.sum(s * s, axis=1, keepdims=True)) + eps
        s = s / norms
        c01 = float(np.dot(s[0], s[1]))
        c02 = float(np.dot(s[0], s[2]))
        c12 = float(np.dot(s[1], s[2]))
        return (c01 + c02 + c12) / 3.0

    a_self_sim = _mean_pairwise_cosine_similarity(a_spec)
    off_spec3 = np.mean(off, axis=1)  # (3,256)
    off_self_sim = _mean_pairwise_cosine_similarity(off_spec3)

    A_spectral_self_similarity_feat = float(a_self_sim)
    off_minus_A_self_similarity_feat = float(off_self_sim - a_self_sim)

    diff_img_mean = np.maximum(a_mean - off_m, 0.0)  # (273,256)
    pix_q_mean_feat = float(np.percentile(diff_img_mean, 99.9))

    a_pix_max = np.max(a, axis=0)  # (273,256)
    off_pix_max = np.max(off, axis=0)  # (273,256)
    diff_img_max = np.maximum(a_pix_max - off_pix_max, 0.0)
    pix_q_max_feat = float(np.percentile(diff_img_max, 99.9))

    a_panel_diff = np.maximum(a - off_mean, 0.0)  # (3,273,256)
    min_over_a = np.min(a_panel_diff, axis=0)  # (273,256)
    pix_q_minA_feat = float(np.percentile(min_over_a, 99.9))

    a_panel_diff_spec = np.mean(a_panel_diff, axis=1)  # (3,256)
    minA_diff_spec = np.min(a_panel_diff_spec, axis=0)  # (256,)
    spec_q_minA_feat = float(np.percentile(minA_diff_spec, 99.5))

    off_med = float(np.median(off_mean_spec))
    mad = float(np.median(np.abs(off_mean_spec - off_med))) + 1e-6
    robust_z = diff_mean_spec / (1.4826 * mad)
    snr_q_feat = float(np.percentile(robust_z, 99.5))

    return np.array(
        [
            global_feat,
            peak_feat,
            grad_peak_feat,
            tmax_peak_feat,
            max_contrast_feat,
            a_consistency_feat,
            global_meanA_feat,
            peak_meanA_feat,
            A_spectral_self_similarity_feat,
            off_minus_A_self_similarity_feat,
            pix_q_mean_feat,
            pix_q_max_feat,
            pix_q_minA_feat,
            spec_q_minA_feat,
            snr_q_feat,
        ],
        dtype=np.float64,
    )




## === cell 2
sample.head()



## === cell 3
pd.DataFrame({"n_sample_rows": [len(sample)], "n_test_files_found": [len(test_files)]})



## === cell 4
feat_mat = np.zeros((len(sample), 15), dtype=np.float64)
for i, _id in enumerate(sample["id"].tolist()):
    feat_mat[i] = compute_features_from_npy(id_to_path[_id])

weights = np.array(
    [
        0.37,
        0.15,
        0.07,
        0.09,
        0.07,
        0.05,
        0.05,
        0.04,
        0.03,
        0.01,
        0.04,
        0.02,
        0.01,
        0.02,
        0.03,
    ],
    dtype=np.float64,
)
weights = weights / weights.sum()

ranks = np.empty_like(feat_mat, dtype=np.float64)
n = feat_mat.shape[0]
for j in range(feat_mat.shape[1]):
    order = np.argsort(feat_mat[:, j], kind="mergesort")
    r = np.empty(n, dtype=np.float64)
    r[order] = np.arange(n, dtype=np.float64)
    ranks[:, j] = r / max(n - 1, 1)

raw = ranks @ weights  # in [0,1]

med = float(np.median(raw))
q75, q25 = np.percentile(raw, [75.0, 25.0])
iqr = float(q75 - q25)
if iqr < 1e-12:
    iqr = 1e-12

z = (raw - med) / (1.35 * iqr)
z = np.clip(z, -8.0, 8.0)
preds = 1.0 / (1.0 + np.exp(-z))

submission = pd.DataFrame(
    {"id": sample["id"].values, "target": preds.astype(np.float64)}
)
submission["target"] = submission["target"].clip(0.0, 1.0)

data6 = submission



## === cell 5
data6.to_csv("submission.csv", index=False)

print(data6.shape)
print(data6.head())
print("Wrote submission.csv")
