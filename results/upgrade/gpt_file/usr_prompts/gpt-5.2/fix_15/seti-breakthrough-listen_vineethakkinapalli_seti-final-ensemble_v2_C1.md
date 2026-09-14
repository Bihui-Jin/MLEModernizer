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

0.7616417998512798

# 6. Current score

0.50717

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook currently fails because it tries to read many external “../input/…” submissions that do not exist in this environment, so no `data*` variables are created and everything downstream crashes. To keep the core “ensemble of submissions” logic but make it runnable end-to-end, I load whichever candidate submission files are actually present (falling back safely), align them to `sample_submission.csv` ids, and then perform the same weighted blend you intended (0.95 * data6 + 0.05 * data8). If those specific files aren’t available, the code fall back to a valid baseline (0.5 for all rows) rather than erroring, ensuring a valid `submission.csv` is always produced in the required format.'
- What this solution (achieved 0.50024) has done: 'Your current 0.5 score indicates you’re effectively submitting constant probabilities, because the notebook can’t find any real prediction CSVs and falls back to 0.5. To move toward the target AUC (~0.762) with minimal change and without altering modeling/training logic (you have none here), I keep your “blend submissions” approach but add a safe, legitimate fallback: train a very small classical model (logistic regression) on lightweight engineered features computed from the provided `.npy` snippets, then predict on test. This uses only the official train/test data (no leaky labels) and should materially improve AUC over 0.5 while staying within time by sampling a limited number of training files. If submission CSVs are present, your original blend is still used; the model fallback only runs when you’d otherwise output 0.5s.'
- What this solution (achieved 0.49227) has done: 'Your current score (0.50024) is far below the target (0.7616), so we should improve (higher-is-better). The main issue is that the “ensemble of external submissions” still usually falls back to constant 0.5, and the fallback logistic regression is trained on the first N ids (not representative), with unscaled features that can make optimization/coefficients unstable. I keep your exact fallback model family (logistic regression) and the same feature extraction, but (1) use a stratified random subset of training ids (same N) to better match the full distribution, and (2) add a StandardScaler in a Pipeline so the logistic regression behaves better numerically. I also fix `_find_npy_by_id` to use the correct shard count (0–15) for this dataset (currently 16 misses shard “15”), which directly increases usable training/test coverage and should lift AUC toward your target.'
- What this solution (achieved 0.4946) has done: 'Your current score is far below the target (0.492 < 0.762), so we should improve (higher-is-better) with the smallest safe change. The biggest likely performance drag is that the fallback feature model is trained on only 8k samples and uses a fairly rigid LR regularization setting; we can move toward the target by (1) increasing the training subset size moderately while keeping the same stratified sampling logic, and (2) tuning the LR regularization strength `C` a bit higher (less underfitting) while keeping the same model family and pipeline. I also fix the shard search to cover 0–15 robustly (including any unexpected shard counts) by discovering shard folders once, which increases file hit-rate and directly improves both train and test feature coverage. These are minimal changes that preserve the same core “blend-if-available, else scaled logistic regression on engineered features” semantics and should lift AUC toward your target.'
- What this solution (achieved 0.49494) has done: 'Your score is far below the target (0.4946 vs 0.7616, higher-is-better), so we should improve with the smallest safe change. The main bottleneck is that the fallback logistic-regression model is trained on a limited subset and the features don’t capture the key “A vs B/C/D” cadence structure strongly enough, so it stays near-random. I keep the exact same fallback model family (scaled LogisticRegression) and the same overall flow, but (1) slightly expand the stratified training subset (still within time), and (2) add a few very cheap, cadence-aware features that directly encode “on-target (A) differs from off-target” patterns without changing the learning approach. This should move AUC upward toward the target while preserving your ensemble-first / model-fallback semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 0.49154) has done: 'Your score is far below the target, so we should improve (higher-is-better) with the smallest safe changes while keeping your existing “blend if available, else scaled LogisticRegression on engineered features” approach intact. The current fallback is likely underperforming because the engineered features don’t strongly encode the key ABACAD cadence logic (A panels vs non‑A panels), so I add a few very cheap, cadence-structure features (A-minus-off summary maps, simple drift/line proxies, and robust energy ratios) without changing the model family or training loop. I also slightly increase the training subset size (still bounded to run within the time limit) and make the subset exactly class-balanced to reduce variance and help AUC. Finally, I keep submission alignment to `sample_submission.csv` ids unchanged and still always write a valid `submission.csv`.'
- What this solution (achieved 0.49719) has done: 'Your current AUC (0.49154) is far below the target (0.7616), so we should improve it (higher-is-better) with the smallest safe changes while keeping the same “blend external submissions if found, else scaled LogisticRegression on engineered features” core logic intact. The main likely issue is that the fallback model is learning weakly from features that are still too generic and not explicitly tied to the ABACAD cadence rule; we add a few ultra-cheap cadence-structure features that directly compare A panels (0/2/4) to off panels (1/3/5) in time/frequency “line strength” summaries. We also keep the exact same model family and pipeline, but slightly adjust regularization (still LogisticRegression) and increase training subset size modestly to reduce variance while staying within the 600s runtime. Submission writing, id alignment, and the ensemble-first behavior remain unchanged, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.49715) has done: 'Your current score is far below the target (0.497 vs 0.762, higher-is-better), and the biggest issue is that the solution still often behaves like a near-random baseline because the fallback model’s features are not strongly discriminative for the ABACAD cadence structure. I keep the exact same overall logic (use external submission blend if found; otherwise train a scaled LogisticRegression on engineered features) but minimally strengthen the *existing* engineered features by adding a few cheap, cadence-specific summaries (A vs B/C/D per-panel differences and simple “A-only consistency vs off-only consistency” ratios). I also add `random_state` to LogisticRegression for determinism (no semantic change) and slightly adjust `C` back toward a more stable value to reduce overfitting risk while improving generalization. All paths, training approach, and submission writing remain the same, and it still always produce a valid `submission.csv`.'
- What this solution (achieved 0.49695) has done: 'Your current score (0.497) is far below the target (0.7616, higher-is-better), so we should legitimately increase model signal with minimal changes while keeping the same “blend external submissions else scaled LogisticRegression on engineered features” core logic. The main problem is the fallback feature model is still too weak; we keep the same model family/training, but (1) add a tiny set of stronger cadence-aware features that measure “A panels contain a coherent track absent in off panels” using very cheap row/col max summaries, and (2) switch LogisticRegression to a slightly less-regularized setting (small change) while keeping everything else the same. We also avoid a potential performance penalty from class_weight="balanced" since we already sample a balanced subset; removing it keeps semantics (LR) but typically improves AUC calibration/ranking. Submission alignment and writing remain unchanged and a valid `submission.csv` is always produced.'
- What this solution (achieved 0.49667) has done: 'Your current score is far below the target (0.49695 vs 0.76164, higher-is-better), so we should improve the fallback (non-external) model with minimal, legitimate changes while preserving the same LogisticRegression-on-engineered-features approach. The biggest issue is that the fallback currently includes all 54k training samples (balanced) which is slow and may still underfit/overfit due to a single C value; instead, we keep the same model family and features but add a tiny, fast internal cross-validation over a small grid of `C` values to pick a better regularization strength (this changes only a hyperparameter, not the approach). We also standardize the missing-test feature fill to use the training-feature mean (instead of zeros), which is a small but often meaningful calibration fix for linear models. Everything else (data discovery, feature extraction, predict_proba, submission alignment and writing) remains the same and still always writes a valid `submission.csv`.'
- What this solution (achieved 0.49746) has done: 'The timeout is dominated by slow per-file disk I/O plus extremely expensive repeated `np.quantile` calls inside `_extract_features`, and by `LogisticRegressionCV` refitting many times on a large (30k) dataset. To keep identical core logic and evaluation semantics, the main speedups are: (1) eliminate repeated shard path probing by building an `id -> file_path` index once for train/test, (2) replace exact `np.quantile` with the provably-equivalent `np.partition` for these fixed quantile levels (0.95/0.99/0.995/0.999) and compute on flattened views to reduce overhead, and (3) preallocate feature matrices and parallelize feature extraction with a deterministic thread pool (NumPy releases the GIL in heavy ops), while keeping the same model, CV, Cs, solver, and sample weights. We also avoid repeated pandas merges for submission alignment and keep all paths unchanged. These changes preserve the feature definitions and the LR-CV training logic, but cut constant factors enough to fit under 600 seconds in typical Kaggle CPU environments.'
- What this solution (achieved 0.49438) has done: 'Your current score (0.49746) is far below the target (0.76164), so we should increase AUC with the smallest change that adds real signal while preserving the same “external blend if available, else scaled LogisticRegressionCV on engineered features” core logic. The main issue is that the fallback training subset is currently an unstratified random sample, which can be class-imbalanced and reduce AUC; we switch to a deterministic class-balanced subset of the same size to stabilize learning without changing the model family or feature definitions. We also make the CV folds stratified (still 3-fold CV, same scoring/solver), which usually improves hyperparameter selection for AUC without changing the approach. Everything else (feature extraction, pipeline, prediction, alignment, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.50717) has done: 'Your current score (0.49438) is far below the target (0.76164), and the biggest reason is that the “external submission blend” is effectively constant (no real external CSVs exist here), so the fallback LR model must carry the performance. I keep the exact same fallback approach (engineered features + StandardScaler + LogisticRegressionCV with stratified CV) but make two minimal, score-relevant fixes: (1) ensure we actually train on the full available labeled data (balanced as before) instead of a capped subset, and (2) add a tiny, deterministic post-fit recalibration (isotonic regression on out-of-fold predictions) to better align probabilities for ROC-AUC ranking without changing the classifier family or feature extraction. These changes are lightweight, don’t alter the core logic/semantics (still LR-on-features), and should move AUC upward toward the target while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

BASE_INPUT = "/kaggle/input"
BASE_DATA = "/kaggle/data"


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


sample_path = _first_existing(
    [
        os.path.join(BASE_INPUT, "sample_submission.csv"),
        os.path.join(BASE_DATA, "sample_submission.csv"),
        os.path.join(BASE_INPUT, "seti-breakthrough-listen", "sample_submission.csv"),
        os.path.join(BASE_DATA, "seti-breakthrough-listen", "sample_submission.csv"),
    ]
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations under /kaggle/input or /kaggle/data"
    )

sample = pd.read_csv(sample_path)
if list(sample.columns)[:2] != ["id", "target"]:
    sample = sample[["id", "target"]].copy()


def load_submission_csv(path):
    """Load a submission file and align it to sample ids. Returns a DataFrame with columns [id, target]."""
    df = pd.read_csv(path)
    if "id" not in df.columns or "target" not in df.columns:
        raise ValueError(
            f"Submission at {path} must contain columns ['id','target']. Found: {df.columns.tolist()}"
        )
    df = df[["id", "target"]].copy()
    df = sample[["id"]].merge(df, on="id", how="left")
    df["target"] = df["target"].astype("float64")
    df["target"] = df["target"].fillna(0.5)
    return df


def find_candidate_submissions(search_root):
    """Find plausible submission-like csvs under a root folder."""
    out = []
    if search_root is None or not os.path.exists(search_root):
        return out
    for root, _, files in os.walk(search_root):
        for fn in files:
            lfn = fn.lower()
            if lfn.endswith(".csv") and ("submission" in lfn or "sub" in lfn):
                out.append(os.path.join(root, fn))
    return out


intended_paths = {
    "data1": "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "data2": "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "data3": "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
    "data4": "../input/seti-learned-image-resizing/submission.csv",
    "data5": "../input/rerun-seti-e-t-resnet18d-baseline/submission.csv",
    "data6": "../input/ensemble-for-seti-competition/submission.csv",
    "data7": "../input/fixed-gradual-warmup-custom-head/submission.csv",
    "data8": "../input/seti-results/learned_resizing_384.csv",
}

loaded = {}
for k, p in intended_paths.items():
    if os.path.exists(p):
        loaded[k] = load_submission_csv(p)

if "data6" not in loaded or "data8" not in loaded:
    candidate_paths = []
    candidate_paths.extend(find_candidate_submissions(BASE_INPUT))
    candidate_paths.extend(find_candidate_submissions(BASE_DATA))
    candidate_paths = sorted(set(candidate_paths))

    def choose_candidate(exclude_paths=()):
        for cp in candidate_paths:
            bn = os.path.basename(cp).lower()
            if bn == "sample_submission.csv":
                continue
            if cp in exclude_paths:
                continue
            return cp
        return None

    if "data6" not in loaded:
        p6 = choose_candidate()
        if p6 is not None:
            loaded["data6"] = load_submission_csv(p6)

    if "data8" not in loaded:
        p8 = choose_candidate()
        if p8 is not None:
            loaded["data8"] = load_submission_csv(p8)

if "data6" not in loaded:
    loaded["data6"] = sample.copy()
    loaded["data6"]["target"] = 0.5
if "data8" not in loaded:
    loaded["data8"] = sample.copy()
    loaded["data8"]["target"] = 0.5

data1 = loaded.get("data1", None)
data2 = loaded.get("data2", None)
data3 = loaded.get("data3", None)
data4 = loaded.get("data4", None)
data5 = loaded.get("data5", None)
data6 = loaded["data6"]
data7 = loaded.get("data7", None)
data8 = loaded["data8"]



## === cell 1
data11 = sample.copy()



## === cell 2
data11 = (
    sample[["id"]]
    .merge(
        data6[["id", "target"]].rename(columns={"target": "t6"}), on="id", how="left"
    )
    .merge(
        data8[["id", "target"]].rename(columns={"target": "t8"}), on="id", how="left"
    )
)

t6 = data11["t6"].astype("float64").fillna(0.5)
t8 = data11["t8"].astype("float64").fillna(0.5)

data11["target"] = (0.95 * t6 + 0.05 * t8).clip(0.0, 1.0)
data11 = data11[["id", "target"]]




## === cell 3
def _resolve_comp_root():
    candidates = [
        os.path.join(BASE_INPUT, "seti-breakthrough-listen"),
        os.path.join(BASE_DATA, "seti-breakthrough-listen"),
        BASE_INPUT,
        BASE_DATA,
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train_labels.csv")) and (
            os.path.exists(os.path.join(c, "train"))
            and os.path.exists(os.path.join(c, "test"))
        ):
            return c
    for c in candidates:
        if os.path.exists(os.path.join(c, "train_labels.csv")) and (
            os.path.exists(os.path.join(c, "train"))
            or os.path.exists(os.path.join(c, "seti-breakthrough-listen", "train"))
        ):
            return c
    return None


def _baseline_is_constant_half(df):
    v = df["target"].to_numpy(dtype=np.float64)
    return np.all(np.isfinite(v)) and np.max(np.abs(v - 0.5)) < 1e-12


def _list_shards(folder):
    shards = []
    try:
        for name in os.listdir(folder):
            p = os.path.join(folder, name)
            if os.path.isdir(p) and name.isdigit():
                shards.append(int(name))
    except FileNotFoundError:
        return []
    shards = sorted(shards)
    return shards


def _build_id_path_index(root_dir, shards):
    idx = {}
    for shard in shards:
        d = os.path.join(root_dir, str(shard))
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".npy"):
                idx[fn[:-4]] = os.path.join(d, fn)
    try:
        for fn in os.listdir(root_dir):
            if fn.endswith(".npy"):
                idx.setdefault(fn[:-4], os.path.join(root_dir, fn))
    except FileNotFoundError:
        pass
    return idx


def _extract_features(arr):
    """
    Same engineered-feature approach as before (cadence-aware summaries).
    """
    x = arr.astype(np.float32, copy=False)  # (6, 273, 256)
    a = x[[0, 2, 4]]
    b = x[[1, 3, 5]]

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    a_mean = float(a_flat.mean())
    b_mean = float(b_flat.mean())
    a_std = float(a_flat.std())
    b_std = float(b_flat.std())
    a_p99 = float(np.quantile(a_flat, 0.99))
    b_p99 = float(np.quantile(b_flat, 0.99))

    diff = a.mean(axis=0) - b.mean(axis=0)  # (273, 256)
    absdiff = np.abs(diff)
    absdiff_flat = absdiff.reshape(-1)
    diff_flat = diff.reshape(-1)

    diff_mean = float(diff_flat.mean())
    diff_std = float(diff_flat.std())
    diff_p99 = float(np.quantile(absdiff_flat, 0.99))

    t_profile = absdiff.mean(axis=1)  # (273,)
    f_profile = absdiff.mean(axis=0)  # (256,)
    diff_tmax = float(t_profile.max())
    diff_fmax = float(f_profile.max())

    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    a_intra = float(
        (np.mean(np.abs(a0 - a1)) + np.mean(np.abs(a0 - a2)) + np.mean(np.abs(a1 - a2)))
        / 3.0
    )
    b_intra = float(
        (np.mean(np.abs(b0 - b1)) + np.mean(np.abs(b0 - b2)) + np.mean(np.abs(b1 - b2)))
        / 3.0
    )

    a_p999 = float(np.quantile(a_flat, 0.999))
    b_p999 = float(np.quantile(b_flat, 0.999))

    thr = float(np.quantile(absdiff_flat, 0.995))
    strong_frac = float((absdiff_flat >= thr).mean())

    a_med = float(np.median(a_flat))
    b_med = float(np.median(b_flat))
    a_l1 = float(np.mean(np.abs(a_flat - a_med)))
    b_l1 = float(np.mean(np.abs(b_flat - b_med)))
    a_over_b_l1 = float(a_l1 / (b_l1 + 1e-6))

    def _corr(u, v):
        u = u.ravel()
        v = v.ravel()
        u = u - u.mean()
        v = v - v.mean()
        denom = np.sqrt((u * u).mean()) * np.sqrt((v * v).mean()) + 1e-6
        return float((u * v).mean() / denom)

    a_corr = float((_corr(a0, a1) + _corr(a0, a2) + _corr(a1, a2)) / 3.0)
    b_corr = float((_corr(b0, b1) + _corr(b0, b2) + _corr(b1, b2)) / 3.0)

    argmax_f = np.argmax(absdiff, axis=1).astype(np.float32)  # (273,)
    t_idx = np.arange(argmax_f.shape[0], dtype=np.float32)
    t0 = t_idx - t_idx.mean()
    f0 = argmax_f - argmax_f.mean()
    slope = float((t0 * f0).sum() / (t0 * t0).sum())
    drift_abs = float(abs(slope))

    absdiff_mean = float(absdiff_flat.mean())
    peak_over_mean = float(float(absdiff_flat.max()) / (absdiff_mean + 1e-6))

    t_line_strength = float(float(t_profile.max()) / (float(t_profile.mean()) + 1e-6))
    f_line_strength = float(float(f_profile.max()) / (float(f_profile.mean()) + 1e-6))

    mean_t_of_fmax = float(np.mean(np.max(absdiff, axis=1)))
    line_conc_t = float(mean_t_of_fmax / (absdiff_mean + 1e-6))

    mean_f_of_tmax = float(np.mean(np.max(absdiff, axis=0)))
    line_conc_f = float(mean_f_of_tmax / (absdiff_mean + 1e-6))

    upper_tail_ratio = float(
        (np.quantile(absdiff_flat, 0.999) + 1e-6)
        / (np.quantile(absdiff_flat, 0.95) + 1e-6)
    )

    a_off_diffs = np.array(
        [
            np.mean(np.abs(a0 - b0)),
            np.mean(np.abs(a0 - b1)),
            np.mean(np.abs(a0 - b2)),
            np.mean(np.abs(a1 - b0)),
            np.mean(np.abs(a1 - b1)),
            np.mean(np.abs(a1 - b2)),
            np.mean(np.abs(a2 - b0)),
            np.mean(np.abs(a2 - b1)),
            np.mean(np.abs(a2 - b2)),
        ],
        dtype=np.float32,
    )
    a_off_mean = float(a_off_diffs.mean())
    a_off_min = float(a_off_diffs.min())
    a_off_max = float(a_off_diffs.max())
    a_off_std = float(a_off_diffs.std())
    a_only_vs_off_ratio = float((a_off_mean + 1e-6) / (b_intra + 1e-6))

    cons_ratio = float((b_intra + 1e-6) / (a_intra + 1e-6))

    a_map = a.mean(axis=0)
    b_map = b.mean(axis=0)

    a_rowmax = a_map.max(axis=1)
    b_rowmax = b_map.max(axis=1)
    a_colmax = a_map.max(axis=0)
    b_colmax = b_map.max(axis=0)

    a_rowmax_strength = float(float(a_rowmax.max()) / (float(a_rowmax.mean()) + 1e-6))
    b_rowmax_strength = float(float(b_rowmax.max()) / (float(b_rowmax.mean()) + 1e-6))
    a_colmax_strength = float(float(a_colmax.max()) / (float(a_colmax.mean()) + 1e-6))
    b_colmax_strength = float(float(b_colmax.max()) / (float(b_colmax.mean()) + 1e-6))

    a_tracks = np.stack(
        [np.argmax(a0, axis=1), np.argmax(a1, axis=1), np.argmax(a2, axis=1)], axis=0
    )
    b_tracks = np.stack(
        [np.argmax(b0, axis=1), np.argmax(b1, axis=1), np.argmax(b2, axis=1)], axis=0
    )
    a_track_disp = float(np.mean(np.std(a_tracks.astype(np.float32), axis=0)))
    b_track_disp = float(np.mean(np.std(b_tracks.astype(np.float32), axis=0)))
    track_disp_ratio = float((b_track_disp + 1e-6) / (a_track_disp + 1e-6))

    return np.array(
        [
            a_mean,
            b_mean,
            a_std,
            b_std,
            a_p99,
            b_p99,
            a_mean - b_mean,
            a_std - b_std,
            a_p99 - b_p99,
            diff_mean,
            diff_std,
            diff_p99,
            diff_tmax,
            diff_fmax,
            a_intra,
            b_intra,
            a_intra - b_intra,
            a_p999,
            b_p999,
            a_p999 - b_p999,
            strong_frac,
            a_over_b_l1,
            a_corr,
            b_corr,
            a_corr - b_corr,
            drift_abs,
            peak_over_mean,
            t_line_strength,
            f_line_strength,
            line_conc_t,
            line_conc_f,
            upper_tail_ratio,
            a_off_mean,
            a_off_min,
            a_off_max,
            a_off_std,
            a_only_vs_off_ratio,
            cons_ratio,
            a_rowmax_strength,
            b_rowmax_strength,
            a_rowmax_strength - b_rowmax_strength,
            a_colmax_strength,
            b_colmax_strength,
            a_colmax_strength - b_colmax_strength,
            a_track_disp,
            b_track_disp,
            track_disp_ratio,
        ],
        dtype=np.float32,
    )


if _baseline_is_constant_half(data11):
    comp_root = _resolve_comp_root()
    if comp_root is None:
        print(
            "Could not resolve competition data root; keeping baseline 0.5 submission."
        )
    else:
        train_labels_path = os.path.join(comp_root, "train_labels.csv")
        if not os.path.exists(train_labels_path):
            train_labels_path = _first_existing(
                [
                    os.path.join(BASE_INPUT, "train_labels.csv"),
                    os.path.join(BASE_DATA, "train_labels.csv"),
                    os.path.join(
                        BASE_INPUT, "seti-breakthrough-listen", "train_labels.csv"
                    ),
                    os.path.join(
                        BASE_DATA, "seti-breakthrough-listen", "train_labels.csv"
                    ),
                ]
            )

        train_dir = os.path.join(comp_root, "train")
        test_dir = os.path.join(comp_root, "test")
        if not os.path.exists(train_dir):
            train_dir = os.path.join(comp_root, "seti-breakthrough-listen", "train")
        if not os.path.exists(test_dir):
            test_dir = os.path.join(comp_root, "seti-breakthrough-listen", "test")

        if (
            (train_labels_path is None)
            or (not os.path.exists(train_dir))
            or (not os.path.exists(test_dir))
        ):
            print("Missing train/test data or labels; keeping baseline 0.5 submission.")
        else:
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import StratifiedKFold
            from sklearn.isotonic import IsotonicRegression
            from concurrent.futures import ThreadPoolExecutor

            labels = pd.read_csv(train_labels_path)[["id", "target"]]
            labels["id"] = labels["id"].astype(str)
            labels["target"] = labels["target"].astype(int)

            N_TRAIN_MAX = int(len(labels))  # was 30000

            rng = np.random.RandomState(42)
            pos = labels[labels["target"] == 1]
            neg = labels[labels["target"] == 0]
            n_each = min(len(pos), len(neg), N_TRAIN_MAX // 2)
            if n_each < 1000:
                labels_sub = labels.sample(
                    n=min(N_TRAIN_MAX, len(labels)), replace=False, random_state=42
                ).reset_index(drop=True)
            else:
                pos_sub = pos.sample(n=n_each, replace=False, random_state=42)
                neg_sub = neg.sample(n=n_each, replace=False, random_state=42)
                labels_sub = (
                    pd.concat([pos_sub, neg_sub], axis=0)
                    .sample(frac=1.0, random_state=42)
                    .reset_index(drop=True)
                )

            train_shards = _list_shards(train_dir)
            test_shards = _list_shards(test_dir)
            if len(train_shards) == 0:
                train_shards = list(range(16))
            if len(test_shards) == 0:
                test_shards = list(range(16))

            train_index = _build_id_path_index(train_dir, train_shards)
            test_index = _build_id_path_index(test_dir, test_shards)

            ids_arr = labels_sub["id"].to_numpy()
            y_all = labels_sub["target"].to_numpy(dtype=int)

            mask_exist = np.fromiter(
                (i in train_index for i in ids_arr), dtype=bool, count=len(ids_arr)
            )
            ids_kept = ids_arr[mask_exist]
            y_kept = y_all[mask_exist]
            missing = int((~mask_exist).sum())

            if len(ids_kept) < 6000:
                print(
                    f"Too few training files found ({len(ids_kept)}); keeping baseline 0.5 submission."
                )
            else:
                first_arr = np.load(train_index[ids_kept[0]], mmap_mode="r")
                f0 = _extract_features(first_arr)
                n_feat = int(f0.shape[0])
                X = np.empty((len(ids_kept), n_feat), dtype=np.float32)
                X[0] = f0
                y = y_kept

                def _load_and_featurize_train(i_id):
                    i, id_ = i_id
                    arr = np.load(train_index[id_], mmap_mode="r")
                    return i, _extract_features(arr)

                max_workers = min(8, (os.cpu_count() or 2))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for i, feat in ex.map(
                        _load_and_featurize_train,
                        enumerate(ids_kept[1:], start=1),
                        chunksize=32,
                    ):
                        X[i] = feat

                n1 = max(1, int((y == 1).sum()))
                n0 = max(1, int((y == 0).sum()))
                w1 = 0.5 / n1
                w0 = 0.5 / n0
                sample_weight = np.where(y == 1, w1, w0).astype(np.float64)

                Cs = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

                clf = Pipeline(
                    steps=[
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        (
                            "lr",
                            LogisticRegressionCV(
                                Cs=Cs,
                                cv=cv,
                                scoring="roc_auc",
                                solver="lbfgs",
                                max_iter=800,
                                n_jobs=1,
                                refit=True,
                                random_state=42,
                            ),
                        ),
                    ]
                )
                clf.fit(X, y, lr__sample_weight=sample_weight)

                oof = np.empty(len(y), dtype=np.float64)
                for tr_idx, va_idx in cv.split(X, y):
                    clf_fold = Pipeline(
                        steps=[
                            ("scaler", StandardScaler(with_mean=True, with_std=True)),
                            (
                                "lr",
                                LogisticRegressionCV(
                                    Cs=Cs,
                                    cv=StratifiedKFold(
                                        n_splits=3, shuffle=True, random_state=42
                                    ),
                                    scoring="roc_auc",
                                    solver="lbfgs",
                                    max_iter=800,
                                    n_jobs=1,
                                    refit=True,
                                    random_state=42,
                                ),
                            ),
                        ]
                    )
                    clf_fold.fit(
                        X[tr_idx],
                        y[tr_idx],
                        lr__sample_weight=sample_weight[tr_idx],
                    )
                    oof[va_idx] = clf_fold.predict_proba(X[va_idx])[:, 1]

                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(oof, y)

                train_feat_mean = X.mean(axis=0).astype(np.float32)

                test_ids = sample["id"].astype(str).to_numpy()
                Xte = np.empty((len(test_ids), n_feat), dtype=np.float32)

                def _load_and_featurize_test(i_id):
                    i, id_ = i_id
                    p = test_index.get(id_, None)
                    if p is None:
                        return i, None
                    arr = np.load(p, mmap_mode="r")
                    return i, _extract_features(arr)

                missing_test = 0
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for i, feat in ex.map(
                        _load_and_featurize_test, enumerate(test_ids), chunksize=32
                    ):
                        if feat is None:
                            missing_test += 1
                            Xte[i] = train_feat_mean
                        else:
                            Xte[i] = feat

                proba = clf.predict_proba(Xte)[:, 1].astype(np.float64)
                proba = iso.transform(proba)
                proba = np.clip(proba, 0.0, 1.0)

                data11 = pd.DataFrame({"id": test_ids, "target": proba})
                chosen_C = float(np.ravel(clf.named_steps["lr"].C_)[0])
                print(
                    "Replaced constant-0.5 baseline with scaled logistic-regression-CV feature model predictions + isotonic recalibration."
                )
                print(
                    "Chosen C:",
                    chosen_C,
                    "Train subset requested:",
                    len(labels_sub),
                    "used:",
                    len(ids_kept),
                    "missing:",
                    missing,
                    "Test rows:",
                    len(data11),
                    "missing_test:",
                    missing_test,
                    "train_shards:",
                    train_shards,
                    "test_shards:",
                    test_shards,
                    "n_features:",
                    n_feat,
                    "workers:",
                    max_workers,
                    "class_balance_sub:",
                    (
                        int((labels_sub["target"] == 0).sum()),
                        int((labels_sub["target"] == 1).sum()),
                    ),
                    "class_balance_used:",
                    (int((y == 0).sum()), int((y == 1).sum())),
                )



## === cell 4
data11.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data11.shape)
print(data11.head())
