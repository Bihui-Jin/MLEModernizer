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

0.7630592962475681

# 6. Current score

0.47964

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to read multiple pre-made ensemble submissions from `../input/...` that are not present in your environment, so nothing downstream is defined and no `submission.csv` is written. To make it run end-to-end while keeping changes minimal, I replace those missing external reads with a self-contained baseline that reads the provided `sample_submission.csv` and outputs a valid file in the required `id,target` format. This yield a valid submission CSV (score likely near 0.5 AUC since it’s constant predictions), unblocking you from “Not yielded” to a scored submission. Once you confirm the achieved score, we can make the smallest legitimate model-based change to move toward the target AUC.'
- What this solution (achieved 0.48464) has done: 'Your current submission predicts a constant 0.5 for every test row, which guarantees an AUC of 0.5; to move toward the target AUC, we need non-constant predictions derived from the provided test `.npy` snippets while keeping changes minimal. I replace the dummy constant targets with a simple, fast heuristic feature computed per snippet (difference in intensity patterns between “on-target” A panels and “off-target” B/C/D panels), then convert that score into probabilities via rank-based normalization so outputs are in [0,1] and non-degenerate. This preserves the overall “read sample_submission → create targets → write submission.csv” flow and stays within the installed packages. The rest of your weighting/ensemble scaffolding is left intact, but now it combine meaningful predictions instead of constants.'
- What this solution (achieved 0.50185) has done: 'Your current code’s “ensemble” is actually just reusing the exact same heuristic 7 times, so it cannot improve beyond whatever that single score provides; the simplest improvement toward your target AUC is to add a second, complementary per-snippet heuristic and blend it with the existing one. I keep your I/O, ranking-to-probability calibration, and overall flow identical, but compute an additional signal based on panel-wise time-derivative energy (captures drifting lines better) and average its rank-normalized probabilities with the existing statistic. This stays fast (single pass over test .npy files) and uses only numpy/pandas, so it should run under the timeout and produce a valid `submission.csv`. The rest of your cells remain structurally the same; only the targets feeding the “ensemble” are made genuinely different.'
- What this solution (achieved 0.48087) has done: 'Your current approach is purely heuristic and the score is stuck near random (AUC≈0.5); to move it toward the 0.763 target without changing the overall “compute per-snippet scores → rank-normalize → blend → write submission.csv” logic, we make the second heuristic more complementary and informative for this competition. Specifically, we keep `_snippet_score` intact, but replace `_snippet_score_drift` with a simple “diagonal-line evidence” statistic using a small set of Doppler slopes and comparing on-target (A panels) vs off-target (B/C/D) panels. We also blend the two rank-normalized probabilities with slightly more weight on the new slope-based feature, while keeping everything deterministic and fast. This should yield a non-trivial lift in AUC toward your target while preserving your pipeline and submission semantics.'
- What this solution (achieved 0.48456) has done: 'We keep your overall pipeline intact (compute two per-snippet heuristic scores → rank-normalize → blend → write `submission.csv`) but fix the main reason performance is near-random: the “drift” feature is indexing the frequency axis incorrectly, so it isn’t actually measuring diagonal line evidence. I replace only the internals of `_snippet_score_drift` with a correct, fast slope-scan that samples along diagonals across time for several Doppler slopes, then contrasts A panels vs B/C/D panels as you already do. I also add a tiny amount of robustness by subtracting a per-panel median and using a high-quantile statistic along the best slope (still a single deterministic pass). This should legitimately increase AUC toward your 0.763 target without changing the core approach or I/O.'
- What this solution (achieved 0.49207) has done: 'We keep your exact “two heuristic scores → rank-normalize → blend → write submission.csv” pipeline, but fix the main weakness: the drift score is currently too sparse (only 3 start frequencies, few slopes, and raw 90th-quantile on single-pixel samples), which makes it noisy and near-random. I minimally strengthen `_snippet_score_drift` by (1) scanning a slightly denser but still small grid of slopes and start frequencies, and (2) scoring each candidate diagonal using a tiny frequency-band aggregation (3-pixel width) to reduce single-pixel noise while preserving the same semantics. Finally, I adjust the blend weight slightly toward the improved drift feature (since your current score indicates it’s the more informative of the two) while keeping everything deterministic and fast. This should move AUC upward toward your 0.763 target without changing the overall approach or output format.'
- What this solution (achieved 0.49709) has done: 'I keep your exact pipeline (two per-snippet heuristic scores → rank-normalize → weighted blend → write `submission.csv`) but make the drift score less noisy and more discriminative with a minimal change: instead of taking a high-quantile of single-point samples along a diagonal, we accumulate a robust “line energy” by averaging across time after clipping to positive (post-median) values. I also scan a slightly denser but still small set of slopes and a few more start frequencies, and (crucially) compare the best line in A panels to the best line in B/C/D panels rather than averaging all panels’ best lines (needles only need to appear in some A panels). Finally, I keep your rank-normalization and overall blending logic, but shift a bit more weight to the improved drift feature to move AUC upward toward the 0.763 target.'
- What this solution (achieved 0.4868) has done: 'Your current score is far below the target (0.497 vs 0.763), so we should legitimately increase AUC while keeping your exact pipeline (two heuristic scores → rank-normalize → weighted blend → write CSV). The smallest likely win is to make the drift feature less noisy and more “line-like” by (1) lightly smoothing in frequency, (2) using a robust top-k mean along each candidate diagonal instead of a plain mean, and (3) expanding the slope/start-frequency grid slightly without blowing runtime. I keep `_snippet_score`, `_rank_to_proba`, file finding, and the overall blending/ensemble scaffolding unchanged, and only adjust `_snippet_score_drift` plus a very small blend-weight tweak toward the improved drift signal. This should move predictions away from near-random ranking and push AUC upward toward your target while remaining deterministic and fast enough.'
- What this solution (achieved 0.48182) has done: 'I keep your exact pipeline (two per-snippet heuristic scores → rank-normalize → weighted blend → write `submission.csv`) and only make the drift heuristic a bit more informative while staying fast. The minimal change is to (1) scan a slightly denser but still small slope grid, (2) test more start frequencies, and (3) reduce noise by scoring a narrow 3-pixel band and using a top‑k mean without the extra `sqrt(k)` amplification (which can over-emphasize noise spikes). I also make the A-vs-(B/C/D) contrast slightly more faithful to the problem by using the max over A panels minus the max over off-target panels (as you already do), but on a more robust line score. Finally, I keep your rank-calibration unchanged and only nudge the blend weight slightly toward the improved drift feature.'
- What this solution (achieved 0.49104) has done: 'Your current score (0.48182) is far below the target (0.76306), so we should increase AUC with the smallest change that preserves your existing “two heuristic scores → rank-normalize → blend → write CSV” pipeline. The main issue is that `_snippet_score_drift` is doing a very expensive/fragile slope scan on raw-ish pixels; we can make it more discriminative (and usually faster) by first collapsing each panel into a 1D time-series per frequency (max over time) and then scoring “line-ness” via a light Hough-like diagonal sum on that 1D summary. This keeps the exact same semantics (A-panels vs off-panels contrast; higher score ⇒ more needle-like), but reduces noise and better matches the competition’s drifting-line signal structure. Finally, we keep your rank-calibration unchanged and only slightly increase the weight on the improved drift feature (since your best-performing signal is already `scores2`-driven).'
- What this solution (achieved 0.48443) has done: 'We keep your exact pipeline (two per-snippet heuristic scores → rank-normalize → weighted blend → write `submission.csv`) but make the drift feature less lossy: right now it throws away most information by reducing each time row to a single argmax frequency before slope voting, which is too noisy and near-random. With a minimal change, we instead compute a small “ridge map” by averaging the top‑k frequencies per time row (still very fast), then do the same slope voting on that richer signal; this preserves the same semantics (drifting line evidence) but stabilizes the ranking. We also keep your score1 unchanged and only mildly adjust the blend weight toward the improved drift score (still dominated by drift, since that’s been your better signal). The output format, paths, determinism, and runtime constraints remain intact, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.48208) has done: 'Your current AUC is far below the target, so we should increase it with the smallest change that keeps your exact “two heuristic scores → rank-normalize → blend → write submission.csv” pipeline intact. The drift scorer is still dominated by noisy per-row maxima/top‑k, so I make it slightly more discriminative by adding a very light time-smoothing (3-tap) and by scoring “line support” using a top‑fraction mean of the Hough-like bins rather than a single max bin (reduces sensitivity to random spikes while keeping the same semantics). I also make the drift vote more stable by using a small fixed set of start-frequency anchors (still fast) and aggregating a narrow 3-bin neighborhood around each anchor. Finally, I keep `_snippet_score` unchanged and only slightly adjust the blend to give the improved drift feature a bit more weight (still within your current design).'
- What this solution (achieved 0.47964) has done: 'Your current score (0.482) is far below the target (0.763), so we should increase AUC with the smallest change that preserves your exact “two heuristic scores → rank-normalize → blend → write submission.csv” pipeline. The biggest low-risk gain is to make the drift heuristic less noisy and more aligned with the known ABACAD structure by explicitly using A panels (0/2/4) vs OFF panels (1/3/5) and adding a simple, robust “ON-only coherence” term (needles repeat across A panels) while keeping the existing slope-vote core intact. I also fix your accidental re-ensemble (cells 2–3) that currently just recombines identical copies (no effect) by directly passing through the already-computed blended prediction (this doesn’t change semantics, it just removes a no-op source of confusion). The output remains a valid `submission.csv` with `id,target`, same paths, deterministic behavior, and still a single pass over test `.npy` files.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_INPUT = "/kaggle/input"
COMP_DIR = os.path.join(BASE_INPUT, "seti-breakthrough-listen")
if not os.path.exists(COMP_DIR):
    COMP_DIR = "/kaggle/data/seti-breakthrough-listen"

sample_path = os.path.join(COMP_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = os.path.join("/kaggle/data", "sample_submission.csv")

sample_sub = pd.read_csv(sample_path)

test_dir = os.path.join(COMP_DIR, "test")
if not os.path.exists(test_dir):
    test_dir = os.path.join("/kaggle/data", "test")


def _find_npy_path(root, _id):
    sub = _id[:2]
    p = os.path.join(root, sub, f"{_id}.npy")
    if os.path.exists(p):
        return p
    for k in range(16):
        p2 = os.path.join(root, str(k), f"{_id}.npy")
        if os.path.exists(p2):
            return p2
    return None


def _snippet_score(arr):
    x = arr.astype(np.float32, copy=False)

    q99 = np.quantile(x.reshape(6, -1), 0.99, axis=1)
    std = x.reshape(6, -1).std(axis=1)

    a = q99[[0, 2, 4]].mean() + 0.5 * std[[0, 2, 4]].mean()
    b = q99[[1, 3, 5]].mean() + 0.5 * std[[1, 3, 5]].mean()

    return float(a - b)


def _snippet_score_drift(arr):
    """
    Minimal, semantics-preserving improvement (still a drift/line-evidence heuristic):
    - keep the slope-vote / "Hough-like" core intact
    - reduce spike noise as before (tiny smoothing, neighborhood anchors, top-fraction mean)
    - add a small ON-only "coherence" term across A panels (needles repeat in A0/A2/A4)
      while OFF panels should not show the same repeated structure.
    """
    x = arr.astype(np.float32, copy=False)  # (6, 273, 256)
    x = x - np.median(x, axis=(1, 2), keepdims=True)

    T, F = x.shape[1], x.shape[2]
    t = np.arange(T, dtype=np.int32)

    slopes = tuple(range(-18, 19, 2))  # -18..18 step 2
    best = np.empty(6, dtype=np.float32)

    k = 6
    n_anchors = 11

    a_profiles = {0: [], 2: [], 4: []}

    for p in range(6):
        panel = x[p]  # (T, F)

        sm = panel.copy()
        sm[:, 1:-1] = 0.25 * panel[:, :-2] + 0.50 * panel[:, 1:-1] + 0.25 * panel[:, 2:]
        panel = sm

        tm = panel.copy()
        tm[1:-1, :] = 0.25 * panel[:-2, :] + 0.50 * panel[1:-1, :] + 0.25 * panel[2:, :]
        panel = tm

        panel = np.maximum(panel, 0.0)

        track = panel.max(axis=1)  # (T,)
        w = track / (track.mean() + 1e-6)

        idx = np.argpartition(panel, -k, axis=1)[:, -k:].astype(np.int32)
        vals = np.take_along_axis(panel, idx, axis=1).astype(np.float32)

        row_sum = vals.sum(axis=1, keepdims=True) + 1e-6
        vals = vals / row_sum

        best_val = -1e9
        best_profile = None

        for s in slopes:
            z = idx - (s * t)[:, None]  # (T, k) in "intercept" coordinates
            zmin = int(z.min())
            zmax = int(z.max())
            nb = zmax - zmin + 1
            if nb <= 4:
                continue

            step = 1
            if nb > 512:
                step = 2

            bi = ((z - zmin) // step).astype(np.int32).reshape(-1)  # (T*k,)
            ww = (w[:, None] * vals).reshape(-1)

            bN = int(bi.max()) + 1
            if bN <= 4:
                continue

            bc = np.bincount(bi, weights=ww, minlength=bN).astype(np.float32)

            anchors = np.linspace(1, bN - 2, num=n_anchors).astype(np.int32)
            neigh = np.vstack([anchors - 1, anchors, anchors + 1]).T.reshape(-1)
            neigh = np.clip(neigh, 0, bN - 1)
            av = bc[neigh].reshape(n_anchors, 3).mean(axis=1)  # (n_anchors,)

            topm = max(2, int(0.30 * n_anchors))
            part = np.partition(av, -topm)[-topm:]
            val = float(part.mean())

            if val > best_val:
                best_val = val
                best_profile = (
                    av  # save the anchor profile for coherence (small vector)
                )

        best[p] = best_val

        if p in a_profiles and best_profile is not None:
            bp = best_profile.astype(np.float32)
            bp = bp - bp.mean()
            bp = bp / (bp.std() + 1e-6)
            a_profiles[p].append(bp)

    a = float(np.max(best[[0, 2, 4]]))
    b = float(np.max(best[[1, 3, 5]]))
    base = float(a - b)

    coh = 0.0
    if len(a_profiles[0]) == 1 and len(a_profiles[2]) == 1 and len(a_profiles[4]) == 1:
        p0 = a_profiles[0][0]
        p2 = a_profiles[2][0]
        p4 = a_profiles[4][0]
        c02 = float(np.dot(p0, p2) / (len(p0) + 1e-6))
        c04 = float(np.dot(p0, p4) / (len(p0) + 1e-6))
        c24 = float(np.dot(p2, p4) / (len(p0) + 1e-6))
        coh = (c02 + c04 + c24) / 3.0
        coh = max(-1.0, min(1.0, coh))

    return float(base + 0.15 * coh)


def _rank_to_proba(scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    proba = (ranks + 0.5) / float(len(scores))  # (0,1) open interval
    if not np.isfinite(proba).all():
        proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
    return proba


scores1 = np.empty(len(sample_sub), dtype=np.float32)
scores2 = np.empty(len(sample_sub), dtype=np.float32)
missing = 0

for i, _id in enumerate(sample_sub["id"].values):
    p = _find_npy_path(test_dir, _id)
    if p is None:
        scores1[i] = 0.0
        scores2[i] = 0.0
        missing += 1
        continue
    arr = np.load(p)
    scores1[i] = _snippet_score(arr)
    scores2[i] = _snippet_score_drift(arr)

proba1 = _rank_to_proba(scores1)
proba2 = _rank_to_proba(scores2)

proba = 0.12 * proba1 + 0.88 * proba2
proba = np.clip(proba, 0.0, 1.0)

sample_sub["target"] = proba

data1 = sample_sub.copy()
data2 = sample_sub.copy()
data3 = sample_sub.copy()
data4 = sample_sub.copy()
data5 = sample_sub.copy()
data6 = sample_sub.copy()
data7 = sample_sub.copy()

print("Computed blended heuristic predictions for test set.")
print("Test dir:", test_dir)
print("Missing files (should be 0):", missing)
print(
    "Pred summary:", float(np.min(proba)), float(np.mean(proba)), float(np.max(proba))
)



## === cell 1
data11 = data1.copy()



## === cell 2
data11 = data11[["id", "target"]]
data11["target"] = data11["target"].clip(0.0, 1.0)



## === cell 3
data11.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data11.shape)
print(data11.head())
