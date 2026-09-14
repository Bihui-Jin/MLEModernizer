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

0.7571668945402937

# 6. Current score

0.50038

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4991) has done: 'The timeout is dominated by loading tens of thousands of small `.npy` files and computing features one file at a time with high multiprocessing overhead and redundant per-file conversions. I keep the exact same features and model, but speed up feature extraction by (1) avoiding repeated Python/NumPy allocations inside `extract_features`, (2) using `np.load(..., mmap_mode="r")` + `np.asarray` to float32 only once per file, and (3) switching to a thread pool (I/O-bound workload) with larger chunking to reduce inter-process pickling/IPC overhead. I also eliminate O(N) existence checks over all files (which is pure overhead) while still asserting shard directories and sampling a few paths for safety. These changes preserve identical semantics (same per-file stats and same classifier training), but cut wall time significantly.'
- What this solution (achieved 0.4991) has done: 'Your current AUC (~0.499) indicates the predictions are effectively random, and the most likely minimal-cause is a systematic mismatch between `id`→file-path mapping and how the train/test files are actually sharded in this dataset. I keep the exact same feature set and LogisticRegression pipeline, but change `_build_paths`/`id_to_path` to locate files via a one-time shard index built from the existing directory structure (still using the same IDs from the CSVs), ensuring each ID loads the correct `.npy`. I also add strict checks that the resolved path’s basename matches the requested id and that we resolve all ids; this is directly score-relevant because wrong-file/wrong-label pairing destroys learnability and yields ~0.5 AUC. Everything else (features, model, training) remains unchanged, and the script still writes `submission.csv` in the required format.'
- What this solution (achieved 0.4991) has done: 'Your AUC being ~0.5 strongly suggests the model is learning on mismatched (id, file) pairs or extracting non-informative global stats; the smallest score-relevant fix is to correct the id→path resolution so it matches the competition’s actual shard layout. I keep the exact same feature vector (means/std/max/min per panel + the same 10 “A vs B” aggregates) and the same LogisticRegression pipeline, but replace the expensive/fragile full-directory glob index with a deterministic `id_to_path = root/shard(id)/id.npy` mapping (and verify it exists). This directly targets the most common cause of “random” AUC in this dataset: reading the wrong file for a given id due to indexing collisions/coverage issues. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.4991) has done: 'I fix the path-resolution logic so every `id` maps to the correct `.npy` file in the actual shard layout (your current assumption of exactly 16 shards is false here, causing massive missing files and label/file mismatches that lead to ~0.5 AUC). To keep changes minimal and score-relevant, I build a one-time index from the directory tree (id → filepath) for train and test, and then use that mapping in `_build_paths_from_ids` with strict checks. I also remove the failing “16 shard dirs must exist” assertion and keep the exact same feature extraction, threading-based featurization, and LogisticRegression pipeline. Finally, I ensure `submission.csv` is always written with the correct columns and 6000 rows.'
- What this solution (achieved 0.4991) has done: 'Your current ~0.499 AUC is consistent with systematic feature/label misalignment rather than “weak model”, so the smallest score-relevant change is to make the featurization order deterministic and provably aligned to the requested `id` list. I keep the exact same feature definitions and the same LogisticRegression pipeline, but change the threaded featurization to return `(id, feats)` and then place each row into `X` by `id→row_index` mapping (instead of relying on `imap` enumeration order matching `paths`). I also harden the disk index to detect duplicate IDs across shards (which can silently map an `id` to the wrong file) and assert 100% coverage for train/test IDs. These changes preserve the core logic and evaluation semantics, but directly address the most likely root cause of random AUC: wrong file paired with a label or prediction row.'
- What this solution (achieved 0.4991) has done: 'Your current AUC (~0.499) is consistent with a systematic train/test misalignment, and in this dataset the most common subtle cause is that IDs in the CSVs are zero-padded strings while pandas may parse them as non-padded/other types, causing incorrect path lookup even if you cast to `str`. I make the ID handling robust by enforcing string dtype at CSV read time and normalizing IDs (strip, lower) consistently for both the disk index and the label/sample IDs, while keeping the exact same features and LogisticRegression pipeline. I also add a strict full-coverage assertion (all train IDs resolve uniquely; all test IDs resolve) before featurization so we fail fast rather than silently training on wrong/missing files. These changes are minimal and directly score-relevant because they ensure each label is paired with the correct `.npy`, which should move performance upward toward your target.'
- What this solution (achieved 0.4991) has done: 'Your current AUC (~0.499) is consistent with a remaining subtle misalignment: the disk index is built from raw filenames, but the sample/train IDs are already canonical and the index-building is not deterministic; if any duplicate/near-duplicate or casing/whitespace variation slips through, you can silently map an `id` to the wrong file and destroy learnability. I make the id→path resolution strictly deterministic by (1) building the index in a sorted order, (2) asserting complete 1:1 mapping between CSV ids and basenames actually loaded, and (3) featurizing by iterating directly over the `ids` list (no id→row remap needed) to remove any possibility of overwriting rows when something unexpected happens. These are minimal, semantics-preserving changes (same features, same model), but they directly target the most likely cause of “random” AUC. The script still run end-to-end and write a valid `submission.csv` with the required 6000 rows.'
- What this solution (achieved 0.4991) has done: 'Your current ~0.499 AUC is most consistent with a remaining systematic mismatch (train features not paired to the correct labels), and in this dataset the most common subtle cause is “duplicate basenames across shards” where your index silently keeps the first occurrence. I keep your exact feature vector and LogisticRegression pipeline, but change the index to be built **only from the ids present in the CSVs** and to **fail fast on duplicates** (instead of silently choosing one), ensuring a strict 1:1 mapping from each requested id to exactly one on-disk file. I also make featurization return `(id, feats)` and place rows by an `id->row` lookup to remove any dependence on iterator ordering, while preserving the same computation per file. These are minimal, score-relevant alignment fixes and should move AUC upward toward your target.'
- What this solution (achieved 0.50038) has done: 'Your AUC being ~0.499 while the pipeline runs and writes a valid submission strongly suggests the model is learning “inverted” signal (or otherwise systematically mis-specified) rather than just being weak. The smallest semantics-preserving change that can move AUC upward without changing features or the model is to calibrate the prediction direction using an out-of-fold check on the training data: if the model’s training AUC is < 0.5, we flip probabilities `p -> 1-p` (equivalent to negating the decision function) for both validation and test predictions. This keeps the exact same feature extraction and LogisticRegression training, only adjusting post-processing in a metric-aligned way. I also add a deterministic CV AUC computation (no extra training loops beyond CV) and keep the submission format/paths unchanged.'
- What this solution (achieved 0.50038) has done: 'Your score (~0.500) is still essentially random, so the most likely remaining issue is train/test feature-label misalignment coming from the id→path index build scanning the whole shard tree and potentially mapping an `id` to the wrong file (or silently missing/duplicating) in a way that doesn’t crash. I keep the exact same features and LogisticRegression pipeline, but change the indexing to resolve each requested id via a targeted search over shard folders (fail-fast, 1:1 mapping) rather than building a global index from filenames. This is a minimal, score-relevant change that preserves core modeling semantics while making it much harder for a subtle mapping bug to destroy learnability. I also add a strict “loaded array basename == requested id” check inside featurization to guarantee alignment without changing the feature values.'
- What this solution (achieved 0.50038) has done: 'Your current ~0.50 AUC suggests the model is producing a consistent ranking but in the wrong direction or with a systematic mapping issue; since you already added an OOF “flip if AUC<0.5”, the next smallest score-relevant step is to (1) ensure we never silently select the wrong shard file for an id (by detecting duplicates across shards instead of “first match wins”), and (2) fit the final model on the same direction that maximizes OOF AUC by optionally negating the LogisticRegression coefficients (equivalent to flipping the decision function) rather than only flipping test probabilities. These keep the exact same feature extraction and LogisticRegression pipeline, but make the id→path mapping provably 1:1 and make the final fitted classifier consistent with the OOF-selected direction, which should move AUC upward toward your target. I also keep runtime bounded by building a per-root index once (sorted, duplicate-checked) and then resolving ids via dictionary lookup rather than per-id shard scanning.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

BASE_PATH = "/kaggle/input"
TRAIN_LABELS_PATH = f"{BASE_PATH}/train_labels.csv"
SAMPLE_SUB_PATH = f"{BASE_PATH}/sample_submission.csv"
TRAIN_DIR = f"{BASE_PATH}/train"
TEST_DIR = f"{BASE_PATH}/test"

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"


def _norm_id(x) -> str:
    return str(x).strip().lower()


train_labels = pd.read_csv(TRAIN_LABELS_PATH, dtype={"id": "string", "target": "int8"})
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": "string"})

train_labels["id"] = train_labels["id"].map(_norm_id)
sample_sub["id"] = sample_sub["id"].map(_norm_id)

train_labels.head(), sample_sub.head()



## === cell 1
A_IDX = (0, 2, 4)
B_IDX = (1, 3, 5)


def extract_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    means = x.mean(axis=(1, 2))
    stds = x.std(axis=(1, 2))
    maxs = x.max(axis=(1, 2))
    mins = x.min(axis=(1, 2))

    a_mean = (means[A_IDX[0]] + means[A_IDX[1]] + means[A_IDX[2]]) / 3.0
    b_mean = (means[B_IDX[0]] + means[B_IDX[1]] + means[B_IDX[2]]) / 3.0
    a_std = (stds[A_IDX[0]] + stds[A_IDX[1]] + stds[A_IDX[2]]) / 3.0
    b_std = (stds[B_IDX[0]] + stds[B_IDX[1]] + stds[B_IDX[2]]) / 3.0
    a_max = (maxs[A_IDX[0]] + maxs[A_IDX[1]] + maxs[A_IDX[2]]) / 3.0
    b_max = (maxs[B_IDX[0]] + maxs[B_IDX[1]] + maxs[B_IDX[2]]) / 3.0

    extra = np.array(
        [
            a_mean,
            b_mean,
            a_std,
            b_std,
            a_max,
            b_max,
            a_mean - b_mean,
            a_std - b_std,
            a_max - b_max,
            (a_max - a_mean) - (b_max - b_mean),
        ],
        dtype=np.float32,
    )

    feats = np.empty(6 * 4 + 10, dtype=np.float32)
    feats[0:6] = means
    feats[6:12] = stds
    feats[12:18] = maxs
    feats[18:24] = mins
    feats[24:34] = extra
    return feats


def build_id_to_path_index(root_dir: str) -> dict:
    shard_dirs = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )
    assert shard_dirs, f"No shard directories found under: {root_dir}"

    id_to_path = {}
    duplicates = []

    for shard in shard_dirs:
        shard_path = os.path.join(root_dir, shard)
        try:
            files = sorted(os.listdir(shard_path))
        except FileNotFoundError:
            continue
        for fn in files:
            if not fn.endswith(".npy"):
                continue
            _id = _norm_id(os.path.splitext(fn)[0])
            p = os.path.join(shard_path, fn)
            if _id in id_to_path:
                duplicates.append((_id, id_to_path[_id], p))
            else:
                id_to_path[_id] = p

    assert (
        not duplicates
    ), f"Duplicate ids across shards under {root_dir}. Ex: {duplicates[:3]}"
    return id_to_path


def resolve_paths_for_ids(root_dir: str, ids: np.ndarray, id_to_path: dict) -> list:
    out = [None] * len(ids)
    missing = []
    for i, raw_id in enumerate(ids.tolist()):
        _id = _norm_id(raw_id)
        p = id_to_path.get(_id)
        if p is None:
            missing.append(_id)
        out[i] = p

    assert (
        not missing
    ), f"Missing {len(missing)} ids on disk under {root_dir}. Ex: {missing[:5]}"

    for s, p in zip(ids[:64], out[:64]):
        assert _norm_id(os.path.splitext(os.path.basename(p))[0]) == _norm_id(s)

    return out


train_ids_all = train_labels["id"].astype(str).values
test_ids_all = sample_sub["id"].astype(str).values

assert len(set(train_ids_all.tolist())) == len(
    train_ids_all
), "Duplicate ids in train_labels.csv after normalization."
assert len(set(test_ids_all.tolist())) == len(
    test_ids_all
), "Duplicate ids in sample_submission.csv after normalization."

train_index = build_id_to_path_index(TRAIN_DIR)
test_index = build_id_to_path_index(TEST_DIR)

train_paths_all = resolve_paths_for_ids(TRAIN_DIR, train_ids_all, train_index)
test_paths_all = resolve_paths_for_ids(TEST_DIR, test_ids_all, test_index)

x0 = np.load(train_paths_all[0], mmap_mode="r")
x0.shape, x0.dtype, extract_features(x0).shape



## === cell 2
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

FEAT_DIM = 6 * 4 + 10


def _featurize_id_path(args):
    _id, p = args
    arr = np.load(p, mmap_mode="r")
    base = _norm_id(os.path.splitext(os.path.basename(p))[0])
    assert base == _id, f"Resolved wrong file: requested={_id}, got={base}, path={p}"
    return _id, extract_features(arr)


def _sanity_check_some_paths(paths: list, n: int = 32):
    if not paths:
        return
    step = max(len(paths) // n, 1)
    for p in paths[::step][:n]:
        assert p is not None and os.path.exists(p), f"Missing file (sampled check): {p}"


def featurize_ids_paths_aligned(
    ids: np.ndarray, paths: list, feat_dim: int, workers: int, chunksize: int
) -> np.ndarray:
    n = len(ids)
    X_local = np.empty((n, feat_dim), dtype=np.float32)
    if n == 0:
        return X_local

    row_of = {_norm_id(i): k for k, i in enumerate(ids.tolist())}
    pairs = [(_norm_id(i), p) for i, p in zip(ids.tolist(), paths)]

    with ThreadPool(processes=workers) as pool:
        for _id, feats in pool.imap(_featurize_id_path, pairs, chunksize=chunksize):
            X_local[row_of[_id]] = feats
    return X_local


train_ids = train_labels["id"].astype(str).values
y = train_labels["target"].astype(np.int32).values

train_paths = resolve_paths_for_ids(TRAIN_DIR, train_ids, train_index)
_sanity_check_some_paths(train_paths, n=32)

workers = min(max(cpu_count(), 1), 12)
chunksize = 256

X = featurize_ids_paths_aligned(
    train_ids, train_paths, FEAT_DIM, workers=workers, chunksize=chunksize
)
X.shape, y.shape, float(y.mean())



## === cell 3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)

clf.fit(X, y)



## === cell 4
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def _oof_auc_and_flip_flag(model, X, y, n_splits=5, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=np.float32)
    for tr, va in cv.split(X, y):
        m = Pipeline(steps=model.steps)  # same architecture/params
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1].astype(np.float32)
    auc = roc_auc_score(y, oof)
    return float(auc), bool(auc < 0.5)


oof_auc, flip_proba = _oof_auc_and_flip_flag(clf, X, y, n_splits=5, seed=42)
oof_auc, flip_proba



## === cell 5
if flip_proba:
    lr = clf.named_steps["lr"]
    lr.coef_ = -lr.coef_
    lr.intercept_ = -lr.intercept_

test_ids = sample_sub["id"].astype(str).values
test_paths = resolve_paths_for_ids(TEST_DIR, test_ids, test_index)
_sanity_check_some_paths(test_paths, n=32)

X_test = featurize_ids_paths_aligned(
    test_ids, test_paths, FEAT_DIM, workers=workers, chunksize=chunksize
)
proba = clf.predict_proba(X_test)[:, 1].astype(np.float32)



## === cell 6
sub = pd.DataFrame({"id": test_ids, "target": proba})
sub["target"] = sub["target"].clip(0.0, 1.0)
sub.to_csv("submission.csv", index=False)

sub.head(), sub.shape



## === cell 7
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["id", "target"]
assert len(check) == 6000
assert check["target"].between(0, 1).all()
check.describe(include="all")
