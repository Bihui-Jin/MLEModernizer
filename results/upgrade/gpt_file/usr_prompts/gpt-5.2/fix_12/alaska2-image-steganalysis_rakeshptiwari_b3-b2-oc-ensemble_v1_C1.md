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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.9112965725853808

# 6. Current score

0.59436

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58571) has done: 'The notebook fails because it tries to read many out-of-environment “../input/model-*” blended submission files that don’t exist here, so nothing downstream is defined and no CSV is written. I keep the same blending-style “produce a submission from existing predictions” core idea, but add a safe fallback: if those external blend files are missing, build a valid submission directly from the provided `sample_submission.csv` (neutral baseline) so a `.csv` is always produced end-to-end. I also make the code robust to different Kaggle directory layouts by probing the known dataset locations you listed. This is score-neutral/low (since current score is not yielded) but guarantees a valid submission file is created.'
- What this solution (achieved 0.58571) has done: 'Your current code falls back to an all-zero submission whenever any blended file is missing, which likely explains the low 0.58571 score; the smallest improvement is to blend whatever submission files are actually present instead of requiring all of them. I keep the same “average-blend existing submissions” core logic, but change the gating to use all available files (with equal weights) and only fall back to zeros if none exist. I also make the merge robust to partial coverage by filling missing predictions with the mean label from the available submissions (instead of 0.0), which typically improves AUC without changing the overall approach. The output path and required `Id,Label` format remain unchanged, and the script still run end-to-end and always write a valid `.csv`.'
- What this solution (achieved 0.58571) has done: 'Your score is low mainly because cell 3 accidentally overwrites `final_sub` and then merges from `merged` (which is not the blended output) and fills missing with 0.0—this effectively collapses predictions toward zeros and harms AUC. I make a minimal fix: keep the blended `final_sub` produced in cell 2 and only align it to `sample_sub`’s Id order (no re-merge from the wide `merged` table). I also change the “no blend files available” fallback from all-zeros to a constant equal to the sample’s mean label (usually 0.5), which is a safe, small AUC lift over zeros without changing the core “blend submissions” approach. The script still run end-to-end and always write a valid `Id,Label` CSV.'
- What this solution (achieved 0.57494) has done: 'Your current blend code is correct structurally, but in this environment none of the referenced `../input/model-*` blend files exist, so you always fall back to a constant prediction (essentially random ranking), which explains the low AUC. To move the score toward the 0.911 target without changing the “blend existing submissions” core idea, I add a minimal, legitimate fallback that generates non-constant, image-specific scores directly from the provided JPEG bytes (a simple steganalysis prior) when no blend files are found. This keeps the rest of your pipeline intact: it still uses `sample_submission.csv` as the Id source, preserves the final alignment/merge, and writes the same submission filename. The heuristic uses per-image compressed-size and a couple of byte-entropy/compressibility proxies to create a ranking signal, which should substantially improve AUC versus a constant baseline while staying lightweight and within the time limit.'
- What this solution (achieved 0.57557) has done: 'Your current score is far below the target because none of the external blend files exist here, so you’re effectively relying on a very weak heuristic built from a small prefix of JPEG bytes. To move the score upward without changing the overall “fallback heuristic submission” core idea, I (1) compute features from the full file (not just the first 120KB), and (2) add a couple of lightweight, image-specific JPEG-tail / marker statistics (near-EOI bytes and count of common JPEG markers) that tend to correlate with embedding artifacts. I keep the same normalization + linear scoring approach and preserve the same submission writing/alignment logic. This should create a stronger ranking signal (better AUC) while staying within the runtime limit for 5k test images.'
- What this solution (achieved 0.57663) has done: 'Your current score is far below the target, and in this environment you’re almost certainly using the heuristic path (no blend files found), so the smallest meaningful improvement is to strengthen that heuristic without changing the overall “generate image-specific scores from JPEG bytes” core idea. I keep the same feature-extraction approach (byte-level stats + normalization + linear scoring), but add a few additional lightweight JPEG-specific signals that often improve ranking: (1) separate header vs tail entropy/compressibility, (2) estimate quantization-table “strength” from DQT segments, and (3) count restart markers (RST) density. I also switch the final score to a rank-based blend of the individual normalized features (still deterministic, still a simple post-processing) to better match AUC’s ranking nature while keeping evaluation semantics identical. The script still run end-to-end and write the same submission CSV.'
- What this solution (achieved 0.59663) has done: 'Your current score is far below the target and (given the missing blend files) you’re effectively relying on the JPEG-byte heuristic, so the smallest way to move toward 0.911 is to strengthen the heuristic’s *ranking* signal without changing its overall approach. I keep the same “extract lightweight byte-level/JPEG-marker stats → rank-normalize → linear score” core logic, but add a robust DHT (Huffman table) size feature and a small DQT-vs-DHT interaction feature, which often improves separation between cover/stego at fixed JPEG quality. I also replace the fixed hand weights with a deterministic PCA-first-component aggregation over the rank-normalized features (still purely unsupervised, still AUC-friendly because it’s rank-based), which typically improves ranking while staying lightweight and within the time limit. Submission writing/format and path probing remain unchanged.'
- What this solution (achieved 0.59482) has done: 'We keep your exact “blend existing submissions if present, otherwise generate an unsupervised JPEG-byte heuristic” core logic, but strengthen the heuristic ranking signal (since your environment likely has 0 blend files, making the fallback dominate and keeping score far below target). The minimal change is to add a few additional, still-lightweight JPEG-structure features that are commonly informative for steganalysis on ALASKA2 (APP/COM segment sizes, SOS scan length, and a coarse “JPEG quality class” inferred from DQT strength), then keep the same rank-normalize → PCA(PC1) aggregation. We also make the DQT/DHT parsing slightly more robust by scanning a bit further (still fast on 5k images) to reduce missing/zero features that can harm ranking. The submission format, Id alignment, and output filename remain unchanged.'
- What this solution (achieved 0.5944) has done: 'Your score is far below the 0.9113 target, so we should improve the fallback heuristic (since no blend files exist here) while keeping the exact same overall pipeline: “blend if available else unsupervised JPEG-byte heuristic → rank-normalize → PCA(PC1) → write submission”. The minimal, high-leverage change is to add a couple more *JPEG-structure* features that are still fast to extract (SOF image dimensions, number of DQT/DHT segments, and scan-header length), because these stabilize the PCA ranking signal across QF {75,90,95}. I also make the SOS scan-length estimation slightly more robust (scan a bit further, still capped) to reduce zero/NaN features that can hurt ranking/AUC. Submission formatting, paths, and the core PCA-on-ranked-features logic remain unchanged.'
- What this solution (achieved 0.59432) has done: 'You’re far below the 0.9113 target (0.5944), so we should cautiously improve the fallback heuristic (since no blend files exist here) while keeping the same pipeline: “blend if present else JPEG-byte heuristic → rank-normalize → PCA(PC1) → submission”. The smallest high-leverage change is to add one more very cheap but informative JPEG-structure feature: an estimate of the *number of non-zero AC coefficients* in the image by partially parsing the entropy-coded scan and counting non-zero coefficient events (the payload is defined per non-zero AC DCT coefficient, so this directly stabilizes the ranking). We keep everything unsupervised and ranking-based (still AUC-friendly), we keep the same PCA aggregation, and we cap the scan to stay within the 600s limit for 5k images. Submission alignment/format and output filename remain unchanged.'
- What this solution (achieved 0.59436) has done: 'We’re far below the target (0.59432 vs 0.91130), and in this environment you’re almost certainly using the heuristic path (no external blend files), so the smallest meaningful improvement is to strengthen the *ranking signal* of that heuristic without changing the overall pipeline (JPEG-byte feature extraction → rank-normalize → PCA(PC1) → submission). I keep all existing features and add two very cheap, JPEG-structure features that often correlate with embedding artifacts: the number of “stuffed” 0xFF00 sequences inside the scan, and the density of restart markers (already counted globally) but specifically within the scan region. These are extracted from the same already-read JPEG bytes, keep everything unsupervised, and only change the feature vector + PCA input dimensionality (core logic unchanged). The submission format, Id alignment, and output filename remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import zlib
from collections import Counter




## === cell 1
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _read_submission_or_none(path):
    if path is None or (not os.path.exists(path)):
        return None
    df = pd.read_csv(path)
    if "Id" not in df.columns:
        for c in ["image_id", "ImageId", "id"]:
            if c in df.columns:
                df = df.rename(columns={c: "Id"})
                break
    if "Label" not in df.columns:
        for c in ["label", "pred", "prediction", "prob"]:
            if c in df.columns:
                df = df.rename(columns={c: "Label"})
                break
    if not {"Id", "Label"}.issubset(df.columns):
        raise ValueError(
            f"Submission file {path} does not contain required columns Id, Label. Found: {df.columns.tolist()}"
        )
    return df[["Id", "Label"]].copy()


blend_paths = {
    "sub_10": "../input/model-1-fold0/submission_fold0_epoch35.csv",
    "sub_20": "../input/model-2-fold0/submission_fold0_epoch38.csv",
    "sub_30": "../input/model-3-fold0/submission_fold0_epoch39.csv",
    "sub_11": "../input/model-1-fold1/submission_fold1_epoch_30.csv",
    "sub_21": "../input/model-2-fold1/submission_fold1_epoch_34.csv",
    "sub_31": "../input/model-3-fold1/submission_fold1_epoch_38.csv",
    "sub_12": "../input/model-1-fold2/submission_fold2_epoch_34.csv",
    "sub_22": "../input/model-2-fold2/submission_fold2_epoch_37.csv",
    "sub_32": "../input/model-3-fold2/submission_fold2_epoch_39.csv",
    "sub_13": "../input/model-1-fold3/submission_fold3_epoch_35.csv",
    "sub_23": "../input/model-2-fold3/submission_fold3_epoch_36.csv",
    "sub_33": "../input/model-3-fold3/submission_fold3_epoch_39.csv",
    "sub_14": "../input/model-1-fold4/submission_fold4_epoch_33.csv",
    "sub_24": "../input/model-2-fold4/submission_fold4_epoch_35.csv",
    "sub_34": "../input/model-3-fold4/submission_fold4_epoch_36.csv",
    "sub_b3_13": "../input/b3-fold3-m1/submission_fold3_b3_m1.csv",
    "sub_b3_23": "../input/b3-fold3-m2/submission_fold3_b3_m2.csv",
    "sub_b3_33": "../input/b3-fold3-m3/submission_fold3_b3_m3.csv",
    "sub_b3_10": "../input/b3-fold0-m1/submission_fold0_b3_m1.csv",
    "sub_b3_20": "../input/b3-fold0-m2/submission_fold0_b3_m2.csv",
    "sub_b3_30": "../input/b3-fold0-m3/submission_fold0_b3_m3.csv",
    "sub_oc_1": "../input/oc-m1/submission_openclose_m1.csv",
    "sub_oc_2": "../input/oc-m2/submission_openclose_m2.csv",
    "sub_oc_3": "../input/oc-m3/submission_openclose_m3.csv",
}

subs = {}
missing = []
for k, p in blend_paths.items():
    df = _read_submission_or_none(p)
    if df is None:
        missing.append(p)
    subs[k] = df

sample_path = _first_existing(
    [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/alaska2-image-steganalysis/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/alaska2-image-steganalysis/sample_submission.csv",
        "../input/sample_submission.csv",
        "../input/alaska2-image-steganalysis/sample_submission.csv",
    ]
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in the expected Kaggle paths."
    )

sample_sub = pd.read_csv(sample_path)
if not {"Id", "Label"}.issubset(sample_sub.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns. Found: {sample_sub.columns.tolist()}"
    )

print(
    f"Found {len(subs) - len(missing)}/{len(subs)} blend files. Missing: {len(missing)}"
)




## === cell 2
def _locate_test_dir():
    return _first_existing(
        [
            "/kaggle/input/alaska2-image-steganalysis/Test",
            "/kaggle/input/Test",
            "/kaggle/data/alaska2-image-steganalysis/Test",
            "/kaggle/data/Test",
            "../input/alaska2-image-steganalysis/Test",
            "../input/Test",
        ]
    )


def _entropy_from_bytes(b):
    n = len(b)
    if n <= 0:
        return 0.0
    cnt = Counter(b)
    p = np.fromiter((v / n for v in cnt.values()), dtype=np.float64)
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _zratio_from_bytes(b, level=6):
    n = len(b)
    if n <= 0:
        return 0.0
    z = zlib.compress(b, level=level)
    return float(len(z) / n)


def _parse_dqt_strength(jpeg_bytes, max_scan_bytes=262144):
    """
    Scan further for DQT markers to reduce zeros and stabilize ranking features for PCA/AUC.
    """
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 4:
        return 0.0

    i = 0
    vals = []
    while i + 4 <= n:
        if b[i] == 0xFF and b[i + 1] == 0xDB:
            seg_len = (b[i + 2] << 8) + b[i + 3]
            seg_start = i + 4
            seg_end = i + 2 + seg_len
            if seg_len < 3 or seg_end > n:
                i += 2
                continue
            j = seg_start
            while j < seg_end:
                pq_tq = b[j]
                pq = (pq_tq >> 4) & 0x0F
                j += 1
                if pq == 0:
                    if j + 64 <= seg_end:
                        vals.extend(b[j : j + 64])
                        j += 64
                    else:
                        break
                elif pq == 1:
                    if j + 128 <= seg_end:
                        for k in range(64):
                            vals.append((b[j + 2 * k] << 8) + b[j + 2 * k + 1])
                        j += 128
                    else:
                        break
                else:
                    break
            i = seg_end
        else:
            i += 1

    if not vals:
        return 0.0
    return float(np.mean(np.array(vals, dtype=np.float64)))


def _parse_dht_total_len_and_count(jpeg_bytes, max_scan_bytes=524288):
    """
    Also return DHT segment count (not just total length).
    """
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 4:
        return (0.0, 0.0)

    i = 0
    total = 0
    cnt = 0
    while i + 4 <= n:
        if b[i] == 0xFF and b[i + 1] == 0xC4:
            seg_len = (b[i + 2] << 8) + b[i + 3]
            seg_end = i + 2 + seg_len
            if seg_len >= 2 and seg_end <= n:
                total += seg_len
                cnt += 1
                i = seg_end
            else:
                i += 2
        else:
            i += 1
    return (float(total), float(cnt))


def _parse_dqt_count(jpeg_bytes, max_scan_bytes=262144):
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 4:
        return 0.0
    i = 0
    cnt = 0
    while i + 2 <= n:
        if b[i] == 0xFF and b[i + 1] == 0xDB:
            cnt += 1
        i += 1
    return float(cnt)


def _parse_app_com_stats(jpeg_bytes, max_scan_bytes=262144):
    """
    Returns: (app_total_len, com_total_len, n_app_segs, n_com_segs)
    """
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 4:
        return (0.0, 0.0, 0.0, 0.0)

    i = 0
    app_total = 0
    com_total = 0
    n_app = 0
    n_com = 0

    while i + 4 <= n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if (0xE0 <= m <= 0xEF) or (m == 0xFE):
            seg_len = (b[i + 2] << 8) + b[i + 3]
            seg_end = i + 2 + seg_len
            if seg_len >= 2 and seg_end <= n:
                if 0xE0 <= m <= 0xEF:
                    app_total += seg_len
                    n_app += 1
                else:
                    com_total += seg_len
                    n_com += 1
                i = seg_end
            else:
                i += 2
        else:
            i += 1

    return (float(app_total), float(com_total), float(n_app), float(n_com))


def _estimate_sos_scan_len(jpeg_bytes, max_scan_bytes=2097152):
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 4:
        return 0.0

    sos = b.find(b"\xFF\xDA")
    if sos < 0 or sos + 4 > n:
        return 0.0
    seg_len = (b[sos + 2] << 8) + b[sos + 3]
    scan_start = sos + 2 + seg_len
    if scan_start >= n:
        return 0.0

    eoi = b.find(b"\xFF\xD9", scan_start)
    if eoi < 0:
        eoi = n
    return float(max(0, eoi - scan_start))


def _parse_scan_header_len(jpeg_bytes, max_scan_bytes=1048576):
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 4:
        return 0.0
    sos = b.find(b"\xFF\xDA")
    if sos < 0 or sos + 4 > n:
        return 0.0
    seg_len = (b[sos + 2] << 8) + b[sos + 3]
    return float(seg_len)


def _parse_sof_dims(jpeg_bytes, max_scan_bytes=262144):
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 10:
        return (0.0, 0.0)

    i = 0
    while i + 9 <= n:
        if b[i] == 0xFF:
            m = b[i + 1]
            if m in (0xC0, 0xC2):  # SOF0 baseline, SOF2 progressive
                seg_len = (b[i + 2] << 8) + b[i + 3]
                seg_end = i + 2 + seg_len
                if seg_len >= 8 and seg_end <= n:
                    h = (b[i + 5] << 8) + b[i + 6]
                    w = (b[i + 7] << 8) + b[i + 8]
                    return (float(h), float(w))
        i += 1
    return (0.0, 0.0)


def _jpeg_quality_class(dqt_strength):
    s = float(dqt_strength)
    if not np.isfinite(s) or s <= 0:
        return 0.0
    if s > 28:
        return 0.0  # likely Q75
    if s > 16:
        return 1.0  # likely Q90
    return 2.0  # likely Q95


def _estimate_nz_ac_events(jpeg_bytes, max_scan_bytes=262144):
    """
    Estimate number of non-zero AC coefficient "events" from scan bytes (lightweight proxy).
    """
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 4:
        return 0.0

    sos = b.find(b"\xFF\xDA")
    if sos < 0 or sos + 4 > n:
        return 0.0
    seg_len = (b[sos + 2] << 8) + b[sos + 3]
    scan_start = sos + 2 + seg_len
    if scan_start >= n:
        return 0.0
    eoi = b.find(b"\xFF\xD9", scan_start)
    if eoi < 0:
        eoi = n
    scan = b[scan_start:eoi]
    if len(scan) == 0:
        return 0.0

    nz = 0
    i = 0
    L = len(scan)
    while i < L:
        v = scan[i]
        if v == 0xFF:
            if i + 1 < L and scan[i + 1] == 0x00:
                i += 2
                continue
            if i + 1 < L and 0xD0 <= scan[i + 1] <= 0xD7:  # restart marker
                i += 2
                continue
            break
        else:
            if v != 0x00:
                nz += 1
            i += 1

    return float(nz)


def _scan_stuff_rst_stats(jpeg_bytes, max_scan_bytes=1048576):
    """
    CHANGE (score ↑ toward target): add two scan-region structure features:
    - stuffed_ff00_density: count of 0xFF00 byte-stuffing pairs per scan byte
    - scan_rst_density: count of RST markers (FFD0..FFD7) per scan byte
    These are cheap to compute and can improve ranking signal across JPEG QFs.
    """
    b = jpeg_bytes[: min(len(jpeg_bytes), max_scan_bytes)]
    n = len(b)
    if n < 4:
        return (0.0, 0.0)

    sos = b.find(b"\xFF\xDA")
    if sos < 0 or sos + 4 > n:
        return (0.0, 0.0)
    seg_len = (b[sos + 2] << 8) + b[sos + 3]
    scan_start = sos + 2 + seg_len
    if scan_start >= n:
        return (0.0, 0.0)

    eoi = b.find(b"\xFF\xD9", scan_start)
    if eoi < 0:
        eoi = n
    scan = b[scan_start:eoi]
    L = len(scan)
    if L <= 1:
        return (0.0, 0.0)

    stuffed = 0
    rst = 0
    i = 0
    while i + 1 < L:
        if scan[i] == 0xFF:
            nxt = scan[i + 1]
            if nxt == 0x00:
                stuffed += 1
                i += 2
                continue
            if 0xD0 <= nxt <= 0xD7:
                rst += 1
                i += 2
                continue
        i += 1

    stuffed_density = float(stuffed / L)
    rst_density = float(rst / L)
    return (stuffed_density, rst_density)


def _jpeg_byte_features(file_path, tail_bytes=4096, head_bytes=65536):
    with open(file_path, "rb") as f:
        b = f.read()
    n = len(b)
    if n == 0:
        return (0.0,) * 24

    head = b[: min(head_bytes, n)]
    tail = b[-min(tail_bytes, n) :]

    cnt = Counter(b)
    ff_ratio = float(cnt.get(0xFF, 0) / n)
    zero_ratio = float(cnt.get(0x00, 0) / n)
    ff0_ratio = ff_ratio + zero_ratio

    ent_all = _entropy_from_bytes(b)
    ent_head = _entropy_from_bytes(head)
    ent_tail = _entropy_from_bytes(tail)

    z_all = _zratio_from_bytes(b, level=6)
    z_head = _zratio_from_bytes(head, level=6)
    z_tail = _zratio_from_bytes(tail, level=6)

    eoi_cnt = float(b.count(b"\xFF\xD9"))
    sos_cnt = float(b.count(b"\xFF\xDA"))
    rst_cnt = 0
    for m in range(0xD0, 0xD8):
        rst_cnt += b.count(bytes([0xFF, m]))
    rst_cnt = float(rst_cnt)

    marker_score = float((eoi_cnt + sos_cnt) / max(1.0, n / 10000.0))
    rst_score = float(rst_cnt / max(1.0, n / 10000.0))

    dqt_strength = _parse_dqt_strength(b)
    dht_total, dht_cnt = _parse_dht_total_len_and_count(b)
    dqt_cnt = _parse_dqt_count(b)
    dqt_dht_ratio = float(dqt_strength / (dht_total + 1.0))

    app_total, com_total, n_app, n_com = _parse_app_com_stats(b)
    sos_scan_len = _estimate_sos_scan_len(b)
    scan_hdr_len = _parse_scan_header_len(b)
    h, w = _parse_sof_dims(b)
    qclass = _jpeg_quality_class(dqt_strength)

    nz_ac = _estimate_nz_ac_events(b)
    nz_ac_norm = float(nz_ac / max(1.0, sos_scan_len))

    stuffed_density, scan_rst_density = _scan_stuff_rst_stats(b)

    return (
        float(n),
        float(ent_all),
        float(z_all),
        float(ff0_ratio),
        float(ent_tail),
        float(marker_score),
        float(ent_head),
        float(z_head),
        float(z_tail),
        float(rst_score),
        float(dqt_strength),
        float(dqt_dht_ratio),
        float(dht_total),
        float(app_total),
        float(com_total),
        float(sos_scan_len / max(1.0, n)),
        float(qclass),
        float(dqt_cnt),
        float(dht_cnt),
        float(scan_hdr_len),
        float((h * w) / 1e6),
        float(nz_ac_norm),
        float(stuffed_density),
        float(scan_rst_density),
    )


def _make_heuristic_submission(sample_sub_df):
    test_dir = _locate_test_dir()
    if test_dir is None:
        ids = sample_sub_df["Id"].astype(str).values
        h = np.array([hash(x) % 10_000 for x in ids], dtype=np.float64)
        h = (h - h.min()) / (h.max() - h.min() + 1e-12)
        out = sample_sub_df[["Id"]].copy()
        out["Label"] = h
        return out

    ids = sample_sub_df["Id"].astype(str).values
    feats = np.zeros((len(ids), 24), dtype=np.float64)

    for i, img_id in enumerate(ids):
        fp = os.path.join(test_dir, img_id)
        if not os.path.exists(fp):
            if (not img_id.lower().endswith(".jpg")) and os.path.exists(fp + ".jpg"):
                fp = fp + ".jpg"
            else:
                feats[i, :] = 0.0
                continue
        feats[i, :] = np.array(_jpeg_byte_features(fp), dtype=np.float64)

    def _rank01(x):
        x = np.asarray(x, dtype=np.float64)
        mask = np.isfinite(x)
        r = np.zeros_like(x, dtype=np.float64)
        if mask.sum() <= 1:
            return r
        idx = np.where(mask)[0]
        order = np.argsort(x[idx], kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(order.size, dtype=np.float64)
        r[idx] = ranks / (order.size - 1.0)
        return r

    Xr = np.column_stack([_rank01(feats[:, j]) for j in range(feats.shape[1])]).astype(
        np.float64
    )

    X = Xr - Xr.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(1.0, (X.shape[0] - 1.0))
    evals, evecs = np.linalg.eigh(cov)
    v1 = evecs[:, np.argmax(evals)]
    pc1 = X @ v1

    score = _rank01(pc1)
    score = np.clip(score, 0.0, 1.0)

    out = sample_sub_df[["Id"]].copy()
    out["Label"] = score.astype(float)
    return out


available_items = [(k, df) for k, df in subs.items() if df is not None]

if len(available_items) > 0:
    cleaned = []
    for k, df in available_items:
        d = df.copy()
        d["Id"] = d["Id"].astype(str)
        d["Label"] = pd.to_numeric(d["Label"], errors="coerce")
        cleaned.append((k, d.sort_values("Id").reset_index(drop=True)))

    merged = sample_sub[["Id"]].copy()
    merged["Id"] = merged["Id"].astype(str)

    for k, d in cleaned:
        merged = merged.merge(
            d.rename(columns={"Label": k}),
            on="Id",
            how="left",
            validate="one_to_one",
        )

    pred_cols = [k for k, _ in cleaned]

    row_mean = merged[pred_cols].mean(axis=1, skipna=True)
    global_mean = (
        float(row_mean.mean(skipna=True))
        if np.isfinite(row_mean.mean(skipna=True))
        else 0.0
    )

    for c in pred_cols:
        merged[c] = merged[c].fillna(row_mean).fillna(global_mean)

    merged["Label"] = merged[pred_cols].mean(axis=1).astype(float)

    final_sub = merged[["Id", "Label"]].copy()
else:
    final_sub = _make_heuristic_submission(sample_sub)

final_sub["Id"] = final_sub["Id"].astype(str)
final_sub["Label"] = pd.to_numeric(final_sub["Label"], errors="coerce").astype(float)

final_sub = (
    sample_sub[["Id"]]
    .copy()
    .assign(Id=sample_sub["Id"].astype(str))
    .merge(
        final_sub[["Id", "Label"]],
        on="Id",
        how="left",
        validate="one_to_one",
    )
)

fill_value = float(pd.to_numeric(final_sub["Label"], errors="coerce").mean())
if not np.isfinite(fill_value):
    fill_value = 0.5
final_sub["Label"] = (
    pd.to_numeric(final_sub["Label"], errors="coerce").fillna(fill_value).astype(float)
)

out_path = "submission_Jul20_blend_b3_b2_oc.csv"
final_sub.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(final_sub.head())
print(f"Rows: {len(final_sub)}, Columns: {final_sub.columns.tolist()}")
print(
    f"Label stats: min={final_sub['Label'].min():.6f}, max={final_sub['Label'].max():.6f}, mean={final_sub['Label'].mean():.6f}"
)
