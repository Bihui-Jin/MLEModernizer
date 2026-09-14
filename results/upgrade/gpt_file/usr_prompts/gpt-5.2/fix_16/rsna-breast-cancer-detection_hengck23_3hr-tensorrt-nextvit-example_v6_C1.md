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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.4517647058823529

# 6. Current score

0.04718

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02342) has done: 'The main blocker is that the notebook depends on external Kaggle input modules (`dicom_reader.py`, `preprocess.py`, `nextvit.py`, and several pip-installed wheels) that are not available in your provided environment, so imports fail and `test_df` is never created. To make this run end-to-end without changing the overall “read test.csv → predict per image → group by prediction_id → write submission.csv” semantics, I replace the missing DICOM/PNG/model pipeline with a lightweight metadata-only probabilistic baseline and keep the submission formatting identical to the competition requirement. I also fix the incorrect cell numbering (your script starts at cell 0) and ensure all required imports are present so `pd`, `Dataset`, etc. are defined. Finally, the code always writes `submission.csv` with columns `prediction_id,cancer` and the correct number of rows matching `sample_submission.csv`.'
- What this solution (achieved 0.0412) has done: 'Your current metadata-only baseline is likely under-scoring because it mixes per-image “proxy risks” in a way that doesn’t match the metric’s need for sharp separation between positives and negatives. To move your pF1 upward toward the 0.45 target without changing the overall pipeline (read CSVs → compute per-image probabilities → aggregate by prediction_id → write submission), I (1) compute smoothed target rates at the *prediction_id* level for train and apply them to test via site/machine/view/laterality, (2) calibrate the final probabilities with a single temperature-like power transform chosen by GroupKFold cross-validation to directly maximize pF1 on out-of-fold predictions, and (3) keep the same submission formatting/alignment with `sample_submission.csv`. These are minimal, metric-aligned changes that keep the “metadata baseline” core logic intact while making probabilities better calibrated for pF1.'
- What this solution (achieved 0.04159) has done: 'Your current pipeline is already valid and score-limited by weak signal; the simplest metric-aligned improvement that preserves your “metadata-only → per-image probs → groupby prediction_id → submission.csv” core logic is to (1) add a *pairwise* smoothed target encoding for `(site_id, machine_id)` (often more informative than either alone) and `(site_id, view)` while keeping the existing single-column encodings, and (2) tune the linear blend weights and the smoothing strength `k` using the same GroupKFold OOF pF1 selection you already do for the power calibration. This keeps architecture/training semantics the same (still a calibrated smoothed-rate baseline), but should materially increase separation and move your public score upward toward the 0.45 target. I also keep submission alignment via `sample_submission.csv` unchanged and ensure deterministic behavior.'
- What this solution (achieved 0.04396) has done: 'I keep your metadata-only smoothed target-encoding baseline intact, but add two high-signal, low-risk encodings that fit the same framework: smoothed target rates for `site_id + laterality` and `site_id + machine_id + view`. Then I minimally extend your existing OOF pF1 grid-search to include these new encodings and re-tune the blend weights (still a simple linear blend of smoothed rates) so the probabilities better separate positives/negatives for pF1. Finally, I keep your current power-calibration step, but slightly widen the power grid (still deterministic, same calibration mechanism) to help pF1 without changing evaluation semantics or requiring any extra packages.'
- What this solution (achieved 0.04395) has done: 'Your current score (0.04396) is far below the target (0.45176), so we should push performance upward with minimal, metric-aligned changes while keeping your “smoothed target-encoding blend + power calibration + mean-by-prediction_id” core logic intact. The biggest likely issue is a train/test mismatch: you train encodings at the *breast (prediction_id)* level but apply them to *image-level* test rows (multiple rows share the same prediction_id), which can dilute signal and hurt pF1. I (1) aggregate test metadata to the same breast level before encoding/scoring, then (2) map the calibrated breast-level probability back onto all test rows so the final groupby mean is consistent (and effectively identity). This preserves your approach but fixes alignment, which is a high-impact correctness/performance improvement toward the target.'
- What this solution (achieved 0.04453) has done: 'Your current solution is valid but far below the target, so we should improve pF1 with minimal, metric-aligned tweaks that keep the same “smoothed target-encoding blend + power calibration + breast-level aggregation + map back to images + submission.csv” core logic. The biggest low-risk gain is to incorporate additional metadata interactions already present in train/test (notably `age` bins and `site_id+age_bin`, plus `site_id+implant`) into the same smoothed-rate encoding framework, then let your existing GroupKFold OOF pF1 selection re-tune blend weights and smoothing `k`. This increases separation while staying deterministic and within the same training/selection approach (no new model, no new loss). I also fix the cell numbering to start at 1 (Kaggle/nbconvert compatibility) without changing behavior.'
- What this solution (achieved 0.04441) has done: 'Your current score is far below the target, so we should cautiously increase pF1 while keeping your existing “breast-level smoothed target-encoding blend + GroupKFold OOF selection + power calibration + map back to images + submission.csv” pipeline intact. The most likely low-risk gain is to add a few more metadata interaction encodings that fit exactly the same framework (no new model), especially ones that capture device/technique differences: `site_id+machine_id+laterality`, `site_id+machine_id+implant`, and `site_id+view+laterality`. Then we minimally extend your existing OOF grid search to (a) include these new encodings and (b) select among a small set of weight vectors that allocate some mass to them, keeping determinism and runtime reasonable. Finally, we keep your submission alignment logic unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.04566) has done: 'Your current solution is already end-to-end valid, so the safest way to move pF1 upward (still far below the 0.45 target) is to keep the same smoothed target-encoding + OOF selection + power calibration pipeline, but add a small amount of extra signal that fits the exact same framework. Concretely, I add a few additional interaction encodings that are commonly strong for this dataset (`site_id+machine_id+view+laterality`, `site_id+laterality+view`, `machine_id+view`, and `site_id+machine_id+age_bin`) and minimally extend the existing OOF weight-grid search to allow allocating a little weight to them. I also add a tiny “bias shift” calibration (a constant logit offset) selected on OOF pF1 after the power is selected; this preserves the same calibration semantics (a deterministic, monotone transform) but often improves pF1 by matching the global positive rate better. The submission writing, alignment with `sample_submission.csv`, breast-level aggregation, and the overall approach remain unchanged.'
- What this solution (achieved 0.04566) has done: 'I keep your metadata-only smoothed target-encoding blend + GroupKFold OOF selection + (power,bias) calibration exactly as-is, but fix a subtle correctness issue that can heavily blunt signal: `age_norm` is computed only once, before train/test aggregation adds `age_bin`, and the test-side `age_norm` currently uses the train min/max without clipping (so out-of-range ages can inflate/deflate probabilities unpredictably). I (1) compute `age_norm` after filling ages and then **clip it to [0,1]** for both train and test, and (2) switch the age-bin edges to be slightly more stable by using quantiles but ensuring strictly increasing edges (otherwise `pd.cut` can create weird bins and reduce encoding quality). These are minimal, deterministic changes that preserve your core logic and should improve calibration/separation, moving pF1 upward toward the target. The submission writing and alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved 0.04566) has done: 'Your current score (0.04566) is far below the target (0.45176), so we should push pF1 upward while keeping the exact same “breast-level smoothed target-encoding blend + GroupKFold OOF selection + monotone (power,bias) calibration + map back to images + submission.csv” core pipeline. The smallest high-impact addition in this framework is to incorporate **a few more metadata interaction encodings** that often carry extra signal (without changing the modeling approach), and to let the existing OOF pF1 search pick how much to use them. Concretely, I add `site_id+view+age_bin`, `site_id+machine_id+view+age_bin`, and `site_id+laterality+age_bin`, then extend the weight grid with minimal additional candidates that allocate a small amount of mass to these new encodings while keeping runtime safe. Submission formatting/alignment remains identical.'
- What this solution (achieved 0.04569) has done: 'Your current pipeline is valid but still very weak relative to the 0.45 target, so the smallest safe way to push pF1 upward is to keep the exact same “breast-level smoothed target-encoding blend + GroupKFold OOF selection + monotone (power,bias) calibration + map back to images + submission.csv” core logic, while (1) adding one more high-signal interaction (`site_id + BIRADS`) that exists in train and (2) slightly improving the existing age effect by also including an explicitly smoothed `site_id+age_bin` and `machine_id+age_bin` signal more directly in the blend search. These additions stay within your existing encoding framework (no new model/training loop) and should increase separation/calibration for pF1. I also fix the notebook cell numbering to start at 1 (Kaggle compatibility) without changing behavior. The submission format/alignment remains identical and still writes `submission.csv`.'
- What this solution (achieved 0.04569) has done: 'Your current score is far below the target, so we should increase pF1 with minimal, metric-aligned adjustments while preserving the exact same smoothed target-encoding blend + GroupKFold OOF selection + (power,bias) monotone calibration pipeline. The most likely low-risk gain is that your OOF selection is optimizing raw pF1, but the final system uses calibrated probabilities; we instead select `(k, weights)` by pF1 *after* calibration (still using the same calibration mechanism), which better matches the leaderboard metric without changing the modeling approach. To keep runtime under control, we reuse the same folds, do a small inner grid over `power` and `bias` during selection (same grids you already use), and then refit once on full train and apply to test as before. Finally, we fix the notebook cell numbering to start at 1 for Kaggle compatibility, keeping I/O paths and submission formatting unchanged.'
- What this solution (achieved 0.04698) has done: 'We keep your exact metadata-only smoothed target-encoding blend + GroupKFold OOF selection + monotone (power,bias) calibration pipeline, but fix a key metric-mismatch: pF1 (like F1) is optimized by *sharper* probabilities, and your current calibration grids are likely too conservative for such a low-base-rate target. I minimally expand the calibration search space (still the same power/bias transforms, no new model) and add a very small final step that selects an additional monotone “sharpening” power on OOF predictions to directly maximize OOF pF1. This is deterministic, fast, and keeps evaluation semantics identical (probabilities, same aggregation, same submission format). The rest of the code (feature keys, smoothing, blend, breast-level aggregation, mapping back to images, and submission writing) stays unchanged.'
- What this solution (achieved 0.04728) has done: 'Your current approach is already a valid end-to-end metadata target-encoding baseline, but it likely under-scores because the pF1 metric rewards *high pTP while heavily penalizing pFP*, so the optimal solution often needs globally *smaller* probabilities with a sharper tail. I keep your exact encoding/blending/calibration pipeline, and add one minimal, deterministic post-processing step: learn a single global multiplicative scale (applied on the logit) on OOF predictions to better match pF1’s preference for low base-rate, without changing ranking/semantics. I also fix the cell numbering to start at 1 (Kaggle compatibility) and keep the submission alignment identical. All changes are small and only affect monotone calibration, so they’re directly aimed at moving the score upward toward your target.'
- What this solution (achieved 0.04718) has done: 'I keep your existing metadata target-encoding + GroupKFold OOF selection + monotone (power/bias/sharpen/logit-scale) calibration pipeline unchanged, but fix one key train/test mismatch that likely depresses pF1: `implant` is breast-level in some cases (notably site 1) and your current breast aggregation uses `max`, which can create inconsistent/noisy implant values between train and test. I switch breast-level `implant` aggregation to a deterministic mode/first strategy (same as other categorical fields) and add a tiny amount of smoothing in the age/implant multiplicative adjustment to prevent over-amplifying rare implant patterns that can increase pFP (hurting pF1). These are minimal, deterministic changes that preserve evaluation semantics and should push pF1 upward from ~0.047 toward your target. The script still run end-to-end and write a valid `submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_CSV = f"{DATA_DIR}/sample_submission.csv"

np.random.seed(42)

print("Using data dir:", DATA_DIR)
print(
    "Files exist:",
    os.path.exists(TRAIN_CSV),
    os.path.exists(TEST_CSV),
    os.path.exists(SAMPLE_SUB_CSV),
)


## === cell 1
mode = ["submit"]  # keep same default behavior
test_df = pd.read_csv(TEST_CSV)
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

print("train_df", train_df.shape)
print("test_df", test_df.shape)
print("sample_sub", sample_sub.shape)
print("test_df columns:", list(test_df.columns))
print("")




## === cell 2
def make_debug_submission(df, out_path="submission.csv", value=0.0):
    submit_df = pd.DataFrame(
        {"prediction_id": df["prediction_id"].values, "cancer": float(value)}
    )
    submit_df = (
        submit_df.groupby("prediction_id", sort=True)["cancer"].mean().reset_index()
    )
    submit_df.to_csv(out_path, index=False)
    print("Wrote", out_path, "shape:", submit_df.shape)
    return submit_df




## === cell 3
def _pf1_score(y_true, y_prob, eps=1e-12):
    """
    Probabilistic F1 from the competition description.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    pTP = float(np.sum(y_prob * y_true))
    pFP = float(np.sum(y_prob * (1.0 - y_true)))
    TP = float(np.sum(y_true))
    if TP <= 0:
        return 0.0

    pPrec = pTP / max(pTP + pFP, eps)
    pRec = pTP / max(TP, eps)
    return 2.0 * pPrec * pRec / max(pPrec + pRec, eps)


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _apply_power_calibration(p, power):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.clip(_sigmoid(_logit(p) / float(power)), 1e-6, 1.0 - 1e-6)


def _apply_bias_shift(p, bias):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.clip(_sigmoid(_logit(p) + float(bias)), 1e-6, 1.0 - 1e-6)


def _apply_sharpening_power(p, gamma):
    """
    Minimal, monotone post-calibration sharpening/flattening transform.
    This keeps the same "calibration by monotone transform" semantics, but can
    better match pF1 which often prefers sharper probabilities.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    g = float(gamma)
    a = np.power(p, g)
    b = np.power(1.0 - p, g)
    out = a / (a + b)
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def _apply_logit_scale(p, scale):
    """
    Change (only) the global probability scale in a monotone way:
      p' = sigmoid(logit(p) * scale)
    For low-prevalence targets and pF1, allowing scale < 1 can reduce pFP
    while keeping strong examples separable, often increasing pF1.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    s = float(scale)
    return np.clip(_sigmoid(_logit(p) * s), 1e-6, 1.0 - 1e-6)


def _mode_or_first(x):
    m = x.mode()
    if len(m):
        return m.iloc[0]
    return x.iloc[0]


def _aggregate_to_breast_level(df, is_train: bool):
    """
    Aggregate to breast (prediction_id) level to match the encoding target level,
    avoiding dilution when test has multiple images per prediction_id.

    Change (score-relevant, minimal): aggregate implant via mode/first instead of max.
    Reason: implant is partly patient-level (esp. site 1) and max over images can
    introduce inconsistent/noisy breast-level implant flags between train and test,
    which can increase pFP under pF1. Mode/first matches other categorical fields.
    """
    d = df.copy()
    if is_train:
        d["prediction_id"] = (
            d["patient_id"].astype(str) + "-" + d["laterality"].astype(str)
        )

    agg_dict = {
        "age": ("age", "median"),
        "implant": ("implant", _mode_or_first),  # changed from "max"
        "site_id": ("site_id", _mode_or_first),
        "machine_id": ("machine_id", _mode_or_first),
        "laterality": ("laterality", _mode_or_first),
        "view": ("view", _mode_or_first),
        "patient_id": ("patient_id", _mode_or_first),
    }
    if is_train:
        agg_dict["cancer"] = ("cancer", "max")
        if "BIRADS" in d.columns:
            agg_dict["BIRADS"] = ("BIRADS", _mode_or_first)

    out = d.groupby("prediction_id", as_index=False).agg(**agg_dict)
    return out


def build_metadata_baseline(train_df, test_df, n_splits=5):
    tr_breast = _aggregate_to_breast_level(train_df, is_train=True)
    te_breast = _aggregate_to_breast_level(test_df, is_train=False)

    required = [
        "age",
        "implant",
        "site_id",
        "machine_id",
        "laterality",
        "view",
        "patient_id",
        "prediction_id",
    ]
    for col in required:
        if col not in tr_breast.columns:
            raise ValueError(f"Missing required column in train breast table: {col}")
        if col not in te_breast.columns:
            raise ValueError(f"Missing required column in test breast table: {col}")

    age_median = float(tr_breast["age"].median())
    tr_breast["age"] = tr_breast["age"].fillna(age_median).astype(float)
    te_breast["age"] = te_breast["age"].fillna(age_median).astype(float)

    a_min, a_max = float(tr_breast["age"].min()), float(tr_breast["age"].max())
    if a_max <= a_min:
        a_max = a_min + 1.0
    tr_breast["age_norm"] = (tr_breast["age"] - a_min) / (a_max - a_min)
    te_breast["age_norm"] = (te_breast["age"] - a_min) / (a_max - a_min)
    tr_breast["age_norm"] = tr_breast["age_norm"].clip(0.0, 1.0)
    te_breast["age_norm"] = te_breast["age_norm"].clip(0.0, 1.0)

    tr_breast["implant"] = tr_breast["implant"].fillna(0).astype(int)
    te_breast["implant"] = te_breast["implant"].fillna(0).astype(int)

    global_rate = float(tr_breast["cancer"].mean())

    def add_pair_keys(df, age_bin_edges):
        df = df.copy()

        df["age_bin"] = pd.cut(df["age"], bins=age_bin_edges, include_lowest=True)
        df["age_bin"] = df["age_bin"].astype(str)

        df["site_machine"] = (
            df["site_id"].astype(str) + "_" + df["machine_id"].astype(str)
        )
        df["site_view"] = df["site_id"].astype(str) + "_" + df["view"].astype(str)
        df["site_lat"] = df["site_id"].astype(str) + "_" + df["laterality"].astype(str)
        df["site_machine_view"] = (
            df["site_id"].astype(str)
            + "_"
            + df["machine_id"].astype(str)
            + "_"
            + df["view"].astype(str)
        )

        df["site_implant"] = df["site_id"].astype(str) + "_" + df["implant"].astype(str)
        df["site_agebin"] = df["site_id"].astype(str) + "_" + df["age_bin"].astype(str)

        df["site_machine_lat"] = (
            df["site_id"].astype(str)
            + "_"
            + df["machine_id"].astype(str)
            + "_"
            + df["laterality"].astype(str)
        )
        df["site_view_lat"] = (
            df["site_id"].astype(str)
            + "_"
            + df["view"].astype(str)
            + "_"
            + df["laterality"].astype(str)
        )
        df["site_machine_implant"] = (
            df["site_id"].astype(str)
            + "_"
            + df["machine_id"].astype(str)
            + "_"
            + df["implant"].astype(str)
        )

        df["site_machine_view_lat"] = (
            df["site_id"].astype(str)
            + "_"
            + df["machine_id"].astype(str)
            + "_"
            + df["view"].astype(str)
            + "_"
            + df["laterality"].astype(str)
        )
        df["site_lat_view"] = (
            df["site_id"].astype(str)
            + "_"
            + df["laterality"].astype(str)
            + "_"
            + df["view"].astype(str)
        )
        df["machine_view"] = df["machine_id"].astype(str) + "_" + df["view"].astype(str)
        df["site_machine_agebin"] = (
            df["site_id"].astype(str)
            + "_"
            + df["machine_id"].astype(str)
            + "_"
            + df["age_bin"].astype(str)
        )

        df["site_view_agebin"] = (
            df["site_id"].astype(str)
            + "_"
            + df["view"].astype(str)
            + "_"
            + df["age_bin"].astype(str)
        )
        df["site_lat_agebin"] = (
            df["site_id"].astype(str)
            + "_"
            + df["laterality"].astype(str)
            + "_"
            + df["age_bin"].astype(str)
        )
        df["site_machine_view_agebin"] = (
            df["site_id"].astype(str)
            + "_"
            + df["machine_id"].astype(str)
            + "_"
            + df["view"].astype(str)
            + "_"
            + df["age_bin"].astype(str)
        )

        df["machine_agebin"] = (
            df["machine_id"].astype(str) + "_" + df["age_bin"].astype(str)
        )

        return df

    q = np.linspace(0, 1, 6)  # 5 bins
    edges = np.quantile(tr_breast["age"].values.astype(float), q)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.array(
            [tr_breast["age"].min(), tr_breast["age"].median(), tr_breast["age"].max()]
        )
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([0.0, 100.0])
    edges = edges.astype(float).copy()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-3

    tr_breast = add_pair_keys(tr_breast, edges)
    te_breast = add_pair_keys(te_breast, edges)

    def _smooth_series(col_series, stats_df, gr, k):
        mean = col_series.map(stats_df["mean"]).astype(float)
        cnt = col_series.map(stats_df["count"]).astype(float)
        post = (mean * cnt + gr * k) / (cnt + k)
        return post.fillna(gr)

    tr_breast["patient_group"] = tr_breast["patient_id"].astype(np.int64)
    gkf = GroupKFold(n_splits=n_splits)

    y_true = tr_breast["cancer"].values.astype(np.float64)

    k_grid = [50.0, 120.0, 200.0, 350.0]

    weight_grid = [
        (
            0.10,
            0.07,
            0.05,
            0.05,
            0.18,
            0.06,
            0.07,
            0.11,
            0.05,
            0.09,
            0.04,
            0.05,
            0.03,
            0.03,
            0.01,
            0.00,
            0.01,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,  # w_machine_agebin
            0.00,  # w_site_birads
        ),
        (
            0.10,
            0.07,
            0.05,
            0.05,
            0.17,
            0.06,
            0.07,
            0.11,
            0.05,
            0.09,
            0.04,
            0.05,
            0.03,
            0.03,
            0.01,
            0.01,
            0.01,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
        ),
        (
            0.10,
            0.07,
            0.05,
            0.05,
            0.17,
            0.06,
            0.07,
            0.08,
            0.05,
            0.10,
            0.04,
            0.05,
            0.03,
            0.03,
            0.01,
            0.01,
            0.01,
            0.01,
            0.00,
            0.00,
            0.00,
            0.01,
            0.00,
        ),
        (
            0.10,
            0.07,
            0.05,
            0.05,
            0.16,
            0.06,
            0.07,
            0.10,
            0.05,
            0.10,
            0.04,
            0.05,
            0.03,
            0.03,
            0.02,
            0.00,
            0.01,
            0.02,
            0.00,
            0.00,
            0.00,
            0.00,
            0.04,
        ),
        (
            0.10,
            0.07,
            0.05,
            0.05,
            0.15,
            0.06,
            0.07,
            0.08,
            0.05,
            0.10,
            0.04,
            0.05,
            0.03,
            0.03,
            0.02,
            0.01,
            0.01,
            0.02,
            0.02,
            0.01,
            0.01,
            0.02,
            0.03,
        ),
    ]

    enc_cols = [
        "site_id",
        "machine_id",
        "laterality",
        "view",
        "site_machine",
        "site_view",
        "site_lat",
        "site_machine_view",
        "age_bin",
        "site_agebin",
        "site_implant",
        "site_machine_lat",
        "site_view_lat",
        "site_machine_implant",
        "site_machine_view_lat",
        "site_lat_view",
        "machine_view",
        "site_machine_agebin",
        "site_view_agebin",
        "site_lat_agebin",
        "site_machine_view_agebin",
        "machine_agebin",
    ]

    has_birads = "BIRADS" in tr_breast.columns
    if has_birads:
        tr_breast["BIRADS"] = tr_breast["BIRADS"].fillna(-1).astype(int)
        site_to_birads = (
            tr_breast.groupby("site_id")["BIRADS"].agg(_mode_or_first).to_dict()
        )
        te_breast["BIRADS"] = (
            te_breast["site_id"].map(site_to_birads).fillna(-1).astype(int)
        )
        tr_breast["site_birads"] = (
            tr_breast["site_id"].astype(str) + "_" + tr_breast["BIRADS"].astype(str)
        )
        te_breast["site_birads"] = (
            te_breast["site_id"].astype(str) + "_" + te_breast["BIRADS"].astype(str)
        )
        enc_cols.append("site_birads")

    power_grid = [0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0]
    bias_grid = [
        -1.5,
        -1.0,
        -0.7,
        -0.5,
        -0.35,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.35,
        0.5,
        0.7,
    ]
    sharpen_grid = [0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
    logit_scale_grid = [0.35, 0.5, 0.7, 0.85, 1.0, 1.2]

    best_cfg = None
    best_pf1_final = -1.0
    best_power = 1.0
    best_bias = 0.0
    best_sharpen = 1.0
    best_logit_scale = 1.0

    for k in k_grid:
        for w in weight_grid:
            if abs((sum(w)) - 1.0) > 1e-9:
                continue

            (
                w_site,
                w_mach,
                w_lat,
                w_view,
                w_sm,
                w_sv,
                w_sl,
                w_smv,
                w_agebin,
                w_sab,
                w_simp,
                w_sml,
                w_svl,
                w_smi,
                w_smvl,
                w_slv,
                w_mv,
                w_sma,
                w_sva,
                w_sla,
                w_smva,
                w_mab,
                w_sbir,
            ) = w

            if not has_birads and w_sbir != 0.0:
                continue

            oof_raw = np.zeros(len(tr_breast), dtype=np.float64)

            for idx_tr, idx_va in gkf.split(
                tr_breast, tr_breast["cancer"], groups=tr_breast["patient_group"]
            ):
                tr_f = tr_breast.iloc[idx_tr]
                va_f = tr_breast.iloc[idx_va]

                gr = float(tr_f["cancer"].mean())

                stats = {}
                for c in enc_cols:
                    stats[c] = tr_f.groupby(c)["cancer"].agg(["mean", "count"])

                p_site = _smooth_series(va_f["site_id"], stats["site_id"], gr, k)
                p_mach = _smooth_series(va_f["machine_id"], stats["machine_id"], gr, k)
                p_lat = _smooth_series(va_f["laterality"], stats["laterality"], gr, k)
                p_view = _smooth_series(va_f["view"], stats["view"], gr, k)
                p_sm = _smooth_series(
                    va_f["site_machine"], stats["site_machine"], gr, k
                )
                p_sv = _smooth_series(va_f["site_view"], stats["site_view"], gr, k)
                p_sl = _smooth_series(va_f["site_lat"], stats["site_lat"], gr, k)
                p_smv = _smooth_series(
                    va_f["site_machine_view"], stats["site_machine_view"], gr, k
                )
                p_agebin = _smooth_series(va_f["age_bin"], stats["age_bin"], gr, k)
                p_sab = _smooth_series(va_f["site_agebin"], stats["site_agebin"], gr, k)
                p_simp = _smooth_series(
                    va_f["site_implant"], stats["site_implant"], gr, k
                )
                p_sml = _smooth_series(
                    va_f["site_machine_lat"], stats["site_machine_lat"], gr, k
                )
                p_svl = _smooth_series(
                    va_f["site_view_lat"], stats["site_view_lat"], gr, k
                )
                p_smi = _smooth_series(
                    va_f["site_machine_implant"], stats["site_machine_implant"], gr, k
                )
                p_smvl = _smooth_series(
                    va_f["site_machine_view_lat"], stats["site_machine_view_lat"], gr, k
                )
                p_slv = _smooth_series(
                    va_f["site_lat_view"], stats["site_lat_view"], gr, k
                )
                p_mv = _smooth_series(
                    va_f["machine_view"], stats["machine_view"], gr, k
                )
                p_sma = _smooth_series(
                    va_f["site_machine_agebin"], stats["site_machine_agebin"], gr, k
                )
                p_sva = _smooth_series(
                    va_f["site_view_agebin"], stats["site_view_agebin"], gr, k
                )
                p_sla = _smooth_series(
                    va_f["site_lat_agebin"], stats["site_lat_agebin"], gr, k
                )
                p_smva = _smooth_series(
                    va_f["site_machine_view_agebin"],
                    stats["site_machine_view_agebin"],
                    gr,
                    k,
                )
                p_mab = _smooth_series(
                    va_f["machine_agebin"], stats["machine_agebin"], gr, k
                )

                if has_birads:
                    p_sbir = _smooth_series(
                        va_f["site_birads"], stats["site_birads"], gr, k
                    ).values
                else:
                    p_sbir = 0.0

                p_fold = (
                    w_site * p_site.values
                    + w_mach * p_mach.values
                    + w_lat * p_lat.values
                    + w_view * p_view.values
                    + w_sm * p_sm.values
                    + w_sv * p_sv.values
                    + w_sl * p_sl.values
                    + w_smv * p_smv.values
                    + w_agebin * p_agebin.values
                    + w_sab * p_sab.values
                    + w_simp * p_simp.values
                    + w_sml * p_sml.values
                    + w_svl * p_svl.values
                    + w_smi * p_smi.values
                    + w_smvl * p_smvl.values
                    + w_slv * p_slv.values
                    + w_mv * p_mv.values
                    + w_sma * p_sma.values
                    + w_sva * p_sva.values
                    + w_sla * p_sla.values
                    + w_smva * p_smva.values
                    + w_mab * p_mab.values
                    + w_sbir * (p_sbir if isinstance(p_sbir, np.ndarray) else 0.0)
                )

                age_mult = 0.92 + 0.16 * va_f["age_norm"].values  # was 0.90 + 0.20*
                imp_mult = 1.00 + 0.03 * va_f["implant"].values  # was +0.05*
                p_fold = p_fold * age_mult * imp_mult

                p_fold = np.clip(p_fold, 1e-6, 1.0 - 1e-6)
                oof_raw[idx_va] = p_fold

            best_local_pf1 = -1.0
            best_local_power = 1.0
            best_local_bias = 0.0
            best_local_sharpen = 1.0
            best_local_lscale = 1.0

            for pw in power_grid:
                oof_pw = _apply_power_calibration(oof_raw, pw)
                for b in bias_grid:
                    oof_pb = _apply_bias_shift(oof_pw, b)
                    for g in sharpen_grid:
                        oof_pbg = _apply_sharpening_power(oof_pb, g)
                        for ls in logit_scale_grid:
                            pf1 = _pf1_score(y_true, _apply_logit_scale(oof_pbg, ls))
                            if pf1 > best_local_pf1:
                                best_local_pf1 = pf1
                                best_local_power = pw
                                best_local_bias = b
                                best_local_sharpen = g
                                best_local_lscale = ls

            if best_local_pf1 > best_pf1_final:
                best_pf1_final = best_local_pf1
                best_cfg = (k, w)
                best_power = best_local_power
                best_bias = best_local_bias
                best_sharpen = best_local_sharpen
                best_logit_scale = best_local_lscale

    k_best, w_best = best_cfg
    print(
        f"Selected blend+calibration via OOF pF1: "
        f"k={k_best}, weights={w_best}, power={best_power}, bias={best_bias}, "
        f"sharpen={best_sharpen}, logit_scale={best_logit_scale}, OOF pF1={best_pf1_final:.6f}"
    )

    def smooth_stats_full(col):
        return tr_breast.groupby(col)["cancer"].agg(["mean", "count"])

    stats_full = {c: smooth_stats_full(c) for c in enc_cols}

    (
        w_site,
        w_mach,
        w_lat,
        w_view,
        w_sm,
        w_sv,
        w_sl,
        w_smv,
        w_agebin,
        w_sab,
        w_simp,
        w_sml,
        w_svl,
        w_smi,
        w_smvl,
        w_slv,
        w_mv,
        w_sma,
        w_sva,
        w_sla,
        w_smva,
        w_mab,
        w_sbir,
    ) = w_best

    te_breast["p_site"] = _smooth_series(
        te_breast["site_id"], stats_full["site_id"], global_rate, k_best
    )
    te_breast["p_machine"] = _smooth_series(
        te_breast["machine_id"], stats_full["machine_id"], global_rate, k_best
    )
    te_breast["p_lat"] = _smooth_series(
        te_breast["laterality"], stats_full["laterality"], global_rate, k_best
    )
    te_breast["p_view"] = _smooth_series(
        te_breast["view"], stats_full["view"], global_rate, k_best
    )
    te_breast["p_site_machine"] = _smooth_series(
        te_breast["site_machine"], stats_full["site_machine"], global_rate, k_best
    )
    te_breast["p_site_view"] = _smooth_series(
        te_breast["site_view"], stats_full["site_view"], global_rate, k_best
    )
    te_breast["p_site_lat"] = _smooth_series(
        te_breast["site_lat"], stats_full["site_lat"], global_rate, k_best
    )
    te_breast["p_site_machine_view"] = _smooth_series(
        te_breast["site_machine_view"],
        stats_full["site_machine_view"],
        global_rate,
        k_best,
    )
    te_breast["p_agebin"] = _smooth_series(
        te_breast["age_bin"], stats_full["age_bin"], global_rate, k_best
    )
    te_breast["p_site_agebin"] = _smooth_series(
        te_breast["site_agebin"], stats_full["site_agebin"], global_rate, k_best
    )
    te_breast["p_site_implant"] = _smooth_series(
        te_breast["site_implant"], stats_full["site_implant"], global_rate, k_best
    )
    te_breast["p_site_machine_lat"] = _smooth_series(
        te_breast["site_machine_lat"],
        stats_full["site_machine_lat"],
        global_rate,
        k_best,
    )
    te_breast["p_site_view_lat"] = _smooth_series(
        te_breast["site_view_lat"], stats_full["site_view_lat"], global_rate, k_best
    )
    te_breast["p_site_machine_implant"] = _smooth_series(
        te_breast["site_machine_implant"],
        stats_full["site_machine_implant"],
        global_rate,
        k_best,
    )
    te_breast["p_site_machine_view_lat"] = _smooth_series(
        te_breast["site_machine_view_lat"],
        stats_full["site_machine_view_lat"],
        global_rate,
        k_best,
    )
    te_breast["p_site_lat_view"] = _smooth_series(
        te_breast["site_lat_view"], stats_full["site_lat_view"], global_rate, k_best
    )
    te_breast["p_machine_view"] = _smooth_series(
        te_breast["machine_view"], stats_full["machine_view"], global_rate, k_best
    )
    te_breast["p_site_machine_agebin"] = _smooth_series(
        te_breast["site_machine_agebin"],
        stats_full["site_machine_agebin"],
        global_rate,
        k_best,
    )
    te_breast["p_site_view_agebin"] = _smooth_series(
        te_breast["site_view_agebin"],
        stats_full["site_view_agebin"],
        global_rate,
        k_best,
    )
    te_breast["p_site_lat_agebin"] = _smooth_series(
        te_breast["site_lat_agebin"], stats_full["site_lat_agebin"], global_rate, k_best
    )
    te_breast["p_site_machine_view_agebin"] = _smooth_series(
        te_breast["site_machine_view_agebin"],
        stats_full["site_machine_view_agebin"],
        global_rate,
        k_best,
    )
    te_breast["p_machine_agebin"] = _smooth_series(
        te_breast["machine_agebin"], stats_full["machine_agebin"], global_rate, k_best
    )

    if has_birads:
        te_breast["p_site_birads"] = _smooth_series(
            te_breast["site_birads"], stats_full["site_birads"], global_rate, k_best
        )
    else:
        te_breast["p_site_birads"] = global_rate

    p_raw = (
        w_site * te_breast["p_site"].values
        + w_mach * te_breast["p_machine"].values
        + w_lat * te_breast["p_lat"].values
        + w_view * te_breast["p_view"].values
        + w_sm * te_breast["p_site_machine"].values
        + w_sv * te_breast["p_site_view"].values
        + w_sl * te_breast["p_site_lat"].values
        + w_smv * te_breast["p_site_machine_view"].values
        + w_agebin * te_breast["p_agebin"].values
        + w_sab * te_breast["p_site_agebin"].values
        + w_simp * te_breast["p_site_implant"].values
        + w_sml * te_breast["p_site_machine_lat"].values
        + w_svl * te_breast["p_site_view_lat"].values
        + w_smi * te_breast["p_site_machine_implant"].values
        + w_smvl * te_breast["p_site_machine_view_lat"].values
        + w_slv * te_breast["p_site_lat_view"].values
        + w_mv * te_breast["p_machine_view"].values
        + w_sma * te_breast["p_site_machine_agebin"].values
        + w_sva * te_breast["p_site_view_agebin"].values
        + w_sla * te_breast["p_site_lat_agebin"].values
        + w_smva * te_breast["p_site_machine_view_agebin"].values
        + w_mab * te_breast["p_machine_agebin"].values
        + w_sbir * te_breast["p_site_birads"].values
    )

    age_mult = 0.92 + 0.16 * te_breast["age_norm"].values
    imp_mult = 1.00 + 0.03 * te_breast["implant"].values
    p_raw = p_raw * age_mult * imp_mult
    p_raw = np.clip(p_raw, 1e-6, 1.0 - 1e-6)

    p_cal = _apply_bias_shift(_apply_power_calibration(p_raw, best_power), best_bias)
    p_cal = _apply_sharpening_power(p_cal, best_sharpen)
    te_breast["cancer_pred"] = _apply_logit_scale(p_cal, best_logit_scale).astype(
        np.float32
    )

    pred_map = te_breast.set_index("prediction_id")["cancer_pred"]
    te_img = test_df[["prediction_id"]].copy()
    te_img["cancer_pred"] = te_img["prediction_id"].map(pred_map).astype(np.float32)

    return te_img


pred_image_df = build_metadata_baseline(train_df, test_df)
print(pred_image_df.head())
print("Per-image preds:", pred_image_df.shape)

gc.collect()


## === cell 4
submit_df = (
    pred_image_df.groupby("prediction_id", sort=True)["cancer_pred"]
    .mean()
    .reset_index()
    .rename(columns={"cancer_pred": "cancer"})
)

submit_df = sample_sub[["prediction_id"]].merge(
    submit_df, on="prediction_id", how="left"
)
submit_df["cancer"] = (
    submit_df["cancer"].fillna(float(train_df["cancer"].mean())).astype(np.float32)
)

print("submit_df", submit_df.shape)
print(submit_df.head())


## === cell 5
out_path = "submission.csv"
submit_df.to_csv(out_path, index=False)

assert list(submit_df.columns) == ["prediction_id", "cancer"]
assert submit_df.shape[0] == sample_sub.shape[0]
assert out_path.endswith(".csv")

print("Wrote submission to:", out_path)
print(
    "cancer min/max/mean:",
    float(submit_df["cancer"].min()),
    float(submit_df["cancer"].max()),
    float(submit_df["cancer"].mean()),
)
print("Done.")
