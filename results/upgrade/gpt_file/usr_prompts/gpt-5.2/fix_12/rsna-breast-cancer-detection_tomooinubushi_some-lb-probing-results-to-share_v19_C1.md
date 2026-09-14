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

0.03

# 6. Current score

0.02431

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02213) has done: 'Your notebook currently produces a submission but the predictions are pure random noise, so the expected pF1 be near zero and unstable; to move toward the 0.03 target we should replace randomness with a simple, deterministic, leakage-free prior based on the training label rate. Because pF1 rewards calibrated probabilities, predicting the global (or slightly smoothed) cancer prevalence for every row is a minimal change that typically scores materially above random without changing the overall “no-image-model” approach. I also ensure the submission exactly matches `sample_submission.csv`’s `prediction_id` order (some `prediction_id`s repeat in `test.csv`, while the submission expects unique IDs). Finally, I set a fixed random seed (even though we remove randomness) to keep the run reproducible.'
- What this solution (achieved 0.02221) has done: 'You’re already below the 0.03 target (0.02213 → need to increase), so the smallest legitimate improvement is to keep your “no-image” approach but replace the single global prior with a slightly more informative, still-leakage-free prior learned from train metadata. Specifically, we compute smoothed cancer rates by `site_id`, `laterality`, and `view` (and their combination) and back off to less-specific priors when a group is rare, which typically lifts pF1 a bit versus a constant probability. We also keep submission alignment strictly to `sample_submission.csv`’s unique `prediction_id` list (your current approach is correct) and clip probabilities for safety. Core logic remains “predict a calibrated prior from train labels”; we’re only making that prior conditional on existing metadata.'
- What this solution (achieved 0.02223) has done: 'We keep your “metadata-only smoothed prior” core logic unchanged, but make it slightly more informative by adding a smoothed `machine_id` cancer-rate prior (it’s present in both train and test and often carries signal). We also replace the hard `min_slv` gate with a soft backoff weight based on subgroup count, so you still trust the specific `(site,laterality,view)` rate when it’s well-supported but smoothly blend toward broader priors otherwise (reduces brittleness and usually nudges pF1 up). Finally, we keep the submission aligned to `sample_submission.csv`’s unique `prediction_id` list and keep probability clipping for safety. These are minimal, deterministic changes aimed at moving 0.02221 closer to the 0.03 target without changing the overall approach.'
- What this solution (achieved 0.02228) has done: 'Your current score (0.02223) is below the 0.03 target, so we should make a small, low-risk lift while keeping your “metadata-only smoothed prior + backoff blending” core logic unchanged. The biggest missing signal you can add without changing approach is **age** (present in both train/test), handled as a **smoothed age-bin prior** and blended into your existing backoff mixture. This keeps everything deterministic, leakage-free, and still just computes priors from train labels conditioned on metadata. I also keep the submission alignment to `sample_submission.csv` intact and clip probabilities as you already do.'
- What this solution (achieved 0.02227) has done: 'To move your 0.02228 pF1 upward toward the 0.03 target without changing the “metadata-only smoothed priors + backoff blending” core logic, I make two minimal, metric-aligned tweaks: (1) add one more leakage-free prior from train labels using `implant` (present in both train/test) and blend it softly like your other priors, and (2) lightly reweight the backoff mixture to keep total weight 1 while giving a small share to the new `implant` signal (this is typically low-risk and can nudge pF1 up). I keep your deterministic setup, subgroup reliability weighting, submission alignment to `sample_submission.csv`, and clipping unchanged. No model/training loop is introduced; we’re still just computing calibrated priors from train metadata.'
- What this solution (achieved 0.02228) has done: 'Your current score (0.02227) is below the 0.03 target, so we should make a small, low-risk increase while keeping the same “metadata-only smoothed priors + backoff blending” core logic. The biggest minimal gain opportunity is to add a stable interaction prior that’s available in both train/test: `(site_id, machine_id)`; this often captures site-specific device effects better than either alone. We blend this new prior into the existing backoff mixture with a small weight (taken from the global `site_id` component) and keep all existing smoothing, reliability weighting, submission alignment, and clipping unchanged. This preserves evaluation semantics while nudging probabilities slightly more informative.'
- What this solution (achieved 0.0223) has done: 'To move your 0.02228 pF1 upward toward the 0.03 target without changing the “metadata-only smoothed priors + backoff blending” core logic, I’m adding one more small, leakage-free conditional prior using the interaction `(site_id, age_bin)`, which is available in both train/test and often captures site-specific screening population differences. I blend this new prior into your existing backoff mixture with a small weight (taken from the global `site_id` component) to keep the total weight at 1 and keep the same reliability-weighted `(site,laterality,view)` override. I’m also making `age_bin` construction consistent between train/test (using the same helper) and ensuring NaNs are handled identically (already mostly the case). This is a minimal, deterministic change expected to slightly improve calibration and ranking without altering overall approach or submission alignment.'
- What this solution (achieved 0.02231) has done: 'We keep your exact “metadata-only smoothed priors + reliability-weighted backoff” approach, but make one minimal, score-relevant calibration adjustment: add a gentle probability “sharpening” around the global prior so images that are above/below prior get slightly pushed further away, which often helps pF1 without changing ranking logic. The transform is deterministic, monotonic (so it preserves ordering), and anchored at the global prior to avoid drifting the overall prevalence too far. We also (very lightly) recalibrate the reliability scale `k` to trust the specific `(site,laterality,view)` subgroup a bit more when it has moderate support, which can add a small lift. Submission alignment and clipping remain unchanged.'
- What this solution (achieved 0.02396) has done: 'Your current score (0.02231) is below the 0.03 target, so we make the smallest deterministic lift while preserving your “metadata-only smoothed priors + reliability-weighted backoff + prior-anchored sharpening” core logic. The most score-relevant low-risk change is to align the post-processing more directly with pF1 by applying a *single global calibration* that slightly increases predicted probabilities (pF1 rewards recall when base rates are very low). Concretely, we add one monotonic logit-shift (no ranking change) and keep your existing sharpening anchored at the global prior; this usually nudges pF1 upward without destabilizing. Everything else (priors, merges, grouping to prediction_id, submission alignment, clipping) remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0242) has done: 'You’re below the 0.03 target (0.02396 → need a small uplift), so we keep your exact “metadata-only smoothed priors + reliability backoff + anchored sharpening + global logit shift” core logic and make only a minimal, metric-aligned calibration tweak. The lowest-risk lever for pF1 here is slightly increasing recall by nudging probabilities upward in a monotonic way (no ranking change), so we very slightly increase the global logit shift `delta` while leaving everything else unchanged. This should move the score upward a bit without destabilizing (and stays far from any architecture/training changes). The submission alignment to `sample_submission.csv` and clipping remain identical, and the script still writes `submission.csv`.'
- What this solution (achieved 0.02431) has done: 'You’re below the 0.03 target (0.0242), so the goal is a small, low-risk uplift without changing your core “metadata-only smoothed priors + reliability backoff + anchored sharpening + monotonic calibration” approach. The safest lever for pF1 is still recall, so I very slightly increase the global monotonic calibration (logit shift) while keeping ranking unchanged and everything else identical. I also compute the logit shift delta from the observed global prior (bounded) so it’s stable if the prevalence changes, but still very close to your current 0.16 value. Submission alignment to `sample_submission.csv` and probability clipping remain unchanged, and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

np.random.seed(42)



## === cell 1
train_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")
sub_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv")

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print("sub_df shape:", sub_df.shape)




## === cell 2
def get_num_unique(train_df, test_df, col):
    all_df = pd.concat([train_df, test_df])
    num_unique_train = len(train_df[col].unique())
    num_unique_test = len(test_df[col].unique())
    num_unique_all = len(all_df[col].unique())
    return num_unique_train, num_unique_test, num_unique_all


def add_count(df, col):
    if type(col) == str:
        aggs = df.groupby(col, as_index=True)[col].count().rename(col + "_count")
    else:
        aggs = (
            df.groupby(col, as_index=False)[col[0]]
            .count()
            .rename("_".join(col) + "_count")
        )
    df = df.merge(aggs, on=col, how="inner")
    return df




## === cell 3
hypotheses = []



## === cell 4
num_unique_train, num_unique_test, num_unique_all = get_num_unique(
    train_df, test_df, "site_id"
)
print(f"num_unique_train: {num_unique_train}")
print(f"num_unique_test: {num_unique_test}")
print(f"num_unique_all: {num_unique_all}")
hypothesis = num_unique_all == 2
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 5
num_unique_train, num_unique_test, num_unique_all = get_num_unique(
    train_df, test_df, "patient_id"
)
print(f"num_unique_train: {num_unique_train}")
print(f"num_unique_test: {num_unique_test}")
print(f"num_unique_all: {num_unique_all}")
hypothesis = num_unique_train + num_unique_test == num_unique_all
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 6
num_unique_train, num_unique_test, num_unique_all = get_num_unique(
    train_df, test_df, "image_id"
)
print(f"num_unique_train: {num_unique_train}")
print(f"num_unique_test: {num_unique_test}")
print(f"num_unique_all: {num_unique_all}")
hypothesis = num_unique_train + num_unique_test == num_unique_all
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 7
num_unique_train, num_unique_test, num_unique_all = get_num_unique(
    train_df, test_df, "laterality"
)
print(f"num_unique_train: {num_unique_train}")
print(f"num_unique_test: {num_unique_test}")
print(f"num_unique_all: {num_unique_all}")
hypothesis = num_unique_test == 2
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 8
num_unique_train, num_unique_test, num_unique_all = get_num_unique(
    train_df, test_df, "machine_id"
)
print(f"num_unique_train: {num_unique_train}")
print(f"num_unique_test: {num_unique_test}")
print(f"num_unique_all: {num_unique_all}")
hypothesis = num_unique_train != num_unique_all
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 9
num_unique_train, num_unique_test, num_unique_all = get_num_unique(
    train_df, test_df, "view"
)
print(f"num_unique_train: {num_unique_train}")
print(f"num_unique_test: {num_unique_test}")
print(f"num_unique_all: {num_unique_all}")
hypothesis = num_unique_all == 6
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 10
temp_df = add_count(test_df, "patient_id").drop_duplicates("patient_id")
hypothesis = temp_df.patient_id_count.min() >= 4
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 11
len1 = len(test_df.drop_duplicates(["patient_id"]))
len2 = len(test_df.drop_duplicates(["patient_id", "site_id"]))
print(f"len1: {len1}")
print(f"len2: {len2}")
hypothesis = len1 == len2
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 12
len1 = len(test_df.drop_duplicates(["patient_id"]))
len2 = len(test_df.drop_duplicates(["patient_id", "age"]))
print(f"len1: {len1}")
print(f"len2: {len2}")
hypothesis = len1 == len2
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 13
len1 = len(test_df.drop_duplicates(["machine_id"]))
len2 = len(test_df.drop_duplicates(["site_id", "machine_id"]))
print(f"len1: {len1}")
print(f"len2: {len2}")
hypothesis = len1 == len2
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 14
len1 = len(test_df.drop_duplicates(["patient_id"]))
len2 = len(test_df.drop_duplicates(["patient_id", "machine_id"]))
print(f"len1: {len1}")
print(f"len2: {len2}")
hypothesis = len1 != len2
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 15
len1 = len(train_df.drop_duplicates(["patient_id"]))
len2 = len(train_df.drop_duplicates(["patient_id", "machine_id"]))
print(f"len1: {len1}")
print(f"len2: {len2}")
hypothesis = len1 != len2
print(f"hyposthesis: {hypothesis}")



## === cell 16
mean_site_id_train = train_df.site_id.mean()
mean_site_id_test = test_df.site_id.mean()
print(f"mean site ID train: {mean_site_id_train}")
print(f"mean site ID test: {mean_site_id_test}")
hypothesis = mean_site_id_test < 1.5
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 17
print(hypotheses)
print(all(hypotheses))



## === cell 18
pos = float(train_df["cancer"].sum())
n = float(len(train_df))
alpha = 1.0  # Laplace smoothing
global_prior = (pos + alpha) / (n + 2 * alpha)
global_prior = float(np.clip(global_prior, 1e-6, 1 - 1e-6))
print("Global cancer prior:", global_prior)


def smoothed_rate(df, keys, y_col="cancer", alpha=1.0):
    g = df.groupby(keys)[y_col].agg(["sum", "count"]).reset_index()
    g["rate"] = (g["sum"] + alpha * global_prior) / (g["count"] + alpha)
    return g.drop(columns=["sum"])


def add_age_bin(df, col="age"):
    out = df.copy()
    age = pd.to_numeric(out[col], errors="coerce")
    out["age_bin"] = (np.floor(age / 5.0) * 5.0).astype("Int64")
    return out


def prior_anchored_sharpen(p, prior, gamma=1.12):
    p = np.asarray(p, dtype=float)
    prior = float(prior)
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    prior = float(np.clip(prior, eps, 1 - eps))
    logit = np.log(p / (1 - p))
    logit_prior = np.log(prior / (1 - prior))
    p2 = 1.0 / (1.0 + np.exp(-(logit_prior + gamma * (logit - logit_prior))))
    return np.clip(p2, eps, 1 - eps)


def logit_shift(p, delta=0.14):
    p = np.asarray(p, dtype=float)
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    p2 = 1.0 / (1.0 + np.exp(-(logit + float(delta))))
    return np.clip(p2, eps, 1 - eps)


train_b = add_age_bin(train_df, "age")
test_b = add_age_bin(test_df, "age")

rate_site = smoothed_rate(train_df, ["site_id"], alpha=20.0)
rate_lat = smoothed_rate(train_df, ["laterality"], alpha=20.0)
rate_view = smoothed_rate(train_df, ["view"], alpha=20.0)
rate_machine = smoothed_rate(train_df, ["machine_id"], alpha=40.0)
rate_site_lat_view = smoothed_rate(
    train_df, ["site_id", "laterality", "view"], alpha=50.0
)

rate_agebin = smoothed_rate(train_b.dropna(subset=["age_bin"]), ["age_bin"], alpha=80.0)

rate_implant = smoothed_rate(
    train_df.dropna(subset=["implant"]), ["implant"], alpha=60.0
)

rate_site_machine = smoothed_rate(train_df, ["site_id", "machine_id"], alpha=70.0)

rate_site_agebin = smoothed_rate(
    train_b.dropna(subset=["age_bin"]), ["site_id", "age_bin"], alpha=120.0
)

t = test_b[
    [
        "prediction_id",
        "site_id",
        "laterality",
        "view",
        "machine_id",
        "age_bin",
        "implant",
    ]
].copy()

t = t.merge(
    rate_site_lat_view, on=["site_id", "laterality", "view"], how="left"
).rename(columns={"rate": "p_slv", "count": "n_slv"})
t = t.merge(rate_site, on=["site_id"], how="left").rename(
    columns={"rate": "p_s", "count": "n_s"}
)
t = t.merge(rate_lat, on=["laterality"], how="left").rename(
    columns={"rate": "p_l", "count": "n_l"}
)
t = t.merge(rate_view, on=["view"], how="left").rename(
    columns={"rate": "p_v", "count": "n_v"}
)
t = t.merge(rate_machine, on=["machine_id"], how="left").rename(
    columns={"rate": "p_m", "count": "n_m"}
)
t = t.merge(rate_agebin, on=["age_bin"], how="left").rename(
    columns={"rate": "p_a", "count": "n_a"}
)
t = t.merge(rate_implant, on=["implant"], how="left").rename(
    columns={"rate": "p_i", "count": "n_i"}
)

t = t.merge(rate_site_machine, on=["site_id", "machine_id"], how="left").rename(
    columns={"rate": "p_sm", "count": "n_sm"}
)

t = t.merge(rate_site_agebin, on=["site_id", "age_bin"], how="left").rename(
    columns={"rate": "p_sa", "count": "n_sa"}
)

k = 160.0  # trust (site,laterality,view) a bit more when moderately supported
n_slv = t["n_slv"].fillna(0.0).astype(float)
w_slv = n_slv / (n_slv + k)  # in [0,1]

backoff = (
    0.20 * t["p_s"].fillna(global_prior)
    + 0.17 * t["p_l"].fillna(global_prior)
    + 0.17 * t["p_v"].fillna(global_prior)
    + 0.17 * t["p_m"].fillna(global_prior)
    + 0.11 * t["p_a"].fillna(global_prior)
    + 0.06 * t["p_i"].fillna(global_prior)
    + 0.08 * t["p_sm"].fillna(global_prior)
    + 0.04 * t["p_sa"].fillna(global_prior)
)

p_slv = t["p_slv"].fillna(backoff)
p = w_slv * p_slv + (1.0 - w_slv) * backoff

p = prior_anchored_sharpen(p.astype(float), global_prior, gamma=1.12)

delta = float(np.clip(0.17 + 0.02 * (0.02 - global_prior), 0.155, 0.185))
p = logit_shift(p.astype(float), delta=delta)

p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)

pred_by_pid = t[["prediction_id"]].copy()
pred_by_pid["cancer"] = p
pred_by_pid = pred_by_pid.groupby("prediction_id", as_index=False)["cancer"].mean()

submission = sub_df[["prediction_id"]].merge(
    pred_by_pid, on="prediction_id", how="left"
)

submission["cancer"] = submission["cancer"].fillna(global_prior).clip(1e-6, 1 - 1e-6)

submission.to_csv("submission.csv", index=False)
print("Used delta:", delta)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
