# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02212) has done: 'Your notebook currently can’t yield a meaningful Kaggle score because it outputs the wrong number of rows/IDs (it uses `test_df['prediction_id'].unique()` which won’t match `sample_submission.csv`), and it also produces pure-random predictions that be very unstable and typically score extremely low. To reliably move score upward toward your target (0.03) with minimal logic change, I keep your “hypotheses” analysis intact but always generate the submission by starting from `sample_submission.csv` to guarantee perfect alignment and row count. Then, instead of random values, I output a simple constant probability equal to the train cancer prevalence, which is a minimal, legitimate baseline and should be stable and typically score better than random under pF1. This preserves your overall approach (no model training added) while ensuring a valid submission file is always produced.'
- What this solution (achieved 0.02224) has done: 'We keep your hypotheses EDA untouched and only adjust the prediction generation to better match the pF1 metric while preserving the “no real model” core logic. Your current constant `base_rate` is stable but often suboptimal for pF1 because pF1 tends to prefer pushing probabilities away from the extremes in a way that increases expected F1. With minimal change, we compute an “F1-optimal constant” directly from the training prevalence (for constant predictions, optimal p is approximately the prevalence), then optionally apply a tiny, deterministic site-based scaling (using only `site_id`, which exists in both train/test) to nudge calibration without adding a model. We still build the submission by starting from `sample_submission.csv` to guarantee perfect row/ID alignment.'
- What this solution (achieved 0.02629) has done: 'To move your pF1 up from 0.02224 toward the 0.03 target with minimal logic change, I keep the same “prior-only” approach but make the constant-probability baseline more pF1-friendly by (1) tuning a single global scaling factor on out-of-fold (patient-level) predictions to maximize pF1 on training data, and (2) keeping your existing site-based prior + shrinkage, but passing it through that calibrated scaler. This preserves your core semantics (no image model, no new features beyond metadata; still just prevalence-based probabilities) while making the probabilities better matched to the metric. I also ensure submission alignment remains exactly driven by `sample_submission.csv` and that no leakage occurs by doing calibration out-of-fold at patient level. The changes are small and should reliably improve score stability and nudge performance upward toward 0.03.'
- What this solution (achieved 0.02629) has done: 'Your current score (0.02629) is below the 0.03 target, so we should make a small, low-risk calibration improvement without changing the “prior-only” core logic. The biggest issue is that the OOF tuning currently evaluates fold scores using site priors computed on the full training set, which leaks label information and can pick a scaler that doesn’t generalize to Kaggle test. I keep the same site+global prior with shrinkage and the same single multiplicative scaler, but tune that scaler in a leakage-free way by recomputing global/site prevalence from each fold’s training split before scoring its validation split. I also slightly densify the scaler grid around 1.0 to nudge performance upward with minimal change, and keep submission alignment anchored to `sample_submission.csv` exactly as you already do.'

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
display(train_df.head())
display(test_df.head())
display(sub_df.head())




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
display(temp_df.head())
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
display(train_df[train_df.patient_id == 22637])



## === cell 16
mean_site_id_train = train_df.site_id.mean()
mean_site_id_test = test_df.site_id.mean()
print(f"mean site ID train: {mean_site_id_train}")
print(f"mean site ID test: {mean_site_id_test}")
hypothesis = mean_site_id_test < 1.5
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 17
len1 = len(test_df.patient_id.unique())
len2 = len(
    test_df[(test_df.laterality == "L") & (test_df.view == "CC")].patient_id.unique()
)
len3 = len(
    test_df[(test_df.laterality == "L") & (test_df.view == "MLO")].patient_id.unique()
)
len4 = len(
    test_df[(test_df.laterality == "R") & (test_df.view == "CC")].patient_id.unique()
)
len5 = len(
    test_df[(test_df.laterality == "R") & (test_df.view == "MLO")].patient_id.unique()
)
print(f"len1: {len1}")
print(f"len2: {len2}")
print(f"len3: {len3}")
print(f"len4: {len4}")
print(f"len5: {len5}")
hypothesis = len1 == len2 == len3 == len4 == len5
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 18
test_machine_49_count = len(test_df.query("machine_id == 49"))
test_len = len(test_df)
test_machine_49_ratio = test_machine_49_count / test_len
print(f"test_machine_49_count: {test_machine_49_count}")
print(f"test_len: {test_len}")
print(f"test_machine_49_ratio: {test_machine_49_ratio}")
hypothesis = test_machine_49_ratio > 0.40
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 19
train_machine_49_count = len(train_df.query("machine_id == 49"))
train_len = len(train_df)
train_machine_49_ratio = train_machine_49_count / train_len
print(f"train_machine_49_count: {train_machine_49_count}")
print(f"train_len: {train_len}")
print(f"train_machine_49_ratio: {train_machine_49_ratio}")



## === cell 20
mean_age_train = train_df.drop_duplicates("patient_id").age.mean()
mean_age_site1_train = (
    train_df[train_df.site_id == 2].drop_duplicates("patient_id").age.mean()
)
mean_age_site2_train = (
    train_df[train_df.site_id == 1].drop_duplicates("patient_id").age.mean()
)
mean_age_test = test_df.drop_duplicates("patient_id").age.mean()
mean_age_site1_test = (
    test_df[test_df.site_id == 2].drop_duplicates("patient_id").age.mean()
)
mean_age_site2_test = (
    test_df[test_df.site_id == 1].drop_duplicates("patient_id").age.mean()
)
print(f"mean_age_train: {mean_age_train}")
print(f"mean_age_site1_train: {mean_age_site1_train}")
print(f"mean_age_site2_train: {mean_age_site2_train}")
print(f"mean_age_test: {mean_age_test}")
print(f"mean_age_site1_test: {mean_age_site1_test}")
print(f"mean_age_site2_test: {mean_age_site2_test}")
hypothesis = (
    (mean_age_test > 56)
    & (61 > mean_age_test)
    & (mean_age_site1_test > mean_age_site2_test)
)
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 21
mean_implant_train = train_df.drop_duplicates(
    ["patient_id", "laterality"]
).implant.mean()
mean_implant_test = test_df.drop_duplicates(["patient_id", "laterality"]).implant.mean()
print(f"mean_implant_train: {mean_implant_train}")
print(f"mean_implant_test: {mean_implant_test}")
hypothesis = (mean_implant_test > 0.01) & (0.02 > mean_implant_test)
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 22
print(hypotheses)
print(all(hypotheses))




## === cell 23
def pf1_score(y_true, y_prob, eps=1e-15):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    pTP = np.sum(y_true * y_prob)
    pFP = np.sum((1.0 - y_true) * y_prob)

    pos = np.sum(y_true)

    pPrecision = pTP / (pTP + pFP + eps)
    pRecall = pTP / (pos + eps)

    return (2.0 * pPrecision * pRecall) / (pPrecision + pRecall + eps)


def make_prior_prediction(df, global_prev, site_prev, shrink):
    p = (
        df["site_id"]
        .map(lambda s: float(site_prev.get(int(s), global_prev)))
        .astype(float)
    )
    p = shrink * global_prev + (1.0 - shrink) * p
    return np.clip(p.to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)


def tune_global_scaler_oof(train_df, shrink, n_folds=5, seed=42):

    patient_ids = train_df["patient_id"].drop_duplicates().to_numpy()
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)
    folds = np.array_split(patient_ids, n_folds)

    scaler_grid = np.array(
        [
            0.85,
            0.9,
            0.93,
            0.95,
            0.97,
            0.985,
            1.0,
            1.015,
            1.03,
            1.05,
            1.075,
            1.1,
            1.15,
            1.2,
        ],
        dtype=np.float64,
    )

    best_s = 1.0
    best_score = -1.0

    for s in scaler_grid:
        oof_scores = []
        for fold_pids in folds:
            fold_mask = train_df["patient_id"].isin(fold_pids)

            trn_df = train_df.loc[
                ~fold_mask, ["site_id", "prediction_id", "cancer"]
            ].copy()
            val_df = train_df.loc[
                fold_mask, ["site_id", "prediction_id", "cancer"]
            ].copy()

            global_prev = float(np.clip(trn_df["cancer"].mean(), 1e-6, 1 - 1e-6))
            site_prev = trn_df.groupby("site_id")["cancer"].mean().to_dict()

            p_img = make_prior_prediction(val_df, global_prev, site_prev, shrink)
            p_img = np.clip(p_img * s, 1e-6, 1.0 - 1e-6)

            val_tmp = val_df[["prediction_id", "cancer"]].copy()
            val_tmp["p"] = p_img

            y_pid = (
                val_tmp.groupby("prediction_id")["cancer"]
                .max()
                .to_numpy(dtype=np.float64)
            )
            p_pid = (
                val_tmp.groupby("prediction_id")["p"].mean().to_numpy(dtype=np.float64)
            )

            score = pf1_score(y_pid, p_pid)
            oof_scores.append(score)

        mean_score = float(np.mean(oof_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_s = float(s)

    return best_s, best_score


global_prev = float(np.clip(train_df["cancer"].mean(), 1e-6, 1 - 1e-6))
site_prev = train_df.groupby("site_id")["cancer"].mean().to_dict()

shrink = 0.25  # 0 -> pure site prior, 1 -> pure global prior

best_scaler, best_oof_pf1 = tune_global_scaler_oof(
    train_df, shrink=shrink, n_folds=5, seed=42
)
print(
    "OOF tuned scaler (prediction_id-level, leakage-free):",
    best_scaler,
    "OOF pF1 (for reference):",
    best_oof_pf1,
)

pid_site = (
    test_df.groupby("prediction_id")["site_id"]
    .agg(lambda x: int(x.mode().iloc[0]))
    .to_dict()
)

submission = sub_df.copy()
submission["site_id"] = submission["prediction_id"].map(pid_site)
submission["site_id"] = submission["site_id"].fillna(
    int(round(test_df["site_id"].mode().iloc[0]))
)

base_p = (
    submission["site_id"]
    .map(lambda s: float(site_prev.get(int(s), global_prev)))
    .astype(float)
)
base_p = shrink * global_prev + (1.0 - shrink) * base_p
base_p = np.clip(base_p.to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)

submission["cancer"] = np.clip(base_p * best_scaler, 1e-6, 1.0 - 1e-6)
submission = submission[["prediction_id", "cancer"]]

assert submission.shape[0] == sub_df.shape[0]
assert list(submission.columns) == ["prediction_id", "cancer"]
assert submission["prediction_id"].isna().sum() == 0
assert submission["cancer"].isna().sum() == 0

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(
    "global_prev:",
    global_prev,
    "site_prev:",
    site_prev,
    "shrink:",
    shrink,
    "best_scaler:",
    best_scaler,
)
display(submission.head())

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2762441566.py in <cell line: 0>()
    108 shrink = 0.25  # 0 -> pure site prior, 1 -> pure global prior
    109 
--> 110 best_scaler, best_oof_pf1 = tune_global_scaler_oof(
    111     train_df, shrink=shrink, n_folds=5, seed=42
    112 )

/tmp/ipykernel_11/2762441566.py in tune_global_scaler_oof(train_df, shrink, n_folds, seed)
     64             fold_mask = train_df["patient_id"].isin(fold_pids)
     65 
---> 66             trn_df = train_df.loc[
     67                 ~fold_mask, ["site_id", "prediction_id", "cancer"]
     68             ].copy()

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['prediction_id'] not in index"
