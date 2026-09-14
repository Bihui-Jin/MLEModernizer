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

0.04

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02195) has done: 'Your current notebook already writes a `submission.csv`, but it uses random probabilities and (in the `all(hypotheses)` branch) risks producing the wrong row count/ordering versus `sample_submission.csv`, which can lead to an invalid or low-scoring submission. I make the submission construction strictly follow `sample_submission.csv`’s `prediction_id` order and row count to guarantee validity. Then, to move the score upward toward the 0.04 target (higher-is-better) with minimal change, I replace the random predictions with a simple, deterministic baseline: per-`prediction_id` cancer prevalence estimated from the training data using the same `patient_id + laterality` key, with a safe global fallback. This keeps core logic minimal (still a lightweight pandas-only approach) and should reliably beat randomness without adding heavy modeling.'
- What this solution (achieved 0.02808) has done: 'Your current score (0.02195) is below the target (0.04), so we should cautiously improve signal without changing the overall “pandas-only metadata baseline” core logic. The biggest minimal win is to avoid using `patient_id` (which is disjoint between train/test) and instead use metadata features that generalize across splits (site_id, machine_id, laterality, view, implant, and binned age) to estimate a smoothed cancer prevalence. We keep the same idea of “group prevalence with global fallback”, but compute it on robust groups and apply simple Bayesian smoothing to reduce overconfidence on rare groups (helps pF1). Submission construction continue to follow `sample_submission.csv` exactly for row count and ordering.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np



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
    train_df[train_df.site_id == 1].drop_duplicates("patient_id").age.mean()
)
mean_age_site2_train = (
    train_df[train_df.site_id == 2].drop_duplicates("patient_id").age.mean()
)
mean_age_test = test_df.drop_duplicates("patient_id").age.mean()
mean_age_site1_test = (
    test_df[test_df.site_id == 1].drop_duplicates("patient_id").age.mean()
)
mean_age_site2_test = (
    test_df[test_df.site_id == 2].drop_duplicates("patient_id").age.mean()
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
    & (mean_age_site1_test < mean_age_site2_test)
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
isna_age_train = train_df.age.isnull().sum()
isna_age_test = test_df.age.isnull().sum()
other_cols = [
    "site_id",
    "patient_id",
    "image_id",
    "laterality",
    "view",
    "implant",
    "machine_id",
]
isna_other_train = train_df[other_cols].isnull().sum().sum()
isna_other_test = test_df[other_cols].isnull().sum().sum()
print(train_df.isnull().sum(axis=0))
print(test_df.isnull().sum(axis=0))
print(f"isna_age_train: {isna_age_train}")
print(f"isna_age_test: {isna_age_test}")
print(f"isna_other_train: {isna_other_train}")
print(f"isna_other_test: {isna_other_test}")
hypothesis = (isna_age_test > 0) & (isna_other_test == 0)
print(f"hyposthesis: {hypothesis}")
hypotheses.append(hypothesis)



## === cell 23
print(hypotheses)
print(all(hypotheses))




## === cell 24
def _prep_age_bin(s: pd.Series) -> pd.Series:
    bins = [0, 39, 49, 59, 69, 120]
    labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    out = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    return out.astype("object")


train_work = train_df.copy()
test_work = test_df.copy()

train_work["age_bin"] = _prep_age_bin(train_work["age"])
test_work["age_bin"] = _prep_age_bin(test_work["age"])


def _breast_level(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    keys = ["patient_id", "laterality"]
    base_cols = ["site_id", "machine_id", "implant", "age_bin"]
    if is_train:
        cols = keys + base_cols + ["view", "cancer"]
    else:
        cols = keys + base_cols + ["view"]
    d = df.loc[:, cols].copy()

    agg = {c: "first" for c in base_cols}
    agg["view"] = lambda s: "|".join(sorted(pd.Series(s).dropna().astype(str).unique()))
    if is_train:
        agg["cancer"] = (
            "max"  # breast is positive if any image for that breast is positive
        )

    out = d.groupby(keys, as_index=False).agg(agg)
    return out


train_breast = _breast_level(train_work, is_train=True)

for c in ["site_id", "machine_id", "implant"]:
    train_breast[c] = train_breast[c].astype("int64")
for c in ["laterality", "view", "age_bin"]:
    train_breast[c] = train_breast[c].astype("object")

global_p = float(train_breast["cancer"].mean())
print("Global cancer prevalence (train breast-level):", global_p)

group_cols = ["site_id", "machine_id", "laterality", "view", "implant", "age_bin"]

g = (
    train_breast.groupby(group_cols, dropna=False)["cancer"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "grp_mean", "count": "grp_n"})
)

m_base = 200.0
g["m_eff"] = m_base / np.sqrt(g["grp_n"].astype(float).clip(lower=1.0))
g["grp_smoothed"] = (g["grp_mean"] * g["grp_n"] + global_p * g["m_eff"]) / (
    g["grp_n"] + g["m_eff"]
)

test_key = test_work.loc[:, ["prediction_id"] + group_cols].copy()

for c in ["site_id", "machine_id", "implant"]:
    test_key[c] = test_key[c].astype("int64")
for c in ["laterality", "view", "age_bin"]:
    test_key[c] = test_key[c].astype("object")

test_key = test_key.groupby(
    ["prediction_id", "site_id", "machine_id", "laterality", "implant", "age_bin"],
    as_index=False,
).agg({"view": lambda s: "|".join(sorted(pd.Series(s).dropna().astype(str).unique()))})

test_key = test_key.merge(g[group_cols + ["grp_smoothed"]], on=group_cols, how="left")

fallback_cols_1 = ["site_id", "machine_id", "laterality", "view"]
g2 = (
    train_breast.groupby(fallback_cols_1, dropna=False)["cancer"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "grp2_mean", "count": "grp2_n"})
)
m2_base = 200.0
g2["m2_eff"] = m2_base / np.sqrt(g2["grp2_n"].astype(float).clip(lower=1.0))
g2["grp2_smoothed"] = (g2["grp2_mean"] * g2["grp2_n"] + global_p * g2["m2_eff"]) / (
    g2["grp2_n"] + g2["m2_eff"]
)
test_key = test_key.merge(
    g2[fallback_cols_1 + ["grp2_smoothed"]], on=fallback_cols_1, how="left"
)

fallback_cols_2 = ["site_id", "laterality", "view"]
g3 = (
    train_breast.groupby(fallback_cols_2, dropna=False)["cancer"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "grp3_mean", "count": "grp3_n"})
)
m3_base = 200.0
g3["m3_eff"] = m3_base / np.sqrt(g3["grp3_n"].astype(float).clip(lower=1.0))
g3["grp3_smoothed"] = (g3["grp3_mean"] * g3["grp3_n"] + global_p * g3["m3_eff"]) / (
    g3["grp3_n"] + g3["m3_eff"]
)
test_key = test_key.merge(
    g3[fallback_cols_2 + ["grp3_smoothed"]], on=fallback_cols_2, how="left"
)

test_key["pred"] = test_key["grp_smoothed"]
test_key["pred"] = test_key["pred"].fillna(test_key["grp2_smoothed"])
test_key["pred"] = test_key["pred"].fillna(test_key["grp3_smoothed"])
test_key["pred"] = test_key["pred"].fillna(global_p).astype(float).clip(0.0, 1.0)

submission = sub_df[["prediction_id"]].merge(
    test_key[["prediction_id", "pred"]],
    on="prediction_id",
    how="left",
)

submission["cancer"] = submission["pred"].fillna(global_p).astype(float).clip(0.0, 1.0)
submission = submission[["prediction_id", "cancer"]]

assert submission.shape[0] == sub_df.shape[0], "Row count mismatch vs sample_submission"
assert submission["prediction_id"].equals(
    sub_df["prediction_id"]
), "prediction_id ordering mismatch"
assert submission["cancer"].between(0.0, 1.0).all(), "cancer out of [0,1]"

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
display(submission.head())
print("cancer summary:", submission["cancer"].describe())

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2991108562.py in <cell line: 0>()
    133 submission = submission[["prediction_id", "cancer"]]
    134 
--> 135 assert submission.shape[0] == sub_df.shape[0], "Row count mismatch vs sample_submission"
    136 assert submission["prediction_id"].equals(
    137     sub_df["prediction_id"]

AssertionError: Row count mismatch vs sample_submission
