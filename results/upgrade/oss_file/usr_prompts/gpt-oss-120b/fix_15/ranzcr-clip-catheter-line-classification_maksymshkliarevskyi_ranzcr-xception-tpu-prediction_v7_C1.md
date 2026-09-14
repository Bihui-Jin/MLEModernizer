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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9535947779316782

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the problematic TensorFlow import, add a robust way to locate the data folder, and rename the cells so they start at 1 while keeping the original logic. This fixes the import error, ensures `train_df`, `sample_sub`, and `label_cols` are defined, and writes a valid `submission.csv` containing constant baseline probabilities.'
- What this solution (achieved 0.5) has done: 'The change parallelizes image loading and uses PIL ImageStat to compute mean intensity faster, cutting the total I/O‑bound work dramatically while keeping all calculations identical. The core logic for deriving probabilities and writing the submission remains unchanged.'
- What this solution (achieved 0.53728) has done: 'I fixed the indexing error caused by trying to use a boolean mask from train_df directly on a Series indexed by StudyInstanceUID. By re‑indexing train_df on StudyInstanceUID and applying the mask to train_intensity, the means are computed correctly for positive and negative cases. This small fix restores the pipeline, allowing it to generate a valid submission.csv without altering the core modeling logic.'
- What this solution (achieved 0.54231) has done: 'I replace the simple linear intensity scaling with a Gaussian likelihood‑ratio probability estimate (using per‑label mean ± std of intensities). This keeps the overall pipeline intact while providing a more discriminative mapping from intensity to probability, which should raise the AUC toward the target. All other steps (data loading, intensity computation, CSV writing) remain unchanged.'
- What this solution (achieved 0.54098) has done: 'I keep the overall pipeline unchanged but improve the probability estimation for each label. After computing the Gaussian likelihood‐ratio (the original method) I also compute a simple linear scaling based on the positive and negative mean intensities. By averaging these two estimates the predictions become more discriminative, which should raise the AUC toward the target while preserving the core logic.'
- What this solution (achieved 0.53486) has done: 'I replace the blended Gaussian + linear probability with a blend of two Gaussian likelihood‑ratio estimates: one computed on the raw mean intensity and one on its log‑transformed values. This keeps the overall pipeline unchanged while giving the model a richer view of the intensity distribution, which should raise the AUC toward the target. The rest of the code (data loading, intensity computation, CSV output) remains identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageStat

default_path = os.path.join(
    os.getcwd(), "data", "ranzcr-clip-catheter-line-classification"
)
if not os.path.isdir(default_path):
    default_path = "/kaggle/input/ranzcr-clip-catheter-line-classification"
WORK_DIR = default_path

train_path = os.path.join(WORK_DIR, "train.csv")
sample_path = os.path.join(WORK_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
sample_sub = pd.read_csv(sample_path)

label_cols = [c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]]

train_img_dir = os.path.join(WORK_DIR, "train")
test_img_dir = os.path.join(WORK_DIR, "test")


def image_mean_intensity(img_path):
    """Return the mean grayscale intensity of an image; fallback to np.nan on error."""
    try:
        with Image.open(img_path) as img:
            img_gray = img.convert("L")
            return ImageStat.Stat(img_gray).mean[0]
    except Exception:
        return np.nan


def compute_intensity_series(df, img_dir, id_col="StudyInstanceUID", max_workers=None):
    """Parallel computation of mean intensities for all images referenced in df."""
    uids = df[id_col].values
    paths = [os.path.join(img_dir, f"{uid}.jpg") for uid in uids]

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        intensities = list(executor.map(image_mean_intensity, paths))

    return pd.Series(intensities, index=uids, name="intensity")


def ecdf_likelihood(test_vals, pos_vals, neg_vals):
    """
    Simple likelihood‑ratio using empirical CDFs:
        p = F_pos / (F_pos + F_neg)
    """
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        return np.full_like(test_vals, 0.5, dtype=float)

    pos_sorted = np.sort(pos_vals)
    neg_sorted = np.sort(neg_vals)

    f_pos = np.searchsorted(pos_sorted, test_vals, side="right") / len(pos_sorted)
    f_neg = np.searchsorted(neg_sorted, test_vals, side="right") / len(neg_sorted)

    prob = f_pos / (f_pos + f_neg + 1e-12)
    return np.clip(prob, 0.0, 1.0)


train_intensity = compute_intensity_series(train_df, train_img_dir)
test_intensity = compute_intensity_series(sample_sub, test_img_dir)

overall_mean_intensity = train_intensity.mean()
train_intensity.fillna(overall_mean_intensity, inplace=True)
test_intensity.fillna(overall_mean_intensity, inplace=True)

train_df_idx = train_df.set_index("StudyInstanceUID")
submission = sample_sub.copy()

for col in label_cols:
    if col not in submission.columns:
        continue

    if col not in train_df_idx.columns:
        fallback_prob = train_df[col].mean() if col in train_df.columns else 0.0
        submission[col] = fallback_prob
        continue

    pos_mask = train_df_idx[col] == 1
    neg_mask = train_df_idx[col] == 0

    pos_mean = train_intensity[pos_mask].mean()
    neg_mean = train_intensity[neg_mask].mean()
    pos_std = train_intensity[pos_mask].std()
    neg_std = train_intensity[neg_mask].std()

    train_intensity_log = np.log1p(train_intensity)
    test_intensity_log = np.log1p(test_intensity)

    pos_mean_log = train_intensity_log[pos_mask].mean()
    neg_mean_log = train_intensity_log[neg_mask].mean()
    pos_std_log = train_intensity_log[pos_mask].std()
    neg_std_log = train_intensity_log[neg_mask].std()

    if any(
        np.isnan(
            [
                pos_mean,
                neg_mean,
                pos_std,
                neg_std,
                pos_mean_log,
                neg_mean_log,
                pos_std_log,
                neg_std_log,
            ]
        )
    ) or ((pos_std == 0 and neg_std == 0) and (pos_std_log == 0 and neg_std_log == 0)):
        fallback_prob = train_df[col].mean()
        submission[col] = fallback_prob
        continue

    sigma_pos = max(pos_std, 1e-6)
    sigma_neg = max(neg_std, 1e-6)
    sigma_pos_log = max(pos_std_log, 1e-6)
    sigma_neg_log = max(neg_std_log, 1e-6)

    pdf_pos = np.exp(-0.5 * ((test_intensity - pos_mean) / sigma_pos) ** 2) / (
        sigma_pos * np.sqrt(2 * np.pi)
    )
    pdf_neg = np.exp(-0.5 * ((test_intensity - neg_mean) / sigma_neg) ** 2) / (
        sigma_neg * np.sqrt(2 * np.pi)
    )
    prob_gauss_raw = pdf_pos / (pdf_pos + pdf_neg)

    pdf_pos_log = np.exp(
        -0.5 * ((test_intensity_log - pos_mean_log) / sigma_pos_log) ** 2
    ) / (sigma_pos_log * np.sqrt(2 * np.pi))
    pdf_neg_log = np.exp(
        -0.5 * ((test_intensity_log - neg_mean_log) / sigma_neg_log) ** 2
    ) / (sigma_neg_log * np.sqrt(2 * np.pi))
    prob_gauss_log = pdf_pos_log / (pdf_pos_log + pdf_neg_log)

    prob_gauss = 0.5 * (prob_gauss_raw + prob_gauss_log)

    prob_ecdf = ecdf_likelihood(
        test_intensity.values,
        train_intensity[pos_mask].values,
        train_intensity[neg_mask].values,
    )

    linear_raw = (test_intensity - neg_mean) / (pos_mean - neg_mean + 1e-12)
    prob_linear = np.clip(linear_raw, 0.0, 1.0)

    midpoint = (pos_mean + neg_mean) / 2.0
    scale = max((pos_std + neg_std) / 2.0, 1e-3)
    prob_sigmoid = 1.0 / (1.0 + np.exp(-(test_intensity - midpoint) / scale))
    prob_sigmoid = np.clip(prob_sigmoid, 0.0, 1.0)

    prob = (
        0.25 * prob_gauss + 0.55 * prob_ecdf + 0.10 * prob_linear + 0.10 * prob_sigmoid
    )
    prob = prob.clip(0.0, 1.0)

    submission[col] = prob




## === cell 1
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
print("First few rows of the submission:")
print(submission.head())
