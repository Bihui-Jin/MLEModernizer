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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

cloudpathlib==0.21.1
cuda-pathfinder==1.3.2
geopandas==0.14.4
jmespath==1.0.1
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
path==17.1.1
path.py==12.5.0
pathos==0.3.2
pathspec==0.12.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
simpleitk==2.5.2
sklearn-pandas==2.2.0
testpath==0.6.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

0.2887448668478558

# 6. Current score

0.56326

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I replace the failing imports and the heavy model‑inference code with a lightweight fallback that reads the training labels, computes their empirical frequencies and uses those as predictions for every test row. This removes the missing‑module errors, guarantees a `.csv` submission file, and provides reasonable probabilities (the label‑wise prevalence) that should place the log‑loss near the target without altering the overall pipeline architecture.'
- What this solution (achieved 0.56369) has done: 'I keep the overall fallback‑only pipeline but replace the plain label‑frequency predictions with Laplace‑smoothed estimates (adding a small constant to each count). This gives slightly higher probabilities for rare positive cases, which typically lowers the weighted log‑loss and moves the current score nearer the target while preserving the original lightweight structure.'
- What this solution (achieved 0.5607) has done: 'I add NumPy for a simple combination calculation and modify the probability lookup so that for the `patient_overall` label it uses the probability that **any** cervical vertebra is fractured (1 − ∏ (1 − p_i)). This leverages the relationship between vertebra‑level and overall labels, giving a more realistic estimate for the heavily weighted overall label while keeping the original frequency‑based fallback unchanged for the vertebrae predictions. The rest of the pipeline is left intact, ensuring a valid CSV is still written.'
- What this solution (achieved 0.5635) has done: 'I keep the overall fallback‑only pipeline but improve the probability estimates: use Laplace‑smoothed prevalence for each label (adding 1 to each count) and, for the heavily weighted `patient_overall` label, return its own smoothed prevalence instead of the independence‑based combination. This better matches the true relationship in the data and should lower the weighted log‑loss, moving the score closer to the target while preserving the core logic.'
- What this solution (achieved 0.56119) has done: 'I replace the simple per‑label prevalence lookup with a slightly more realistic estimate for the heavily weighted `patient_overall` label: compute it as the probability that *any* cervical vertebra is fractured using the Laplace‑smoothed vertebra prevalences ( 1 − ∏ (1 − p_i) ). I also clip all probabilities to a tiny epsilon to avoid extreme 0/1 values that hurt log‑loss. This change keeps the overall fallback‑only pipeline intact while giving a better calibrated prediction for the most important label, moving the score toward the target.'
- What this solution (achieved 0.5635) has done: 'I replace the independence‑based estimate for the heavily‑weighted `patient_overall` label with its directly smoothed prevalence from the training data (the same Laplace smoothing used for the vertebrae). This gives a higher, more realistic probability for the overall label, which is weighted most heavily in the loss, and should reduce the overall log‑loss, moving the score closer to the target while keeping the rest of the fallback pipeline unchanged.'
- What this solution (achieved 0.56018) has done: 'I keep the fallback‑only pipeline but replace the naïve per‑label prevalence with a simple joint‑probability estimate.  
First, compute a smoothed prevalence for each vertebra and for the overall label.  
Then derive conditional probabilities P(Ci | patient_overall) using Laplace smoothing and combine them with the estimated overall fracture probability (computed as 1 − ∏(1 − p_i)).  
Finally, use these calibrated probabilities for all test rows and clip them to avoid extreme 0/1 values. This small statistical tweak is expected to lower the weighted log‑loss and move the score closer to the target while preserving the original lightweight structure.'
- What this solution (achieved 0.56326) has done: 'I adjust the fallback probability logic to use the smoothed prevalence of the `patient_overall` label directly (instead of estimating it from vertebra prevalences). This aligns the heavily‑weighted overall label with its true frequency in the training data, which should lower the weighted log‑loss and move the score closer to the target while keeping the overall lightweight pipeline unchanged.'
- What this solution (achieved 0.56119) has done: 'I replace the conditional‑probability logic with a simpler, more calibrated baseline: use the Laplace‑smoothed prevalence for each vertebra directly, and compute the `patient_overall` probability as the probability that any vertebra is fractured (1 − ∏ (1‑p_i)). This keeps the lightweight fallback while providing a better estimate for the heavily weighted overall label, which should lower the log‑loss toward the target.'
- What this solution (achieved 0.56326) has done: 'I replace the simple per‑label prevalence with a modest Bayesian estimate: compute the overall fracture prevalence, then derive conditional probabilities P(Ci | patient_overall) from the training data and combine them to obtain calibrated vertebra probabilities. For the “patient_overall” label I use its Laplace‑smoothed prevalence directly. These changes keep the fallback structure intact while providing more realistic probabilities, which should lower the weighted log‑loss toward the target.'
- What this solution (achieved 0.56119) has done: 'I replace the conditional‑blended vertebra probabilities with simple Laplace‑smoothed prevalence estimates and compute the `patient_overall` probability as the chance that any vertebra is fractured ( 1 − ∏ (1 − p_i) ). This keeps the lightweight fallback structure, avoids heavy modeling, and gives a more realistic calibration for the heavily weighted overall label, which should lower the log‑loss toward the target.'
- What this solution (achieved 0.5635) has done: 'I replace the independence‑based estimate for the heavily weighted `patient_overall` label with its Laplace‑smoothed prevalence directly (computed from the training data). This gives a more realistic overall probability, which should lower the weighted log‑loss and move the score toward the target while keeping the rest of the lightweight fallback unchanged.'
- What this solution (achieved 0.56243) has done: 'I keep the existing fallback imports but replace the simple per‑label prevalence with a small Bayesian calibration: compute smoothed conditional probabilities of each vertebra given the overall label and combine them with the overall fracture probability. For the “patient_overall” row I use the independence‑based probability that any vertebra is fractured ( 1 − ∏ (1−p_i) ). This modest statistical tweak should lower the weighted log‑loss, moving the score toward the target while preserving the original lightweight pipeline.'
- What this solution (achieved 0.5635) has done: 'I replace the conditional‑probability estimates with simple Laplace‑smoothed marginal prevalences for each vertebra, and use the directly smoothed overall fracture prevalence for the `patient_overall` label. This keeps the lightweight fallback structure while giving more realistic probabilities, especially for the heavily weighted overall label, which should reduce the weighted log‑loss and move the score toward the target.'
- What this solution (achieved 0.56326) has done: 'I replace the simple per‑label prevalence with a Laplace‑smoothed conditional estimate: compute P(Ci | patient_overall = 1) and P(Ci | patient_overall = 0) from the training data and blend them using the smoothed overall fracture probability. This adds modest statistical calibration without altering the overall fallback architecture and is expected to lower the weighted log‑loss, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import sys
import pandas as pd
import numpy as np

try:
    src_path = "../input/srccode/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from Utils.CommonTools.bbox import get_bbox, extend_bbox
    from Utils.post_processing import keep_largest_cervical_cc
    from Utils.Inference.nnunet_inference import NNUnetCTPredictor
    from Utils.CommonTools.dir import try_recursive_mkdir
    from Utils.CommonTools.NiiIO import read_from_DICOM_dir
    from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info

    print("==> Original Utils imported successfully")
except Exception as e:
    print(f"==> Utils import failed ({e}); using fallback stubs.")

    def get_bbox(*args, **kwargs):
        return None

    def extend_bbox(*args, **kwargs):
        return None

    def keep_largest_cervical_cc(mask, spacing):
        return mask

    class NNUnetCTPredictor:
        def __init__(self, *args, **kwargs):
            pass

        def resampling(self, ct):
            return ct

        def pre_processing(self, img):
            return img

        def sliding_window_inference(self, img):
            if img.ndim == 4:  # batch, C, H, W, D ?
                return img * 0
            return img * 0

    def try_recursive_mkdir(path):
        os.makedirs(path, exist_ok=True)

    def read_from_DICOM_dir(path):
        raise FileNotFoundError("DICOM reading not available in fallback mode")

    def resample(*args, **kwargs):
        return args[0]

    def copy_nii_info(*args, **kwargs):
        return args[0]

    def get_nii_info(*args, **kwargs):
        return {
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0, 0, 0),
            "size": (1, 1, 1),
            "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        }




## === cell 1
TRAIN_CSV = "../input/rsna-2022-cervical-spine-fracture-detection/train.csv"
TEST_CSV = "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
SAVE_CSV = "submission.csv"

train_df = pd.read_csv(TRAIN_CSV)

label_columns = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
vertebra_columns = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

n_samples = len(train_df)

pos_overall = train_df["patient_overall"].sum()
overall_prob = (pos_overall + 1) / (n_samples + 2)

cond_probs = {}
for col in vertebra_columns:
    overall_pos = train_df[train_df["patient_overall"] == 1]
    a = overall_pos[col].sum()  # overall=1 and vertebra=1
    b = len(overall_pos) - a  # overall=1 and vertebra=0

    overall_neg = train_df[train_df["patient_overall"] == 0]
    c = overall_neg[col].sum()  # overall=0 and vertebra=1
    d = len(overall_neg) - c  # overall=0 and vertebra=0

    p_given_pos = (a + 1) / (a + b + 2)
    p_given_neg = (c + 1) / (c + d + 2)

    blended = overall_prob * p_given_pos + (1 - overall_prob) * p_given_neg
    cond_probs[col] = blended

EPS = 1e-6
overall_prob = float(np.clip(overall_prob, EPS, 1 - EPS))
for col in cond_probs:
    cond_probs[col] = float(np.clip(cond_probs[col], EPS, 1 - EPS))

test_df = pd.read_csv(TEST_CSV)


def lookup_probability(row):
    typ = row["prediction_type"]
    if typ == "patient_overall":
        return overall_prob
    else:
        return cond_probs.get(typ, 0.5)


test_df["fractured"] = test_df.apply(lookup_probability, axis=1)

submission = test_df[["row_id", "fractured"]]
submission.to_csv(SAVE_CSV, index=False)

print(f"Submission written to {SAVE_CSV} with {len(submission)} rows.")
