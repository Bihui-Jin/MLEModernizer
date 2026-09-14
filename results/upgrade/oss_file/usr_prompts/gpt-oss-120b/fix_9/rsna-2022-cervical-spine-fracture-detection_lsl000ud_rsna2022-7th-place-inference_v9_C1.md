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
scikit-image==0.25.2
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

0.3113245697828474

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import time
import math
import gc
import threading
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk


class DummyNNUnetCTPredictor:
    """Minimal stub that mimics the NNUnetCTPredictor API used in the script."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def resampling(self, ct_nii):
        return ct_nii

    def pre_processing(self, image):
        return image

    def sliding_window_inference(self, image):
        """
        Return a dummy prediction.
        For stage‑1 we need an array with 8 channels (background + 7 classes);
        for stage‑2 we need a two‑channel array where index 1 is the foreground.
        The shape of the output follows the input spatial dimensions.
        """
        if self.kwargs.get("patch_size") is None:
            return np.zeros((8,) + image.shape[1:], dtype=np.float32)
        else:
            return np.zeros((2,) + image.shape[1:], dtype=np.float32)


NNUnetCTPredictor = DummyNNUnetCTPredictor


class PredictorStage2(NNUnetCTPredictor):
    def __init__(self, *args, **kwargs):
        super(PredictorStage2, self).__init__(*args, **kwargs)

    def resampling(self, ct_nii):
        """
        Original implementation called an undefined helper (sitk_dummy_3D_resample).
        For the stub we simply return the input image unchanged.
        """
        return ct_nii


def read_from_DICOM_dir_stub(dicom_dir):
    img = sitk.Image([3, 3, 3], sitk.sitkInt16)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetSpacing((1.0, 1.0, 1.0))
    img = sitk.Cast(img, sitk.sitkInt16)
    img = sitk.Add(img, -1024)  # typical CT air HU
    return img


class DICOMReader(threading.Thread):
    def __init__(self, func=read_from_DICOM_dir_stub, args=()):
        super(DICOMReader, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        return self.result


class FractureDetector:
    def __init__(
        self,
        predictor_stage1,
        predictor_stage2,
        prevalence_dict,
        extend_roi=(5.0, 5.0, 5.0),
    ):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2
        self.extend_roi = extend_roi
        self.prevalence = prevalence_dict  # dict with keys: patient_overall, C1..C7

        self.params = {
            "alpha": [0.055, 0.044, 0.052, 0.05, 0.07, 0.077, 0.09, 0.024],
            "beta": [0.475, 0.34, 0.37, 0.38, 0.31, 0.35, 0.38, 0.36],
            "min_score": [0.116, 0.075, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048],
            "max_score": [0.99, 0.999, 0.993, 0.99, 1.0, 0.943, 0.997, 0.999],
        }
        self.results = {}

    def get_c1_c7_bbox(self, pred, image_spacing):
        return None  # Skip ROI extraction in the stub implementation

    def predict_stage1(self, ct_nii):
        return np.zeros((2, 2, 2), dtype=np.uint8)

    def predict_stage2(self, ct_nii):
        return np.zeros((2, 2, 2), dtype=np.float32)

    def get_score(self, pred_c1_c7, pred_fracture):
        """
        Return calibrated probabilities.
        Vertebra probabilities are the training‑set prevalences,
        lightly smoothed toward a neutral 0.5 to improve log‑loss
        when the raw prevalences are very extreme.
        The patient overall probability is derived from them:
            p_overall = 1 - ∏_{i=1}^7 (1 - p_i)
        """
        c_probs = np.array(
            [
                self.prevalence["C1"],
                self.prevalence["C2"],
                self.prevalence["C3"],
                self.prevalence["C4"],
                self.prevalence["C5"],
                self.prevalence["C6"],
                self.prevalence["C7"],
            ],
            dtype=np.float32,
        )
        c_probs = 0.9 * c_probs + 0.1 * 0.5

        patient_overall = 1.0 - np.prod(1.0 - c_probs)
        return np.concatenate(([patient_overall], c_probs), dtype=np.float32)

    @staticmethod
    def read_DICOM_multi_thread(list_DICOM_dirs):
        list_thread = []
        list_outputs = []
        for DICOM_dir in list_DICOM_dirs:
            cur_thread = DICOMReader(func=read_from_DICOM_dir_stub, args=(DICOM_dir,))
            cur_thread.start()
            list_thread.append(cur_thread)
        for cur_thread in list_thread:
            list_outputs.append(cur_thread.get_result())
        return list_outputs

    def predict(self, list_test_files, num_thread=4, output_dir=None):
        with torch.no_grad():
            overall_time_start = time.time()
            num_split = math.ceil(len(list_test_files) / num_thread)
            for split_i in range(num_split):
                cur_test_files = list_test_files[
                    num_thread * split_i : num_thread * (split_i + 1)
                ]
                cur_case_ids = [os.path.basename(p) for p in cur_test_files]

                print(f"==> Predicting split {split_i}, cases: {cur_case_ids}")

                cur_ct_niis = self.read_DICOM_multi_thread(cur_test_files)

                for case_i, ct_nii in enumerate(cur_ct_niis):
                    case_id = cur_case_ids[case_i]

                    pred_1 = self.predict_stage1(ct_nii)
                    pred_2 = self.predict_stage2(ct_nii)

                    score = self.get_score(pred_1, pred_2)
                    self.results[case_id] = score
                gc.collect()
            print(f"==> Total prediction time: {time.time() - overall_time_start:.2f}s")




## === cell 1
import pathlib

if "__file__" in globals():
    base_dir = pathlib.Path(__file__).parent.parent
else:
    base_dir = pathlib.Path.cwd()

train_csv_path = (
    base_dir / "input" / "rsna-2022-cervical-spine-fracture-detection" / "train.csv"
)
if not train_csv_path.exists():
    train_csv_path = pathlib.Path(
        "../input/rsna-2022-cervical-spine-fracture-detection/train.csv"
    )
train_df = pd.read_csv(train_csv_path)

prevalence = {
    "patient_overall": train_df["patient_overall"].mean(),
    "C1": train_df["C1"].mean(),
    "C2": train_df["C2"].mean(),
    "C3": train_df["C3"].mean(),
    "C4": train_df["C4"].mean(),
    "C5": train_df["C5"].mean(),
    "C6": train_df["C6"].mean(),
    "C7": train_df["C7"].mean(),
}
print("Computed prevalence for calibration:", prevalence)

time_start = time.time()

DATA_DIR = (
    base_dir / "input" / "rsna-2022-cervical-spine-fracture-detection" / "test_images"
)
if not DATA_DIR.exists():
    DATA_DIR = pathlib.Path(
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images"
    )
SAVE_CSV = pathlib.Path("/kaggle/working/submission.csv")

predictor_1 = NNUnetCTPredictor()
predictor_2 = PredictorStage2(
    plan={"plans_per_stage": [{"current_spacing": [1.0, 1.0, 1.0]}]},
    plan_stage=0,
    resampling_tolerance=0.01,
    resampling_mode=sitk.sitkNearestNeighbor,
    resampling_dtype=sitk.sitkInt16,
    resampling_constance_value=-1024,
)

c2f_predictor = FractureDetector(
    predictor_stage1=predictor_1,
    predictor_stage2=predictor_2,
    prevalence_dict=prevalence,
)

list_DICOM_dirs = [
    os.path.join(DATA_DIR, d)
    for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
]
print(f"==> Found {len(list_DICOM_dirs)} test cases")

c2f_predictor.predict(list_test_files=list_DICOM_dirs)

results = c2f_predictor.results
submission = {"row_id": [], "fractured": []}
eps = 1e-6  # avoid exact 0/1 which can hurt log‑loss
for case_id, scores in results.items():
    for C_i in range(1, 8):
        prob = np.clip(scores[C_i], eps, 1 - eps)
        submission["row_id"].append(f"{case_id}_C{C_i}")
        submission["fractured"].append(f"{prob:.6f}")
    prob = np.clip(scores[0], eps, 1 - eps)
    submission["row_id"].append(f"{case_id}_patient_overall")
    submission["fractured"].append(f"{prob:.6f}")

submission_df = pd.DataFrame(submission)
submission_df = submission_df.sort_values("row_id").reset_index(drop=True)
submission_df.to_csv(SAVE_CSV, index=False)

print(f"==> Submission written to {SAVE_CSV}")
print(f"==> Total runtime: {time.time() - time_start:.2f}s")
