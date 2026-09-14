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

3.11

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

0.2659648081841107

# 6. Current score

0.56032

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.00166) has done: 'I fix the import/path issues causing `NNUnetCTPredictor` to be undefined by replacing the missing `../input/src2-final/src` dependency with a safe, self-contained fallback that still produces a valid submission. Since the external model code and weights paths referenced in your script are not present in this environment, the minimal end-to-end fix is to generate predictions directly from `test.csv` with well-calibrated constant probabilities (low for C1–C7 and derived patient_overall), ensuring the submission has the correct `row_id` alignment and numeric `fractured` values. This run reliably within the time limit and create `submission.csv` in the working directory. The model/loop structure is left in place but guarded so it only runs if the required external assets actually exist.'
- What this solution (achieved 0.56032) has done: 'Your current score (1.00166, lower-is-better) is far worse than the target (0.26596), so we should improve (decrease) loss with the smallest legitimate change to the fallback path. Since the external nnUNet assets are missing, the only controllable lever is the constant probabilities; the best minimal improvement is to set those constants to match the empirical fracture prevalences from `train.csv` (a standard baseline that usually cuts logloss a lot). We also make `patient_overall` consistent by computing it from the per-level probabilities via `1 - ∏(1-p_Ci)` instead of a max, which is better calibrated for a “any fracture” label while keeping semantics (probabilities only). Everything else (I/O paths, submission alignment via `test.csv`/`sample_submission.csv`, and the guarded full pipeline) stays unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import math
import gc
import threading
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

src_path = "../input/src2-final/src"
HAS_EXTERNAL_SRC = os.path.isdir(src_path)

if HAS_EXTERNAL_SRC and (src_path not in sys.path):
    sys.path.insert(0, src_path)

print(f"HAS_EXTERNAL_SRC={HAS_EXTERNAL_SRC}")
if HAS_EXTERNAL_SRC:
    print("External src contents:", os.listdir(src_path)[:10])

NNUnetCTPredictor = None
ModelStage3 = None
if HAS_EXTERNAL_SRC:
    try:
        from Utils.CommonTools import sitk_base  # noqa: F401
        from Utils.CommonTools.NiiIO import read_from_DICOM_dir  # noqa: F401
        from Utils.PreProcessing.resampling import sitk_dummy_3D_resample  # noqa: F401
        from Utils.CommonTools.bbox import get_bbox, extend_bbox  # noqa: F401
        from Utils.post_processing import keep_largest_cervical_cc  # noqa: F401
        from Utils.Inference.nnunet_inference import (
            NNUnetCTPredictor as _NNUnetCTPredictor,
        )
        from Utils.CommonTools.dir import try_recursive_mkdir  # noqa: F401
        from Utils.CommonTools.sitk_base import (
            resample,
            copy_nii_info,
            get_nii_info,
        )  # noqa: F401
        from Training.Task_301_PostProcessing_Overall.model import Model as _ModelStage3

        NNUnetCTPredictor = _NNUnetCTPredictor
        ModelStage3 = _ModelStage3
        print("==> External imports success")
    except Exception as e:
        print(
            "==> External imports failed; will use fallback submission. Error:", repr(e)
        )
        NNUnetCTPredictor = None
        ModelStage3 = None



## === cell 1
if NNUnetCTPredictor is not None:
    import SimpleITK as sitk  # guaranteed above in this branch

    class PredictorStage2(NNUnetCTPredictor):
        def __init__(self, *args, **kwargs):
            super(PredictorStage2, self).__init__(*args, **kwargs)

        def resampling(self, ct_nii):
            ori_spacing = ct_nii.GetSpacing()[::-1]  # to z,y,x
            ori_size = ct_nii.GetSize()[::-1]
            new_spacing = self.plan["plans_per_stage"][self.plan_stage][
                "current_spacing"
            ]

            new_size = [int(math.ceil(ori_size[0] * ori_spacing[0] / 0.8)), 224, 224]

            new_spacing[0] = 0.8
            new_spacing[1] = ori_size[1] * ori_spacing[1] / 224.0
            new_spacing[2] = ori_size[2] * ori_spacing[2] / 224.0

            do_resampling = np.any(
                np.abs(np.array(ori_spacing) - np.array(new_spacing))
                > self.resampling_tolerance
            )
            if do_resampling:
                ct_nii = sitk_dummy_3D_resample(
                    ct_nii,
                    new_spacing=new_spacing[::-1],
                    new_size=new_size[::-1],
                    interp_xy=self.resampling_mode,
                    interp_z=sitk.sitkNearestNeighbor,
                    out_dtype=self.resampling_dtype,
                    constant_value=self.resampling_constance_value,
                )
            else:
                print(
                    f"==> No necessary to do resampling ori {ori_spacing}, new: {new_spacing}"
                )
            return ct_nii

    class PredictorStage3:
        def __init__(self, list_model_pth, device, tta=False, tta_flip_axis=(4,)):
            self.list_model_pth = list_model_pth
            self.device = device
            self.tta = tta
            self.tta_flip_axis = tta_flip_axis

            self.list_model = None

            self.in_ch = 2
            self.out_ch = 1
            self.list_ch = [-1, 16, 32, 64, 128]

            self.init_model()

        def init_model(self):
            import torch

            with torch.no_grad():
                self.list_model = []
                for i in range(len(self.list_model_pth)):
                    model = ModelStage3(
                        in_ch=self.in_ch,
                        out_ch=self.out_ch,
                        list_ch=self.list_ch,
                        random_init=False,
                    )
                    ckpt = torch.load(self.list_model_pth[i], map_location="cpu")
                    model.load_state_dict(ckpt)
                    model.eval()
                    model = model.to(self.device)
                    self.list_model.append(model)
                    print(
                        f"==> Init model from {self.list_model_pth[i]} to device {self.device}"
                    )

        def predict(self, image):
            import torch

            with torch.no_grad():
                list_pred = []
                input_ori = image.copy()

                if self.tta:
                    p_flip_z = (0, 1) if 2 in self.tta_flip_axis else (0,)
                    p_flip_y = (0, 1) if 3 in self.tta_flip_axis else (0,)
                    p_flip_x = (0, 1) if 4 in self.tta_flip_axis else (0,)
                else:
                    p_flip_z = (0,)
                    p_flip_y = (0,)
                    p_flip_x = (0,)

                for flip_z in p_flip_z:
                    for flip_y in p_flip_y:
                        for flip_x in p_flip_x:
                            patch_input = (
                                torch.from_numpy(input_ori).to(self.device).unsqueeze(0)
                            )

                            flip_axis = []
                            if flip_z == 1:
                                flip_axis.append(2)
                            if flip_y == 1:
                                flip_axis.append(3)
                            if flip_x == 1:
                                flip_axis.append(4)

                            do_flip = (flip_z == 1) or (flip_y == 1) or (flip_x == 1)
                            if do_flip:
                                patch_input = torch.flip(patch_input, dims=flip_axis)

                            for model in self.list_model:
                                pred = model(patch_input)[0]
                                pred = torch.sigmoid(pred[0, 0])
                                list_pred.append(pred.cpu().numpy())
                return float(np.mean(list_pred))

    class DICOMReader(threading.Thread):
        def __init__(self, func, args=()):
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
            predictor_stage3,
            extend_roi=(5.0, 5.0, 5.0),
        ):
            self.predictor_stage1 = predictor_stage1
            self.predictor_stage2 = predictor_stage2
            self.predictor_stage3 = predictor_stage3
            self.extend_roi = extend_roi

            self.params = {
                "alpha": [0.055, 0.044, 0.052, 0.05, 0.07, 0.077, 0.09, 0.024],
                "beta": [0.475, 0.34, 0.37, 0.38, 0.31, 0.35, 0.38, 0.36],
                "min_score": [0.116, 0.01, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048],
                "max_score": [0.99, 0.999, 0.993, 0.99, 1.0, 0.943, 0.997, 0.999],
            }
            self.results = {}

        def get_c1_c7_bbox(self, pred, image_spacing):
            c1_c7_bbox = get_bbox(pred > 0)
            if c1_c7_bbox is None:
                return None
            c1_c7_bbox = extend_bbox(
                c1_c7_bbox,
                max_shape=pred.shape,
                list_extend_length=self.extend_roi,
                spacing=image_spacing,
                approximate_method=np.ceil,
            )
            return c1_c7_bbox

        def predict_stage1(self, ct_nii):
            time_start = time.time()
            ori_nii_info = get_nii_info(ct_nii)
            ct_nii = self.predictor_stage1.resampling(ct_nii)
            print(f"                  Resampling use: {time.time() - time_start}")

            time_start = time.time()
            image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
            image = self.predictor_stage1.pre_processing(image)
            print(f"                  Pre_processing use: {time.time() - time_start}")

            time_start = time.time()
            pred = self.predictor_stage1.sliding_window_inference(image)
            print(f"                  Model forward use: {time.time() - time_start}")

            time_start = time.time()
            pred = np.argmax(pred, axis=0)
            pred = keep_largest_cervical_cc(pred, ct_nii.GetSpacing()[::-1])
            pred[pred > 7] = 0
            print(f"                  Post processing use: {time.time() - time_start}")

            time_start = time.time()
            pred_nii = sitk.GetImageFromArray(np.uint8(pred))
            pred_nii = copy_nii_info(ct_nii, pred_nii)
            pred_nii = resample(
                pred_nii,
                new_spacing=ori_nii_info["spacing"],
                new_origin=ori_nii_info["origin"],
                new_size=ori_nii_info["size"],
                new_direction=ori_nii_info["direction"],
                center_origin=None,
                interp=sitk.sitkNearestNeighbor,
                dtype=sitk.sitkUInt8,
                constant_value=0,
            )
            pred = sitk.GetArrayFromImage(pred_nii)
            print(f"                  Resampling back use: {time.time() - time_start}")
            return pred

        def predict_stage2(self, ct_nii):
            time_start = time.time()
            ori_nii_info = get_nii_info(ct_nii)
            ct_nii = self.predictor_stage2.resampling(ct_nii)
            print(f"                  Resampling use: {time.time() - time_start}")

            time_start = time.time()
            image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
            image = self.predictor_stage2.pre_processing(image)
            print(f"                  Pre_processing use: {time.time() - time_start}")

            time_start = time.time()
            pred = self.predictor_stage2.sliding_window_inference(image)
            print(f"                  Model forward use: {time.time() - time_start}")

            time_start = time.time()
            pred = pred[1]  # 0 for background, 1 for foreground
            print(f"                  Post processing use: {time.time() - time_start}")

            time_start = time.time()
            pred_nii = sitk.GetImageFromArray(pred)
            pred_nii = copy_nii_info(ct_nii, pred_nii)
            pred_nii = resample(
                pred_nii,
                new_spacing=ori_nii_info["spacing"],
                new_origin=ori_nii_info["origin"],
                new_size=ori_nii_info["size"],
                new_direction=ori_nii_info["direction"],
                center_origin=None,
                interp=sitk.sitkLinear,
                dtype=sitk.sitkFloat32,
                constant_value=0.0,
            )
            pred = sitk.GetArrayFromImage(pred_nii)
            print(f"                  Resampling back use: {time.time() - time_start}")
            return pred

        def predict_stage3(self, pred_c1_c7, pred_fracture):
            pred_c1_c7_nii = sitk.GetImageFromArray(pred_c1_c7)
            pred_fracture_nii = sitk.GetImageFromArray(pred_fracture)

            new_size = (96, 96, 96)
            new_spacing = list(
                np.array(pred_fracture_nii.GetSize()[::-1]) / np.array(new_size)
            )

            pred_fracture_nii = sitk_base.resample(
                pred_fracture_nii,
                new_spacing[::-1],
                new_origin=None,
                new_size=new_size[::-1],
                new_direction=None,
                center_origin=None,
                interp=sitk.sitkLinear,
                dtype=sitk.sitkFloat32,
                constant_value=0,
            )
            pred_c1_c7_nii = sitk_base.resample(
                pred_c1_c7_nii,
                new_spacing[::-1],
                new_origin=None,
                new_size=new_size[::-1],
                new_direction=None,
                center_origin=None,
                interp=sitk.sitkNearestNeighbor,
                dtype=sitk.sitkUInt8,
                constant_value=0,
            )

            pred_c1_c7 = sitk.GetArrayFromImage(pred_c1_c7_nii)
            pred_fracture = sitk.GetArrayFromImage(pred_fracture_nii)

            input_ = np.concatenate(
                (pred_fracture[np.newaxis], pred_c1_c7[np.newaxis]), axis=0
            )
            score = self.predictor_stage3.predict(input_)
            return score

        def get_score(self, pred_c1_c7, pred_fracture):
            output = np.zeros(8, np.float32)  # Overall, C1-C7

            if (pred_c1_c7 is not None) and (pred_fracture is not None):
                for C_i in range(8):
                    if C_i == 0:
                        roi_fracture = pred_fracture[
                            np.logical_and(
                                pred_fracture >= self.params["alpha"][C_i],
                                pred_c1_c7 > 0,
                            )
                        ]
                    else:
                        roi_fracture = pred_fracture[
                            np.logical_and(
                                pred_fracture >= self.params["alpha"][C_i],
                                pred_c1_c7 == C_i,
                            )
                        ]

                    if len(roi_fracture) == 0:
                        output[C_i] = self.params["min_score"][C_i]
                    else:
                        output[C_i] = max(
                            self.params["min_score"][C_i],
                            min(
                                self.params["max_score"][C_i],
                                np.percentile(
                                    roi_fracture, 100 * self.params["beta"][C_i]
                                ),
                            ),
                        )
            else:
                for C_i in range(8):
                    output[C_i] = self.params["min_score"][C_i]
            output[0] = max(self.params["min_score"][0], np.max(output[1:]))
            return output

        @staticmethod
        def read_DICOM_multi_thread(list_DICOM_dirs):
            list_thread = []
            list_outputs = []
            for DICOM_dir in list_DICOM_dirs:
                cur_thread = DICOMReader(func=read_from_DICOM_dir, args=(DICOM_dir,))
                cur_thread.start()
                list_thread.append(cur_thread)
            for cur_thread in list_thread:
                list_outputs.append(cur_thread.get_result())
            list_thread.clear()
            return list_outputs

        def predict(self, list_test_files, num_thread=4, output_dir=None):
            import torch

            with torch.no_grad():
                overall_time_start = time.time()
                num_split = math.ceil(len(list_test_files) / num_thread)
                for split_i in range(num_split):
                    cur_test_files = list_test_files[
                        num_thread * split_i : num_thread * (split_i + 1)
                    ]
                    cur_case_ids = [
                        test_file.split("/")[-1] for test_file in cur_test_files
                    ]

                    print(f"==> Predicting {split_i}: {cur_case_ids}")

                    time_start = time.time()
                    cur_ct_niis = self.read_DICOM_multi_thread(cur_test_files)
                    print(
                        f"    Finish Reading use : {time.time() - time_start} seconds"
                    )

                    time_start = time.time()
                    for case_i in range(len(cur_ct_niis)):
                        case_id = cur_case_ids[case_i]
                        ct_nii = cur_ct_niis[case_i]

                        ori_nii_info = get_nii_info(ct_nii)
                        pred_1 = self.predict_stage1(ct_nii)
                        c1_c7_bbox = self.get_c1_c7_bbox(
                            pred_1, ori_nii_info["spacing"][::-1]
                        )

                        if c1_c7_bbox is not None:
                            bz, ez, by, ey, bx, ex = c1_c7_bbox
                            roi_ct_nii = ct_nii[bx : ex + 1, by : ey + 1, bz : ez + 1]
                            roi_pred_1 = pred_1[bz : ez + 1, by : ey + 1, bx : ex + 1]
                            roi_pred_2 = self.predict_stage2(roi_ct_nii)
                        else:
                            roi_pred_1 = None
                            roi_pred_2 = None

                        if roi_pred_1 is None:
                            roi_pred_1 = np.zeros((2, 2, 2), np.uint8)
                            roi_pred_2 = np.zeros((2, 2, 2), np.float32)

                        score = self.get_score(roi_pred_1, roi_pred_2)
                        score[0] = self.predict_stage3(roi_pred_1, roi_pred_2)
                        self.results[case_id] = score
                        print(f"     {case_id} -> {score}")

                    print(
                        f"    Finish this split use : {time.time() - time_start} seconds"
                    )
                    print(
                        f"    Overall use : {time.time() - overall_time_start} seconds"
                    )
                    gc.collect()




## === cell 2
time_start = time.time()

BASE_DIR = "/kaggle/data/rsna-2022-cervical-spine-fracture-detection"
if not os.path.exists(BASE_DIR):
    BASE_DIR = "../input/rsna-2022-cervical-spine-fracture-detection"

TEST_CSV = os.path.join(BASE_DIR, "test.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
SAVE_CSV = "submission.csv"

test_df = pd.read_csv(TEST_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

USE_FULL_PIPELINE = False
if NNUnetCTPredictor is not None and torch is not None and sitk is not None:
    required_paths = [
        "../input/models-final/models/stage1_0.model",
        "../input/models-final/models/stage2_0.model",
        "../input/plans-nnunet/stage1.pkl",
        "../input/plans-nnunet/stage2.pkl",
    ]
    USE_FULL_PIPELINE = all(os.path.exists(p) for p in required_paths)

if USE_FULL_PIPELINE:
    DATA_DIR = os.path.join(BASE_DIR, "test_images")
    list_model_C1_C7_segmentation = [
        "../input/models-final/models/stage1_0.model",
        "../input/models-final/models/stage1_1.model",
        "../input/models-final/models/stage1_2.model",
    ]
    plan_C1_C7_segmentation = "../input/plans-nnunet/stage1.pkl"

    list_model_fracture_detection = [
        "../input/models-final/models/stage2_0.model",
    ]
    plan_fracture_detection = "../input/plans-nnunet/stage2.pkl"
    list_model_post_processing = [
        "../input/models-final/models/stage3_111.model",
        "../input/models-final/models/stage3_222.model",
        "../input/models-final/models/stage3_333.model",
        "../input/models-final/models/stage3_444.model",
        "../input/models-final/models/stage3_555.model",
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predictor_1 = NNUnetCTPredictor(
        list_model_pth=list_model_C1_C7_segmentation,
        plan_file=plan_C1_C7_segmentation,
        plan_stage=-1,
        device=device,
        use_gaussian_for_sliding_window=True,
        patch_size=None,
        stride=None,
        tta=False,
        tta_flip_axis=(4,),
        resampling_tolerance=0.01,
        resampling_mode=sitk.sitkNearestNeighbor,
        resampling_dtype=sitk.sitkInt16,
        resampling_constance_value=-1024,
        remove_air_CT=True,
    )
    predictor_2 = PredictorStage2(
        list_model_pth=list_model_fracture_detection,
        plan_file=plan_fracture_detection,
        plan_stage=-1,
        device=device,
        use_gaussian_for_sliding_window=True,
        patch_size=(96, 224, 224),
        stride=(96, 224, 224),
        tta=True,
        tta_flip_axis=(4,),
        resampling_tolerance=0.01,
        resampling_mode=sitk.sitkNearestNeighbor,
        resampling_dtype=sitk.sitkInt16,
        resampling_constance_value=-1024,
        remove_air_CT=False,
        save_dtype=np.float32,
    )
    predictor_3 = PredictorStage3(
        list_model_pth=list_model_post_processing,
        device=device,
        tta=True,
        tta_flip_axis=(4,),
    )
    c2f_predictor = FractureDetector(
        predictor_stage1=predictor_1,
        predictor_stage2=predictor_2,
        predictor_stage3=predictor_3,
    )

    list_DICOM_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)]
    c2f_predictor.predict(list_test_files=list_DICOM_dirs)

    pred_map = {}
    for case_id, score in c2f_predictor.results.items():
        pred_map[(case_id, "patient_overall")] = float(score[0])
        for i in range(1, 8):
            pred_map[(case_id, f"C{i}")] = float(score[i])

    sub = sample_df.copy()
    fractured = []
    for _, r in test_df.iterrows():
        fractured.append(
            pred_map.get((r["StudyInstanceUID"], r["prediction_type"]), 0.05)
        )
    sub["fractured"] = np.clip(fractured, 1e-5, 1 - 1e-5)
    sub.to_csv(SAVE_CSV, index=False)
else:
    train_df = pd.read_csv(TRAIN_CSV)
    label_cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]
    prevalence = train_df[label_cols].mean(numeric_only=True).to_dict()

    default_prob_by_type = {
        f"C{i}": float(prevalence.get(f"C{i}", 0.03)) for i in range(1, 8)
    }

    studies = test_df["StudyInstanceUID"].unique()
    per_study = {}
    for sid in studies:
        c_probs = [default_prob_by_type[f"C{i}"] for i in range(1, 8)]
        p_any_from_levels = 1.0 - float(np.prod([1.0 - p for p in c_probs]))
        p_overall = max(
            float(prevalence.get("patient_overall", 0.08)), p_any_from_levels
        )

        per_study[sid] = {f"C{i}": default_prob_by_type[f"C{i}"] for i in range(1, 8)}
        per_study[sid]["patient_overall"] = p_overall

    sub = sample_df.copy()
    pred = np.empty(len(test_df), dtype=np.float32)
    for idx, r in enumerate(test_df.itertuples(index=False)):
        pred[idx] = per_study[r.StudyInstanceUID][r.prediction_type]

    sub["fractured"] = np.clip(pred, 1e-5, 1 - 1e-5).astype(float)
    sub.to_csv(SAVE_CSV, index=False)

print(
    f"Wrote {SAVE_CSV} with shape {pd.read_csv(SAVE_CSV).shape} in {time.time()-time_start:.1f}s"
)
print(pd.read_csv(SAVE_CSV).head())
