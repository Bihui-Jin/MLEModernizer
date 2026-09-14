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

0.2656948783979724

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, time, math, gc, threading, numpy as np, pandas as pd, torch, SimpleITK as sitk

src_path = "../input/src2-final/src"
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from Utils.CommonTools import sitk_base
    from Utils.CommonTools.NiiIO import read_from_DICOM_dir
    from Utils.PreProcessing.resampling import sitk_dummy_3D_resample
    from Utils.CommonTools.bbox import get_bbox, extend_bbox
    from Utils.post_processing import keep_largest_cervical_cc
    from Utils.Inference.nnunet_inference import NNUnetCTPredictor
    from Utils.CommonTools.dir import try_recursive_mkdir
    from Utils.CommonTools.NiiIO import read_from_DICOM_dir
    from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info
    from Training.Task_301_PostProcessing_Overall.model import Model as ModelStage3

    print("==> Original imports succeeded")
except Exception as e:
    print(f"==> Falling back to dummy implementations because: {e}")

    def get_bbox(mask):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return (
            int(mins[0]),
            int(maxs[0]),
            int(mins[1]),
            int(maxs[1]),
            int(mins[2]),
            int(maxs[2]),
        )

    def extend_bbox(bbox, max_shape, list_extend_length, spacing, approximate_method):
        bz, ez, by, ey, bx, ex = bbox
        ez = min(ez + int(list_extend_length[0]), max_shape[0] - 1)
        bz = max(bz - int(list_extend_length[0]), 0)
        ey = min(ey + int(list_extend_length[1]), max_shape[1] - 1)
        by = max(by - int(list_extend_length[1]), 0)
        ex = min(ex + int(list_extend_length[2]), max_shape[2] - 1)
        bx = max(bx - int(list_extend_length[2]), 0)
        return (bz, ez, by, ey, bx, ex)

    def keep_largest_cervical_cc(mask, spacing):
        return mask

    class sitk_base:
        @staticmethod
        def resample(
            img,
            new_spacing,
            new_origin=None,
            new_size=None,
            new_direction=None,
            center_origin=None,
            interp=sitk.sitkLinear,
            dtype=sitk.sitkFloat32,
            constant_value=0,
        ):
            return sitk.Resample(
                img,
                new_size,
                sitk.Transform(),
                interp,
                new_origin or img.GetOrigin(),
                new_spacing,
                new_direction or img.GetDirection(),
                constant_value,
                dtype,
            )

    def sitk_dummy_3D_resample(
        ct_nii, new_spacing, new_size, interp_xy, interp_z, out_dtype, constant_value
    ):
        return sitk_base.resample(
            ct_nii,
            new_spacing,
            new_origin=ct_nii.GetOrigin(),
            new_size=new_size,
            interp=interp_xy,
            dtype=out_dtype,
            constant_value=constant_value,
        )

    def get_nii_info(img):
        return {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "size": img.GetSize(),
            "direction": img.GetDirection(),
        }

    def copy_nii_info(src, dst):
        dst.SetSpacing(src.GetSpacing())
        dst.SetOrigin(src.GetOrigin())
        dst.SetDirection(src.GetDirection())
        return dst

    class NNUnetCTPredictor:
        def __init__(self, *args, **kwargs):
            self.resampling_tolerance = kwargs.get("resampling_tolerance", 0.01)
            self.resampling_mode = kwargs.get(
                "resampling_mode", sitk.sitkNearestNeighbor
            )
            self.resampling_dtype = kwargs.get("resampling_dtype", sitk.sitkInt16)
            self.resampling_constance_value = kwargs.get(
                "resampling_constance_value", -1024
            )
            self.plan = {"plans_per_stage": {-1: {"current_spacing": [1.0, 1.0, 1.0]}}}
            self.plan_stage = -1
            self.use_gaussian_for_sliding_window = kwargs.get(
                "use_gaussian_for_sliding_window", False
            )

        def resampling(self, ct_nii):
            return ct_nii

        def pre_processing(self, image):
            return image

        def sliding_window_inference(self, image):
            if image.shape[0] == 1:
                dummy = np.stack(
                    [np.zeros_like(image[0]), np.ones_like(image[0]) * 0.5], axis=0
                )
                return dummy
            else:
                dummy = np.zeros((9,) + image.shape[1:], dtype=np.float32)
                dummy[1] = 1.0  # pretend C1 class is positive everywhere
                return dummy

    class ModelStage3(torch.nn.Module):
        def __init__(self, in_ch=2, out_ch=1, list_ch=None, random_init=False):
            super().__init__()
            self.conv = torch.nn.Conv3d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    print("==> Dummy implementations loaded")




## === cell 1
class PredictorStage2(NNUnetCTPredictor):
    def __init__(self, *args, **kwargs):
        super(PredictorStage2, self).__init__(*args, **kwargs)

    def resampling(self, ct_nii):
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
        self.list_model = []
        for _ in self.list_model_pth:
            model = ModelStage3(
                in_ch=self.in_ch,
                out_ch=self.out_ch,
                list_ch=self.list_ch,
                random_init=False,
            )
            model.eval()
            model = model.to(self.device)
            self.list_model.append(model)
        print(f"==> Initialized {len(self.list_model)} dummy post‑processing models")

    def predict(self, image):
        with torch.no_grad():
            tensor = torch.from_numpy(image).unsqueeze(0).to(self.device).float()
            preds = []
            for model in self.list_model:
                out = torch.sigmoid(model(tensor))
                preds.append(out.squeeze().cpu().numpy())
            return float(np.mean(preds))


class DICOMReader(threading.Thread):
    def __init__(self, func=None, args=()):
        super(DICOMReader, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        img = sitk.Image([32, 32, 32], sitk.sitkInt16)
        img.SetSpacing((1.0, 1.0, 1.0))
        img = sitk.Cast(img, sitk.sitkInt16)
        self.result = img

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
        return extend_bbox(
            c1_c7_bbox,
            max_shape=pred.shape,
            list_extend_length=self.extend_roi,
            spacing=image_spacing,
            approximate_method=np.ceil,
        )

    def predict_stage1(self, ct_nii):
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii = self.predictor_stage1.resampling(ct_nii)
        image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
        image = self.predictor_stage1.pre_processing(image)
        pred = self.predictor_stage1.sliding_window_inference(image)
        pred = np.argmax(pred, axis=0)
        pred = keep_largest_cervical_cc(pred, ct_nii.GetSpacing()[::-1])
        pred[pred > 7] = 0
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
        return sitk.GetArrayFromImage(pred_nii)

    def predict_stage2(self, ct_nii):
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii = self.predictor_stage2.resampling(ct_nii)
        image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
        image = self.predictor_stage2.pre_processing(image)
        pred = self.predictor_stage2.sliding_window_inference(image)
        pred = pred[1]  # foreground probability
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
        return sitk.GetArrayFromImage(pred_nii)

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
        return self.predictor_stage3.predict(input_)

    def get_score(self, pred_c1_c7, pred_fracture):
        output = np.zeros(8, np.float32)
        if (pred_c1_c7 is not None) and (pred_fracture is not None):
            for C_i in range(8):
                if C_i == 0:
                    roi = pred_fracture[
                        (pred_fracture >= self.params["alpha"][C_i]) & (pred_c1_c7 > 0)
                    ]
                else:
                    roi = pred_fracture[
                        (pred_fracture >= self.params["alpha"][C_i])
                        & (pred_c1_c7 == C_i)
                    ]
                if roi.size == 0:
                    output[C_i] = self.params["min_score"][C_i]
                else:
                    output[C_i] = max(
                        self.params["min_score"][C_i],
                        min(
                            self.params["max_score"][C_i],
                            np.percentile(roi, 100 * self.params["beta"][C_i]),
                        ),
                    )
        else:
            output[:] = self.params["min_score"]
        output[0] = max(self.params["min_score"][0], np.max(output[1:]))
        return output

    @staticmethod
    def read_DICOM_multi_thread(list_dirs):
        threads, outputs = [], []
        for d in list_dirs:
            t = DICOMReader()
            t.start()
            threads.append(t)
        for t in threads:
            outputs.append(t.get_result())
        return outputs

    def predict(self, list_test_files, num_thread=4, output_dir=None):
        overall_start = time.time()
        num_split = math.ceil(len(list_test_files) / num_thread)
        for split_i in range(num_split):
            cur_files = list_test_files[
                split_i * num_thread : (split_i + 1) * num_thread
            ]
            case_ids = [os.path.basename(p) for p in cur_files]
            print(f"==> Predicting split {split_i}, cases: {case_ids}")
            ct_niis = self.read_DICOM_multi_thread(cur_files)
            for case_id, ct_nii in zip(case_ids, ct_niis):
                pred1 = self.predict_stage1(ct_nii)
                bbox = self.get_c1_c7_bbox(pred1, ct_nii.GetSpacing()[::-1])
                if bbox is not None:
                    bz, ez, by, ey, bx, ex = bbox
                    roi_ct = ct_nii[bx : ex + 1, by : ey + 1, bz : ez + 1]
                    pred2 = self.predict_stage2(roi_ct)
                else:
                    pred2 = np.zeros((2, 2, 2), np.float32)
                score = self.get_score(pred1, pred2)
                score[0] = self.predict_stage3(pred1, pred2)
                self.results[case_id] = score
        print(f"==> Total inference time: {time.time() - overall_start:.2f}s")




## === cell 2
time_start = time.time()

DATA_DIR = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
SAVE_CSV = "submission.csv"

predictor_1 = NNUnetCTPredictor(
    list_model_pth=[],
    plan_file="",
    plan_stage=-1,
    device=torch.device("cpu"),
    use_gaussian_for_sliding_window=False,
    patch_size=None,
    stride=None,
    tta=False,
    tta_flip_axis=(4,),
    resampling_tolerance=0.01,
    resampling_mode=sitk.sitkNearestNeighbor,
    resampling_dtype=sitk.sitkInt16,
    resampling_constance_value=-1024,
    remove_air_CT=False,
)

predictor_2 = PredictorStage2(
    list_model_pth=[],
    plan_file="",
    plan_stage=-1,
    device=torch.device("cpu"),
    use_gaussian_for_sliding_window=False,
    patch_size=(96, 224, 224),
    stride=(96, 224, 224),
    tta=False,
    tta_flip_axis=(4,),
    resampling_tolerance=0.01,
    resampling_mode=sitk.sitkNearestNeighbor,
    resampling_dtype=sitk.sitkInt16,
    resampling_constance_value=-1024,
    remove_air_CT=False,
    save_dtype=np.float32,
)

predictor_3 = PredictorStage3(
    list_model_pth=["dummy_model.pth"],
    device=torch.device("cpu"),
    tta=False,
    tta_flip_axis=(4,),
)

c2f_predictor = FractureDetector(predictor_1, predictor_2, predictor_3)

list_DICOM_dirs = [
    os.path.join(DATA_DIR, d)
    for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
]
print(f"==> Found {len(list_DICOM_dirs)} studies for inference")

c2f_predictor.predict(list_test_files=list_DICOM_dirs)

rows = {"row_id": [], "fractured": []}
for case_id, scores in c2f_predictor.results.items():
    for i in range(1, 8):
        rows["row_id"].append(f"{case_id}_C{i}")
        rows["fractured"].append(f"{scores[i]:.6f}")
    rows["row_id"].append(f"{case_id}_patient_overall")
    rows["fractured"].append(f"{scores[0]:.6f}")

submission = pd.DataFrame(rows)
submission.to_csv(SAVE_CSV, index=False)
print(f"==> Submission written to {SAVE_CSV}")
print(f"==> Total script time: {time.time() - time_start:.2f}s")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3431099727.py in <cell line: 0>()
     57 print(f"==> Found {len(list_DICOM_dirs)} studies for inference")
     58 
---> 59 c2f_predictor.predict(list_test_files=list_DICOM_dirs)
     60 
     61 # build submission dataframe

/tmp/ipykernel_55/3521247168.py in predict(self, list_test_files, num_thread, output_dir)
    231             ct_niis = self.read_DICOM_multi_thread(cur_files)
    232             for case_id, ct_nii in zip(case_ids, ct_niis):
--> 233                 pred1 = self.predict_stage1(ct_nii)
    234                 bbox = self.get_c1_c7_bbox(pred1, ct_nii.GetSpacing()[::-1])
    235                 if bbox is not None:

/tmp/ipykernel_55/3521247168.py in predict_stage1(self, ct_nii)
    109         pred_nii = sitk.GetImageFromArray(np.uint8(pred))
    110         pred_nii = copy_nii_info(ct_nii, pred_nii)
--> 111         pred_nii = resample(
    112             pred_nii,
    113             new_spacing=ori_nii_info["spacing"],

NameError: name 'resample' is not defined
