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

3.10

# 3. Installed packages

cloudpathlib==0.21.1
cuda-pathfinder==1.3.2
geopandas==0.14.4
jmespath==1.0.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nibabel==5.3.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
protobuf==6.33.0
pydicom==3.0.1
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
testpath==0.6.0
tqdm==4.67.1

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

8.0739

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 3.02703) has done: 'The changes focus on speeding up the expensive DICOM loading loops. Instead of listing every file in each study folder, the code now tries to open the common first slice (`1.dcm`) directly and falls back to a fast `os.scandir` iterator only if that file is missing. This avoids loading large directory listings for thousands of studies while preserving identical image‑selection logic, so model training, inference, and the final submission remain unchanged.'
- What this solution (achieved 12.41164) has done: 'The fixes address two runtime errors: (1) handling cases where `predict_proba` returns a 1‑dimensional array because a label has only one class in the training subset, and (2) ensuring the `study_preds` dictionary is created before the submission mapping runs. The updated cell 8 now safely extracts positive‑class probabilities for each label, and the rest of the pipeline proceeds to generate a valid `submission.csv` file.'
- What this solution (achieved 12.41164) has done: 'The changes increase the training data from the small 50‑study subset to the full training set, allowing the logistic models to learn from all available examples, and replace the simple average used for the `patient_overall` prediction with the maximum vertebra probability, which better reflects the “any fracture” label that carries higher weight in the loss. Both adjustments are minimal, keep the original architecture and pipelines untouched, and are expected to lower the weighted log‑loss toward the target score.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import cv2 as cv
import os
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(42)




## === cell 1
train_df = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/train.csv")




## === cell 2
def load_dicom(path):
    from pydicom import dcmread

    img = dcmread(path)
    data = img.pixel_array.astype(np.float32)
    data = data - np.min(data)
    max_val = np.max(data)
    if max_val != 0:
        data = data / max_val
    data = (data * 255).astype(np.uint8)
    return data




## === cell 3
train_dir = "../input/rsna-2022-cervical-spine-fracture-detection/train_images"




## === cell 4
trainset = []
trainlabel = []
limit = len(train_df)  # was 50
for i in tqdm(range(min(limit, len(train_df))), desc="Loading train studies"):
    study_id = train_df.loc[i, "StudyInstanceUID"]
    study_path = os.path.join(train_dir, study_id)
    if not os.path.isdir(study_path):
        continue
    first_slice = os.path.join(study_path, "1.dcm")
    if os.path.isfile(first_slice):
        candidates = [first_slice]
    else:
        candidates = (entry.path for entry in os.scandir(study_path) if entry.is_file())
    for im_path in candidates:
        try:
            img = load_dicom(im_path)
        except Exception:
            continue
        img = cv.resize(img, (64, 64))
        img = np.expand_dims(img, -1).astype(np.float32) / 255.0
        trainset.append(img)
        trainlabel.append(
            [
                train_df.loc[i, "C1"],
                train_df.loc[i, "C2"],
                train_df.loc[i, "C3"],
                train_df.loc[i, "C4"],
                train_df.loc[i, "C5"],
                train_df.loc[i, "C6"],
                train_df.loc[i, "C7"],
            ]
        )
        break  # keep only one slice per study

X_train = np.array(trainset)  # shape (n, 64, 64, 1)
Y_train = np.array(trainlabel)  # shape (n, 7)




## === cell 5
test_df = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/test.csv")
test_dir = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"

test_images = []
test_study_ids = []

unique_studies = test_df["StudyInstanceUID"].unique()
for study_id in tqdm(unique_studies, desc="Loading test studies"):
    study_path = os.path.join(test_dir, study_id)
    if not os.path.isdir(study_path):
        continue
    first_slice = os.path.join(study_path, "1.dcm")
    if os.path.isfile(first_slice):
        candidates = [first_slice]
    else:
        candidates = (entry.path for entry in os.scandir(study_path) if entry.is_file())
    found = False
    for im_path in candidates:
        try:
            img = load_dicom(im_path)
        except Exception:
            continue
        img = cv.resize(img, (64, 64))
        img = np.expand_dims(img, -1).astype(np.float32) / 255.0
        test_images.append(img)
        test_study_ids.append(study_id)
        found = True
        break  # only one slice per study
    if not found:
        continue

X_test = np.array(test_images)  # shape (num_studies, 64, 64, 1)




## === cell 6
def create_classifier():
    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    )
    pipeline = make_pipeline(
        StandardScaler(),  # default with_mean=True
        classifier,
    )
    return pipeline


X_train_flat = X_train.reshape((X_train.shape[0], -1))

clf_vertebrae = create_classifier()
clf_vertebrae.fit(X_train_flat, Y_train)

y_overall = train_df["patient_overall"].values
clf_overall = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
)
clf_overall.fit(X_train_flat, y_overall)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3205940250.py in <cell line: 0>()
     24     LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
     25 )
---> 26 clf_overall.fit(X_train_flat, y_overall)
     27 
     28 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    403             if self._final_estimator != "passthrough":
    404                 fit_params_last_step = fit_params_steps[self.steps[-1][0]]
--> 405                 self._final_estimator.fit(Xt, y, **fit_params_last_step)
    406 
    407         return self

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
-> 1124     check_consistent_length(X, y)
   1125 
   1126     return X, y

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [149, 202]

## === cell 7
X_test_flat = X_test.reshape((X_test.shape[0], -1))

prob_lists = clf_vertebrae.predict_proba(X_test_flat)


def _positive_class_probs(p):
    """
    Return the probability of the positive class.
    Handles the case where the classifier was trained on a single class
    (resulting in a 1‑D array of probabilities for the only class).
    """
    if p.ndim == 1:
        return np.zeros(p.shape[0])
    else:
        return p[:, 1]


probs = np.column_stack([_positive_class_probs(p) for p in prob_lists])

overall_prob_full = clf_overall.predict_proba(X_test_flat)
overall_probs = _positive_class_probs(overall_prob_full)  # shape (num_studies,)

study_preds = {sid: prob for sid, prob in zip(test_study_ids, probs)}
study_overall = {sid: prob for sid, prob in zip(test_study_ids, overall_probs)}




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/3106105096.py in <cell line: 0>()
     21 
     22 # Probability for patient_overall (single column).
---> 23 overall_prob_full = clf_overall.predict_proba(X_test_flat)
     24 overall_probs = _positive_class_probs(overall_prob_full)  # shape (num_studies,)
     25 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict_proba(self, X, **predict_proba_params)
    545         for _, name, transform in self._iter(with_final=False):
    546             Xt = transform.transform(Xt)
--> 547         return self.steps[-1][1].predict_proba(Xt, **predict_proba_params)
    548 
    549     @available_if(_final_estimator_has("decision_function"))

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in predict_proba(self, X)
   1360             where classes are ordered as they are in ``self.classes_``.
   1361         """
-> 1362         check_is_fitted(self)
   1363 
   1364         ovr = self.multi_class in ["ovr", "warn"] or (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LogisticRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 8
submission = test_df.copy()


def map_prob(row):
    sid = row["StudyInstanceUID"]
    ptype = row["prediction_type"]
    if sid not in study_preds:
        return 0.0
    if ptype == "patient_overall":
        return float(study_overall.get(sid, 0.0))
    prob_vec = study_preds[sid]
    label_map = {
        "C1": prob_vec[0],
        "C2": prob_vec[1],
        "C3": prob_vec[2],
        "C4": prob_vec[3],
        "C5": prob_vec[4],
        "C6": prob_vec[5],
        "C7": prob_vec[6],
    }
    return float(label_map.get(ptype, 0.0))


submission["fractured"] = submission.apply(map_prob, axis=1)
submission["fractured"] = submission["fractured"].clip(0.0, 1.0)

submission[["row_id", "fractured"]].to_csv(
    "submission.csv", index=False, float_format="%.6f"
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/908537147.py in <cell line: 0>()
     23 
     24 
---> 25 submission["fractured"] = submission.apply(map_prob, axis=1)
     26 submission["fractured"] = submission["fractured"].clip(0.0, 1.0)
     27 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_55/908537147.py in map_prob(row)
      5     sid = row["StudyInstanceUID"]
      6     ptype = row["prediction_type"]
----> 7     if sid not in study_preds:
      8         return 0.0
      9     if ptype == "patient_overall":

NameError: name 'study_preds' is not defined
