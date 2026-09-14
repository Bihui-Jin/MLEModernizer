# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import glob
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)




## === cell 1
class DataLoader:
    """
    Minimal generator-like loader to match original core flow:
    yields dict(modality -> {'images': [list_of_slices], 'ids': [patid]})
    """

    def __init__(
        self,
        base_dir="/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
        mods=("FLAIR",),
        batch_size=1,
    ):
        self.batch_size = batch_size
        self.base_dir = base_dir
        self.pat_ids = sorted(glob.glob(os.path.join(base_dir, "*")))
        self.modalities = list(mods)
        print("PAT IDS:", len(self.pat_ids), " | Modalities:", self.modalities)

    def load_dicom_image(self, path):
        dicom = pydicom.dcmread(path, force=True)
        data = apply_voi_lut(dicom.pixel_array, dicom)

        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            data = np.amax(data) - data

        data = data.astype(np.float32)
        data = data - np.min(data)
        mx = np.max(data)
        if mx > 0:
            data = data / mx
        else:
            data = np.zeros_like(data, dtype=np.float32)

        data = (data * 255.0).clip(0, 255).astype(np.uint8)[:, :, np.newaxis]
        data = np.repeat(data, 3, axis=2)
        return data

    def __getitem__(self, index):
        batch_patids = self.pat_ids[index : index + self.batch_size]
        all_images = {k: {"images": [], "ids": []} for k in self.modalities}

        for patid_path in batch_patids:
            patid = patid_path.replace("\\", "/").split("/")[-1]
            for MOD in all_images.keys():
                dicom_pngs = []
                dicom_files = sorted(glob.glob(os.path.join(patid_path, MOD, "*.dcm")))
                for dicom_path in dicom_files:
                    try:
                        png = self.load_dicom_image(dicom_path)
                        dicom_pngs.append(png)
                    except Exception:
                        continue

                if len(dicom_pngs) != 0:
                    all_images[MOD]["images"].append(dicom_pngs)
                    all_images[MOD]["ids"].append(patid)

        return all_images

    def __len__(self):
        return (len(self.pat_ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]




## === cell 2
class StageOne:
    """
    Original code loads a CNN filter model from /kaggle/input/models/... (not available).
    To keep pipeline running end-to-end, we provide a deterministic "filter" that selects
    mid-slices, similar to the original fallback when nothing passes threshold.
    """

    def __init__(self, modelpath="", height=256, width=256, score_min_thresh=0.5):
        self.modelpath = modelpath
        self.score_min_thresh = score_min_thresh
        self.height = height
        self.width = width
        self.offset_perc = 0.1

        self._has_model = False
        if modelpath and os.path.exists(modelpath):
            self._has_model = False

    def infer(self, image_batch, filter_batchsize=16):
        filtered_images = {k: {"images": [], "ids": []} for k in image_batch.keys()}

        for K in image_batch.keys():
            for imagesbatch, patid in zip(
                image_batch[K]["images"], image_batch[K]["ids"]
            ):
                n = len(imagesbatch)
                if n == 0:
                    continue
                offset = max(1, math.ceil(n * self.offset_perc))
                filtered_batch_images = (
                    imagesbatch[offset:-offset]
                    if (n - 2 * offset) >= 8
                    else imagesbatch
                )

                filtered_images[K]["images"].append(filtered_batch_images)
                filtered_images[K]["ids"].append(patid)

        return filtered_images




## === cell 3
class StageTwo:
    """
    Original code loads a 3D classifier model from /kaggle/input/models/... (not available).
    We keep the same interface but use a deterministic image-statistics-based probability
    to produce a valid ROC-AUC submission (probabilities in [0,1]).
    """

    def __init__(self, modelpath="", height=224, width=224):
        self.modelpath = modelpath
        self.height = height
        self.width = width

        self._has_model = False
        if modelpath and os.path.exists(modelpath):
            self._has_model = False

    def preprocess(self, image):
        return cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

    def infer(self, imagebatch_5d):
        """
        imagebatch_5d: (B, H, W, D, C) as created by the original pipeline.
        Return shape (B, 2) like a softmax over [class0, class1].
        """
        x = imagebatch_5d.astype(np.float32) / 255.0
        mean_int = x.mean(axis=(1, 2, 3, 4))
        std_int = x.std(axis=(1, 2, 3, 4))

        logit = (mean_int - 0.45) * 5.0 + (std_int - 0.20) * 3.0
        p1 = 1.0 / (1.0 + np.exp(-logit))
        p1 = np.clip(p1, 1e-4, 1 - 1e-4)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)




## === cell 4
mods = ["FLAIR", "T1w", "T1wCE", "T2w"]
generator = DataLoader(
    base_dir="/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
    mods=mods,
)

stage_one = StageOne(modelpath="/kaggle/input/models/FINAL_MODELALL_acc0.9825_ep26.h5")
stage_two = StageTwo(modelpath="/kaggle/input/models/3d_image_classification.hdf5")

dset = tqdm(enumerate(generator), total=len(generator), position=0, leave=True)
dset.set_description("Loading_test")

all_results = {}
for i, sample in dset:
    filtered_images = stage_one.infer(sample)
    for K in filtered_images.keys():
        for batchimg, patid in zip(
            filtered_images[K]["images"], filtered_images[K]["ids"]
        ):
            if len(batchimg) == 0:
                continue

            batchimg = [stage_two.preprocess(img) for img in batchimg]
            batchimg = np.asarray(batchimg, dtype=np.uint8)

            batchimg = np.transpose(batchimg, (1, 2, 0, 3))[None, :, :, :, :]

            out = stage_two.infer(batchimg)  # (1,2)
            prob1 = float(out[0, 1])

            patidkey = str(patid).zfill(5)
            all_results.setdefault(patidkey, []).append(prob1)

allpatids = []
allpreds = []
for K in sorted(all_results.keys()):
    allpatids.append(K)
    allpreds.append(float(np.mean(all_results[K])))

sample_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sub = pd.read_csv(sample_path)
    sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
    pred_map = dict(zip(allpatids, allpreds))
    sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).fillna(0.5).astype(float)
    df = sub[["BraTS21ID", "MGMT_value"]]
else:
    df = pd.DataFrame({"BraTS21ID": allpatids, "MGMT_value": allpreds})

outcsv = "submission.csv"
df.to_csv(outcsv, index=False)
print(f"Wrote {outcsv} with shape={df.shape} and columns={list(df.columns)}")



## === cell 5
df.head()
