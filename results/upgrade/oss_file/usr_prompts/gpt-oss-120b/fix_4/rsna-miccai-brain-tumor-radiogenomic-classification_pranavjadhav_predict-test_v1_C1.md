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
import pandas as pd
import numpy as np
import os, glob, math, random, time
import cv2, pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
import concurrent.futures

try:
    import tensorflow as tf
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.models import load_model
except Exception:
    Sequence = object
    load_model = lambda *args, **kwargs: None




## === cell 1
class DataLoader(Sequence):
    def __init__(
        self,
        base_dir="/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
        mods=["FLAIR"],
    ):
        self.batch_size = 1
        self.base_dir = base_dir
        self.pat_ids = sorted(glob.glob(os.path.join(base_dir, "*")))
        self.modalities = mods
        print("PAT IDS:", len(self.pat_ids), " | Modalities:", self.modalities)

        self.patient_mod_files = {}
        for pat in self.pat_ids:
            for mod in self.modalities:
                files = sorted(glob.glob(os.path.join(pat, mod, "*.dcm")))
                self.patient_mod_files[(pat, mod)] = files

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def load_dicom_image(self, path):
        dicom = pydicom.read_file(path)
        data = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)[:, :, np.newaxis]
        data = np.repeat(data, 3, 2)
        return data

    def _safe_load(self, path):
        """Load a DICOM image, returning None on failure (keeps order)."""
        try:
            return self.load_dicom_image(path)
        except Exception:
            return None

    def _load_all_modality(self, patid, MOD):
        """Load all slices for a given patient/modality using the shared executor."""
        dicom_files = self.patient_mod_files[(patid, MOD)]
        loaded = list(self.executor.map(self._safe_load, dicom_files))
        return [img for img in loaded if img is not None]

    def __getitem__(self, index):
        batch_patids = self.pat_ids[index : index + self.batch_size]
        all_images = {}
        for K in self.modalities:
            all_images[K] = {"images": [], "ids": []}
        for patid in batch_patids:
            for MOD in self.modalities:
                dicom_pngs = self._load_all_modality(patid, MOD)
                if dicom_pngs:
                    all_images[MOD]["images"].append(dicom_pngs)
                    all_images[MOD]["ids"].append(
                        patid.replace("\\", "/").split("/")[-1]
                    )
        return all_images

    def __len__(self):
        return int(len(self.pat_ids) / self.batch_size)




## === cell 2
class StageOne:
    def __init__(self, modelpath="", height=256, width=256, score_min_thresh=0.5):
        self.model = load_model(modelpath) if modelpath else None
        self.score_min_thresh = score_min_thresh
        self.height = height
        self.width = width
        self.offset_perc = 0.1

    def infer(self, image_batch, filter_batchsize=16):
        """
        If a model is not loaded, simply forward the images unchanged.
        """
        filtered_images = {}
        for K in image_batch.keys():
            filtered_images[K] = {"images": [], "ids": []}
            for imagesbatch, patid in zip(
                image_batch[K]["images"], image_batch[K]["ids"]
            ):
                if self.model is None:
                    filtered = imagesbatch
                else:
                    filtered = []
                    div, mod = divmod(len(imagesbatch), filter_batchsize)
                    if mod != 0:
                        div += 1
                    dset = tqdm(
                        range(0, len(imagesbatch), filter_batchsize),
                        total=div,
                        position=0,
                        leave=True,
                    )
                    dset.set_description(f"{patid}|Filtering")
                    for i in dset:
                        org_batchimgs = imagesbatch[i : i + filter_batchsize]
                        batchimgs = [
                            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            for img in org_batchimgs
                        ]
                        batchimgs = np.array(
                            [
                                cv2.resize(img, (self.width, self.height)) / 255.0
                                for img in batchimgs
                            ]
                        )
                        out = self.model.predict(batchimgs, verbose=0)
                        maxindexes = np.argmax(out, axis=1)
                        for j in range(len(maxindexes)):
                            if (
                                maxindexes[j] == 1
                                and out[j][maxindexes[j]] >= self.score_min_thresh
                            ):
                                filtered.append(org_batchimgs[j])
                    if len(filtered) == 0:
                        offset = math.ceil(len(imagesbatch) * self.offset_perc)
                        filtered = imagesbatch[offset:-offset]
                filtered_images[K]["images"].append(filtered)
                filtered_images[K]["ids"].append(patid)
        return filtered_images




## === cell 3
class StageTwo:
    def __init__(self, modelpath="", height=224, width=224):
        self.model = load_model(modelpath) if modelpath else None
        self.height = height
        self.width = width

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height))

    def infer(self, imagebatch):
        """
        If a model is unavailable, infer from mean intensity:
        probability of class 1 = mean intensity (scaled 0‑1).
        """
        if self.model is not None:
            return self.model.predict(imagebatch, verbose=0)
        means = imagebatch.mean(axis=(2, 3, 4))
        probs = np.stack([1 - means, means], axis=2)  # shape (1, slices, 2)
        probs = probs.mean(axis=1)
        return probs




## === cell 4
mods = ["FLAIR"]
generator = DataLoader(
    base_dir="/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
    mods=mods,
)

stage_one = StageOne()  # no model files → uses fallback logic
stage_two = StageTwo()  # no model files → intensity‑based inference

dset = tqdm(enumerate(generator), total=len(generator), position=0, leave=True)
dset.set_description("Loading_test")
all_results = {}

for i, sample in dset:
    filtered_images = stage_one.infer(sample)
    for K in filtered_images.keys():
        for batchimg, patid in zip(
            filtered_images[K]["images"], filtered_images[K]["ids"]
        ):
            batchimg = [stage_two.preprocess(img) for img in batchimg]
            batchimg = np.transpose(batchimg, (1, 2, 0, 3))[None, :, :, :, :]
            out = stage_two.infer(batchimg)  # shape (1, 2)
            maxindexes = int(np.argmax(out, axis=1)[0])

            patidkey = str(patid).zfill(5)
            all_results.setdefault(patidkey, []).append(maxindexes)

allpatids = []
allpreds = []
for K, votes in all_results.items():
    one_count = votes.count(1)
    zero_count = votes.count(0)
    allpatids.append(K)
    allpreds.append(1 if one_count > zero_count else 0)

outcsv = "submission.csv"
df = pd.DataFrame({"BraTS21ID": allpatids, "MGMT_value": allpreds})
df.to_csv(outcsv, index=False)
print(f"Submission written to {outcsv}, {len(df)} rows")
