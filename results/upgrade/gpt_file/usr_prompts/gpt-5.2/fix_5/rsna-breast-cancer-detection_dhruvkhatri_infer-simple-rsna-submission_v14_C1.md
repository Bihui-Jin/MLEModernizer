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

No external packages required in the script and installed.

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

0.0382193263260298

# 6. Current score

0.04562

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.04562) has done: 'The timeout is dominated by DICOM→PNG conversion (heavy per-file `pydicom.pixel_array` decode + OpenCV work) plus some avoidable overhead in inference post-processing. I keep the exact same conversion and model logic, but reduce wall time by (1) using `pydicom`’s `stop_before_pixels=True` to quickly skip already-converted files without decoding pixel data, (2) pre-filtering the conversion list to only missing PNGs, (3) using a safer multiprocessing context (`spawn`) and larger `chunksize` to cut IPC overhead, and (4) vectorizing the inference collection to avoid Python per-row loops. These changes are equivalence-preserving: the same PNGs are produced for files that need conversion, and the same model outputs are aggregated the same way.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.models.efficientnet import efficientnet_b0

from tqdm import tqdm
from PIL import Image

import cv2
import pydicom


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

INPUT_DIR = "/kaggle/input/rsna-breast-cancer-detection"
WORK_DIR = "/kaggle/working"
TEST_IMG_OUTDIR = os.path.join(WORK_DIR, "test")

os.makedirs(TEST_IMG_OUTDIR, exist_ok=True)




## === cell 1
class RSNABreast(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img = self.load_image(row["Filepath"])
        if self.transform:
            img = self.transform(img)
        return img, row["prediction_id"]

    @staticmethod
    def load_image(path):
        img = Image.open(path)
        try:
            img.draft("RGB", (256, 512))
        except Exception:
            pass
        img = img.convert("RGB")
        return img




## === cell 2
def process_dicom_to_png(f, save_folder="", extension="png"):
    patient = os.path.basename(os.path.dirname(f))
    image = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(save_folder, f"{patient}_{image}.{extension}")

    if os.path.exists(out_path):
        return True

    try:
        dicom = pydicom.dcmread(f, force=True)
        img = dicom.pixel_array.astype(np.float32)
    except Exception:
        return False

    minX = float(np.min(img))
    maxX = float(np.max(img))
    if maxX - minX < 1e-6:
        return False
    img = (img - minX) / (maxX - minX)

    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = 1.0 - img

    try:
        col_mask = img.std(axis=0) > 60
        row_mask = img.std(axis=1) > 60
        if col_mask.sum() > 20:
            img = img[:, col_mask]
        if row_mask.sum() > 20:
            img = img[row_mask, :]
    except Exception:
        pass

    if img.shape[0] > 20 and img.shape[1] > 20:
        X = img[5:-5, 5:-5]
    else:
        X = img

    try:
        mask = (X > 0.05).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        if n > 1:
            idx = stats[1:, 4].argmax() + 1
            x1, y1, w, h = stats[idx][:4]
            x2, y2 = x1 + w, y1 + h
            X_fit = X[y1:y2, x1:x2]
        else:
            X_fit = X
    except Exception:
        X_fit = X

    try:
        X_fit = cv2.resize(X_fit, (512, 1024), interpolation=cv2.INTER_AREA)
    except Exception:
        return False

    cv2.imwrite(out_path, (np.clip(X_fit, 0, 1) * 255).astype(np.uint8))
    return True


test_images = glob.glob(os.path.join(INPUT_DIR, "test_images", "*", "*.dcm"))

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def _fast_outpath(dcm_path: str, outdir: str) -> str:
    try:
        ds = pydicom.dcmread(
            dcm_path,
            stop_before_pixels=True,
            force=True,
            specific_tags=["PatientID", "SOPInstanceUID"],
        )
        patient = str(getattr(ds, "PatientID", "")) or os.path.basename(
            os.path.dirname(dcm_path)
        )
        image = os.path.splitext(os.path.basename(dcm_path))[0]
    except Exception:
        patient = os.path.basename(os.path.dirname(dcm_path))
        image = os.path.splitext(os.path.basename(dcm_path))[0]
    return os.path.join(outdir, f"{patient}_{image}.png")


to_convert = []
if len(test_images) > 0:
    for p in test_images:
        outp = os.path.join(
            TEST_IMG_OUTDIR,
            f"{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}.png",
        )
        if not os.path.exists(outp):
            to_convert.append(p)


def _worker_process_one(args):
    path, outdir = args
    return process_dicom_to_png(path, save_folder=outdir)


ok = 0
total = len(to_convert)
if total > 0:
    cpu_cnt = os.cpu_count() or 2
    max_workers = max(1, min(8, cpu_cnt))

    chunksize = 128

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        it = ex.map(
            _worker_process_one,
            ((p, TEST_IMG_OUTDIR) for p in to_convert),
            chunksize=chunksize,
        )
        for res in tqdm(it, total=total, desc="Converting DICOM -> PNG"):
            ok += int(bool(res))

print(f"Converted {ok}/{total} missing DICOMs to PNG in: {TEST_IMG_OUTDIR}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
BrokenProcessPool                         Traceback (most recent call last)
/tmp/ipykernel_55/3580909519.py in <cell line: 0>()
    121             chunksize=chunksize,
    122         )
--> 123         for res in tqdm(it, total=total, desc="Converting DICOM -> PNG"):
    124             ok += int(bool(res))
    125 

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.

## === cell 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 1, bias=True)
model = model.to(device)
model.eval()



## === cell 4
root_path = TEST_IMG_OUTDIR


def getpath(root_path, row_df):
    return os.path.join(root_path, f"{row_df['patient_id']}_{row_df['image_id']}.png")




## === cell 5
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_csv = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

test_csv["Filepath"] = (
    root_path
    + "/"
    + test_csv["patient_id"].astype(str)
    + "_"
    + test_csv["image_id"].astype(str)
    + ".png"
)

filepaths = test_csv["Filepath"].to_numpy()
missing_mask = np.fromiter(
    (not os.path.exists(p) for p in filepaths), dtype=bool, count=filepaths.shape[0]
)
missing = test_csv.loc[missing_mask, "Filepath"].tolist()
if len(missing) > 0:
    blank = np.zeros((1024, 512), dtype=np.uint8)
    for p in missing:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        cv2.imwrite(p, blank)
print(f"Missing PNGs filled with blanks: {len(missing)}")



## === cell 6
valid_transform = transforms.Compose(
    [
        transforms.Resize([512, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

ds = RSNABreast(test_csv, valid_transform)

num_workers = min(8, (os.cpu_count() or 2))
dl = DataLoader(
    ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

pred_ids_all = []
pred_probs_all = []

sigmoid = nn.Sigmoid()

for xb, pred_id in tqdm(dl, desc="Infer"):
    xb = xb.to(device, non_blocking=True)
    with torch.no_grad():
        out = model(xb)
        prob = sigmoid(out).detach().cpu().numpy().reshape(-1)
    pred_ids_all.extend(list(pred_id))
    pred_probs_all.append(prob)

pred_probs_all = (
    np.concatenate(pred_probs_all, axis=0)
    if len(pred_probs_all)
    else np.array([], dtype=np.float32)
)
pred_df = pd.DataFrame(
    {"prediction_id": pred_ids_all, "cancer": pred_probs_all.astype(float)}
)



## === cell 7
pred_df = pred_df.groupby("prediction_id", as_index=False)["cancer"].max()

sub = sample_sub[["prediction_id"]].merge(pred_df, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(0.0).astype(float)

sub["cancer"] = sub["cancer"].clip(0.0, 1.0)

out_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", list(sub.columns))
