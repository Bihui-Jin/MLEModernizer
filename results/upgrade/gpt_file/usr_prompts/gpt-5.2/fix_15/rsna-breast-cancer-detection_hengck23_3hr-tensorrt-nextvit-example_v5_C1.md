# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import sys
import gc
from glob import glob
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import cv2

from tqdm.notebook import tqdm

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import timm  # kept because original solution uses timm.efficientnet_b4

try:
    import pydicom
except Exception as e:
    raise RuntimeError(
        "pydicom is required in this environment but failed to import."
    ) from e


def time_to_str(t, mode="sec"):
    if mode == "sec":
        if t < 60:
            return f"{t:0.1f} sec"
        if t < 3600:
            return f"{t/60:0.1f} min"
        return f"{t/3600:0.1f} hour"
    return str(t)


def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.device_count() = %d" % torch.cuda.device_count())
    print(
        "torch.cuda.get_device_properties() = %s"
        % str(torch.cuda.get_device_properties(0))[21:]
    )
print("timm", timm.__version__)
print("import ok!")




## === cell 1
mode = [
    "submit",  # submit #local
]

convert_height = 1536
image_height = 1536
image_width = 960

if "local" in mode:
    csv_file = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/train_images"

if "submit" in mode:
    csv_file = "/kaggle/input/rsna-breast-cancer-detection/test.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/test_images"

test_df = pd.read_csv(csv_file)
test_df.loc[:, "i"] = np.arange(len(test_df))

print("test_df", test_df.shape)
print(test_df.head())
print("")




## === cell 2
def make_debug_submission():
    sample_path = "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
    sample = pd.read_csv(sample_path)
    submit_df = sample.copy()
    submit_df["cancer"] = 0.0
    submit_df.to_csv("submission.csv", index=False)
    print("Wrote debug submission.csv with shape:", submit_df.shape)




## === cell 3
png_dir = "/kaggle/working/~png"

cache_dir = "/kaggle/working/~dcm_uint8_cache"
os.makedirs(cache_dir, exist_ok=True)

from functools import lru_cache


def _dicom_to_uint8(dcm_path: str, out_h: int):
    try:
        ds = pydicom.dcmread(dcm_path, force=True, defer_size=1 << 20)
        arr = ds.pixel_array.astype(np.float32)

        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut

            arr = apply_voi_lut(arr, ds).astype(np.float32)
        except Exception:
            pass

        try:
            photometric = getattr(ds, "PhotometricInterpretation", "")
            if photometric == "MONOCHROME1":
                arr = arr.max() - arr
        except Exception:
            pass

        arr = arr - np.min(arr)
        mx = np.max(arr)
        if mx > 0:
            arr = arr / mx
        img = (arr * 255.0).clip(0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Invalid pixel array shape")

        if h != out_h:
            s = out_h / float(h)
            new_w = max(1, int(round(w * s)))
            img = cv2.resize(
                img,
                (new_w, out_h),
                interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
            )
        return img
    except Exception:
        return np.zeros((out_h, max(1, int(out_h * 0.66))), np.uint8)


def run_dicom_to_png():
    os.makedirs(png_dir, exist_ok=True)

    patient_ids = test_df["patient_id"].astype(str).unique()
    for pid in patient_ids:
        os.makedirs(os.path.join(png_dir, pid), exist_ok=True)

    pid_arr = test_df["patient_id"].astype(str).values
    iid_arr = test_df["image_id"].astype(str).values

    start_timer = timer()
    missing = 0

    for pid, iid in tqdm(zip(pid_arr, iid_arr), total=len(pid_arr), desc="dicom->png"):
        out_path = f"{png_dir}/{pid}/{iid}.png"
        if os.path.exists(out_path):
            continue

        dcm_path = f"{dcm_dir}/{pid}/{iid}.dcm"
        if not os.path.exists(dcm_path):
            missing += 1
            img = np.zeros((convert_height, int(convert_height * 0.66)), np.uint8)
        else:
            img = _dicom_to_uint8(dcm_path, convert_height)

        cv2.imwrite(out_path, img)

    print(
        "dicom_to_png done:",
        time_to_str(timer() - start_timer, "sec"),
        "missing_dcm:",
        missing,
    )


def prebuild_dicom_uint8_cache(df: pd.DataFrame, out_h: int):
    pid_arr = df["patient_id"].astype(str).values
    iid_arr = df["image_id"].astype(str).values

    start = timer()
    missing = 0
    wrote = 0
    skipped = 0

    existing = set()
    try:
        for p in glob(f"{cache_dir}/*_{out_h}.npy"):
            existing.add(os.path.basename(p))
    except Exception:
        existing = set()

    for pid, iid in tqdm(
        zip(pid_arr, iid_arr), total=len(pid_arr), desc="build .npy cache"
    ):
        fname = f"{pid}_{iid}_{out_h}.npy"
        if fname in existing:
            skipped += 1
            continue

        dcm_path = f"{dcm_dir}/{pid}/{iid}.dcm"
        if not os.path.exists(dcm_path):
            missing += 1
            m = np.zeros((out_h, int(out_h * 0.66)), np.uint8)
        else:
            m = _dicom_to_uint8(dcm_path, out_h)

        np.save(os.path.join(cache_dir, fname), m.astype(np.uint8, copy=False))
        wrote += 1

    print(
        "prebuild cache done:",
        time_to_str(timer() - start, "sec"),
        "wrote:",
        wrote,
        "skipped:",
        skipped,
        "missing_dcm:",
        missing,
    )


print(
    "Skipping up-front DICOM cache build (use lazy per-sample cache during inference) ..."
)
print("gc.collect", gc.collect())
print("")




## === cell 4
def run_add_breast_box_fallback(df):
    df = df.copy()
    df.loc[:, "pad_breast_box"] = [[0.0, 0.0, 1.0, 1.0] for _ in range(len(df))]
    df.loc[:, "max_pad_breast_shape"] = [[1.0, 1.0] for _ in range(len(df))]
    return df


if not ("skip-add-breast-box" in mode):
    test_df = run_add_breast_box_fallback(test_df)

print(test_df.iloc[0], "\n")
print("gc.collect", gc.collect())
print("")




## === cell 5
def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return eval(x)
    return x


def _to_float_array_2d(col_values, width):
    out = np.zeros((len(col_values), width), dtype=np.float32)
    for i, v in enumerate(col_values):
        vv = to_list(v)
        out[i, :] = np.asarray(vv, dtype=np.float32).reshape(-1)[:width]
    return out


@lru_cache(maxsize=256)
def _load_uint8_npy_cached(path: str) -> np.ndarray:
    return np.load(path)  # regular load is faster than mmap for many small files


class RsnaDataset(Dataset):
    def __init__(self, df):
        self.length = len(df)

        self.patient_id = df["patient_id"].astype(str).values
        self.image_id = df["image_id"].astype(str).values
        self.prediction_id = (
            df["prediction_id"].astype(str).values
            if "prediction_id" in df.columns
            else None
        )

        self.pad_breast_box = _to_float_array_2d(df["pad_breast_box"].values, 4)
        self.max_pad_breast_shape = _to_float_array_2d(
            df["max_pad_breast_shape"].values, 2
        )

        self._cache_path = np.empty(self.length, dtype=object)

        out_h = int(convert_height)

        pid = self.patient_id
        iid = self.image_id

        self._cache_exists = np.zeros(self.length, dtype=bool)
        for i in range(self.length):
            fname = f"{pid[i]}_{iid[i]}_{out_h}.npy"
            cp = os.path.join(cache_dir, fname)
            self._cache_path[i] = cp
            self._cache_exists[i] = os.path.exists(cp)

        self._has_box = np.ones(self.length, dtype=bool)
        self._xmin = np.zeros(self.length, dtype=np.int32)
        self._xmax = np.zeros(self.length, dtype=np.int32)
        self._ymin = np.zeros(self.length, dtype=np.int32)
        self._ymax = np.zeros(self.length, dtype=np.int32)
        self._mh = np.zeros(self.length, dtype=np.int32)
        self._mw = np.zeros(self.length, dtype=np.int32)

        for i in range(self.length):
            try:
                box = self.pad_breast_box[i]
                xmin, ymin, xmax, ymax = (box * out_h).astype(np.int32)
                xmin = int(max(0, min(xmin, 10**9)))
                xmax = int(max(1, min(xmax, 10**9)))
                ymin = int(max(0, min(ymin, 10**9)))
                ymax = int(max(1, min(ymax, 10**9)))
                if xmax <= xmin or ymax <= ymin:
                    raise ValueError("bad crop")
                self._xmin[i], self._xmax[i], self._ymin[i], self._ymax[i] = (
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )

                shape = self.max_pad_breast_shape[i]
                mh, mw = (shape * out_h).astype(np.int32)
                self._mh[i] = max(1, int(mh))
                self._mw[i] = max(1, int(mw))
            except Exception:
                self._has_box[i] = False
                self._mh[i] = out_h
                self._mw[i] = max(1, int(out_h * 0.66))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pid = self.patient_id[index]
        iid = self.image_id[index]

        cache_path = self._cache_path[index]

        if self._cache_exists[index]:
            m = _load_uint8_npy_cached(cache_path)
        else:
            dcm_path = f"{dcm_dir}/{pid}/{iid}.dcm"
            if not os.path.exists(dcm_path):
                m = np.zeros((convert_height, int(convert_height * 0.66)), np.uint8)
            else:
                m = _dicom_to_uint8(dcm_path, convert_height)

            tmp_base = cache_path + f".tmp{os.getpid()}"
            np.save(tmp_base, m.astype(np.uint8, copy=False))
            tmp_path = tmp_base + ".npy"
            try:
                os.replace(tmp_path, cache_path)
                self._cache_exists[index] = True
                _load_uint8_npy_cached.cache_clear()  # ensure future loads see the new file
            except Exception:
                pass

        if m is None:
            m = np.zeros((convert_height, int(convert_height * 0.66)), np.uint8)
        h, w = m.shape

        image = np.zeros((image_height, image_width), np.uint8)

        try:
            if self._has_box[index]:
                xmin = int(max(0, min(self._xmin[index], w - 1)))
                xmax = int(max(1, min(self._xmax[index], w)))
                ymin = int(max(0, min(self._ymin[index], h - 1)))
                ymax = int(max(1, min(self._ymax[index], h)))
                if xmax <= xmin or ymax <= ymin:
                    raise ValueError("bad crop")
                crop = m[ymin:ymax, xmin:xmax]

                mh = int(self._mh[index])
                mw = int(self._mw[index])

                scale = image_height / mh
                if (scale * mw) > image_width:
                    scale = image_width / mw
            else:
                crop = m
                mh, mw = m.shape
                scale = image_height / max(1, mh)
                if (scale * mw) > image_width:
                    scale = image_width / max(1, mw)

            dsize = (
                min(image_width, max(1, int(scale * crop.shape[1]))),
                min(image_height, max(1, int(scale * crop.shape[0]))),
            )
            if dsize != (crop.shape[1], crop.shape[0]):
                crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)

            ch, cw = crop.shape
            x = (image_width - cw) // 2
            y = (image_height - ch) // 2
            image[y : y + ch, x : x + cw] = crop
        except Exception:
            crop = m
            mh, mw = m.shape
            scale = image_height / max(1, mh)
            if (scale * mw) > image_width:
                scale = image_width / max(1, mw)

            dsize = (
                min(image_width, max(1, int(scale * crop.shape[1]))),
                min(image_height, max(1, int(scale * crop.shape[0]))),
            )
            if dsize != (crop.shape[1], crop.shape[0]):
                crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            ch, cw = crop.shape
            x = (image_width - cw) // 2
            y = (image_height - ch) // 2
            image[y : y + ch, x : x + cw] = crop

        r = {}
        r["index"] = index
        r["d"] = {"patient_id": pid, "image_id": iid}
        r["image"] = torch.from_numpy(image)
        return r


def null_collate(batch):
    d = {
        "index": [b["index"] for b in batch],
        "d": [b["d"] for b in batch],
    }
    images = torch.stack([b["image"] for b in batch], dim=0).unsqueeze(1)
    d["image"] = images
    if "cancer" in batch[0]:
        d["cancer"] = torch.tensor(
            [float(b["cancer"]) for b in batch], dtype=torch.float32
        ).view(-1)
    return d


class EffB4Net(nn.Module):
    def __init__(self):
        super(EffB4Net, self).__init__()
        self.register_buffer(
            "mean", torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        )

        self.encoder = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
            num_classes=0,
            global_pool="",
        )
        self.cancer = nn.Linear(1792, 1)

    def forward_logits(self, batch):
        x = batch["image"]
        x = (x - self.mean) / self.std
        e = self.encoder.forward_features(x)
        x = F.adaptive_avg_pool2d(e, 1)
        x = torch.flatten(x, 1, 3)
        logits = self.cancer(x).reshape(-1)
        return logits

    def forward(self, batch):
        logits = self.forward_logits(batch)
        cancer = torch.sigmoid(logits)
        return cancer


print("define model ok")




## === cell 6
def _resolve_checkpoint_path(original_path: str) -> str | None:
    if original_path and os.path.exists(original_path):
        return original_path

    fname = os.path.basename(original_path) if original_path else ""
    if not fname:
        return None

    direct_candidates = [
        f"/kaggle/input/{fname}",
        f"/kaggle/input/rsna-breast-cancer-detection/{fname}",
    ]
    for p in direct_candidates:
        if os.path.exists(p):
            return p

    parts = [p for p in (original_path or "").split("/") if p]
    if len(parts) >= 3 and parts[0] == "kaggle" and parts[1] == "input":
        dataset_slug = parts[2]
        base = f"/kaggle/input/{dataset_slug}"
        if os.path.isdir(base):
            candidates = glob(f"{base}/**/{fname}", recursive=True)
            if candidates:
                candidates = sorted(candidates, key=lambda p: (len(p), p))
                return candidates[0]

    candidates = glob(f"/kaggle/input/**/{fname}", recursive=True)
    if candidates:
        candidates = sorted(candidates, key=lambda p: (len(p), p))
        return candidates[0]

    return None


class TrainRsnaDataset(RsnaDataset):
    def __init__(self, df):
        super().__init__(df)
        self.cancer_label = df["cancer"].astype(np.float32).values

    def __getitem__(self, index):
        r = super().__getitem__(index)
        r["cancer"] = torch.tensor(self.cancer_label[index], dtype=torch.float32)
        return r


def _seed_worker(worker_id: int):
    seed = 42 + worker_id
    np.random.seed(seed)


def train_local_checkpoint_if_needed(
    local_ckpt_path: str, max_train_images: int = 4096
):
    if os.path.exists(local_ckpt_path):
        print("Found existing local checkpoint:", local_ckpt_path)
        return local_ckpt_path

    train_csv = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
    train_images_dir = "/kaggle/input/rsna-breast-cancer-detection/train_images"
    if not (os.path.exists(train_csv) and os.path.exists(train_images_dir)):
        print("Training data not found; cannot train local checkpoint.")
        return None

    df = pd.read_csv(train_csv)
    df = run_add_breast_box_fallback(df)

    if len(df) > max_train_images:
        df = df.sample(n=max_train_images, random_state=42).reset_index(drop=True)

    global dcm_dir
    old_dcm_dir = dcm_dir
    dcm_dir = train_images_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = EffB4Net().to(device)
    net.train()

    cpu = os.cpu_count() or 2
    num_workers = 2 if cpu >= 4 else 0

    train_ds = TrainRsnaDataset(df)
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=null_collate,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    bce_logits = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    start = timer()
    running = 0.0
    nstep = 0

    for batch in train_loader:
        image0 = batch["image"].to(device, non_blocking=True).float() / 255.0
        image0 = image0.expand(-1, 3, -1, -1).contiguous()

        y = batch["cancer"].to(device, non_blocking=True).float().view(-1)

        opt.zero_grad(set_to_none=True)
        with amp.autocast(enabled=torch.cuda.is_available()):
            logits = net.forward_logits({"image": image0})
            loss = bce_logits(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running += float(loss.detach().cpu().item())
        nstep += 1
        if nstep % 50 == 0:
            print(
                f"\rtrain step {nstep:5d} loss {running/nstep:0.4f}", end="", flush=True
            )

        if timer() - start > 240:
            break

    print("")
    os.makedirs(os.path.dirname(local_ckpt_path), exist_ok=True)
    torch.save({"state_dict": net.state_dict()}, local_ckpt_path)
    print(
        "Saved local checkpoint:",
        local_ckpt_path,
        "steps:",
        nstep,
        "avg_loss:",
        running / max(1, nstep),
    )

    dcm_dir = old_dcm_dir
    net.eval()
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return local_ckpt_path


def run_submit():
    model = [
        [
            EffB4Net,
            "/kaggle/input/rsna-breast-mammography-weight-10/effb4-1536-baseline-gpu-aug0-01-swa.model.pth",
        ],
    ]
    num_net = len(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = []
    for i in range(num_net):
        Net, checkpoint = model[i]
        ckpt_path = _resolve_checkpoint_path(checkpoint)

        if ckpt_path is None:
            local_ckpt = "/kaggle/working/local_effb4_1536_1epoch.pth"
            print("WARNING: external checkpoint not found:", checkpoint)
            print(
                "Training a minimal local checkpoint to avoid 0.0 debug submission..."
            )
            ckpt_path = train_local_checkpoint_if_needed(
                local_ckpt_path=local_ckpt, max_train_images=4096
            )

        if ckpt_path is None:
            print("WARNING: no viable checkpoint could be produced/found.")
            print("Falling back to debug submission (all zeros).")
            make_debug_submission()
            return

        print("Loading checkpoint:", ckpt_path)
        n = Net()
        try:
            f = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print("WARNING: checkpoint found but failed to load:", ckpt_path)
            print("Exception:", repr(e))
            print("Falling back to debug submission (all zeros).")
            make_debug_submission()
            return

        n.load_state_dict(f.get("state_dict", f), strict=False)
        n.to(device)
        n.eval()
        net.append(n)

    print(
        f"Loaded {len(net)}/{num_net} checkpoints successfully; proceeding with inference."
    )

    test_dataset = RsnaDataset(test_df)

    cpu = os.cpu_count() or 2
    if torch.cuda.is_available():
        num_workers = min(6, max(2, cpu // 2))
        batch_size = 8
        prefetch_factor = 2
    else:
        num_workers = min(4, max(0, cpu // 2))
        prefetch_factor = 2 if num_workers > 0 else None
        batch_size = 4

    g = torch.Generator()
    g.manual_seed(42)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=null_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )

    result_prob = []
    test_num = 0
    start_timer = timer()

    for t, batch in enumerate(test_loader):
        bs = len(batch["index"])
        image0 = batch["image"].to(device, non_blocking=True).float() / 255.0
        image0 = image0.expand(-1, 3, -1, -1).contiguous()

        batch1 = {"image": image0}
        batch2 = {"image": torch.flip(image0, dims=[3])}  # TTA (unchanged)

        p = 0.0
        count = 0
        with torch.no_grad():
            with amp.autocast(enabled=torch.cuda.is_available()):
                for i in range(num_net):
                    p = p + net[i](batch1)
                    count += 1
                    p = p + net[i](batch2)
                    count += 1

        p = p / float(count)
        p = p.float().detach().cpu().numpy()
        result_prob.append(p)

        test_num += bs
        print(
            "\r %8d / %d  %s"
            % (test_num, len(test_dataset), time_to_str(timer() - start_timer, "sec")),
            end="",
            flush=True,
        )

    print("")
    probability = np.concatenate(result_prob)
    probability = np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)

    pred_df = pd.DataFrame(
        {"prediction_id": test_df.prediction_id.values, "cancer": probability}
    )
    pred_df = pred_df.groupby("prediction_id", as_index=False).mean()

    sample_path = "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
    sample = pd.read_csv(sample_path)
    submit_df = sample[["prediction_id"]].merge(pred_df, on="prediction_id", how="left")
    submit_df["cancer"] = submit_df["cancer"].fillna(0.0).astype(np.float32)

    submit_df = submit_df[["prediction_id", "cancer"]]
    submit_df.to_csv("submission.csv", index=False)
    print("submission.csv written:", submit_df.shape)
    print(submit_df.head())


print(mode)
run_submit()
print("***************ok!")
