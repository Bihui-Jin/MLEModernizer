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


def _dicom_to_uint8(dcm_path: str, out_h: int):
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)

        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut

            arr = apply_voi_lut(arr, ds).astype(np.float32)
        except Exception:
            pass

        try:
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
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

    start_timer = timer()
    missing = 0
    for _, d in tqdm(test_df.iterrows(), total=len(test_df), desc="dicom->png"):
        out_dir = f"{png_dir}/{d.patient_id}"
        os.makedirs(out_dir, exist_ok=True)

        out_path = f"{out_dir}/{d.image_id}.png"
        if os.path.exists(out_path):
            continue

        dcm_path = f"{dcm_dir}/{d.patient_id}/{d.image_id}.dcm"
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


if not ("skip-dicom-to-png" in mode):
    run_dicom_to_png()

print("glob png", len(glob(f"{png_dir}/**/*.png", recursive=True)))
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


class RsnaDataset(Dataset):
    def __init__(self, df):
        self.length = len(df)
        self.df = df

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        m = cv2.imread(
            f"{png_dir}/{d.patient_id}/{d.image_id}.png", cv2.IMREAD_GRAYSCALE
        )
        if m is None:
            m = np.zeros((convert_height, int(convert_height * 0.66)), np.uint8)
        h, w = m.shape

        image = np.zeros((image_height, image_width), np.uint8)
        try:
            xmin, ymin, xmax, ymax = (
                np.array(to_list(d.pad_breast_box), dtype=np.float32) * h
            ).astype(int)
            xmin = max(0, min(xmin, w - 1))
            xmax = max(1, min(xmax, w))
            ymin = max(0, min(ymin, h - 1))
            ymax = max(1, min(ymax, h))
            if xmax <= xmin or ymax <= ymin:
                raise ValueError("bad crop")
            crop = m[ymin:ymax, xmin:xmax]

            mh, mw = (
                np.array(to_list(d.max_pad_breast_shape), dtype=np.float32) * h
            ).astype(int)
            mh = max(1, mh)
            mw = max(1, mw)

            scale = image_height / mh
            if (scale * mw) > image_width:
                scale = image_width / mw

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
        r["d"] = d
        r["image"] = torch.from_numpy(image)
        return r


def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]
    d["image"] = torch.stack(d["image"]).unsqueeze(1)
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
        self.encoder = timm.models.efficientnet.efficientnet_b4(
            pretrained=False, drop_rate=0, drop_path_rate=0
        )
        self.cancer = nn.Linear(1792, 1)

    def forward(self, batch):
        x = batch["image"]
        x = (x - self.mean) / self.std
        e = self.encoder.forward_features(x)
        x = F.adaptive_avg_pool2d(e, 1)
        x = torch.flatten(x, 1, 3)
        cancer = self.cancer(x).reshape(-1)
        cancer = torch.sigmoid(cancer)
        return cancer


print("define model ok")




## === cell 6
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
        n = Net()
        f = torch.load(checkpoint, map_location="cpu")
        n.load_state_dict(f.get("state_dict", f), strict=False)
        n.to(device)
        n.eval()
        net.append(n)

    test_dataset = RsnaDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=8,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=null_collate,
    )

    result_prob = []
    test_num = 0
    start_timer = timer()

    for t, batch in enumerate(test_loader):
        batch_size = len(batch["index"])
        image0 = batch["image"].to(device).float() / 255.0
        image0 = image0.repeat(1, 3, 1, 1)

        batch1 = {"image": image0}
        batch2 = {
            "image": torch.flip(
                image0,
                dims=[
                    3,
                ],
            )
        }  # TTA

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

        test_num += batch_size
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

    submit_df.to_csv("submission.csv", index=False)
    print("submission.csv written:", submit_df.shape)
    print(submit_df.head())


print(mode)
run_submit()
print("***************ok!")



## === cell 7
pass
