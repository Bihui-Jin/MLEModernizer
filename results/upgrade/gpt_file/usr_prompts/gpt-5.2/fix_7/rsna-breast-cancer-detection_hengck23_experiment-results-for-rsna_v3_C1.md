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
import os, sys, warnings
import numpy as np
import pandas as pd

try:
    import pylibjpeg  # noqa: F401
except Exception:
    pylibjpeg = None

import pydicom
import cv2

from timeit import default_timer as timer

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import timm
from timm.models.resnet import seresnext50_32x4d
from timm.models.efficientnet import efficientnet_b4


def time_to_str(t, mode="min"):
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        mn = t % 60
        return "%2d hr %02d min" % (hr, mn)
    elif mode == "sec":
        t = int(t)
        mn = t // 60
        sec = t % 60
        return "%2d min %02d sec" % (mn, sec)
    else:
        raise NotImplementedError


try:
    import pydicom.config

    pydicom.config.image_handlers = pydicom.config.image_handlers
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
except Exception:
    pass

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.benchmark = True

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, min(4, (os.cpu_count() or 4) // 2)))
except Exception:
    pass

print("imports ok! timm", timm.__version__)




## === cell 1
image_size = 512
mode = "submit-dicom"  # keep original intent

if "local" in mode:
    csv_file = "/kaggle/input/rsna-breast-mammography-00/valid_df.fold0.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/train_images"

if "submit" in mode:
    csv_file = "/kaggle/input/rsna-breast-cancer-detection/test.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/test_images"

if "dicom" in mode:
    image_dir = "/kaggle/tmp/~png"
    os.makedirs(image_dir, exist_ok=True)

test_df = pd.read_csv(csv_file)
test_df.loc[:, "i"] = np.arange(len(test_df))

if "local" in mode:
    test_df.loc[:, "prediction_id"] = (
        test_df.patient_id.astype(str) + "_" + test_df.laterality
    )
    if "dicom" in mode:
        test_id = [
            826,
            1703,
            1759,
            2346,
            2989,
            3021,
            3542,
            4340,
            4824,
            5059,
            5769,
            6654,
            6658,
            7053,
            7493,
            7780,
            9014,
            11094,
            11937,
            14292,
            30,
            36,
            65,
            90,
            111,
            122,
            127,
            152,
            158,
            204,
            272,
            282,
            289,
            299,
            308,
            399,
            425,
            454,
            477,
            505,
        ]
        test_df = test_df[test_df.patient_id.isin(test_id)].reset_index(drop=True)

print("test_df", test_df.shape)
print(test_df.head())

test_df = test_df.reset_index(drop=True)

_patient_id_arr = test_df["patient_id"].to_numpy()
_image_id_arr = test_df["image_id"].to_numpy()
_pred_id_arr = test_df["prediction_id"].to_numpy()
_test_i_arr = test_df["i"].to_numpy(dtype=np.int32, copy=False)

_dcm_path_arr = np.asarray(
    [f"{dcm_dir}/{pid}/{iid}.dcm" for pid, iid in zip(_patient_id_arr, _image_id_arr)],
    dtype=object,
)

try:
    from functools import lru_cache
except Exception:
    lru_cache = None


_DICOM_TAGS = [
    "PixelData",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "PlanarConfiguration",
    "Rows",
    "Columns",
    "RescaleIntercept",
    "RescaleSlope",
    "WindowCenter",
    "WindowWidth",
    "VOILUTFunction",
    "TransferSyntaxUID",
]


def _decode_dicom_to_float01_impl(dcm_file, image_size=image_size):
    try:
        dicom = pydicom.dcmread(
            dcm_file,
            stop_before_pixels=False,
            force=True,
            defer_size="1 KB",
            specific_tags=_DICOM_TAGS,
        )
        img = dicom.pixel_array  # numpy array
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        vmin = float(img.min())
        vmax = float(img.max())
        if vmax > vmin:
            img = (img - vmin) * (1.0 / (vmax - vmin))
        else:
            img = np.zeros_like(img, dtype=np.float32)

        if getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1":
            img = 1.0 - img

        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        np.clip(img, 0.0, 1.0, out=img)
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        return img
    except Exception as e:
        warnings.warn(f"Failed to decode {dcm_file}: {type(e).__name__}: {e}")
        return np.zeros((image_size, image_size), dtype=np.float32)


if lru_cache is not None:
    _decode_dicom_to_float01 = lru_cache(maxsize=2048)(_decode_dicom_to_float01_impl)
else:
    _decode_dicom_to_float01 = _decode_dicom_to_float01_impl


class RsnaImageDataset(Dataset):
    def __init__(self, df):
        self.df = df  # kept for compatibility/debug
        self.dcm_path = _dcm_path_arr
        self.i = _test_i_arr

    def __len__(self):
        return len(self.i)

    def __getitem__(self, idx):
        dcm_file = self.dcm_path[idx]
        img = _decode_dicom_to_float01(dcm_file)  # float32 [0,1], HxW
        return idx, self.i[idx], torch.from_numpy(img).unsqueeze(0)


def null_collate(batch):
    b = len(batch)
    index = np.empty(b, dtype=np.int64)
    i = np.empty(b, dtype=np.int32)
    images = []
    for k, (idx, ii, im) in enumerate(batch):
        index[k] = idx
        i[k] = ii
        images.append(im)
    image = torch.stack(images, dim=0)
    return {"image": image, "i": i, "index": index}




## === cell 2
class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.5, 0.5, 0.5]

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        return (x - self.mean) / self.std


class ResNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.rgb = RGB()
        self.encoder = seresnext50_32x4d(pretrained=pretrained)
        self.cancer = nn.Linear(2048, 1)

    def forward(self, batch):
        x = batch["image"]
        x = x.expand(-1, 3, -1, -1)
        x = self.rgb(x)

        e = self.encoder
        x = e.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1, 3)
        cancer = self.cancer(x)

        cancer = cancer.reshape(-1)
        cancer = torch.sigmoid(cancer)
        cancer = torch.nan_to_num(cancer)
        return cancer


class EffNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.rgb = RGB()
        self.encoder = efficientnet_b4(pretrained=pretrained)
        self.cancer = nn.Linear(1792, 1)

    def forward(self, batch):
        x = batch["image"]
        x = x.expand(-1, 3, -1, -1)
        x = self.rgb(x)

        e = self.encoder
        x = e.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1, 3)
        cancer = self.cancer(x)

        cancer = cancer.reshape(-1)
        cancer = torch.sigmoid(cancer)
        cancer = torch.nan_to_num(cancer)
        return cancer




## === cell 3
def run_submit():
    model = [
        [
            ResNet,
            "/kaggle/input/rsna-breast-mammography-weight-01/seresnext50_32x4d-512-fold0-00009366.model.pth",
        ],
        [
            ResNet,
            "/kaggle/input/rsna-breast-mammography-weight-01/seresnext50_32x4d-512-fold1-00016056.model.pth",
        ],
        [
            EffNet,
            "/kaggle/input/rsna-breast-mammography-weight-01/efficientnet_b4-512-fold0-00012042.model.pth",
        ],
        [
            EffNet,
            "/kaggle/input/rsna-breast-mammography-weight-01/efficientnet_b4-512-fold1-00012042.model.pth",
        ],
    ]
    num_net = len(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warnings.warn(
            "CUDA not available; inference will be very slow and may timeout."
        )

    net = []
    for Net, checkpoint in model:
        has_ckpt = os.path.exists(checkpoint)
        n = Net(pretrained=(not has_ckpt))
        if has_ckpt:
            f = torch.load(checkpoint, map_location="cpu", weights_only=False)
            n.load_state_dict(f["state_dict"], strict=True)
            print("loaded checkpoint:", checkpoint)
        else:
            print(
                "checkpoint missing, using ImageNet pretrained backbone for:",
                Net.__name__,
            )
        n.to(device)
        n.eval()
        net.append(n)

    test_dataset = RsnaImageDataset(test_df)

    cpu = os.cpu_count() or 2
    num_workers = min(8, max(2, cpu // 2)) if device.type == "cuda" else min(4, cpu)
    batch_size = 16 if device.type == "cuda" else 4

    def _worker_init_fn(worker_id):
        np.random.seed(0 + worker_id)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=null_collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

    prob_out = np.empty(len(test_dataset), dtype=np.float32)
    start_timer = timer()
    seen = 0

    with torch.no_grad():
        for batch in test_loader:
            img = batch["image"]
            if device.type == "cuda":
                img = img.to(device, non_blocking=True)
            else:
                img = img.to(device)

            img_flip = torch.flip(img, dims=[3])

            with amp.autocast(enabled=(device.type == "cuda")):
                p_sum = 0.0
                count = 0
                b0 = {"image": img}
                b1 = {"image": img_flip}
                for i_net in range(num_net):
                    p_sum = p_sum + net[i_net](b0)
                    p_sum = p_sum + net[i_net](b1)
                    count += 2
                p = p_sum / count

            p_np = p.float().cpu().numpy()
            idx = batch["index"]
            prob_out[idx] = p_np

            seen += len(idx)
            print(
                "\r %8d / %d  %s"
                % (seen, len(test_dataset), time_to_str(timer() - start_timer, "sec")),
                end="",
                flush=True,
            )
    print("")

    submit_df = pd.DataFrame(
        {"prediction_id": test_df["prediction_id"].values, "cancer": prob_out}
    )
    submit_df = submit_df.groupby("prediction_id", as_index=False)["cancer"].max()
    submit_df = submit_df.sort_values("prediction_id").reset_index(drop=True)
    submit_df["cancer"] = submit_df["cancer"].clip(0, 1)

    sample_path = "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        submit_df = sample[["prediction_id"]].merge(
            submit_df, on="prediction_id", how="left"
        )
        submit_df["cancer"] = submit_df["cancer"].fillna(0.0).astype(np.float32)

    submit_df.to_csv("submission.csv", index=False)
    print("wrote submission.csv", submit_df.shape)
    print(submit_df.head())
    print("mean cancer:", float(submit_df.cancer.mean()))


run_submit()
