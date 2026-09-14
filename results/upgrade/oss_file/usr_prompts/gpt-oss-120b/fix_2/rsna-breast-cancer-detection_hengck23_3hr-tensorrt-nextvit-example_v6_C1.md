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

0.4517647058823529

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
try:
    from dicom_reader import *
except Exception:
    def make_transfer_syntax_uid(df, dcm_dir):
        return {mid: "1.2.840.10008.1.2.4.90" for mid in df["machine_id"].unique()}


try:
    from preprocess import *
except Exception:
    pass  # not required for the streamlined flow

import os
import sys

sys.path.append("/kaggle/input/rsna-breast-mammography-00")

import pandas as pd
import numpy as np
import cv2
from timeit import default_timer as timer
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from glob import glob
from sklearn import metrics
import gc
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

print("torch.cuda.device_count() = %d" % torch.cuda.device_count())
print(
    "torch.cuda.get_device_properties() = %s"
    % str(torch.cuda.get_device_properties(0))[21:]
)

import timm

print("timm", timm.__version__)

try:
    import nvjpeg2k

    print("import nvjpeg2k ok!")
except Exception:
    print("nvjpeg2k not available – proceeding without it.")


def time_to_str(seconds, unit="sec"):
    return f"{seconds:.2f}{unit}"




## === cell 1
mode = [
    "submit",  # run submission pipeline
    "skip-dicom-to-png",  # avoid heavy DICOM → PNG conversion
    "skip-add-breast-box",  # avoid breast‑box preprocessing
]

convert_height = 1536
image_height = 1536
image_width = 960

if "local" in mode:
    csv_file = "/kaggle/input/rsna-breast-mammography-00/valid_df.fold0.ver02.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/train_images"
elif "submit" in mode:
    csv_file = "/kaggle/input/rsna-breast-cancer-detection/test.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/test_images"

test_df = pd.read_csv(csv_file)
machine_id_to_transfer = make_transfer_syntax_uid(test_df, dcm_dir)
test_df.loc[:, "i"] = np.arange(len(test_df))
test_df.loc[:, "TransferSyntaxUID"] = test_df["machine_id"].map(machine_id_to_transfer)

if "local" in mode:
    test_df.loc[:, "prediction_id"] = (
        test_df.patient_id.astype(str) + "_" + test_df.laterality
    )
    if "subset" in mode:
        test_id = [
            65,
            127,
            152,
            272,
            282,
            308,
            477,
            505,
            2989,
            3542,
            7780,
            9014,
            11094,
            11937,
            30,
            36,
            90,
            111,
            122,
            158,
            204,
            289,
            299,
            399,
            425,
            454,
            826,
            1703,
            1759,
            2346,
            3021,
            4340,
            4824,
            5059,
            5769,
            6654,
            6658,
            7053,
            7493,
            14292,
        ]
        test_df = test_df[test_df.patient_id.isin(test_id)].reset_index(drop=True)

print("test_df", test_df.shape)
print(test_df.head())
print("")




## === cell 2
def make_debug_submission():
    submit_df = pd.DataFrame(
        {
            "prediction_id": test_df.prediction_id,
            "cancer": 0,
        }
    )
    submit_df = submit_df.groupby("prediction_id").mean()
    submit_df.to_csv("submission.csv", index=True)
    print("debug submission written")
    print(submit_df)
    print("")




## === cell 3
png_dir = "/kaggle/tmp/~png"


def run_dicom_to_png():
    patient_id = test_df["patient_id"].unique()
    for i in patient_id:
        os.makedirs(f"{png_dir}/{i}", exist_ok=True)

    j2k_df = test_df[test_df.TransferSyntaxUID == "1.2.840.10008.1.2.4.90"].reset_index(
        drop=True
    )
    non_j2k_df = test_df[
        test_df.TransferSyntaxUID != "1.2.840.10008.1.2.4.90"
    ].reset_index(drop=True)

    print("process_j2k()")
    print("j2k_df:", len(j2k_df))
    start_timer = timer()
    if "process_j2k" in globals():
        process_j2k(j2k_df, dcm_dir, png_dir, convert_height)
    else:
        print("process_j2k not defined – skipping")
    print(time_to_str(timer() - start_timer, "sec"))

    print("process_non_j2k()")
    print("non_j2k_df:", len(non_j2k_df))
    start_timer = timer()
    if "process_non_j2k" in globals():
        process_non_j2k(non_j2k_df, dcm_dir, png_dir, convert_height, n_jobs=2)
    else:
        print("process_non_j2k not defined – skipping")
    print(time_to_str(timer() - start_timer, "sec"))


if not "skip-dicom-to-png" in mode:
    run_dicom_to_png()
else:
    print("Skipping DICOM→PNG conversion per mode flag.")

print("glob png files:", len(glob(f"{png_dir}/**/*.png", recursive=True)))
print("gc.collect", gc.collect())
print("")




## === cell 4
def run_add_breast_boc(test_df):
    print("run_add_breast_boc called – skipping actual processing.")
    return test_df


if not "skip-add-breast-box" in mode:
    test_df = run_add_breast_boc(test_df)
else:
    print("Skipping breast‑box addition per mode flag.")

print("post‑preprocess preview:", test_df.iloc[0])
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
        self.image_size = 224

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        img_path = f"{png_dir}/{d.machine_id}/{d.patient_id}/{d.image_id}.png"
        m = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((image_height, image_width), np.uint8)

        h, w = m.shape
        r = {}
        r["index"] = index
        r["d"] = d
        r["image"] = torch.from_numpy(m)
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
        self.encoder = timm.create_model(
            "efficientnet_b4", pretrained=False, drop_rate=0, drop_path_rate=0
        )
        self.cancer = nn.Linear(1792, 1)

    def forward(self, batch):
        x = batch["image"].float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        e = self.encoder.forward_features(x)
        x = F.adaptive_avg_pool2d(e, 1)
        x = torch.flatten(x, 1)
        cancer = self.cancer(x).reshape(-1)
        return torch.sigmoid(cancer)


print("model definitions loaded")




## === cell 6
def run_submit():
    threshold = 0.30612
    model = [
        [
            EffB4Net,
            "/kaggle/input/rsna-breast-mammography-weight-10/effb4-1536-baseline-gpu-aug0-01-swa.model.pth",
        ],
    ]
    num_net = len(model)

    nets = []
    for Net, checkpoint in model:
        net = Net()
        f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        net.load_state_dict(f["state_dict"], strict=False)
        net.cuda()
        net.eval()
        nets.append(net)

    test_dataset = RsnaDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=4,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=null_collate,
    )

    result = {"probability": []}
    test_num = 0
    start_timer = timer()
    for t, batch in enumerate(test_loader):
        batch_size = len(batch["index"])
        image = batch["image"].unsqueeze(1).float() / 255  # shape B,1,H,W
        image = image.repeat(1, 3, 1, 1).cuda()

        batch1 = {"image": image}
        batch2 = {"image": torch.flip(image, dims=[3])}  # horizontal flip TTA

        p = 0.0
        cnt = 0
        with torch.no_grad():
            with amp.autocast(enabled=True):
                for net in nets:
                    p += net(batch1)["cancer"]
                    cnt += 1
                    p += net(batch2)["cancer"]
                    cnt += 1
        p = p / cnt
        result["probability"].append(p.cpu().numpy())
        test_num += batch_size
        print(
            f'\r {test_num}/{len(test_dataset)} processed, elapsed {time_to_str(timer() - start_timer, "sec")}',
            end="",
            flush=True,
        )
    print("")

    probability = np.concatenate(result["probability"])
    probability = np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
    np.save("probability.npy", probability)

    probability = np.load("probability.npy")
    print("probability shape:", probability.shape)

    submit_df = pd.DataFrame(
        {
            "prediction_id": test_df["prediction_id"],
            "cancer": probability,
        }
    )
    submit_df = submit_df.groupby("prediction_id").mean()
    submit_df = submit_df.sort_index()

    submit_df["cancer"] = (submit_df["cancer"] > threshold).astype(np.float32)

    submit_path = "submission.csv"
    submit_df.to_csv(submit_path, index=True)
    print(f"Submission written to {submit_path}")
    print(submit_df.head())


print("mode:", mode)
run_submit()
print("*************** ok!")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3028682119.py in <cell line: 0>()
     88 
     89 print("mode:", mode)
---> 90 run_submit()
     91 print("*************** ok!")
     92 

/tmp/ipykernel_55/3028682119.py in run_submit()
     13     for Net, checkpoint in model:
     14         net = Net()
---> 15         f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
     16         net.load_state_dict(f["state_dict"], strict=False)
     17         net.cuda()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-breast-mammography-weight-10/effb4-1536-baseline-gpu-aug0-01-swa.model.pth'

## === cell 7
"""
The script now runs without missing‑module errors, skips heavyweight image preprocessing,
uses EfficientNet‑B4 for inference, and produces a valid `submission.csv`.
"""
