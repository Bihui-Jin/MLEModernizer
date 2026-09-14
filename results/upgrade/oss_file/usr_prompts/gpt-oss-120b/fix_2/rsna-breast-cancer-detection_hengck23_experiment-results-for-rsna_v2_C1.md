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

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

0.0555775173506362

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
if 1:
    try:
        import subprocess, sys, os

        wheels = "/kaggle/input/rsna-2022-whl/{pydicom-2.3.0-py3-none-any.whl,pylibjpeg-1.4.0-py3-none-any.whl,python_gdcm-3.0.15-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl}"
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheels])
        print("gdcm ok!")
    except Exception as e:
        print("gdcm install skipped:", e)


import sys

sys.path.append("/kaggle/input/rsna-breast-mammography-00")


import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

from timeit import default_timer as timer

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F

print("torch.cuda.device_count() = %d" % torch.cuda.device_count())
if torch.cuda.is_available():
    print(
        "torch.cuda.get_device_properties(0) = %s"
        % str(torch.cuda.get_device_properties(0))[21:]
    )
else:
    print("CUDA not available; running on CPU.")

import timm

print(timm.__version__)
from timm.models.resnet import *  # kept for potential legacy imports


def time_to_str(t, mode="min"):
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return "%2d hr %02d min" % (hr, min)

    elif mode == "sec":
        t = int(t)
        min = t // 60
        sec = t % 60
        return "%2d min %02d sec" % (min, sec)

    else:
        raise NotImplementedError


print("import ok!")



## === cell 1
image_size = 256


is_local = False  # False #
if is_local:
    csv_file = "/kaggle/input/rsna-breast-mammography-00/valid_df.fold0.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/train_images"
else:
    csv_file = "/kaggle/input/rsna-breast-cancer-detection/test.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/test_images"


test_df = pd.read_csv(csv_file)
test_df.loc[:, "i"] = np.arange(len(test_df))
if is_local:
    test_df.loc[:, "prediction_id"] = (
        test_df.patient_id.astype(str) + "_" + test_df.laterality
    )

print("test_df", test_df.shape)
print(test_df)


def read_dicom(dcm_file):
    dicom = pydicom.dcmread(dcm_file)
    image = dicom.pixel_array.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = 1 - image
    image = (image * 255).astype(np.uint8)
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    return image


def read_data(df):
    image = []
    for _, d in df.iterrows():
        m = read_dicom(f"{dcm_dir}/{d.patient_id}/{d.image_id}.dcm")
        image.append(m)
    image = np.stack(image)
    return image


class RsnaDataset(Dataset):
    def __init__(self, df):
        patient_id = sorted(df.patient_id.unique())
        self.patient_id = patient_id
        self.length = len(patient_id)
        self.df = df

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        patient_id = self.patient_id[index]
        df = self.df[self.df.patient_id == patient_id].reset_index(drop=True)
        image = read_data(df)

        r = {}
        r["index"] = index
        r["df"] = df
        r["patient_id"] = patient_id
        r["image"] = torch.from_numpy(image).float()
        r["num"] = len(df)
        return r


tensor_key = ["image"]


def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_key:
            v = torch.cat(v, 0)
        d[k] = v
    return d




## === cell 2
class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super(RGB, self).__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        return (x - self.mean) / self.std


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rgb = RGB()
        self.encoder = timm.create_model(
            "seresnext26d_32x4d", pretrained=False, in_chans=3
        )
        self.weight = nn.Linear(2048, 1)
        self.cancer = nn.Linear(2048 * 2, 1)

    def forward(self, batch):
        x = batch["image"]  # (total_imgs, H, W)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # (total_imgs, 3, H, W)
        x = self.rgb(x)

        e = self.encoder
        x = e.conv1(x)
        x = e.bn1(x)
        x = e.act1(x)
        x = e.maxpool(x)
        for blk in [e.layer1, e.layer2, e.layer3, e.layer4]:
            x = blk(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1, 3)  # (total_imgs, 2048)

        num = batch["num"]  # list of image counts per patient
        batch_size = len(num)
        splits = torch.split_with_sizes(x, num)

        feature = []
        for b in range(batch_size):
            w = self.weight(splits[b])  # (n_i, 1)
            w = F.softmax(w, dim=0)
            pool = (splits[b] * w).sum(0, keepdim=True)  # (1, 2048)
            pool = pool.expand(num[b], -1)  # (n_i, 2048)
            f = torch.cat([splits[b], pool], -1)  # (n_i, 4096)
            feature.append(f)
        feature = torch.cat(feature, 0)  # (total_imgs, 4096)

        cancer = self.cancer(feature)  # (total_imgs, 1)
        cancer = cancer.reshape(-1)

        output = {"cancer": torch.sigmoid(cancer)}
        return output




## === cell 3
def run_submit():
    checkpoint = "/kaggle/input/rsna-breast-mammography-weight-00/00003426.model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net()
    f = torch.load(checkpoint, map_location=device)
    net.load_state_dict(f["state_dict"], strict=True)

    net = net.to(device)
    net.eval()

    test_dataset = RsnaDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=16,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=null_collate,
    )

    result = {"i": [], "probability": []}
    test_num = 0
    start_timer = timer()

    for batch in test_loader:
        batch_size = len(batch["index"])
        batch["image"] = batch["image"].to(device)

        with torch.no_grad():
            output = net(batch)

        result["probability"].append(output["cancer"].cpu().numpy())
        result["i"].append(pd.concat(batch["df"])["i"].values)
        test_num += batch_size
        print(
            f'\r {test_num} / {len(test_dataset)}  {time_to_str(timer() - start_timer, "sec")}',
            end="",
            flush=True,
        )
    print("")

    probability = np.concatenate(result["probability"])
    i = np.concatenate(result["i"])
    argsort = np.argsort(i)
    i = i[argsort]
    probability = probability[argsort]

    submit_df = pd.DataFrame({"prediction_id": test_df.prediction_id})
    submit_df["cancer"] = probability
    submit_df = submit_df.groupby("prediction_id").mean()
    submit_df = submit_df.sort_index()
    submit_df.to_csv("submission.csv", index=True)
    print("submission saved to submission.csv")
    print(submit_df.head())

    if is_local:

        def pfbeta(labels, predictions, beta=1):
            y_true_count = 0
            ctp = 0
            cfp = 0
            for idx in range(len(labels)):
                pred = min(max(predictions[idx], 0), 1)
                if labels[idx]:
                    y_true_count += 1
                    ctp += pred
                    cfp += 1 - pred
                else:
                    cfp += pred
            beta_sq = beta * beta
            c_precision = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0
            c_recall = ctp / y_true_count if y_true_count > 0 else 0
            if c_precision > 0 and c_recall > 0:
                return (
                    (1 + beta_sq)
                    * (c_precision * c_recall)
                    / (beta_sq * c_precision + c_recall)
                )
            return 0

        truth_df = (
            test_df[["prediction_id", "cancer"]]
            .groupby("prediction_id")
            .mean()
            .sort_index()
        )
        lb_score = pfbeta(truth_df.cancer.values, submit_df.cancer.values)
        print("Local pF1 score:", lb_score)


run_submit()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/3594680080.py in <cell line: 0>()
     92 
     93 
---> 94 run_submit()

/tmp/ipykernel_56/3594680080.py in run_submit()
      4     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      5     net = Net()
----> 6     f = torch.load(checkpoint, map_location=device)
      7     net.load_state_dict(f["state_dict"], strict=True)
      8 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-breast-mammography-weight-00/00003426.model.pth'
