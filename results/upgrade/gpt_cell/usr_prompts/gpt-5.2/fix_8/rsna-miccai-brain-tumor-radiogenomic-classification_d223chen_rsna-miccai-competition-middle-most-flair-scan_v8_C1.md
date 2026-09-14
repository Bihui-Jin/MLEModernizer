# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 3. Data file paths

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

# 4. Code solution

## === cell 0
from fastai.vision.all import *
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)



## === cell 1
import pandas as pd
import os
import random

labels_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
df = pd.read_csv(labels_path, dtype={"BraTS21ID": str, "MGMT_value": np.int64})
df = df.rename(columns={"BraTS21ID": "id", "MGMT_value": "value"})
df["id"] = df["id"].str.zfill(5)
df = df[~df.id.isin(["00109", "00123", "00709"])].reset_index(drop=True)



## === cell 2
df.head()



## === cell 3
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import torch

INPUT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

os.makedirs("./train", exist_ok=True)
os.makedirs("./test", exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass


def process_dicom(path, outpath):
    dicom = pydicom.dcmread(path)
    data = apply_voi_lut(dicom.pixel_array, dicom)
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        data = np.amax(data) - data
    data = data.astype(np.float32)
    data = data - np.min(data)
    mx = np.max(data)
    if mx > 0:
        data = data / mx
    data = (data * 255.0).clip(0, 255).astype(np.uint8)

    image_out = Image.fromarray(data, mode="L")
    image_out.save(outpath)


def get_dicom_files(input_dir, ds="train"):
    for subdir, dirs, files in os.walk(f"{input_dir}/{ds}"):
        if len(files) == 0:
            continue
        flair_files = [f for f in files if f.endswith(".dcm") and "Image" in f]
        if len(flair_files) == 0:
            continue

        if "FLAIR" not in subdir:
            continue

        filename = random.choice(flair_files)
        filepath = os.path.join(subdir, filename)

        cur_id = subdir.split("/")[-2]
        outpath = os.path.join(f"./{ds}", f"{cur_id}.png")
        if not os.path.exists(outpath):
            process_dicom(filepath, outpath)


get_dicom_files(INPUT, "train")
get_dicom_files(INPUT, "test")



## === cell 4
df["file"] = df["id"].apply(lambda x: f"./train/{x}.png")
df = df[df["file"].apply(os.path.exists)].reset_index(drop=True)
df.head()



## === cell 5
dls = ImageDataLoaders.from_df(
    df,
    item_tfms=Resize(224),
    bs=64,
    label_col="value",
    fn_col="file",
    path="",
    y_block=CategoryBlock,
)



## === cell 6
dls.show_batch(max_n=9)



## === cell 7
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        return x




## === cell 8
learn = cnn_learner(
    dls, Net, metrics=[RocAucBinary(), accuracy, error_rate], model_dir="/tmp/model/"
).to_fp16()



## === cell 9
learn.lr_find()



## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/844731991.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mlearn[0m[0;34m.[0m[0mlr_find[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py[0m in [0;36mlr_find[0;34m(self, start_lr, end_lr, num_it, stop_div, show_plot, suggest_funcs)[0m
[1;32m    294[0m     [0mn_epoch[0m [0;34m=[0m [0mnum_it[0m[0;34m//[0m[0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdls[0m[0;34m.[0m[0mtrain[0m[0;34m)[0m [0;34m+[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    295[0m     [0mcb[0m[0;34m=[0m[0mLRFinder[0m[0;34m([0m[0mstart_lr[0m[0;34m=[0m[0mstart_lr[0m[0;34m,[0m [0mend_lr[0m[0;34m=[0m[0mend_lr[0m[0;34m,[0m [0mnum_it[0m[0;34m=[0m[0mnum_it[0m[0;34m,[0m [0mstop_div[0m[0;34m=[0m[0mstop_div[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 296[0;31m     [0;32mwith[0m [0mself[0m[0;34m.[0m[0mno_logging[0m[0;34m([0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mn_epoch[0m[0;34m,[0m [0mcbs[0m[0;34m=[0m[0mcb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    297[0m     [0;32mif[0m [0msuggest_funcs[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    298[0m         [0mlrs[0m[0;34m,[0m [0mlosses[0m [0;34m=[0m [0mtensor[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mrecorder[0m[0;34m.[0m[0mlrs[0m[0;34m[[0m[0mnum_it[0m[0;34m//[0m[0;36m10[0m[0;34m:[0m[0;34m-[0m[0;36m5[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0mtensor[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mrecorder[0m[0;34m.[0m[0mlosses[0m[0;34m[[0m[0mnum_it[0m[0;34m//[0m[0;36m10[0m[0;34m:[0m[0;34m-[0m[0;36m5[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mfit[0;34m(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)[0m
[1;32m    270[0m             [0mself[0m[0;34m.[0m[0mopt[0m[0;34m.[0m[0mset_hypers[0m[0;34m([0m[0mlr[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlr[0m [0;32mif[0m [0mlr[0m [0;32mis[0m [0;32mNone[0m [0;32melse[0m [0mlr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    271[0m             [0mself[0m[0;34m.[0m[0mn_epoch[0m [0;34m=[0m [0mn_epoch[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 272[0;31m             [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_fit[0m[0;34m,[0m [0;34m'fit'[0m[0;34m,[0m [0mCancelFitException[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_end_cleanup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    273[0m [0;34m[0m[0m
[1;32m    274[0m     [0;32mdef[0m [0m_end_cleanup[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mdl[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mxb[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0myb[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mpred[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mloss[0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m([0m[0;32mNone[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m[0;34m([0m[0;32mNone[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m[0;32mNone[0m[0;34m,[0m[0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_fit[0;34m(self)[0m
[1;32m    259[0m         [0;32mfor[0m [0mepoch[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mn_epoch[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    260[0m             [0mself[0m[0;34m.[0m[0mepoch[0m[0;34m=[0m[0mepoch[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 261[0;31m             [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_epoch[0m[0;34m,[0m [0;34m'epoch'[0m[0;34m,[0m [0mCancelEpochException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    262[0m [0;34m[0m[0m
[1;32m    263[0m     [0;32mdef[0m [0mfit[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mn_epoch[0m[0;34m,[0m [0mlr[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mwd[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mcbs[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mreset_opt[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mstart_epoch[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_epoch[0;34m(self)[0m
[1;32m    253[0m [0;34m[0m[0m
[1;32m    254[0m     [0;32mdef[0m [0m_do_epoch[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 255[0;31m         [0mself[0m[0;34m.[0m[0m_do_epoch_train[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    256[0m         [0mself[0m[0;34m.[0m[0m_do_epoch_validate[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    257[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_epoch_train[0;34m(self)[0m
[1;32m    245[0m     [0;32mdef[0m [0m_do_epoch_train[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    246[0m         [0mself[0m[0;34m.[0m[0mdl[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdls[0m[0;34m.[0m[0mtrain[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 247[0;31m         [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mall_batches[0m[0;34m,[0m [0;34m'train'[0m[0;34m,[0m [0mCancelTrainException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    248[0m [0;34m[0m[0m
[1;32m    249[0m     [0;32mdef[0m [0m_do_epoch_validate[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mds_idx[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m [0mdl[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mall_batches[0;34m(self)[0m
[1;32m    211[0m     [0;32mdef[0m [0mall_batches[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    212[0m         [0mself[0m[0;34m.[0m[0mn_iter[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 213[0;31m         [0;32mfor[0m [0mo[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdl[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mone_batch[0m[0;34m([0m[0;34m*[0m[0mo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    214[0m [0;34m[0m[0m
[1;32m    215[0m     [0;32mdef[0m [0m_backward[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mloss_grad[0m[0;34m.[0m[0mbackward[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mone_batch[0;34m(self, i, b)[0m
[1;32m    241[0m         [0mb[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_set_device[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    242[0m         [0mself[0m[0;34m.[0m[0m_split[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 243[0;31m         [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_one_batch[0m[0;34m,[0m [0;34m'batch'[0m[0;34m,[0m [0mCancelBatchException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    244[0m [0;34m[0m[0m
[1;32m    245[0m     [0;32mdef[0m [0m_do_epoch_train[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_one_batch[0;34m(self)[0m
[1;32m    222[0m [0;34m[0m[0m
[1;32m    223[0m     [0;32mdef[0m [0m_do_one_batch[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 224[0;31m         [0mself[0m[0;34m.[0m[0mpred[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mmodel[0m[0;34m([0m[0;34m*[0m[0mself[0m[0;34m.[0m[0mxb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    225[0m         [0mself[0m[0;34m([0m[0;34m'after_pred'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    226[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0myb[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    248[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    249[0m         [0;32mfor[0m [0mmodule[0m [0;32min[0m [0mself[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m             [0minput[0m [0;34m=[0m [0mmodule[0m[0;34m([0m[0minput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    251[0m         [0;32mreturn[0m [0minput[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    248[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    249[0m         [0;32mfor[0m [0mmodule[0m [0;32min[0m [0mself[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m             [0minput[0m [0;34m=[0m [0mmodule[0m[0;34m([0m[0minput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    251[0m         [0;32mreturn[0m [0minput[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    123[0m [0;34m[0m[0m
[1;32m    124[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m         [0;32mreturn[0m [0mF[0m[0;34m.[0m[0mlinear[0m[0;34m([0m[0minput[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mweight[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mbias[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m     [0;32mdef[0m [0mextra_repr[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mstr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 10
learn.fit_one_cycle(10, lr_max=1e-2)
