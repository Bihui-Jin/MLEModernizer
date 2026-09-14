# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

3.10

# 3. Installed packages

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
protobuf==6.33.0
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
setuptools==75.2.0
setuptools-scm==9.2.2
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
types-setuptools==80.9.0.20250529

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

# 5. Target score

-1.0

# 6. Current score

0.58706

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.58706) has done: 'I fix the runtime errors preventing data preparation and training: remove the TensorFlow seeding call that crashes due to protobuf incompatibility, update `pydicom.read_file` to `pydicom.dcmread`, and add missing `re` import plus guards for corrupted/empty DICOMs. Then I ensure PNG generation is complete by skipping bad slices and dropping any training rows whose PNGs weren’t created, which resolves the `FileNotFoundError` and lets the FastAI dataloaders build. Finally, I generate predictions for exactly the IDs in `sample_submission.csv` (not `os.listdir`), preserving row count/order and writing a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import sys
import os
import platform

print(sys.version)
print(os.name)
print(platform.system())
print(platform.release())



## === cell 1
import torch

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))



## === cell 2
import shutil, subprocess, textwrap

nvcc = shutil.which("nvcc")
if nvcc:
    print(subprocess.check_output([nvcc, "--version"], text=True))
else:
    print("nvcc not found")



## === cell 3
smi = shutil.which("nvidia-smi")
if smi:
    print(subprocess.check_output([smi], text=True))
else:
    print("nvidia-smi not found")



## === cell 4
import re
import pydicom
import pandas as pd
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
from PIL import Image

from fastai.vision.all import *
import numpy as np
import random

np.set_printoptions(threshold=sys.maxsize)



## === cell 5
torch.cuda.empty_cache()



## === cell 6
EPOCHS = 10
INPUT_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
LABELS_PATH = os.path.join(INPUT_PATH, "train_labels.csv")

df = pd.read_csv(LABELS_PATH, dtype={"BraTS21ID": str, "MGMT_value": int})
df = df.rename(columns={"BraTS21ID": "id", "MGMT_value": "value"})
df["id"] = df["id"].str.zfill(5)

exclude_cases = ["00109", "00123", "00709"]
df = df[~df.id.isin(exclude_cases)].reset_index(drop=True)
df.head()



## === cell 7
os.makedirs("./train", exist_ok=True)
print("Train folder created")
os.makedirs("./test", exist_ok=True)
print("Test folder created")




## === cell 8
def seed_everything(seed=2021):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seed done!")


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def process_dicom(path):
    dicom = pydicom.dcmread(path, force=True)

    if not hasattr(dicom, "pixel_array"):
        return None

    data = apply_voi_lut(dicom.pixel_array, dicom)

    if getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1":
        data = np.amax(data) - data

    data = data.astype(np.float32)
    data = data - np.min(data)
    max_val = np.max(data)
    if max_val <= 0 or not np.isfinite(max_val):
        return None

    data = data / max_val
    data = (data * 255.0).clip(0, 255).astype(np.uint8)
    return data


def save_image(data, outpath):
    if data is None:
        return False

    img = Image.fromarray(data, mode="L")
    img.save(outpath)
    return True


def resolve_dicom_files(input_dir, dataset="train"):
    dataset_dir = os.path.join(input_dir, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    made = 0
    failed = 0

    for subdir, dirs, files in os.walk(dataset_dir):
        if len(files) == 0:
            continue

        if os.path.basename(subdir) != "FLAIR":
            continue

        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) == 0:
            continue

        filename = natural_sort(dcm_files)[len(dcm_files) // 2]
        filepath = os.path.join(subdir, filename)

        cur_id = os.path.basename(os.path.dirname(subdir))  # subject folder name
        outpath = os.path.join(f"./{dataset}", f"{cur_id}.png")

        data = process_dicom(filepath)
        ok = save_image(data, outpath)
        if ok:
            made += 1
        else:
            failed += 1

    print(f"[{dataset}] PNGs written: {made}, failed/skipped: {failed}")




## === cell 9
seed_everything()



## === cell 10
resolve_dicom_files(INPUT_PATH, "train")
resolve_dicom_files(INPUT_PATH, "test")



## === cell 11
df["file"] = df["id"].apply(lambda x: f"./train/{x}.png")
exists_mask = df["file"].apply(os.path.exists)
missing = (~exists_mask).sum()
if missing:
    print(
        f"Dropping {missing} training rows with missing PNGs (conversion skipped/failed)."
    )
df = df[exists_mask].reset_index(drop=True)
df.head()



## === cell 12
df["value"] = df["value"].astype(str)
dls = ImageDataLoaders.from_df(
    df, item_tfms=Resize(224), bs=64, label_col="value", fn_col="file", path=""
)
dls.show_batch(max_n=6)



## === cell 13
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
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


net = Net()
print(net)



## === cell 14
params = list(net.parameters())
print(len(params))
print(params[0].size())



## === cell 15
learn = cnn_learner(dls, Net, metrics=[accuracy], model_dir="/tmp/model/")
learn = learn.to_fp16() if torch.cuda.is_available() else learn
learn



## === cell 16
learn.lr_find()



## === cell 17
learn.fit_one_cycle(EPOCHS, lr_max=1e-2)



## === cell 18
learn.show_results(max_n=6)



## === cell 19
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15, 11))



## === cell 20
SAMPLE_PATH = os.path.join(INPUT_PATH, "sample_submission.csv")
sample_sub = pd.read_csv(SAMPLE_PATH, dtype={"BraTS21ID": str})
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].str.zfill(5)

df_test = pd.DataFrame({"id": sample_sub["BraTS21ID"].values})
df_test.head(), df_test.shape



## === cell 21
probs = []
for id_num in tqdm(df_test["id"].tolist()):
    full_path = f"./test/{id_num}.png"
    if not os.path.exists(full_path):
        probs.append(0.5)
        continue
    pred = learn.predict(full_path)  # returns (pred_class, pred_idx, probs)
    p = pred[2]
    try:
        if hasattr(learn.dls, "vocab") and "1" in learn.dls.vocab:
            idx1 = list(learn.dls.vocab).index("1")
            probs.append(float(p[idx1]))
        else:
            probs.append(float(p.max()))
    except Exception:
        probs.append(float(p.max()))
df_test["value"] = probs
df_test["value"].min(), df_test["value"].max()



## === cell 22
df_output = df_test.rename(columns={"id": "BraTS21ID", "value": "MGMT_value"})

df_output = (
    df_output.set_index("BraTS21ID").reindex(sample_sub["BraTS21ID"]).reset_index()
)

df_output["MGMT_value"] = df_output["MGMT_value"].astype(float).clip(0.0, 1.0)

out_path = "submission.csv"
df_output.to_csv(out_path, index=False)
print(df_output.head())
print("Wrote:", out_path, "rows:", len(df_output))



## === cell 23
chk = pd.read_csv(out_path)
print(chk.shape)
print(chk.columns.tolist())
print(chk.isna().sum().to_dict())
