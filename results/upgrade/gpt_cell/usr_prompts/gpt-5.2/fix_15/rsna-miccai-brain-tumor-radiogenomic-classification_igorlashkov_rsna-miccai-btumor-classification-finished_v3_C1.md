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

0.51882

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51882) has done: 'Diagnosis: Cell 25 contains plain English text (with smart quotes) that is not commented or wrapped in a string, so Python attempts to parse it and raises a `SyntaxError`. This prevents the intended inference loop from running and stops `df_test['value']` from being populated. The correct fix is to replace the cell contents with valid Python that performs prediction for each test PNG created earlier. Additionally, filtering `df_test` to IDs that actually have `./test/{id}.png` avoids runtime errors if any IDs didn’t get converted.

Patch summary: Replace the invalid prose in cell 25 with executable Python that (1) filters to existing PNGs, (2) runs `learn.predict` per test image path, and (3) writes the positive-class probability into `df_test.value`, preserving the existing `df_test` interface for cell 26.

Updated cells: Cell 25 only.

Compatibility notes for cell k+1: Cell 26 expects `df_test` to exist and be a DataFrame; this patch keeps `df_test` defined and populates/retains the `id` and `value` columns so `df_test.head()` continues to work.

Assumptions: The test PNGs were generated in `./test/` by cell 11; if some are missing, they should be skipped rather than crashing.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 0 is not valid Python code; it contains plain English prose (including a Unicode right-quote ’) that Python tries to parse, triggering a `SyntaxError: invalid character '’' (U+2019)`. The root cause is that the notebook export accidentally placed narrative text as executable code. The minimal fix is to turn that content into a Python comment/docstring so the kernel can execute the cell without changing any runtime logic. This unblocks execution for cell 1 and beyond without altering variables or model behavior.

Patch summary: Replace the prose in cell 0 with a harmless module-level docstring (or comments) so the cell executes cleanly and does not affect downstream cells.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 30 crashes with a `SyntaxError` because it contains a stray Markdown code-fence token ('
- What this solution (achieved 0.51882) has done: 'Diagnosis: The notebook is failing in cell 0 because it contains plain narrative prose (with a Unicode right-quote ’) that Python tries to parse as code, producing a SyntaxError before any real cells can run. The earliest unblock is to make cell 0 valid Python without changing any runtime behavior or variables. The safest minimal fix is to wrap that narrative text in a triple-quoted string so it becomes a harmless module-level string literal.

Patch summary: Convert the non-Python narrative in cell 0 into a docstring (or string literal) so the cell executes successfully and does not affect later cells.

Updated cells: Only cell 0 is changed.

Compatibility notes for cell k+1: Cell 1 already contains a docstring explaining the same issue; leaving it unchanged is compatible and does not depend on any variables from cell 0.

Assumptions: No later cell depends on side effects from the original narrative content in cell 0 (since it never executed successfully).'
- What this solution (achieved 0.51882) has done: 'Diagnosis: Cell 31 is failing with a `SyntaxError` because it contains a Markdown code fence ('

# 9. Code solution

## === cell 0
"""
Your current submission is stuck at 0.5 because the pipeline never actually performs inference:
cell 26 overwrites `df_test` and sets every prediction to a constant 0.5. To move the AUC upward
toward your target (higher is better), the minimal change is to keep your existing training setup
intact and only add a small inference loop that uses `learn.predict` on the already-created
`./test/{id}.png` files, writing the positive-class probability into `df_test["value"]`.
I’ll also ensure IDs are consistently treated as zero-padded strings and keep the submission
format identical. Everything else (DICOM-to-PNG, DataLoaders, model definition, training loop,
loss/metrics) remains unchanged.
"""


## === cell 1
"""
NOTE: This cell previously contained non-Python narrative text (including a Unicode right-quote ’),
which caused a SyntaxError when executed. It has been converted into a docstring so execution
can proceed without changing any runtime logic or variables used by later cells.
"""


## === cell 2
import sys
import os
import platform
print(sys.version)
print(os.name)

print(platform.system())
print(platform.release())


## === cell 3
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


## === cell 4
!nvcc --version


## === cell 5
!nvidia-smi


## === cell 6
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import pydicom
import pandas as pd
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
import binascii
from PIL import Image

from fastai.vision.all import *
import numpy as np
import pandas as pd
import random
np.set_printoptions(threshold=sys.maxsize)


## === cell 7
torch.cuda.empty_cache()


## === cell 8
EPOCHS = 10
INPUT_PATH = '../input/rsna-miccai-brain-tumor-radiogenomic-classification'
LABELS_PATH = os.path.join(INPUT_PATH, 'train_labels.csv')
df = pd.read_csv(LABELS_PATH, header=0, names=['id','value'], dtype=object)
exclude_cases = ["00109", "00123", "00709"]
df = df[~df.id.isin(exclude_cases)]


## === cell 9
df.head()


## === cell 10
os.makedirs('./train', exist_ok = True)
print('Train folder created')

os.makedirs('./test', exist_ok = True)
print('Test folder created')


## === cell 11
def seed_everything(seed=2021):
    import random
    import os
    import tensorflow as tf
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('Seed done!')
    
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)    
    
def process_dicom(path):
    dicom = pydicom.read_file(path)
    data = apply_voi_lut(dicom.pixel_array, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    max_val = np.max(data)
    if max_val == 0:  # RuntimeWarning: invalid value encountered in true_divide
        return None
    
    data = data - np.min(data)
    data = data / max_val
    data = (data * 255).astype(np.uint8)
    return data
    
def save_image(data, outpath):
    height = len(data)
    width = len(data[0])
    
    pixels_out = []
    for row in data:
        pixels_out.extend(row)
    assert(len(pixels_out) == height * width)
    
    image_out = Image.new('L', (width, height))
    image_out.putdata(pixels_out)
    image_out.save(outpath)
    
def resolve_dicom_files(input_dir, dataset='train'):
    for subdir, dirs, files in os.walk(f"{input_dir}/{dataset}"):
        if len(files) == 0:
            continue
        filename = natural_sort(files)[len(files)//2] #take middle most image -- FLAIR DCM file per training item.
        filepath = os.path.join(subdir, filename)
        
        if filepath.endswith(".dcm") and "FLAIR" in filepath:
            cur_id = subdir.split('/')[-2]
            outpath = os.path.join(f'./{dataset}',f'{cur_id}.png')
            
            data = process_dicom(filepath)
            save_image(data, outpath)


## === cell 12
def seed_everything(seed=2021):
    import random
    import os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("Seed done!")


seed_everything()


## === cell 13
import re

if not hasattr(pydicom, "read_file") and hasattr(pydicom, "dcmread"):
    pydicom.read_file = pydicom.dcmread

resolve_dicom_files(INPUT_PATH, "train")
resolve_dicom_files(INPUT_PATH, "test")


## === cell 14
for id_num in df.id:
    full_path = f'./train/{id_num}.png'
    df.loc[df.id == id_num, 'file'] = full_path


## === cell 15
df


## === cell 16
dls = ImageDataLoaders.from_df(df, item_tfms=Resize(224), bs=64, label_col =1, fn_col=2, path='')


## === cell 17
dls.show_batch()


## === cell 18
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
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


## === cell 19
net = Net()
print(net)


## === cell 20
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


## === cell 21
learn = cnn_learner(dls, Net, metrics=[error_rate, accuracy], model_dir="/tmp/model/").to_fp16()


## === cell 22
learn.lr_find()


## === cell 23
%%time
learn.fit_one_cycle(EPOCHS, lr_max=1e-2)


## === cell 24
learn.show_results()


## === cell 25
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


## === cell 26
df_test = pd.DataFrame(columns=['id', 'value'])
df_test.id = os.listdir(os.path.join(INPUT_PATH, "test/"))


## === cell 27
import os

sample_path = os.path.join(INPUT_PATH, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path, dtype={"BraTS21ID": str})

df_test = pd.DataFrame({"id": sample_sub["BraTS21ID"].astype(str)})

df_test["value"] = 0.5

existing_mask = df_test["id"].map(lambda x: os.path.exists(f"./test/{x}.png"))
existing_ids = df_test.loc[existing_mask, "id"].tolist()

print(f"Found ./test/{{id}}.png for: {len(existing_ids)}/{len(df_test)}")

learn.model.eval()
with torch.no_grad():
    for _id in tqdm(existing_ids, desc="Predicting test images"):
        img_path = f"./test/{_id}.png"
        pred_class, pred_idx, probs = learn.predict(img_path)

        vocab = list(getattr(learn.dls.vocab, "items", learn.dls.vocab))
        if "1" in vocab:
            pos_idx = vocab.index("1")
        elif 1 in vocab:
            pos_idx = vocab.index(1)
        else:
            pos_idx = min(1, len(probs) - 1)

        df_test.loc[df_test["id"] == _id, "value"] = float(probs[pos_idx].cpu().item())

missing_png = (~existing_mask).sum()
print(f"Missing ./test/{{id}}.png files (submitting 0.5 for those): {missing_png}/{len(df_test)}")
print("Pred range:", df_test["value"].min(), df_test["value"].max())


## === cell 28
df_test.head()


## === cell 29
df_test.value.min(), df_test.value.max()


## === cell 30
df_output = df_test.rename(columns={'id':'BraTS21ID','value':'MGMT_value'})
df_output.to_csv('submission.csv', index=False)
df_output.head()


## === cell 31
import pkg_resources
import types


def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            name = val.__name__.split(".")[0]
        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        poorly_named_packages = {"PIL": "Pillow", "sklearn": "scikit-learn"}
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name


imports = list(set(get_imports()))

requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name != "pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))
