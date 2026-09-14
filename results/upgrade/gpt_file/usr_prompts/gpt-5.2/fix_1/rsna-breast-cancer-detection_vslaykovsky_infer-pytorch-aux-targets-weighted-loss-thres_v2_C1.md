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

0.041385369226159

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
try:
    import pylibjpeg
except:
    !mkdir -p /root/.cache/torch/hub/checkpoints/
    !cp ../input/rsna-2022-whl/efficientnet_v2_s-dd5fe13b.pth  /root/.cache/torch/hub/checkpoints/

    !pip install /kaggle/input/rsna-2022-whl/{pydicom-2.3.0-py3-none-any.whl,pylibjpeg-1.4.0-py3-none-any.whl,python_gdcm-3.0.15-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl}
    !pip install /kaggle/input/rsna-2022-whl/{torch-1.12.1-cp37-cp37m-manylinux1_x86_64.whl,torchvision-0.13.1-cp37-cp37m-manylinux1_x86_64.whl}

## === cell 3
import gc
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.notebook import tqdm

import wandb

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
plt.rcParams['figure.figsize'] = (20, 5)


WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT
RSNA_2022_PATH = '/kaggle/input/rsna-breast-cancer-detection'
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'
EFFNET_CHECKPOINTS_PATH = '/kaggle/input/breast-cancer-models'


MODEL_NAMES = [f'effnetv2-f{i}' for i in range(5)]
 
try:
    from kaggle_secrets import UserSecretsClient
    IS_KAGGLE = True
except:
    IS_KAGGLE = False


os.environ["WANDB_MODE"] = "offline"

if os.environ["WANDB_MODE"] == "online":
    if IS_KAGGLE:
        os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret("WANDB_API_KEY")

if not IS_KAGGLE:
    print('Running locally')
    RSNA_2022_PATH = '/mnt/rsna2022'
    TRAIN_IMAGES_PATH = '/mnt/rsna2022/train_images'
    TEST_IMAGES_PATH = '/mnt/rsna2022/test_images'
    METADATA_PATH = '/home/vslaykovsky/Downloads/'
    EFFNET_CHECKPOINTS_PATH = 'checkpoints'
    os.environ['WANDB_API_KEY'] = 'yourkeyhere'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 2

## === cell 5
def load_df_test():
    df_test = pd.read_csv(f'{RSNA_2022_PATH}/test.csv')
    return df_test

df_test = load_df_test()
df_test

## === cell 7
def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images. 
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img=dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array    
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data=(data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img


for fname in glob.glob(f'/kaggle/input/rsna-breast-cancer-detection/train_images/10006/*.dcm'):
    im, meta = load_dicom(fname)
    plt.figure()
    plt.imshow(im)
    break

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/64436654.py in <cell line: 0>()
     15 
     16 for fname in glob.glob(f'/kaggle/input/rsna-breast-cancer-detection/train_images/10006/*.dcm'):
---> 17     im, meta = load_dicom(fname)
     18     plt.figure()
     19     plt.imshow(im)

/tmp/ipykernel_11/64436654.py in load_dicom(path)
      6     img=dicom.dcmread(path)
      7     img.PhotometricInterpretation = 'YBR_FULL'
----> 8     data = img.pixel_array
      9     data = data - np.min(data)
     10     if np.max(data) != 0:

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
   1009             # Processing may give us a new writeable array anyway, so do
   1010             #   it first to avoid an unnecessary ndarray.copy()
-> 1011             arr, overrides = runner.process(arr)
   1012 
   1013         arr = arr.copy() if not arr.flags.writeable and as_writeable else arr

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in process(self, arr)
    568         changes: dict[str, str | int] = {}
    569         for func in PROCESSORS:
--> 570             arr = func(arr, self, changes)
    571 
    572         return arr, changes

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _process_color_space(arr, runner, changes)
    134         changes["photometric_interpretation"] = PI.YBR_FULL
    135     elif to_rgb:
--> 136         arr = convert_color_space(arr, PI.YBR_FULL, PI.RGB)
    137         changes["photometric_interpretation"] = PI.RGB
    138 

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/processing.py in convert_color_space(arr, current, desired, per_frame)
    862     """
    863     if arr.dtype != np.dtype("u1"):
--> 864         raise ValueError(
    865             f"Invalid ndarray.dtype '{arr.dtype}' for color space conversion, "
    866             "must be 'uint8' or an equivalent"

ValueError: Invalid ndarray.dtype 'uint16' for color space conversion, must be 'uint8' or an equivalent

## === cell 8
class EffnetDataSet(torch.utils.data.Dataset):    
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        
    def __getitem__(self, i):
        path = os.path.join(self.path, str(self.df.iloc[i].patient_id), f'{self.df.iloc[i].image_id}.dcm')        
        
        try:
            img = load_dicom(path)[0]         
            img = np.transpose(img, (2, 0, 1))  # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            if self.transforms is not None:
                img = self.transforms(torch.as_tensor(img))
        except Exception as ex:
            print(ex)
            return None
        
        return img        
    
    def __len__(self):
        return len(self.df)
    
    
ds_test = EffnetDataSet(df_test, TEST_IMAGES_PATH, WEIGHTS.transforms())
X = ds_test[2]
X.shape

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4141453358.py in <cell line: 0>()
     27 ds_test = EffnetDataSet(df_test, TEST_IMAGES_PATH, WEIGHTS.transforms())
     28 X = ds_test[2]
---> 29 X.shape

AttributeError: 'NoneType' object has no attribute 'shape'

## === cell 10
class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_cancer = torch.nn.Sequential(
            torch.nn.Linear(1280, 1),
        )

    def forward(self, x):
        x = self.model(x)['flatten']
        return self.nn_cancer(x)

    def predict(self, x):
        canc = self.forward(x)
        return torch.sigmoid(canc)

model = EffnetModel()
model.predict(torch.randn(1, 3, 512, 512))
del model

## === cell 11
def load_model(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE)
    model.load_state_dict(data)
    return model

## === cell 12
effnet_models = [load_model(EffnetModel(), name, EFFNET_CHECKPOINTS_PATH).to(DEVICE) for name in MODEL_NAMES]

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2231486210.py in <cell line: 0>()
----> 1 effnet_models = [load_model(EffnetModel(), name, EFFNET_CHECKPOINTS_PATH).to(DEVICE) for name in MODEL_NAMES]

/tmp/ipykernel_11/2231486210.py in <listcomp>(.0)
----> 1 effnet_models = [load_model(EffnetModel(), name, EFFNET_CHECKPOINTS_PATH).to(DEVICE) for name in MODEL_NAMES]

/tmp/ipykernel_11/3758234228.py in load_model(model, name, path)
      1 def load_model(model, name, path='.'):
----> 2     data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE)
      3     model.load_state_dict(data)
      4     return model

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/breast-cancer-models/effnetv2-f0.tph'

## === cell 14
from typing import List


def predict_effnet(models: List[EffnetModel], ds, max_batches=1e9):
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    for m in models:
        m.eval()

    with torch.no_grad():
        predictions = []
        for idx, X in enumerate(tqdm(dl_test, miniters=10)):
            pred = torch.zeros(len(X)).to(DEVICE)
            for m in models:
                y1 = m.predict(X.to(DEVICE)).squeeze()
                pred += y1 / len(models)
            predictions.append(pred)
            if idx >= max_batches:
                break
        return torch.concat(predictions).cpu().numpy()

predict_effnet([EffnetModel().to(DEVICE)], ds_test, max_batches=2).shape

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1893259397.py in <cell line: 0>()
     20 
     21 # Quick test
---> 22 predict_effnet([EffnetModel().to(DEVICE)], ds_test, max_batches=2).shape

/tmp/ipykernel_11/1893259397.py in predict_effnet(models, ds, max_batches)
      9     with torch.no_grad():
     10         predictions = []
---> 11         for idx, X in enumerate(tqdm(dl_test, miniters=10)):
     12             pred = torch.zeros(len(X)).to(DEVICE)
     13             for m in models:

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 398, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 240, in collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>


## === cell 15
effnet_pred = predict_effnet(effnet_models, ds_test)

df_effnet_pred = pd.DataFrame(
    data=effnet_pred, columns=['cancer']
)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/81808427.py in <cell line: 0>()
----> 1 effnet_pred = predict_effnet(effnet_models, ds_test)
      2 
      3 df_effnet_pred = pd.DataFrame(
      4     data=effnet_pred, columns=['cancer']
      5 )

NameError: name 'effnet_models' is not defined

## === cell 16
df_effnet_pred

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2291379558.py in <cell line: 0>()
----> 1 df_effnet_pred

NameError: name 'df_effnet_pred' is not defined

## === cell 17
df_test_pred = pd.concat([df_test, df_effnet_pred], axis=1).sort_values(['patient_id', 'image_id'])
df_test_pred = df_test_pred.groupby('prediction_id')['cancer'].mean().reset_index()
df_test_pred

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3253675718.py in <cell line: 0>()
----> 1 df_test_pred = pd.concat([df_test, df_effnet_pred], axis=1).sort_values(['patient_id', 'image_id'])
      2 df_test_pred = df_test_pred.groupby('prediction_id')['cancer'].mean().reset_index()
      3 df_test_pred

NameError: name 'df_effnet_pred' is not defined

## === cell 18
df_test_pred.to_csv('submission.csv', index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2120184940.py in <cell line: 0>()
----> 1 df_test_pred.to_csv('submission.csv', index=False)

NameError: name 'df_test_pred' is not defined
