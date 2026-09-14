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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.1403747355696585

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 5
import fastai
import shutil

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from fastai.vision.all import *
from fastai.vision.core import *

InteractiveShell.ast_node_interactivity = "all"

train = False

## === cell 8
model_loc = Path('~/.torch').expanduser()
if not model_loc.exists():
    model_loc.mkdir()
model_loc.ls()

## === cell 10
Path("/root/.cache/torch/hub/checkpoints/").mkdir(exist_ok=True, parents=True)
Path("../input/cassava-leaf-disease-classification/").exists() == Path("/root/.cache/torch/hub/checkpoints/").exists()
shutil.copy("../input/cassava-modelresnet34fine-tune10pkl/resnet34-333f7ec4.pth", "/root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth")
shutil.copy("../input/cassava-modelresnet34fine-tune10pkl/resnet50-19c8e357.pth", "/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1330540879.py in <cell line: 0>()
      1 Path("/root/.cache/torch/hub/checkpoints/").mkdir(exist_ok=True, parents=True)
      2 Path("../input/cassava-leaf-disease-classification/").exists() == Path("/root/.cache/torch/hub/checkpoints/").exists()
----> 3 shutil.copy("../input/cassava-modelresnet34fine-tune10pkl/resnet34-333f7ec4.pth", "/root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth")
      4 shutil.copy("../input/cassava-modelresnet34fine-tune10pkl/resnet50-19c8e357.pth", "/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth")

/usr/lib/python3.11/shutil.py in copy(src, dst, follow_symlinks)
    429     if os.path.isdir(dst):
    430         dst = os.path.join(dst, os.path.basename(src))
--> 431     copyfile(src, dst, follow_symlinks=follow_symlinks)
    432     copymode(src, dst, follow_symlinks=follow_symlinks)
    433     return dst

/usr/lib/python3.11/shutil.py in copyfile(src, dst, follow_symlinks)
    254         os.symlink(os.readlink(src), dst)
    255     else:
--> 256         with open(src, 'rb') as fsrc:
    257             try:
    258                 with open(dst, 'wb') as fdst:

FileNotFoundError: [Errno 2] No such file or directory: '../input/cassava-modelresnet34fine-tune10pkl/resnet34-333f7ec4.pth'

## === cell 12
fastai.__version__

## === cell 13
path = Path('../input/cassava-leaf-disease-classification')
os.listdir(path)

## === cell 14
labels = pd.read_csv(path / 'train.csv')
labels["image_id"] = labels["image_id"].apply(
    lambda x: f'train_images/{x}')
labels.head()

## === cell 17
dls = ImageDataLoaders.from_df(
    labels, 
    path='../input/cassava-leaf-disease-classification/',
    bs=64,
    fn_col=0, 
    label_col=1, 
    seed=42,
    valid_pct=0.2,
    item_tfms=RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.)),
    batch_tfms=aug_transforms())

dls.valid_ds.items[:3]

## === cell 20
learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy]).to_native_fp16()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1689400858.py in <cell line: 0>()
----> 1 learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy]).to_native_fp16()

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __getattr__(self, k)
    551         if self._component_attr_filter(k):
    552             attr = getattr(self,self._default,None)
--> 553             if attr is not None: return getattr(attr,k)
    554         raise AttributeError(k)
    555     def __dir__(self): return custom_dir(self,self._dir())

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'Sequential' object has no attribute 'to_native_fp16'

## === cell 23
if train:
    learn.fine_tune(10, cbs=[MixUp(0.5)])

## === cell 26
model_name = "resnet50-fine_tune-10"

if train:
    learn.export(Path(f"{Path(os.getcwd()).parent}/output/cassava-leaf-disease-classification/{model_name}.pkl"))
else:
    learn = load_learner(Path(f"{Path(os.getcwd()).parent}/input/cassava-modelresnet34fine-tune10pkl/{model_name}.pkl"), cpu=False)
    
learn = learn.to_native_fp32()
learn

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3918404726.py in <cell line: 0>()
      5 else:
      6     # The learner should be loaded on gpu to enable test time augumentation. Thanks to @muellerzr for the tip
----> 7     learn = load_learner(Path(f"{Path(os.getcwd()).parent}/input/cassava-modelresnet34fine-tune10pkl/{model_name}.pkl"), cpu=False)
      8 
      9 learn = learn.to_native_fp32()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in load_learner(fname, cpu, pickle_module)
    455         warn("load_learner` uses Python's insecure pickle module, which can execute malicious arbitrary code when loading. Only load files you trust.\nIf you only need to load model weights and optimizer state, use the safe `Learner.load` instead.")
    456         load_kwargs = {"weights_only": False} if ismin_torch("2.6") else {}
--> 457         res = torch.load(fname, map_location=map_loc, pickle_module=pickle_module, **load_kwargs)
    458     except ImportError as e:
    459         if any(o in str(e) for o in ("fastcore.transform","fastcore.dispatch")):

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/cassava-modelresnet34fine-tune10pkl/resnet50-fine_tune-10.pkl'

## === cell 28
if train:
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()

## === cell 30
if train:
    from fastai.vision.widgets import *
    cleaner = ImageClassifierCleaner(learn)
    cleaner

## === cell 32
test_labels = pd.read_csv(path / 'sample_submission.csv')
test_labels["image_id"] = test_labels["image_id"].apply(
    lambda x: f'test_images/{x}')

test_labels.head()

## === cell 34
test_dl = dls.test_dl(test_labels, with_labels=True)
test_dl.show_batch()
test_dl.items

## === cell 35

test_dl = dls.test_dl(test_labels, with_labels=True)
preds = learn.tta(dl=test_dl, n=12, beta=0) # I learnt this trick from fastai forums
test_predictions = preds[1]
test_predictions[0]

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1985298215.py in <cell line: 0>()
      1 test_dl = dls.test_dl(test_labels, with_labels=True)
      2 # preds = learn.get_preds(dl=test_dl)
----> 3 preds = learn.tta(dl=test_dl, n=12, beta=0) # I learnt this trick from fastai forums
      4 test_predictions = preds[1]
      5 test_predictions[0]

NameError: name 'learn' is not defined

## === cell 36
submission = pd.read_csv(path / 'sample_submission.csv')
submission['label'] = test_predictions
submission.head()
submission.to_csv('submission.csv',index=False)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/980432034.py in <cell line: 0>()
      1 submission = pd.read_csv(path / 'sample_submission.csv')
----> 2 submission['label'] = test_predictions
      3 submission.head()
      4 submission.to_csv('submission.csv',index=False)

NameError: name 'test_predictions' is not defined
