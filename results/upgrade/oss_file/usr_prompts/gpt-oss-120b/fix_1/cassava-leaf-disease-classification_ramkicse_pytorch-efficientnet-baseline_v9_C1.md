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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.8434572378362043

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
!pip install ../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar

## === cell 3
!ls -lrt ../input/ramki-cassava-weights/weight-at-epoch-61-acc-0.85794.pth

## === cell 4
!ls ../input/cassava-leaf-disease-classification

## === cell 6
!ls ../input/cassava-leaf-disease-classification

## === cell 7
!pip install albumentations

## === cell 8
TRAINING = False
WEIGHT_FILE = '../input/ramki-cassava-weights/weight-at-epoch-61-acc-0.85794.pth'

## === cell 9
import numpy as np 
import pandas as pd 
import os
from PIL import Image, ImageFilter
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import *

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import models
import time
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import classification_report


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3348972216.py in <cell line: 0>()
     11 from torch.optim import *
     12 
---> 13 from sklearn.metrics import roc_auc_score
     14 from sklearn.model_selection import train_test_split, StratifiedKFold
     15 from torchvision import models

/usr/local/lib/python3.11/dist-packages/sklearn/__init__.py in <module>
     80     from . import _distributor_init  # noqa: F401
     81     from . import __check_build  # noqa: F401
---> 82     from .base import clone
     83     from .utils._show_versions import show_versions
     84 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in <module>
     15 from . import __version__
     16 from ._config import get_config
---> 17 from .utils import _IS_32BIT
     18 from .utils._set_output import _SetOutputMixin
     19 from .utils._tags import (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py in <module>
     23 from .deprecation import deprecated
     24 from .discovery import all_estimators
---> 25 from .fixes import parse_version, threadpool_info
     26 from ._estimator_html_repr import estimator_html_repr
     27 from .validation import (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/fixes.py in <module>
     17 import numpy as np
     18 import scipy
---> 19 import scipy.stats
     20 import threadpoolctl
     21 

/usr/local/lib/python3.11/dist-packages/scipy/stats/__init__.py in <module>
    622 from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
    623                                DegenerateDataWarning, FitError)
--> 624 from ._stats_py import *
    625 from ._variation import variation
    626 from .distributions import *

/usr/local/lib/python3.11/dist-packages/scipy/stats/_stats_py.py in <module>
     37 
     38 from scipy import sparse
---> 39 from scipy.spatial import distance_matrix
     40 
     41 from scipy.optimize import milp, LinearConstraint

/usr/local/lib/python3.11/dist-packages/scipy/spatial/__init__.py in <module>
    114 from ._plotutils import *
    115 from ._procrustes import procrustes
--> 116 from ._geometric_slerp import geometric_slerp
    117 
    118 # Deprecated namespaces, to be removed in v2.0.0

/usr/local/lib/python3.11/dist-packages/scipy/spatial/_geometric_slerp.py in <module>
      5 
      6 import numpy as np
----> 7 from scipy.spatial.distance import euclidean
      8 
      9 if TYPE_CHECKING:

/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py in <module>
    119 from . import _hausdorff
    120 from ..linalg import norm
--> 121 from ..special import rel_entr
    122 
    123 from . import _distance_pybind

/usr/local/lib/python3.11/dist-packages/scipy/special/__init__.py in <module>
    824     chdtr, chdtrc, betainc, betaincc, stdtr)
    825 
--> 826 from . import _basic
    827 from ._basic import *
    828 

/usr/local/lib/python3.11/dist-packages/scipy/special/_basic.py in <module>
     20 from . import _specfun
     21 from ._comb import _comb_int
---> 22 from ._multiufuncs import (assoc_legendre_p_all,
     23                            legendre_p_all)
     24 from scipy._lib.deprecation import _deprecated

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in <module>
    140 
    141 
--> 142 sph_legendre_p = MultiUFunc(
    143     sph_legendre_p,
    144     r"""sph_legendre_p(n, m, theta, *, diff_n=0)

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in __init__(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)
     39             for ufunc in ufuncs_iter:
     40                 if not isinstance(ufunc, np.ufunc):
---> 41                     raise ValueError("All ufuncs must have type `numpy.ufunc`."
     42                                      f" Received {ufunc_or_ufuncs}")
     43                 seen_input_types.add(frozenset(x.split("->")[0] for x in ufunc.types))

ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)

## === cell 10
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
from albumentations import Rotate 
import albumentations as A


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/796341250.py in <cell line: 0>()
----> 1 from efficientnet_pytorch import EfficientNet
      2 from albumentations.pytorch import ToTensorV2
      3 from albumentations import Rotate
      4 import albumentations as A

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 11
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

## === cell 12
if torch.cuda.is_available():
    print(torch.cuda.device_count())

## === cell 13
writer = SummaryWriter('logs/8-final')

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1379888584.py in <cell line: 0>()
----> 1 writer = SummaryWriter('logs/8-final')

NameError: name 'SummaryWriter' is not defined

## === cell 14
SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001
NUM_CLASSES = 5


## === cell 15
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/502751144.py in <cell line: 0>()
     16 
     17 
---> 18 seed_everything(SEED)

/tmp/ipykernel_11/502751144.py in seed_everything(seed)
      6         seed {int} -- Number of the seed
      7     """
----> 8     random.seed(seed)
      9     os.environ["PYTHONHASHSEED"] = str(seed)
     10     np.random.seed(seed)

NameError: name 'random' is not defined

## === cell 16
base_path = '../input/cassava-leaf-disease-classification/'


## === cell 17
train_path =base_path + 'train_images/'
test_path = base_path + 'test_images/'

train_csv = pd.read_csv(base_path + 'train.csv')
sample = pd.read_csv(base_path + 'sample_submission.csv')

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/79555994.py in <cell line: 0>()
      2 test_path = base_path + 'test_images/'
      3 
----> 4 train_csv = pd.read_csv(base_path + 'train.csv')
      5 sample = pd.read_csv(base_path + 'sample_submission.csv')

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    624 
    625     with parser:
--> 626         return parser.read(nrows)
    627 
    628 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read(self, nrows)
   1966                 new_col_dict = col_dict
   1967 
-> 1968             df = DataFrame(
   1969                 new_col_dict,
   1970                 columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    441         from pandas.core.series import Series
    442 
--> 443         arrays = Series(data, index=columns, dtype=object)
    444         missing = arrays.isna()
    445         if index is None:

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __init__(self, data, index, dtype, name, copy, fastpath)
    488 
    489         if index is not None:
--> 490             index = ensure_index(index)
    491 
    492         if dtype is not None:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in ensure_index(index_like, copy)
   7645             return MultiIndex.from_arrays(index_like)
   7646         else:
-> 7647             return Index(index_like, copy=copy, tupleize_cols=False)
   7648     else:
   7649         return Index(index_like, copy=copy)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in __new__(cls, data, dtype, copy, name, tupleize_cols)
    563 
    564         try:
--> 565             arr = sanitize_array(data, None, dtype=dtype, copy=copy)
    566         except ValueError as err:
    567             if "index must be specified when data is not list-like" in str(err):

/usr/local/lib/python3.11/dist-packages/pandas/core/construction.py in sanitize_array(data, index, dtype, copy, allow_2d)
    652 
    653         else:
--> 654             subarr = maybe_convert_platform(data)
    655             if subarr.dtype == object:
    656                 subarr = cast(np.ndarray, subarr)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/cast.py in maybe_convert_platform(values)
    136     if arr.dtype == _dtype_obj:
    137         arr = cast(np.ndarray, arr)
--> 138         arr = lib.maybe_convert_objects(arr)
    139 
    140     return arr

lib.pyx in pandas._libs.lib.maybe_convert_objects()

TypeError: Cannot convert numpy.ndarray to numpy.ndarray

## === cell 18
train_csv.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1606115182.py in <cell line: 0>()
----> 1 train_csv.head()

NameError: name 'train_csv' is not defined

## === cell 19

train_csv_disease = train_csv.label.map({0:"Cassava Bacterial Blight (CBB)",
1:"Cassava Brown Streak Disease (CBSD)",
2:"Cassava Green Mottle (CGM)",
3:"Cassava Mosaic Disease (CMD)",
4:"Healthy"})
diseases = train_csv_disease.value_counts()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/419007262.py in <cell line: 0>()
----> 1 train_csv_disease = train_csv.label.map({0:"Cassava Bacterial Blight (CBB)",
      2 1:"Cassava Brown Streak Disease (CBSD)",
      3 2:"Cassava Green Mottle (CGM)",
      4 3:"Cassava Mosaic Disease (CMD)",
      5 4:"Healthy"})

NameError: name 'train_csv' is not defined

## === cell 20
diseases

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2146108635.py in <cell line: 0>()
----> 1 diseases

NameError: name 'diseases' is not defined

## === cell 21
diseases.plot.pie()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1823557202.py in <cell line: 0>()
----> 1 diseases.plot.pie()

NameError: name 'diseases' is not defined

## === cell 23
class CasavaDataset(Dataset):
    
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe
        self.transforms = transforms
        self.test = test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.iloc[idx].label
        p = self.df.iloc[idx].image_id
        
        if self.test == False:
            p_path = train_path + p
        else:
            p_path = test_path + p
            
        image =  Image.open(p_path).convert('RGB')
        image = np.array(image)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        return image, label

## === cell 24
    



transforms_train = A.Compose([
        A.RandomResizedCrop(height=300, width=300, p=1.0),
        A.Rotate(20),
        A.Flip(),
        A.Transpose(),
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
        ], p=1.0)

transforms_valid = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/532394141.py in <cell line: 0>()
     13 
     14 
---> 15 transforms_train = A.Compose([
     16         A.RandomResizedCrop(height=300, width=300, p=1.0),
     17         A.Rotate(20),

NameError: name 'A' is not defined

## === cell 25
train_csv.shape

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2410063919.py in <cell line: 0>()
----> 1 train_csv.shape

NameError: name 'train_csv' is not defined

## === cell 26
model_name = 'efficientnet-b7'

if TRAINING:
    model = EfficientNet.from_pretrained(model_name, num_classes=5) 
    print('model created with imagenet weights')
else:
    model = EfficientNet.from_name(model_name, num_classes=5) 
    weights_file = '../input/ramki-cassava-weights/weight.pt'
    model.load_state_dict(torch.load(weights_file)) 
    model.to(device)
    print('model loaded from weight file')

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4071103618.py in <cell line: 0>()
      5     print('model created with imagenet weights')
      6 else:
----> 7     model = EfficientNet.from_name(model_name, num_classes=5)
      8     weights_file = '../input/ramki-cassava-weights/weight.pt'
      9     model.load_state_dict(torch.load(weights_file))

NameError: name 'EfficientNet' is not defined

## === cell 28
layer =0
for child in model.children():
    layer+=1
print(layer)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3824757308.py in <cell line: 0>()
      1 layer =0
----> 2 for child in model.children():
      3     layer+=1
      4 print(layer)

NameError: name 'model' is not defined

## === cell 29
model = model.to(device)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1795457377.py in <cell line: 0>()
      1 # model = nn.DataParallel(model.cuda())
----> 2 model = model.to(device)

NameError: name 'model' is not defined

## === cell 30
trainset      = CasavaDataset(train_csv, transforms=transforms_train, test=False)
train_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset      = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/564013970.py in <cell line: 0>()
----> 1 trainset      = CasavaDataset(train_csv, transforms=transforms_train, test=False)
      2 train_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
      3 
      4 testset      = CasavaDataset(sample, transforms=transforms_valid, test=True)
      5 test_loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

NameError: name 'train_csv' is not defined

## === cell 31
len(train_loader)

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4006441096.py in <cell line: 0>()
----> 1 len(train_loader)

NameError: name 'train_loader' is not defined

## === cell 32
BATCH_SIZE

## === cell 33
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

## === cell 34
def show_metrics(model, dataloader, criterion ):
    model.eval() 
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    complete_outputs = []
    complete_labels = []
    tk = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        
        imgs_train, labels_train = imgs.to(device), labels.to(device).long()
        output_train = model(imgs_train)

        loss = criterion(output_train, labels_train)
        
        
        predicted_classes = output_train.argmax(1)
        complete_outputs.extend(predicted_classes.tolist())
        complete_labels.extend(labels_train.tolist())
        
        correctly_identified_sum = (predicted_classes==labels_train).sum().item()
        number_of_images = imgs_train.size(0)
                                    
        accs.update(correctly_identified_sum / number_of_images, number_of_images)
        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg,acc=accs.avg)
                    
    cf_matrix = confusion_matrix(complete_outputs, complete_labels)
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="YlGnBu")

    
    target_names = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)", "Cassava Green Mottle (CGM)","Cassava Mosaic Disease (CMD)","Healthy"]
    print(classification_report(complete_outputs, complete_labels, target_names=target_names))

   
    return losses.avg, complete_outputs, complete_labels

## === cell 35
def train_model(model, epoch, dataloader_train, criterion, optimizer ):
    model.train() 
    
    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        
        imgs_train, labels_train = imgs.to(device), labels.to(device).long()
        output_train = model(imgs_train)

        loss = criterion(output_train, labels_train)
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
        predicted_classes = output_train.argmax(1)
        correctly_identified_sum = (predicted_classes==labels_train).sum().item()
        number_of_images = imgs_train.size(0)
                                    
        accs.update(correctly_identified_sum/number_of_images, number_of_images)
        
        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg,acc=accs.avg)
        
    return losses.avg, accs.avg


def test_model(model, dataloader_valid, criterion):    
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid, labels_valid = imgs.to(device), labels.to(device).long()
            output_valid = model(imgs_valid)
            
            loss = criterion(output_valid, labels_valid)

            losses.update(loss.item(), imgs_valid.size(0))
            accs.update((output_valid.argmax(1)==labels_valid).sum().item()/imgs_valid.size(0),imgs_valid.size(0))
            
            tk.set_postfix(loss=losses.avg,acc=accs.avg)
    

            
    return losses.avg,accs.avg , loss

## === cell 37
train_csv

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3533528771.py in <cell line: 0>()
----> 1 train_csv

NameError: name 'train_csv' is not defined

## === cell 38
X_train, X_test = train_test_split(train_csv, test_size=0.25, random_state=SEED, stratify= train_csv['label'])

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/324854062.py in <cell line: 0>()
----> 1 X_train, X_test = train_test_split(train_csv, test_size=0.25, random_state=SEED, stratify= train_csv['label'])

NameError: name 'train_test_split' is not defined

## === cell 39
train_csv['label'].value_counts()

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3746598334.py in <cell line: 0>()
----> 1 train_csv['label'].value_counts()

NameError: name 'train_csv' is not defined

## === cell 40
X_train['label'].value_counts()

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1427094722.py in <cell line: 0>()
----> 1 X_train['label'].value_counts()

NameError: name 'X_train' is not defined

## === cell 41
X_test['label'].value_counts()

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3524793613.py in <cell line: 0>()
----> 1 X_test['label'].value_counts()

NameError: name 'X_test' is not defined

## === cell 42
dataset_train = CasavaDataset( X_train, transforms=transforms_train)
dataset_valid = CasavaDataset( X_test, transforms=transforms_valid)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True) #sampler=sampler
dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2668408055.py in <cell line: 0>()
----> 1 dataset_train = CasavaDataset( X_train, transforms=transforms_train)
      2 dataset_valid = CasavaDataset( X_test, transforms=transforms_valid)
      3 
      4 dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True) #sampler=sampler
      5 dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

NameError: name 'X_train' is not defined

## === cell 43
if TRAINING:











    


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, \
                                                           patience=4, verbose=True, min_lr=1e-5)

    best_acc = 0

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_model(model, epoch, dataloader_train, criterion, optimizer)
        val_loss, val_acc, loss = test_model(model, dataloader_valid, criterion)

        writer.add_scalar('training acc', train_acc, epoch+1)
        writer.add_scalar('training loss', train_loss, epoch+1)

        writer.add_scalar('validation loss', val_loss, epoch+1)
        writer.add_scalar('validation Acc', val_acc, epoch+1)

        writer.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'weights/weight-at-epoch-{}-acc-{:.5}.pth'.format(epoch, best_acc))

        print('current_val_acc:', val_acc, 'best_val_acc:', best_acc)

    show_metrics(model, dataloader_train, criterion )


## === cell 44
test_pred = []

model.eval()

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, position=0, leave=True)):
        images, _ = data
        images = images.to(device)

        pred = model(images)

        pred = pred.argmax(1).cpu().detach().numpy().astype('int')

        test_pred.extend(pred)

sample.label = test_pred
sample.to_csv('submission.csv',index=False)

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1300551936.py in <cell line: 0>()
      1 test_pred = []
      2 
----> 3 model.eval()
      4 
      5 with torch.no_grad():

NameError: name 'model' is not defined

## === cell 45
sample

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1007176509.py in <cell line: 0>()
----> 1 sample

NameError: name 'sample' is not defined

## === cell 46
!cat submission.csv
