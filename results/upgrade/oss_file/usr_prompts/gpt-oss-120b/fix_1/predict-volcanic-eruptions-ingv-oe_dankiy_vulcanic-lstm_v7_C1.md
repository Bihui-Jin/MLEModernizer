# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Given readings from several seismic sensors around a volcano, estimate how long it will be until the next eruption.

## Metric
Mean absolute error (MAE) between the predicted loss and the actual loss.

## Submission Format
For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Data
### Dataset Description

#### Files
**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
        input/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
        working/
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
```

-> data/predict-volcanic-eruptions-ingv-oe/sample_submission.csv has 444 rows and 2 columns.
The columns are: segment_id, time_to_eruption

-> data/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1004346803.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1007996426.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1009749143.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1016956864.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1024522044.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1028325789.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> (stopped after 10 files for performance)

# 5. Target score

5793085.557252489

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

## === cell 2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

## === cell 3
def normalize(X_train, X_valid, X_test, normalize_opt, excluded_feat):
    feats = [f for f in X_train.columns if f not in excluded_feat]
    if normalize_opt is not None:
        if normalize_opt == 'min_max':
            scaler = preprocessing.MinMaxScaler()
        scaler = scaler.fit(X_train[feats])
        X_train[feats] = scaler.transform(X_train[feats])
        X_valid[feats] = scaler.transform(X_valid[feats])
        X_test[feats] = scaler.transform(X_test[feats])
    return X_train, X_valid, X_test

## === cell 4
PATH_DATASET = '/kaggle/input/vulcanic-preprocessing/'

train_sample = pd.read_csv(f'{PATH_DATASET}/train_sample.csv')
targets = pd.read_csv(f'{PATH_DATASET}/targets.csv')
test = pd.read_csv(f'{PATH_DATASET}/test.csv').iloc[:,1:]

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2820058884.py in <cell line: 0>()
      1 PATH_DATASET = '/kaggle/input/vulcanic-preprocessing/'
      2 
----> 3 train_sample = pd.read_csv(f'{PATH_DATASET}/train_sample.csv')
      4 targets = pd.read_csv(f'{PATH_DATASET}/targets.csv')
      5 test = pd.read_csv(f'{PATH_DATASET}/test.csv').iloc[:,1:]

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/vulcanic-preprocessing//train_sample.csv'

## === cell 5
train_x, valid_x, train_y, valid_y = train_test_split(train_sample, targets, test_size=0.2, random_state=0)

train_x, valid_x, test_scaled = normalize(train_x.copy(), valid_x.copy(), test.copy(), 'min_max', [])

train_x = train_x.values.reshape(train_x.shape[0], 1, train_x.shape[1])
valid_x = valid_x.values.reshape(valid_x.shape[0], 1, valid_x.shape[1])
train_y = train_y.to_numpy()
valid_y = valid_y.to_numpy()
test_scaled = test_scaled.values.reshape(test_scaled.shape[0], 1, test_scaled.shape[1])

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3633252476.py in <cell line: 0>()
----> 1 train_x, valid_x, train_y, valid_y = train_test_split(train_sample, targets, test_size=0.2, random_state=0)
      2 
      3 train_x, valid_x, test_scaled = normalize(train_x.copy(), valid_x.copy(), test.copy(), 'min_max', [])
      4 
      5 train_x = train_x.values.reshape(train_x.shape[0], 1, train_x.shape[1])

NameError: name 'train_sample' is not defined

## === cell 7
!pip install pywick

## === cell 8
import torch

from sklearn import preprocessing
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pywick.optimizers.nadam import Nadam
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error as mse

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/401430604.py in <cell line: 0>()
      4 from torch.nn import functional as F
      5 from torch import nn
----> 6 from pytorch_lightning.core.lightning import LightningModule
      7 from pywick.optimizers.nadam import Nadam
      8 from torch.utils.data import TensorDataset, DataLoader

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/__init__.py in <module>
     25 from lightning_fabric.utilities.seed import seed_everything  # noqa: E402
     26 from lightning_fabric.utilities.warnings import disable_possible_user_warnings  # noqa: E402
---> 27 from pytorch_lightning.callbacks import Callback  # noqa: E402
     28 from pytorch_lightning.core import LightningDataModule, LightningModule  # noqa: E402
     29 from pytorch_lightning.trainer import Trainer  # noqa: E402

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/callbacks/__init__.py in <module>
     12 # See the License for the specific language governing permissions and
     13 # limitations under the License.
---> 14 from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder
     15 from pytorch_lightning.callbacks.callback import Callback
     16 from pytorch_lightning.callbacks.checkpoint import Checkpoint

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/callbacks/batch_size_finder.py in <module>
     24 
     25 import pytorch_lightning as pl
---> 26 from pytorch_lightning.callbacks.callback import Callback
     27 from pytorch_lightning.tuner.batch_size_scaling import _scale_batch_size
     28 from pytorch_lightning.utilities.exceptions import MisconfigurationException, _TunerExitException

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/callbacks/callback.py in <module>
     20 
     21 import pytorch_lightning as pl
---> 22 from pytorch_lightning.utilities.types import STEP_OUTPUT
     23 
     24 

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/types.py in <module>
     34 from torch.optim import Optimizer
     35 from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
---> 36 from torchmetrics import Metric
     37 from typing_extensions import NotRequired, Required
     38 

/usr/local/lib/python3.11/dist-packages/torchmetrics/__init__.py in <module>
     35         scipy.signal.hamming = scipy.signal.windows.hamming
     36 
---> 37 from torchmetrics import functional  # noqa: E402
     38 from torchmetrics.aggregation import (  # noqa: E402
     39     CatMetric,

/usr/local/lib/python3.11/dist-packages/torchmetrics/functional/__init__.py in <module>
    127 from torchmetrics.functional.retrieval._deprecated import _retrieval_recall as retrieval_recall
    128 from torchmetrics.functional.retrieval._deprecated import _retrieval_reciprocal_rank as retrieval_reciprocal_rank
--> 129 from torchmetrics.functional.text._deprecated import _bleu_score as bleu_score
    130 from torchmetrics.functional.text._deprecated import _char_error_rate as char_error_rate
    131 from torchmetrics.functional.text._deprecated import _chrf_score as chrf_score

/usr/local/lib/python3.11/dist-packages/torchmetrics/functional/text/__init__.py in <module>
     48 
     49 if _TRANSFORMERS_GREATER_EQUAL_4_4:
---> 50     from torchmetrics.functional.text.bert import bert_score
     51     from torchmetrics.functional.text.infolm import infolm
     52 

/usr/local/lib/python3.11/dist-packages/torchmetrics/functional/text/bert.py in <module>
     54 
     55 if _TRANSFORMERS_GREATER_EQUAL_4_4:
---> 56     from transformers import AutoModel, AutoTokenizer
     57 
     58     def _download_model_for_bert_score() -> None:

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in __getattr__(self, name)
   2152         elif name in self._class_to_module.keys():
   2153             try:
-> 2154                 module = self._get_module(self._class_to_module[name])
   2155                 value = getattr(module, name)
   2156             except (ModuleNotFoundError, RuntimeError) as e:

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
-> 2184             raise e
   2185 
   2186     def __reduce__(self):

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2180     def _get_module(self, module_name: str):
   2181         try:
-> 2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
   2184             raise e

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/modeling_auto.py in <module>
     19 
     20 from ...utils import logging
---> 21 from .auto_factory import (
     22     _BaseAutoBackboneClass,
     23     _BaseAutoModelClass,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in <module>
     41 
     42 if is_torch_available():
---> 43     from ...generation import GenerationMixin
     44 
     45 

ImportError: cannot import name 'GenerationMixin' from 'transformers.generation' (/usr/local/lib/python3.11/dist-packages/transformers/generation/__init__.py)

## === cell 9
NUM_MODELS = 1
BATCH_SIZE = 8192
NUM_EPOCHS = 2000
PATH_MODEL = '/kaggle/working/models/'
PATH_DATA = '/kaggle/input/predict-volcanic-eruptions-ingv-oe/'

## === cell 10
class VolcanicLSTM(LightningModule):
    
    def __init__(self, num_features):
        super().__init__()
        
        self.bn = nn.BatchNorm1d(num_features=num_features)
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=128, num_layers=1)
        
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, padding=1, stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=84, kernel_size=2, padding=1, stride=2)
        self.conv3 = nn.Conv1d(in_channels=84, out_channels=64, kernel_size=2, padding=1, stride=2)
        
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(in_features=64, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=32)
        self.lin3 = nn.Linear(in_features=32, out_features=1)
        
        
    def forward(self, x):
        batch_size, _, _ = x.size()
        x = x.view(batch_size, -1)
        x = self.bn(x)
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm(x)
        
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = self.flat(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)

        return x
    
    def configure_optimizers(self):
        return Nadam(self.parameters(), lr=0.005)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.l1_loss(preds, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.l1_loss(preds, y)
        self.log('val_loss', loss)
        return loss
    
    def train_dataloader(self):
        tensor_x = torch.Tensor(train_x)
        tensor_y = torch.Tensor(train_y)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        
        return DataLoader(dataset, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        tensor_x = torch.Tensor(valid_x)
        tensor_y = torch.Tensor(valid_y)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        
        return DataLoader(dataset, batch_size=BATCH_SIZE)


    def test_dataloader(self):
        tensor_x = torch.Tensor(test_scaled)
        
        dataset = TensorDataset(tensor_x)
        
        return DataLoader(dataset, batch_size=BATCH_SIZE)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3942407255.py in <cell line: 0>()
----> 1 class VolcanicLSTM(LightningModule):
      2 
      3     def __init__(self, num_features):
      4         super().__init__()
      5 

NameError: name 'LightningModule' is not defined

## === cell 11
submission = pd.read_csv(f'{PATH_DATA}/sample_submission.csv')

sub_final = np.zeros(len(submission))
oof_final = np.zeros(len(valid_x))

i = 0

while i < NUM_MODELS:
    print('Running model ', i+1)

    if not os.path.exists(PATH_MODEL):
        os.makedirs(PATH_MODEL)
        
    model = VolcanicLSTM(num_features=train_x.shape[-1])
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=PATH_MODEL,
        filename=f'best_epoch-{i+1}'
    )
    trainer = Trainer(gpus=1, callbacks=[checkpoint_callback], min_epochs=1, max_epochs=NUM_EPOCHS, progress_bar_refresh_rate=0)
    
    trainer.fit(model)
    
    test_model = VolcanicLSTM.load_from_checkpoint(checkpoint_callback.best_model_path, num_features=train_x.shape[-1])
    
    oof_final += torch.squeeze(test_model(torch.Tensor(valid_x))).detach().numpy()
    sub_final += torch.squeeze(test_model(torch.unsqueeze(torch.Tensor(test.to_numpy()), 1))).detach().numpy()
    
    i += 1

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2671543739.py in <cell line: 0>()
      2 
      3 sub_final = np.zeros(len(submission))
----> 4 oof_final = np.zeros(len(valid_x))
      5 
      6 i = 0

NameError: name 'valid_x' is not defined

## === cell 12
oof_final /= NUM_MODELS
sub_final /= NUM_MODELS

print(f"\nMAE for NN: {mse(valid_y, oof_final, squared=False):.0f}")

submission['time_to_eruption'] = sub_final
submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4118914396.py in <cell line: 0>()
----> 1 oof_final /= NUM_MODELS
      2 sub_final /= NUM_MODELS
      3 
      4 print(f"\nMAE for NN: {mse(valid_y, oof_final, squared=False):.0f}")
      5 

NameError: name 'oof_final' is not defined
