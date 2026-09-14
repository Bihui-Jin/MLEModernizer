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
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

1.023523051943233

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import os,gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import glob
import torch
import warnings

warnings.filterwarnings('ignore')

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'
df = pd.read_csv(PATH + 'train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape )
print('Targets', list(TARGETS))
df.head()

## === cell 5
train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spectrogram_id','min']

tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape )

train.head(20)

## === cell 6
import numpy as np

ycol = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
cd = {'Seizure': 'seizure_vote', 'GPD': 'gpd_vote', 'LRDA': 'lrda_vote', 'Other': 'other_vote', 'GRDA': 'grda_vote', 'LPD': 'lpd_vote'}

eeg_id_col = train.iloc[:, 0]  # 第一列是 eeg_id 列
prob_cols = train.iloc[:, -7:-1]  # 倒数第七到倒数第二列是概率列
label_col = train.iloc[:, -1]  # 最后一列是标签列

prob_cols = prob_cols.astype("float32")

prob_cols_normalized = prob_cols.div(prob_cols.sum(axis=1), axis=0)

normalized_data = pd.concat([eeg_id_col, prob_cols_normalized, label_col], axis=1)

normalized_data.head(20)

## === cell 7
normalized_data.to_csv("normalized_data.csv", index=False)

## === cell 8
EEG_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/'
train_path = '/kaggle/input/processed/normalized_data.csv'

## === cell 9
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
from scipy.signal import butter, sosfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, csv_file, eeg_path):
        self.csv = pd.read_csv(csv_file)  # 加载 CSV 文件
        self.eeg_path = eeg_path
        self.sos = self.butter_bandpass_filter_init()
        self.FEATS = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']  # 指定需要使用的特征列

    def butter_bandpass_filter_init(self):
        lowcut = 0.5  # 设置带通滤波的低频截止频率
        highcut = 40.0  # 设置带通滤波的高频截止频率
        fs = 200.0  # 采样频率
        order = 5  # 滤波器阶数

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self, data):
        y = sosfilt(self.sos, data)
        return y
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        eeg_id = self.csv.loc[idx, 'eeg_id']
        eeg_file_path = f"{self.eeg_path}{eeg_id}.parquet"  # 构建 EEG 数据文件的路径
        eeg_data = pd.read_parquet(eeg_file_path)[self.FEATS].values  # 从 Parquet 文件中加载指定的特征列
        if np.isnan(eeg_data).any():
            imputer = SimpleImputer(strategy='mean')
            eeg_data = imputer.fit_transform(eeg_data)
        eeg_data = self.butter_bandpass_filter(eeg_data)  # 对数据进行滤波
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)  # 转换为 PyTorch 张量
                    
        mid_index = eeg_data.shape[0] // 2
        start_index = mid_index - 5000  # 从中间向左偏移5000个时间点
        end_index = mid_index + 5000  # 到中间向右偏移5000个时间点
        eeg_data = eeg_data[start_index:end_index]
        eeg_data = torch.transpose(eeg_data, 0, 1)
        
        labels = torch.tensor(self.csv.loc[idx, ycol].values.astype(np.float32), dtype=torch.float32)  # 加载对应的标签
        
        return eeg_data, labels

dataset = EEGDataset(train_path, EEG_PATH)


dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1039720845.py in <cell line: 0>()
     65 
     66 # 创建 EEG 数据集实例
---> 67 dataset = EEGDataset(train_path, EEG_PATH)
     68 
     69 

/tmp/ipykernel_11/1039720845.py in __init__(self, csv_file, eeg_path)
     14 class EEGDataset(Dataset):
     15     def __init__(self, csv_file, eeg_path):
---> 16         self.csv = pd.read_csv(csv_file)  # 加载 CSV 文件
     17         self.eeg_path = eeg_path
     18         self.sos = self.butter_bandpass_filter_init()

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/processed/normalized_data.csv'

## === cell 10
X, y = dataset[0]
print(f"Sample {0 + 1}: X shape {X.shape}, y shape {y.shape}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/608596393.py in <cell line: 0>()
----> 1 X, y = dataset[0]
      2 print(f"Sample {0 + 1}: X shape {X.shape}, y shape {y.shape}")

NameError: name 'dataset' is not defined

## === cell 12
import torch.nn.functional as F

class EEGNet(nn.Module):

    def __init__(self, in_channels=8, num_classes=6):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        batch_size, channels, seq_length = x.size()
        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)

        x, _ = self.lstm2(x)
        
        att_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(att_weights * x, dim=1)

        x = self.fc(x)

        return x

## === cell 13
input_channels = 8
num_classes = 6  # 类别数

model = EEGNet(in_channels=input_channels, num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1281598769.py in <cell line: 0>()
     19     model.train()
     20     running_loss = 0.0
---> 21     for i, (inputs, labels) in enumerate(dataloader):
     22         inputs, labels = inputs.to(device), labels.to(device)
     23         optimizer.zero_grad()

NameError: name 'dataloader' is not defined

## === cell 15
test_path = '/kaggle/input/hms-harmful-brain-activity-classification/test.csv'
TEST_EEG_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/'

class TestEEGDataset(Dataset):
    def __init__(self, csv_file, eeg_path):
        self.csv = pd.read_csv(csv_file)  # Load the CSV file
        self.eeg_path = eeg_path
        self.sos = self.butter_bandpass_filter_init()  # Initialize the Butterworth filter parameters
        self.FEATS = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
        
    def __len__(self):
        return len(self.csv)
    
    def butter_bandpass_filter_init(self):
        lowcut = 0.5  # Set the low-frequency cutoff for bandpass filtering
        highcut = 45.0  # Set the high-frequency cutoff for bandpass filtering
        fs = 200.0  # Sampling frequency
        order = 5  # Filter order

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')  # Create second-order sections for the Butterworth filter
        return sos

    def butter_bandpass_filter(self, data):
        y = sosfilt(self.sos, data)
        return y

    def __getitem__(self, idx):
        eeg_id = self.csv.loc[idx, 'eeg_id']
        eeg_file_path = f"{self.eeg_path}{eeg_id}.parquet"  # Build the EEG data file path
        eeg_data = pd.read_parquet(eeg_file_path)[self.FEATS].values  # Load EEG data from Parquet file

        eeg_data = self.butter_bandpass_filter(eeg_data)  # Apply filtering to the data
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)  # Convert to PyTorch tensor
         
        mid_index = eeg_data.shape[0] // 2
        start_index = mid_index - 5000  # Offset 5000 data points to the left from the middle
        end_index = mid_index + 5000  # Offset 5000 data points to the right from the middle
        eeg_data = eeg_data[start_index:end_index]
        eeg_data = torch.transpose(eeg_data, 0, 1)
        
        return eeg_data

testdataset = TestEEGDataset(test_path, TEST_EEG_PATH)

test_dataloader = DataLoader(testdataset, batch_size=32, shuffle=True, num_workers=4)

model.eval()

predictions = []
with torch.no_grad():
    for inputs in test_dataloader:  # 注意这里不需要 labels
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        predictions.append(probabilities.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)

print(predictions)

## === cell 16
columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
results_df = pd.DataFrame(predictions, columns=columns)

print(results_df)

## === cell 17
test_data = pd.read_csv(test_path)
sub = pd.DataFrame({'eeg_id':test_data.eeg_id.values})
sub[TARGETS] = results_df
sub.to_csv('submission.csv',index=False)
print('Submissionn shape',sub.shape)
sub.head()

## === cell 18
sub.iloc[:,-6:].sum(axis=1)

## --- ERROR in outputing the csv:
Invalid submission: Submission probabilities must add to 1 for each row
