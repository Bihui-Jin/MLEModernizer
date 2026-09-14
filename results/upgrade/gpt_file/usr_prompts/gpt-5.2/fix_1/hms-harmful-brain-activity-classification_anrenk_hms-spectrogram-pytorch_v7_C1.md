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

geopandas==0.14.4
numpy==1.26.4
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

1.1274117538279391

# 6. Current score

1.40938

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import numpy as np
import pandas as pd

import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

## === cell 3
INPUT_IMAGE_SIZE = (400, 400)
TRAIN_SIZE = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {DEVICE}')

## === cell 5
train_eeg_files = glob.glob('/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/*')
test_eeg_files = glob.glob('/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/*')
train_spectrogram_files = glob.glob('/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/*')
test_spectrogram_files = glob.glob('/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/*')
print(f'Total number of files in train_eegs: {len(train_eeg_files)}')
print(f'Total number of files in test_eegs: {len(test_eeg_files)}')
print(f'Total number of files in train_spectrograms: {len(train_spectrogram_files)}')
print(f'Total number of files in test_spectrograms: {len(test_spectrogram_files)}')

## === cell 6
train_csv = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
test_csv = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
sample_csv = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv')
print(f'number of entries in train_csv {len(train_csv)}')
print(f'number of entries in test_csv {len(test_csv)}')
print(f'number of entries in sample_csv {len(sample_csv)}')

## === cell 8
columns_to_drop = ['eeg_id', 
                   'eeg_sub_id', 
                   'eeg_label_offset_seconds', 
                   'spectrogram_sub_id', 
                   'spectrogram_label_offset_seconds',
                   'label_id',
                   'patient_id',
                   'expert_consensus']

## === cell 9
spectro_csv = train_csv[train_csv['spectrogram_sub_id']==0].drop(columns_to_drop, axis=1).reset_index(drop=True)

## === cell 10
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, 0, 'constant')

## === cell 11
class HMS_dataset(Dataset):
    def __init__(self,csv_file, root_dir, image_size = (400,400), targets_available=True):
        self.hms_dataset = csv_file
        self.root_dir = root_dir
        self.image_size = image_size
        self.targets_available = targets_available
        self.transform = transforms.Compose([
                            SquarePad(),
                            transforms.Resize(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5),
                            ])
        
    def __len__(self):
        return len(self.hms_dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name = os.path.join(self.root_dir, str(self.hms_dataset.iloc[idx, 0]) + '.parquet')
        spectro = np.log(pd.read_parquet(file_name).drop('time', axis=1).fillna(0).replace(0, 1e-6).to_numpy().T)
        spectro_pil = Image.fromarray(spectro)
        spectro_torch = self.transform(spectro_pil)
        if self.targets_available:
            votes_label = self.hms_dataset.iloc[idx, 1:].to_numpy()
            torch_labels = torch.tensor(votes_label, dtype=torch.float) # labels to tensor transform
            sample = (spectro_torch, torch_labels)
        else:
            sample = spectro_torch
            
        return sample

## === cell 13
root_train = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
hms_data = HMS_dataset(spectro_csv,root_train, INPUT_IMAGE_SIZE)
print(len(hms_data))

## === cell 14
train_size = int(TRAIN_SIZE * len(hms_data))
test_size = len(hms_data) - train_size
train_set, test_set = torch.utils.data.random_split(hms_data, [train_size, test_size])
print(len(train_set))
print(len(test_set))

## === cell 15
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

## === cell 17
class Conv_net(nn.Module):
    def __init__(self, n_classes):
        super(Conv_net, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        
        self.fc_net = nn.Sequential(
            nn.Linear(128*10*10, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        return x

## === cell 18
model = Conv_net(6).to(DEVICE)

## === cell 21
'''
def train(model, optimizer, loss_fn, data_loader):
    total_loss = 0
    data_len = len(data_loader)
    
    for batch, (x,y) in enumerate(data_loader, 1):
        preds = F.log_softmax(model(x.to(DEVICE)), dim=1)
        y = F.softmax(y, dim=1).to(DEVICE)
        loss = loss_fn(preds, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            print(f'training: {batch}/{data_len}, loss={(total_loss/10):.3f}')
            total_loss = 0

def val(model, loss_fn, data_loader):
    total_loss = 0
    data_len = len(data_loader)
    
    for x,y in data_loader:
        with torch.no_grad():
            preds = F.log_softmax(model(x.to(DEVICE)), dim=1)
            y = F.softmax(y, dim=1).to(DEVICE)
            loss = loss_fn(preds, y)
            total_loss += loss.item()
    
    return total_loss/data_len
        
'''        

## === cell 22
'''
for e in tqdm(range(EPOCHS), ncols=100, desc='training...'):
    train(model,optimizer,loss_fn, train_loader)
    test_results = val(model, loss_fn, test_loader)
    print(test_results)
'''

## === cell 24
os.listdir('/kaggle/input/hms_spectrogram_classifier/pytorch/model1/1')

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/77919841.py in <cell line: 0>()
----> 1 os.listdir('/kaggle/input/hms_spectrogram_classifier/pytorch/model1/1')

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms_spectrogram_classifier/pytorch/model1/1'

## === cell 25
model.load_state_dict(torch.load('/kaggle/input/hms_spectrogram_classifier/pytorch/model1/1/trained_model.pt', map_location=torch.device(DEVICE)))

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1913630196.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load('/kaggle/input/hms_spectrogram_classifier/pytorch/model1/1/trained_model.pt', map_location=torch.device(DEVICE)))

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms_spectrogram_classifier/pytorch/model1/1/trained_model.pt'

## === cell 26
model.eval()

## === cell 28
spectro_test_csv = test_csv.drop(['eeg_id', 'patient_id'], axis=1)
root_test = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
test_hms_data = HMS_dataset(spectro_test_csv, root_test, targets_available=False)

## === cell 29
with torch.no_grad():
    test_preds = F.softmax(model(test_hms_data[0].unsqueeze_(0).to(DEVICE)), dim=1).detach().cpu().numpy()

## === cell 30
test_preds

## === cell 31
test_preds.sum()

## === cell 33
i=0
for col in sample_csv.columns[1:]:
    sample_csv[col] = test_preds[0][i]
    i+=1

## === cell 34
sample_csv.to_csv('submission.csv', index = False)
