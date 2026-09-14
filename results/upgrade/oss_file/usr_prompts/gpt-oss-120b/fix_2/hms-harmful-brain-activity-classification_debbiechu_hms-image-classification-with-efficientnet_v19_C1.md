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

1.0598384737455746

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will fix the dataset drop to avoid a KeyError, add a tiny probability smoothing after softmax to help lower the KL‑divergence, and reorder the submission columns so that `eeg_id` comes first, matching the required format. These minimal changes ensure the script runs end‑to‑end and creates a valid `submission.csv` that is more likely to achieve a score closer to the target.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/2701598227.py", line 1
    I will fix the dataset drop to avoid a KeyError, add a tiny probability smoothing after softmax to help lower the KL‑divergence, and reorder the submission columns so that `eeg_id` comes first, matching the required format. These minimal changes ensure the script runs end‑to‑end and creates a valid `submission.csv` that is more likely to achieve a score closer to the target.
                                                                                                                        ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
import zipfile
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
%matplotlib inline



## === cell 2
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

set_seed(42)



## === cell 3
base_path = "/kaggle/input/hms-harmful-brain-activity-classification"
train_csv_path = os.path.join(base_path, 'train.csv')
test_csv_path = os.path.join(base_path, 'test.csv')

df = pd.read_csv(train_csv_path)

temp = df[['eeg_id', 'expert_consensus']].drop_duplicates()
df = df.loc[temp.index].reset_index(drop=True)

df['eeg_path'] = base_path + '/train_eegs/' + df['eeg_id'].astype(str) + '.parquet'
df['spec_path'] = base_path + '/train_spectrograms/' + df['spectrogram_id'].astype(str) + '.parquet'
df['class_name'] = df['expert_consensus'].copy()

class_name_to_label = {
    'Seizure': 0,
    'GPD': 2,
    'LRDA': 3,
    'Other': 5,
    'GRDA': 4,
    'LPD': 1
}

df['class_label'] = df['class_name'].map(class_name_to_label)

test = pd.read_csv(test_csv_path)
test['eeg_path'] = base_path + '/test_eegs/' + test['eeg_id'].astype(str) + '.parquet'
test['spec_path'] = base_path + '/test_spectrograms/' + test['spectrogram_id'].astype(str) + '.parquet'
if 'spectrogram_label_offset_seconds' not in test.columns:
    test['spectrogram_label_offset_seconds'] = 0
if 'class_label' not in test.columns:
    test['class_label'] = 0

eeg_ids = []
for class_label in df['class_label'].unique():
    eeg_id = df[df['class_label'] == class_label].iloc[0]['eeg_id']
    eeg_ids.append(eeg_id)

test_df = df[df['eeg_id'].isin(eeg_ids)] # for test
df = df[~df['eeg_id'].isin(eeg_ids)] # train & val
df = df.iloc[:1000,]



## === cell 4
df.head()



## === cell 5
from torchvision import models, transforms
import torch.nn as nn

model = models.efficientnet_v2_l(weights=None)
num_classes = 6
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)

weights_path = '/kaggle/input/efficientnet-v2-l/efficientnet_v2_l-59c71312.pth'
state_dict = torch.load(weights_path, map_location='cpu')
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.1')}
model.load_state_dict(state_dict, strict=False)
model = model.to(device)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2966774209.py in <cell line: 0>()
      8 
      9 weights_path = '/kaggle/input/efficientnet-v2-l/efficientnet_v2_l-59c71312.pth'
---> 10 state_dict = torch.load(weights_path, map_location='cpu')
     11 state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.1')}
     12 model.load_state_dict(state_dict, strict=False)

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/efficientnet-v2-l/efficientnet_v2_l-59c71312.pth'

## === cell 6
def read_parquet_subset(parquet_file_path, offset_seconds, length, is_eeg=True):
    offset_seconds = int(offset_seconds)  # Ensure offset_seconds integer, currently it's float
    start_row = int(offset_seconds * 200) if is_eeg else int(offset_seconds / 2) # The starting row
    end_row = start_row + (10000 if is_eeg else 300)  # The ending row
    df = pd.read_parquet(parquet_file_path)
    df_subset = df.iloc[start_row:end_row]  # Select the subset based on start and end row indices
    return df_subset

def convert_2d_to_3d(data_2d):
    data_2d = data_2d.to_numpy()
    data_2d_clipped = np.clip(data_2d, 0, 255) # ensure value falls within 0 and 255
    data_2d_clipped = np.nan_to_num(data_2d_clipped) # fillna with 0 (for numpy)
    data_3d = np.repeat(data_2d_clipped[:, :, np.newaxis], 3, axis=2)
    data_3d_uint8 = data_3d.astype(np.uint8)
    return data_3d_uint8

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SPECTROGRAM_Dataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        spec_data = read_parquet_subset(row['spec_path'], row['spectrogram_label_offset_seconds'], 300, is_eeg=False)
        spec_data = spec_data.drop(columns=['time'], errors='ignore')
        spec_data_3d = convert_2d_to_3d(spec_data)
        data_img = Image.fromarray(spec_data_3d)
        feature_vector = self.transform(data_img)
        label = row['class_label']
        return feature_vector, label
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



## === cell 7
lr=0.001
num_epochs = 100
batch_size=32
factor=0.8
num_classes = 6
num_workers=2



## === cell 8
from sklearn.model_selection import StratifiedGroupKFold

labels = df['class_label'].values
groups = df['eeg_id'].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in sgkf.split(X=df, y=labels, groups=groups):
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    
train_dataset = SPECTROGRAM_Dataset(train_df, transform)
val_dataset = SPECTROGRAM_Dataset(val_df, transform)
test_dataset = SPECTROGRAM_Dataset(test_df, transform)
test2_dataset = SPECTROGRAM_Dataset(test, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print(f"Size of train: {len(train_dataset)}")
print(f"Size of val: {len(val_dataset)}")
print(f"Size of test: {len(test_dataset)}")
print(f"Size of test2: {len(test2_dataset)}")
print(f"Number of train batches: {len(train_loader)}")
print(f"Number of val batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")
print(f"Number of test2 batches: {len(test2_loader)}")



## === cell 9
import torch.nn.functional as F
from torch.optim import Adam
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

def to_one_hot(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels]

criterion = torch.nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=1)



## === cell 10
best_val_loss = float('inf')
training_losses = []
validation_losses = []
epochs_no_improve = 0
n_patience = 1
epochs_no_improve = 0
accumulation_steps = 4
scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    batch_count = 0
    for i, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
            
        with autocast():
            outputs = model(features)
            log_probs = F.log_softmax(outputs, dim=1)# Convert to log probabilities
            one_hot_labels = to_one_hot(labels, num_classes=num_classes) # Convert labels to one-hot for KLDivLoss
            loss = criterion(log_probs, one_hot_labels) / accumulation_steps # Normalize loss to account for accumulation
        
        scaler.scale(loss).backward()
            
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                
        train_loss += loss.item() * features.size(0)      
        batch_count += 1
        if batch_count % 10 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}')  
            
    train_loss /= len(train_loader.dataset)
    training_losses.append(train_loss)
    
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad(), autocast():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            log_probs = F.log_softmax(outputs, dim=1)
            one_hot_labels = to_one_hot(labels, num_classes=num_classes)
            loss = criterion(log_probs, one_hot_labels)
            val_loss += loss.item() * features.size(0)
    
    val_loss /= len(val_loader.dataset)
    validation_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0 # reset to 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        print(f"Best model saved with Validation Loss: {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve == n_patience:
        print("Early stopping triggered.")
        break



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3256779227.py in <cell line: 0>()
     17 
     18         with autocast():
---> 19             outputs = model(features)
     20             log_probs = F.log_softmax(outputs, dim=1)# Convert to log probabilities
     21             one_hot_labels = to_one_hot(labels, num_classes=num_classes) # Convert labels to one-hot for KLDivLoss

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/efficientnet.py in forward(self, x)
    341 
    342     def forward(self, x: Tensor) -> Tensor:
--> 343         return self._forward_impl(x)
    344 
    345 

/usr/local/lib/python3.11/dist-packages/torchvision/models/efficientnet.py in _forward_impl(self, x)
    331 
    332     def _forward_impl(self, x: Tensor) -> Tensor:
--> 333         x = self.features(x)
    334 
    335         x = self.avgpool(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.FloatTensor) should be the same

## === cell 11

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
all_targets = []
all_probs = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device).float()
        labels = labels.to(device)
        outputs = model(features)
        
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

from sklearn.metrics import classification_report
classification_report(all_targets, all_preds, target_names=["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"])



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3994095167.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load('best_model.pth'))
      2 model.eval()
      3 
      4 all_preds = []
      5 all_targets = []

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

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pth'

## === cell 12
print(all_preds)
print(all_targets)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1231107439.py in <cell line: 0>()
----> 1 print(all_preds)
      2 print(all_targets)
      3 

NameError: name 'all_preds' is not defined

## === cell 13

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
all_probs = []

eps = 1e-2

with torch.no_grad():
    for features, labels in test2_loader:
        features = features.to(device).float()
        labels = labels.to(device)
        outputs = model(features)
        
        probs = F.softmax(outputs, dim=1)
        probs = (probs + eps) / (1.0 + eps * probs.shape[1])
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        _, preds = torch.max(probs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

print(all_preds)
print(all_probs)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3003520651.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load('best_model.pth'))
      2 model.eval()
      3 
      4 all_preds = []
      5 all_probs = []

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

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pth'

## === cell 14
sample_submission_csv_path = os.path.join(base_path, 'sample_submission.csv')
sub = pd.read_csv(sample_submission_csv_path)
sub



## === cell 15
submission = pd.DataFrame(all_probs, columns=['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'])
submission



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/983677687.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(all_probs, columns=['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'])
      2 submission
      3 

NameError: name 'all_probs' is not defined

## === cell 16
submission.insert(0, 'eeg_id', test['eeg_id'].values)
submission



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3201914767.py in <cell line: 0>()
      1 # Insert eeg_id as the first column to match the required format
----> 2 submission.insert(0, 'eeg_id', test['eeg_id'].values)
      3 submission
      4 

NameError: name 'submission' is not defined

## === cell 17
row_sums = submission.iloc[:, 1:].sum(axis=1)
print(f"Mean row sum (should be ~1.0): {row_sums.mean():.6f}")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/957983205.py in <cell line: 0>()
      1 # Verify that each row sums to 1 (allow a tiny numerical tolerance)
----> 2 row_sums = submission.iloc[:, 1:].sum(axis=1)
      3 print(f"Mean row sum (should be ~1.0): {row_sums.mean():.6f}")
      4 

NameError: name 'submission' is not defined

## === cell 18
submission.to_csv('submission.csv', index=False)
```

## --- ERROR in cell 18, traceback:
  File "/tmp/ipykernel_55/1146646358.py", line 2
    ```
    ^
SyntaxError: invalid syntax
