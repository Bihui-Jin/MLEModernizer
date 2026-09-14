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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sentence-transformers==4.1.0
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
transformers==4.53.3

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

1.132911383265961

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

data_path = '/kaggle/input/hms-harmful-brain-activity-classification'
sg_path = f"{data_path}/train_spectrograms"
sg_test_path = f"{data_path}/test_spectrograms"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

## === cell 2
train_df_M = pd.read_csv(f'{data_path}/train.csv')
print(train_df_M.shape)
train_df_M.head()

## === cell 3
print('Train Data:\n', train_df_M['expert_consensus'].value_counts().rename('Count').to_frame().assign(Percentage=lambda x: round((x / x.sum()) * 100)))
plot = train_df_M['expert_consensus'].value_counts().plot(kind='bar', title='Distribution of expert_consensus')

## === cell 5
train_df = train_df_M.sample(300, random_state=52).reset_index(drop=True)
print('Train Data:\n', train_df['expert_consensus'].value_counts().rename('Count').to_frame().assign(Percentage=lambda x: round((x / x.sum()) * 100)))

## === cell 7
train_checked_count = 0
null_count = 0
for index, row in train_df.iterrows():
    parquet_path = f"{sg_path}/{row['spectrogram_id']}.parquet"
    parquet_df = pd.read_parquet(parquet_path)
    filtered_df = parquet_df[(parquet_df['time'] >= row['spectrogram_label_offset_seconds']) & 
                             (parquet_df['time'] < row['spectrogram_label_offset_seconds'] + 600)]
    if filtered_df.isnull().any().any():
        train_df.at[index, 'SG_NullData_Ind'] = True
    else:
        train_df.at[index, 'SG_NullData_Ind'] = False

print(train_df['SG_NullData_Ind'].value_counts())
print('\ncheck expert consensus where sg is not null...')
print(train_df[train_df['SG_NullData_Ind'] == False]['expert_consensus'].value_counts().rename('Count').to_frame().assign(Percentage=lambda x: round((x / x.sum()) * 100)))

## === cell 9
train_df = train_df[train_df['SG_NullData_Ind'] != True]
train_df.shape

## === cell 10


train_df = train_df.sort_values(by='spectrogram_id')
sg_records_referred_list = []
spectrogram_id = 0
for index, row in train_df.iterrows():
    if row['spectrogram_id'] != spectrogram_id:
        parquet_path = f"{sg_path}/{row['spectrogram_id']}.parquet"
        parquet_df = pd.read_parquet(parquet_path)
        spectrogram_id = row['spectrogram_id']
    filtered_df = parquet_df[(parquet_df['time'] >= row['spectrogram_label_offset_seconds']) & 
                             (parquet_df['time'] < row['spectrogram_label_offset_seconds'] + 600)].copy()
    filtered_df['spectrogram_id'] = row['spectrogram_id']
    sg_records_referred_list.append(filtered_df)

sg_records_df = pd.concat(sg_records_referred_list, ignore_index=True)
sg_records_df = sg_records_df.drop_duplicates()

print(sg_records_df.shape)
sg_records_df.head()

## === cell 11
corr = sg_records_df.drop(columns=['time', 'spectrogram_id']).corr()
corr.style.background_gradient(cmap='coolwarm')

## === cell 13
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

linked = linkage(corr, 'ward')
plt.figure(figsize=(50, 5))
dendrogram(linked, labels=corr.index.tolist(), leaf_rotation=90)

thresholds = [1, 2, 3, 4]  
for thr in thresholds:
    plt.axhline(y=thr, color='r', linestyle='--')

plt.title('Hierarchical Clustering Dendrogram with Multiple Thresholds')
plt.xlabel('Feature')
plt.ylabel('Distance')
plt.show()

for thr in thresholds:
    clusters = fcluster(linked, thr, criterion='distance')
    num_unique_clusters = len(set(clusters))
    print(f'Number of unique features at threshold {thr}: {num_unique_clusters}')

## === cell 15
thr = 0.4286 #this 0.264 if you run on entire train SG records 

clusters = fcluster(linked, thr, criterion='distance')

num_unique_clusters = len(set(clusters))
print(f'Number of unique features at threshold {thr}: {num_unique_clusters}')

cluster_assignments = pd.DataFrame({'Feature': corr.columns, 'Cluster': clusters})

grouped_features = cluster_assignments.groupby('Cluster')['Feature'].apply(list)
grouped_features

## === cell 16

principal_components_df = pd.DataFrame()

for cluster_label, features in grouped_features.items():
    cluster_data = sg_records_df[features]
    
    scaler = StandardScaler()
    cluster_data_standardized = scaler.fit_transform(cluster_data)
    
    pca = PCA(n_components=1)
    
    principal_component = pca.fit_transform(cluster_data_standardized)
    
    principal_component_df = pd.DataFrame(principal_component, columns=[f'PCA_{cluster_label}'])
    
    principal_components_df = pd.concat([principal_components_df, principal_component_df], axis=1)

keys = sg_records_df[['spectrogram_id', 'time']].reset_index(drop=True)

final_pc_df = pd.concat([keys, principal_components_df], axis=1)
print(final_pc_df.shape)
final_pc_df.head()


## === cell 18
vote_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
prob_columns = ['seizure_prob', 'lpd_prob', 'gpd_prob', 'lrda_prob', 'grda_prob', 'other_prob']

train_df['total_votes'] = train_df[vote_columns].sum(axis=1)

for vote_col, prob_col in zip(vote_columns, prob_columns):
    train_df[prob_col] = train_df[vote_col] / train_df['total_votes']
train_df.head()

## === cell 19
processed_data = {} 
EmptySGs = []
for index, row in train_df.iterrows():
    label_id = row['label_id']
    spectrogram_id = row['spectrogram_id']
    offset_seconds = row['spectrogram_label_offset_seconds'] + 76 # skip 76 seconds to get the middle 448 seconds
    labels = row[['seizure_prob', 'lpd_prob', 'gpd_prob', 'lrda_prob', 'grda_prob', 'other_prob']]
    
    mask = (final_pc_df['spectrogram_id'] == spectrogram_id) & \
           (final_pc_df['time'] >= offset_seconds) & \
           (final_pc_df['time'] < offset_seconds + 448)
    
    temp_df = final_pc_df[mask].reset_index(drop=True)
    
    if temp_df.shape[0] == 224 and temp_df.isnull().sum().sum() == 0:
        processed_data[label_id] = {'spectrogram_id': spectrogram_id}
        for label in labels.index:
            processed_data[label_id][label] = labels[label]

        for i in range(1, 225):  # 224 PCA features
            processed_data[label_id][f'PCA_{i}'] = []
        
        for i in range(1, 225):  
            processed_data[label_id][f'PCA_{i}'] += temp_df[f'PCA_{i}'].tolist()
    
    else :
        EmptySGs.append(spectrogram_id)
    
final_training_data_byPCA = pd.DataFrame.from_dict(processed_data, orient='index')
final_training_data_byPCA.reset_index(inplace=True)

print(final_training_data_byPCA.shape)
print('Null or incorrect size #: ', EmptySGs)
final_training_data_byPCA.head()

## === cell 20
def transform_to_3channel_with_zeros(dataframe):
    images = []
    for _, row in dataframe.iterrows():
        img_list = []
        for i in range(1, 225):  # Assuming feature names are 'PCA_Cluster_1' to 'PCA_Cluster_224'
            cluster_data = ast.literal_eval(row[f'PCA_{i}']) if isinstance(row[f'PCA_{i}'], str) else row[f'PCA_{i}']
            img_list.append(np.array(cluster_data))
        
        single_channel_img = np.stack(img_list, axis=0)
        zeros_channel = np.zeros_like(single_channel_img)
        three_channel_img = np.stack([single_channel_img, zeros_channel, zeros_channel], axis=0)
        
        img_tensor = torch.tensor(three_channel_img, dtype=torch.float32)
        images.append(img_tensor.unsqueeze(0))  # Add batch dimension
    
    return torch.cat(images, dim=0)

images_tensor = transform_to_3channel_with_zeros(final_training_data_byPCA)
print(images_tensor.shape)
images_tensor[0]

## === cell 21
from sklearn.model_selection import train_test_split
labels = final_training_data_byPCA[['seizure_prob', 'lpd_prob', 'gpd_prob', 'lrda_prob', 'grda_prob', 'other_prob']].values
train_images, val_images, train_labels, val_labels = train_test_split(images_tensor, labels, test_size=0.2, random_state=52)
print('** distribution of highest prob labels in train - check if it is skewed **')
print(pd.DataFrame(train_labels, columns=prob_columns)[prob_columns].idxmax(axis=1).value_counts())
print('\n** distribution of highest prob labels in validation - check if it is skewed **')
print(pd.DataFrame(val_labels, columns=prob_columns)[prob_columns].idxmax(axis=1).value_counts())

## === cell 23
config_path = '/kaggle/input/mobilenet_v2_1.0_224_class6/pytorch/google_mobilenet_v2_1.0_224_class6/2/config.pkl'
model_path = '/kaggle/input/mobilenet_v2_1.0_224_class6/pytorch/google_mobilenet_v2_1.0_224_class6/2/mobilenet_v2_1.0_224_class6.pth'

from transformers import AutoModelForImageClassification
import pickle

with open(config_path, 'rb') as f:
    config = pickle.load(f)

model = AutoModelForImageClassification.from_config(config)
model.load_state_dict(torch.load(model_path))


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3907166657.py in <cell line: 0>()
      7 
      8 # Load the configuration
----> 9 with open(config_path, 'rb') as f:
     10     config = pickle.load(f)
     11 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/mobilenet_v2_1.0_224_class6/pytorch/google_mobilenet_v2_1.0_224_class6/2/config.pkl'

## === cell 24
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

train_dataset = TensorDataset(train_images, torch.tensor(train_labels, dtype=torch.float))
val_dataset = TensorDataset(val_images, torch.tensor(val_labels, dtype=torch.float))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

criterion = torch.nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr=0.00001)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3001598042.py in <cell line: 0>()
     11 # Set loss function to Kullback Liebler divergence
     12 criterion = torch.nn.KLDivLoss(reduction='batchmean')
---> 13 optimizer = optim.Adam(model.parameters(), lr=0.00001)

NameError: name 'model' is not defined

## === cell 25
train_losses = []
val_losses = []

num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(F.log_softmax(outputs.logits, dim=1), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(F.log_softmax(outputs.logits, dim=1), labels)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/495467619.py in <cell line: 0>()
      6 
      7 for epoch in range(num_epochs):
----> 8     model.train()  # Set model to training mode
      9     total_loss = 0
     10     for images, labels in tqdm(train_loader):

NameError: name 'model' is not defined

## === cell 26
plt.figure(figsize=(8, 3))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

## === cell 27
import sys



## === cell 28
del sg_records_df, final_training_data_byPCA, final_pc_df, principal_components_df, train_df_M

## === cell 30
test_df = pd.read_csv(f'{data_path}/test.csv')
print(test_df.shape)
test_df.head()

## === cell 31
test_preds = pd.DataFrame(columns = ['eeg_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']) 
for eeg_id in test_df['eeg_id']:
    if eeg_id not in test_preds['eeg_id'].values:
        new_row = pd.DataFrame([[eeg_id, 0.166, 0.166, 0.166, 0.166, 0.166, 0.170]],
                               columns=['eeg_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'])
        test_preds = pd.concat([test_preds, new_row], ignore_index=True)

test_preds.head()

## === cell 32
test_preds.to_csv("submission.csv", index=False)
print('submission file generated')

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length
