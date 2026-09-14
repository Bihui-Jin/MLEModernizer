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

1.1101675291376314

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I add a fallback that uses the overall class‑vote distribution from the training data as a constant probability prediction for every test row. This guarantees a valid CSV with rows that sum to 1, fixing the previous failure where the model file was missing. The fallback is only used if loading or running the pretrained model fails, so the original logic is preserved when the model is available.'
- What this solution (achieved 1.40627) has done: 'I improve the fallback probability calculation by blending two sensible priors: (1) the overall vote‑distribution derived from the raw annotation counts and (2) the distribution of the expert consensus labels. This still keeps the original constant‑prediction fallback (so the core logic, model loading, and inference remain unchanged) but provides a slightly more informative prior that should reduce the KL‑divergence toward the target score. The change is limited to the probability‑setup cells and does not affect model architecture or training.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
from torch.nn.functional import softmax, one_hot, log_softmax
import numpy as np
import matplotlib.cm as cm



## === cell 1
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
classes = train["expert_consensus"].unique()
mapping = {c: i for i, c in enumerate(classes)}
num_classes = classes.shape[0]

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
train_votes = train[vote_cols].astype(float)

class_totals = train_votes.sum()
vote_dist = (class_totals / class_totals.sum()).values  # shape (6,)

consensus_counts = train["expert_consensus"].value_counts()
consensus_dist = np.array(
    [consensus_counts.get(col.split("_")[0], 0) for col in vote_cols], dtype=float
)
consensus_dist = (
    consensus_dist / consensus_dist.sum()
    if consensus_dist.sum() > 0
    else np.full_like(consensus_dist, 1 / len(consensus_dist))
)

fallback_probs = 0.6 * vote_dist + 0.4 * consensus_dist  # shape (6,)

patient_vote_sums = train.groupby("patient_id")[vote_cols].sum()
patient_totals = patient_vote_sums.sum(axis=1).replace(0, np.nan)  # avoid div by zero
patient_probs_df = patient_vote_sums.div(patient_totals, axis=0).fillna(fallback_probs)
patient_probs = {
    pid: row.values.astype(float) for pid, row in patient_probs_df.iterrows()
}



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2225616157.py in <cell line: 0>()
     34 patient_vote_sums = train.groupby("patient_id")[vote_cols].sum()
     35 patient_totals = patient_vote_sums.sum(axis=1).replace(0, np.nan)  # avoid div by zero
---> 36 patient_probs_df = patient_vote_sums.div(patient_totals, axis=0).fillna(fallback_probs)
     37 # Convert to a dict of numpy arrays for fast lookup.
     38 patient_probs = {

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7438                 new_data = self.where(self.notna(), value)._mgr
   7439             else:
-> 7440                 raise ValueError(f"invalid fill value with a {type(value)}")
   7441 
   7442         result = self._constructor_from_mgr(new_data, axes=new_data.axes)

ValueError: invalid fill value with a <class 'numpy.ndarray'>

## === cell 2
cmap = cm.get_cmap("viridis")




## === cell 3
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        specto_id = self.data.loc[idx, "spectrogram_id"]
        specto_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{specto_id}.parquet"
        specto = pd.read_parquet(specto_path)
        spectrogram = Image.fromarray((cmap(specto) * 255).astype(np.uint8))
        if self.transform:
            spectrogram = self.transform(spectrogram)[:3, :, :]
        return spectrogram




## === cell 4
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)



## === cell 5
test_dataset = ImageDataset(test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




## === cell 6
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = softmax(x, dim=1)
        return x




## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(num_classes).to(device)

model_path = "/kaggle/input/hms-train-spectogram-images/trained_model.pt"
model_loaded = False
try:
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model_loaded = True
except Exception as e:
    print(f"Model loading failed ({e}); will use fallback probabilities.")



## === cell 8
if model_loaded:
    model.eval()
    out = []
    pbar = tqdm(test_loader)
    for images in pbar:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        out.append(outputs.detach().cpu().numpy())
    outputs = np.vstack(out)  # shape (num_test, num_classes)
else:
    fallback_array = np.empty((len(test), 6), dtype=float)
    for i, pid in enumerate(test["patient_id"]):
        fallback_array[i] = patient_probs.get(pid, fallback_probs)
    outputs = fallback_array



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/644525095.py in <cell line: 0>()
     13     fallback_array = np.empty((len(test), 6), dtype=float)
     14     for i, pid in enumerate(test["patient_id"]):
---> 15         fallback_array[i] = patient_probs.get(pid, fallback_probs)
     16     outputs = fallback_array
     17 

NameError: name 'patient_probs' is not defined

## === cell 9
outputs = outputs / outputs.sum(axis=1, keepdims=True)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/476507560.py in <cell line: 0>()
----> 1 outputs = outputs / outputs.sum(axis=1, keepdims=True)
      2 

NameError: name 'outputs' is not defined

## === cell 10
mapping  # retained for reference; not used in final submission



## === cell 11
submission = test[["eeg_id"]].copy()
submission["seizure_vote"] = outputs[:, 0]
submission["lpd_vote"] = outputs[:, 1]
submission["gpd_vote"] = outputs[:, 2]
submission["lrda_vote"] = outputs[:, 3]
submission["grda_vote"] = outputs[:, 4]
submission["other_vote"] = outputs[:, 5]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3518613781.py in <cell line: 0>()
      1 submission = test[["eeg_id"]].copy()
----> 2 submission["seizure_vote"] = outputs[:, 0]
      3 submission["lpd_vote"] = outputs[:, 1]
      4 submission["gpd_vote"] = outputs[:, 2]
      5 submission["lrda_vote"] = outputs[:, 3]

NameError: name 'outputs' is not defined

## === cell 12
submission.to_csv("submission.csv", index=False)
submission

## --- ERROR in outputing the csv:
Invalid submission: Submission must contain all target columns: ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
