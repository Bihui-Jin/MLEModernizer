# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Develop a model to tag audio data automatically using a diverse vocabulary of 80 categories.

## Metric
The task consists of predicting the audio labels (tags) for every test clip. Some test clips bear one label while others bear several labels. The predictions are to be done at the clip level, i.e., no start/end timestamps for the sound events are required.

The primary metric is label-weighted label-ranking average precision. 

The  "label-weighted" part means that the overall score is the average over all the *labels* in the test set, where each label receives equal weight (by contrast, plain *lrap* gives each *test item* equal weight).

## Submission Format
For each `fname` in the test set, you must predict the probability of each label. The file should contain a header and have the following format:

```
fname,Accelerating_and_revving_and_vroom,...Zipper_(clothing)
000ccb97.wav,0.1,....,0.3
0012633b.wav,0.0,...,0.8
```

## Dataset
The following 5 audio files in the curated train set have a wrong label, due to a bug in the file renaming process:\
`f76181c4.wav, 77b925c2.wav, 6a1f682a.wav, c7db12aa.wav, 7752cc8a.wav`

The audio file `1d44b0bd.wav` in the curated train set was found to be corrupted (contains no signal) due to an error in format conversion.

- **train_curated.csv** - ground truth labels for the curated subset of the training audio files (see Data Fields below)
- **train_noisy.csv** - ground truth labels for the noisy subset of the training audio files (see Data Fields below)
- **sample_submission.csv** - a sample submission file in the correct format, including the correct sorting of the sound categories; it contains the list of audio files found in the test.zip folder (corresponding to the public leaderboard)
- **train_curated.zip** - a folder containing the audio (.wav) training files of the curated subset
- **train_noisy.zip** - a folder containing the audio (.wav) training files of the noisy subset
- **test.zip** - a folder containing the audio (.wav) test files for the public leaderboard

### Columns
Each row of the train_curated.csv and train_noisy.csv files contains the following information:

- **fname**: the audio file name, eg, `0006ae4e.wav`
- **labels**: the audio classification label(s) (ground truth). Note that the number of labels per clip can be one, eg, `Bark` or more, eg, `"Walk_and_footsteps,Slam"`.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
librosa==0.11.0
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
scipy==1.15.3
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
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        input/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        working/
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
```

-> data/freesound-audio-tagging-2019/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/freesound-audio-tagging-2019/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/freesound-audio-tagging-2019/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> data/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.1407

# 6. Current score

0.10102

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.10102) has done: 'I make the pipeline produce a valid submission by fixing three issues that currently prevent a meaningful Kaggle score: (1) label column order must exactly match `sample_submission.csv`, but your `set()`-based `All_class` is unordered and can misalign columns; (2) your test DataLoader uses `shuffle=True` and then manually reconstructs `Names/Dataout`, which breaks row alignment—Igenerate predictions in the exact sample-submission order and write them directly; (3) training is missing entirely, so the model is effectively random—Iadd a minimal training loop (same model/loss/optimizer) and then predict with `sigmoid` logits. These are minimal, directly score-relevant changes that preserve your core MFCC->CNN logic and produce `submission.csv` in the required format within the time limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import librosa
import librosa.display

import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim

print(os.listdir("../input"))
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")



## === cell 1
Labels = pd.read_csv("../input/train_curated.csv")
WavPath = "../input/train_curated/"

Fils = os.listdir(WavPath)
sound, sample_rate = torchaudio.load(WavPath + Fils[5])



## === cell 2
x, sr = librosa.load(WavPath + Fils[4], sr=None)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=23)
delta2_mfcc = librosa.feature.delta(MFCC, order=2)
delta2_mfcc.shape



## === cell 3
x



## === cell 4
FilesS = np.zeros(len(Fils))
for i, File in enumerate(Fils):
    FilesS[i] = os.path.getsize(WavPath + File)



## === cell 5
sample_sub_path = "../input/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
label_cols = [c for c in sample_sub.columns if c != "fname"]
NumClasses = len(label_cols)

Fils_2 = Labels["fname"].reset_index(drop=True)
OneHot_All = np.zeros((len(Fils_2), NumClasses), dtype=np.float32)

for i, labs in enumerate(Labels["labels"].astype(str).values):
    labs_set = set(labs.split(","))
    for j, clas in enumerate(label_cols):
        OneHot_All[i, j] = 1.0 if clas in labs_set else 0.0

print("NumClasses:", NumClasses)



## === cell 6
split_frac = 0.92
batch_size = 32

n = len(Fils_2)
split_idx = int(n * split_frac)
split_idx1 = int(batch_size * np.floor(split_idx / batch_size))
split_idx2 = int(batch_size * np.floor((n - split_idx1) / batch_size))

train_x, val_x = Fils_2[:split_idx1], Fils_2[split_idx1 : split_idx1 + split_idx2]
train_y, val_y = (
    OneHot_All[:split_idx1, :],
    OneHot_All[split_idx1 : split_idx1 + split_idx2, :],
)

print(len(train_x) / batch_size, len(val_x) / batch_size)



## === cell 7
torch.zeros((2, 3)).type(torch.FloatTensor)



## === cell 8
from scipy.io import wavfile


class Dataset(data.Dataset):
    def __init__(
        self, list_IDs, labels, DataPath, RecLen, DecNum=5, fft_Samp=256, Im_3D=False
    ):
        self.labels = labels
        self.list_IDs = list_IDs
        self.DataPath = DataPath
        self.RecLen = RecLen
        self.fft_Samp = fft_Samp
        self.Im_3D = Im_3D

        self.NFCC_Num = 23
        self.TimeSamp = 1024

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        fs, aud = wavfile.read(self.DataPath + ID)

        S = librosa.feature.melspectrogram(y=np.float32(aud), sr=fs, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=self.NFCC_Num)

        y = torch.from_numpy(self.labels[ID]).float()

        Im = torch.zeros((self.NFCC_Num, self.TimeSamp), dtype=torch.float32)
        if MFCC.shape[1] > self.TimeSamp:
            Im = torch.from_numpy(MFCC[:, : self.TimeSamp]).float()
        else:
            start = int(self.TimeSamp / 2 - int(MFCC.shape[1] / 2))
            Im[:, start : start + MFCC.shape[1]] = torch.from_numpy(MFCC).float()

        return Im, y, ID




## === cell 9
class CnnAudioNet(nn.Module):
    def __init__(self, NumClasses):
        super(CnnAudioNet, self).__init__()
        self.NumClasses = NumClasses
        self.Fc_features = 128
        self.C1 = nn.Conv2d(1, 8, 3, padding=1)
        self.C2 = nn.Conv2d(8, 16, 3, padding=1)
        self.C3 = nn.Conv2d(16, 16, 3, padding=1)
        self.C4 = nn.Conv2d(16, 32, 3, padding=1)
        self.C5 = nn.Conv2d(32, 32, 3, padding=1)

        self.BN1 = nn.BatchNorm2d(8)
        self.BN2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d((1, 2), (1, 2))

        self.fc1 = nn.Linear(32 * 23 * 16, 128)
        self.fc2 = nn.Linear(128, self.NumClasses)
        self.dropout = nn.Dropout(0.25)
        self.Bat1 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.maxpool2(F.relu(self.BN1(self.C1(x))))
        x = self.maxpool2(F.relu(self.BN2(self.C2(x))))
        x = self.maxpool2(F.relu(self.BN2(self.C3(x))))
        x = self.maxpool2(F.relu(self.C4(x)))
        x = self.maxpool2(F.relu(self.C5(x)))
        x = self.maxpool2(F.relu(self.C5(x)))
        x = x.view(-1, 32 * 23 * 16)
        x = self.dropout(self.Bat1(self.fc1(x)))
        x = self.fc2(x)
        return x




## === cell 10
from torchvision import models


class CnnTransferNet(nn.Module):
    def __init__(self):
        super(CnnTransferNet, self).__init__()
        self.vgg = models.vgg16_bn().to(device)
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, NumClasses)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        Features = self.dropout(self.vgg(x))
        Features = F.relu(self.fc1(Features))
        Features = self.fc2(Features)
        return Features




## === cell 11
model = CnnAudioNet(NumClasses).to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.005, amsgrad=False)



## === cell 12
labelsDict_train = dict(zip(train_x.tolist(), train_y))
labelsDict_val = dict(zip(val_x.tolist(), val_y))

RecLen = 176400

train_params = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": 2,  # keep modest for reliability in this environment
    "pin_memory": train_on_gpu,
}

val_params = {
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": 2,
    "pin_memory": train_on_gpu,
}

training_set = Dataset(train_x.tolist(), labelsDict_train, WavPath, RecLen)
training_generator = data.DataLoader(training_set, **train_params)

val_set = Dataset(val_x.tolist(), labelsDict_val, WavPath, RecLen)
val_generator = data.DataLoader(val_set, **val_params)



## === cell 13
torch.manual_seed(0)
np.random.seed(0)
if train_on_gpu:
    torch.cuda.manual_seed_all(0)

epochs = 2  # minimal to keep within 600s; preserves training approach (standard loop)

model.train()
for ep in range(epochs):
    running_loss = 0.0
    for xb, yb, _ in training_generator:
        xb = xb.unsqueeze(1).to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb, _ in val_generator:
            xb = xb.unsqueeze(1).to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            logits = model(xb)
            val_loss += criterion(logits, yb).item()
    model.train()

    print(
        f"Epoch {ep+1}/{epochs} - train_loss={running_loss/len(training_generator):.4f} - val_loss={val_loss/max(1,len(val_generator)):.4f}"
    )



## === cell 14
WavPath_test = "../input/test/"

test_files = sample_sub["fname"].tolist()
test_files = [f for f in test_files if os.path.isfile(os.path.join(WavPath_test, f))]

dummy_labels = {f: np.zeros((NumClasses,), dtype=np.float32) for f in test_files}

test_params = {
    "batch_size": 8,
    "shuffle": False,  # critical for correct fname alignment
    "num_workers": 2,
    "pin_memory": train_on_gpu,
}

test_set = Dataset(test_files, dummy_labels, WavPath_test, RecLen)
test_generator = data.DataLoader(test_set, **test_params)



## === cell 15
model.eval()
preds = np.zeros((len(test_files), NumClasses), dtype=np.float32)

offset = 0
with torch.no_grad():
    for xb, _, ids in test_generator:
        bs = xb.shape[0]
        xb = xb.unsqueeze(1).to(device, dtype=torch.float32)
        logits = model(xb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        preds[offset : offset + bs, :] = prob
        offset += bs



## === cell 16
sub = pd.DataFrame(preds, columns=label_cols)
sub.insert(0, "fname", test_files)

print("Submission shape:", sub.shape)
sub.head()



## === cell 17
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
