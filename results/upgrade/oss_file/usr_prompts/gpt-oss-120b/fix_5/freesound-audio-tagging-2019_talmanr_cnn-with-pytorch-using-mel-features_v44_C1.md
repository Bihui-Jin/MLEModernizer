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

0.14312

# 6. Current score

0.12758

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07516) has done: 'Implemented fixes to stop runtime errors caused by mismatched tensor dimensions in the CNN model.  
- Updated `CnnAudioNet` to use the correct flatten size after convolutions and removed the duplicated final convolution/pooling step.  
- Adjusted the `forward` method to flatten dynamically (`x.view(x.size(0), -1)`).  
These changes allow the model to train, generate predictions, and write a valid `submission.csv` without altering the core training logic.'
- What this solution (achieved 0.12758) has done: 'I increase the training duration and use a slightly smaller learning rate so the model can fit the data a bit better without changing its architecture or overall pipeline. Specifically, I set the Adam optimizer’s learning rate to 0.001 instead of 0.005 and raise the number of training epochs from 1 to 3. These minimal adjustments should push the validation performance upward, moving the score closer to the target while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

print(os.listdir("../input"))

import torchaudio
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim

train_on_gpu = torch.cuda.is_available()




## === cell 1
Labels = pd.read_csv("../input/train_curated.csv")
WavPath = "../input/train_curated/"
Fils = os.listdir(WavPath)
sound, sample_rate = torchaudio.load(os.path.join(WavPath, Fils[5]))
ipd.Audio(data=sound[0, :], rate=sample_rate)  # load a local WAV file




## === cell 2
x, sr = librosa.load(os.path.join(WavPath, Fils[4]))
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
plt.title("Spectrogram (dB)")
plt.imshow(Xdb, aspect="auto", origin="lower")
plt.colorbar()
plt.tight_layout()

S = librosa.feature.melspectrogram(y=x, sr=sample_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=23)
delta2_mfcc = librosa.feature.delta(MFCC, order=2)

librosa.display.specshow(delta2_mfcc)
plt.colorbar()
plt.tight_layout()




## === cell 3
x  # just to show the waveform array in the notebook




## === cell 4
FilesS = np.zeros(len(Fils))
for i, File in enumerate(Fils):
    FilesS[i] = os.path.getsize(os.path.join(WavPath, File))

plt.figure(figsize=(20, 8))
plt.hist(FilesS, bins=50)




## === cell 5
Fils_2 = Labels["fname"].values
Class = set(Labels["labels"])
All_class = []
for i in Class:
    for j in i.split(","):
        All_class.append(j)
All_class = sorted(set(All_class))  # keep a deterministic order

NumClasses = len(All_class)
OneHot_All = np.zeros((len(Fils_2), NumClasses))

for i, file_labels in enumerate(Labels["labels"]):
    for j, clas in enumerate(All_class):
        OneHot_All[i, j] = int(clas in file_labels)




## === cell 6
print(NumClasses)
split_frac = 0.92
batch_size = 32

split_idx = int(len(Fils) * split_frac)
split_idx1 = int(batch_size * np.floor(split_idx / batch_size))
split_idx2 = int(batch_size * np.floor((len(Fils) - split_idx1) / batch_size))
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
    def __init__(self, list_IDs, labels, DataPath, RecLen, transform=None):
        """
        list_IDs: list/array of filenames
        labels: dict mapping filename -> one‑hot np.array
        DataPath: directory containing wav files
        RecLen: not used directly (kept for compatibility)
        transform: optional transform applied to the spectrogram tensor
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.DataPath = DataPath
        self.RecLen = RecLen
        self.NFCC_Num = 23
        self.TimeSamp = 1024
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        fs, data = wavfile.read(os.path.join(self.DataPath, ID))

        S = librosa.feature.melspectrogram(y=np.float32(data), sr=fs, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        MFCC_1 = librosa.feature.mfcc(S=log_S, n_mfcc=23)
        MFCC = librosa.feature.delta(MFCC_1, order=2) / 6

        LabelOut = torch.from_numpy(self.labels[ID]).double()

        Im = torch.zeros((self.NFCC_Num, self.TimeSamp), dtype=torch.float32)

        if MFCC.shape[1] > self.TimeSamp:
            Im = torch.from_numpy(MFCC[:, : self.TimeSamp]).float()
        else:
            start = int(self.TimeSamp / 2 - MFCC.shape[1] / 2)
            Im[:, start : start + MFCC.shape[1]] = torch.from_numpy(MFCC).float()

        if self.transform:
            Im = self.transform(Im)

        return Im, LabelOut, ID




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
        self.maxpoll = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d((1, 2), (1, 2))

        self.fc1 = nn.Linear(32 * 1 * 32, 128)
        self.fc2 = nn.Linear(128, self.NumClasses)
        self.dropout = nn.Dropout(0.25)
        self.Bat1 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.maxpool2(F.relu(self.BN1(self.C1(x))))
        x = self.maxpoll(F.relu(self.BN2(self.C2(x))))
        x = self.maxpoll(F.relu(self.BN2(self.C3(x))))
        x = self.maxpoll(F.relu(self.C4(x)))
        x = self.maxpoll(F.relu(self.C5(x)))  # single C5 + pooling
        x = x.view(x.size(0), -1)  # flatten dynamically
        x = self.dropout(self.Bat1(self.fc1(x)))
        x = self.fc2(x)
        return x




## === cell 10
class CnnTransferNet(nn.Module):
    def __init__(self):
        super(CnnTransferNet, self).__init__()
        self.vgg = nn.Sequential()  # placeholder – not used in the current pipeline

    def forward(self, x):
        return x




## === cell 11
model = CnnAudioNet(NumClasses)
if train_on_gpu:
    model.cuda()
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001, amsgrad=False)




## === cell 12
labelsDict_train = dict(zip(train_x, train_y))
labelsDict_val = dict(zip(val_x, val_y))

params = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": 4,  # reduce workers for portability
    "pin_memory": True if train_on_gpu else False,
}

RecLen = 176400  # retained for Dataset signature compatibility

training_set = Dataset(train_x, labelsDict_train, WavPath, RecLen, transform=None)
training_generator = data.DataLoader(training_set, **params)

val_set = Dataset(val_x, labelsDict_val, WavPath, RecLen, transform=None)
val_generator = data.DataLoader(val_set, **params)




## === cell 13
import time

start_time = time.time()

n_epochs = 3
print("Start training:")

for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0

    for dataBatch, target, _ in training_generator:
        if train_on_gpu:
            dataBatch = dataBatch.unsqueeze(1).float().cuda()
            target = target.cuda()
        else:
            dataBatch = dataBatch.unsqueeze(1).float()

        optimizer.zero_grad()
        output = model(dataBatch)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * dataBatch.size(0)

    model.eval()
    valid_loss = 0.0
    SumCorrectVal = 0
    TotVal = 0

    for dataBatch_v, target_v, _ in val_generator:
        if train_on_gpu:
            dataBatch_v = dataBatch_v.unsqueeze(1).float().cuda()
            target_v = target_v.cuda()
        else:
            dataBatch_v = dataBatch_v.unsqueeze(1).float()

        output_v = model(dataBatch_v)
        loss_v = criterion(output_v, target_v.float())

        pred = torch.max(output_v, 1)[1]
        correct = torch.sum(pred == torch.squeeze(torch.argmax(target_v, dim=-1)))
        SumCorrectVal += correct.item()
        valid_loss += loss_v.item() * dataBatch_v.size(0)
        TotVal += dataBatch_v.size(0)

    train_loss /= len(training_generator.dataset)
    valid_loss /= len(val_generator.dataset)

    print(
        f"Epoch: {epoch} \t Training Loss: {train_loss:.6f} \t Validation Loss: {valid_loss:.6f}"
    )
    print(
        f"Epoch: {epoch} \t Validation Correct: {SumCorrectVal:.0f} / {TotVal:.0f} "
        f"Acc: {100.0 * SumCorrectVal / TotVal:.2f}%"
    )
    print(f"--- {time.time() - start_time:.2f} seconds ---")




## === cell 14
if "target" in locals() and "output" in locals():
    plt.plot(target[5, :].detach().cpu().numpy())
    plt.plot(torch.sigmoid(output[5, :]).detach().cpu().numpy())




## === cell 15
WavPath_test = "../input/test/"
Fils_test = [f for f in os.listdir(WavPath_test) if f.lower().endswith(".wav")]

one_hot_test = np.zeros((len(Fils_test), NumClasses))
labelsDict_test = dict(zip(Fils_test, one_hot_test))

test_params = {
    "batch_size": 4,
    "shuffle": False,
    "num_workers": 4,
    "pin_memory": True if train_on_gpu else False,
}

test_set = Dataset(Fils_test, labelsDict_test, WavPath_test, RecLen, transform=None)
test_generator = data.DataLoader(test_set, **test_params)




## === cell 16
model.eval()
all_outputs = []
all_fnames = []

with torch.no_grad():
    for dataBatch, _, batch_names in test_generator:
        if train_on_gpu:
            dataBatch = dataBatch.unsqueeze(1).float().cuda()
        else:
            dataBatch = dataBatch.unsqueeze(1).float()
        out = torch.sigmoid(model(dataBatch))  # probabilities
        all_outputs.append(out.cpu())
        all_fnames.extend(batch_names)

preds = torch.cat(all_outputs, dim=0).numpy()  # shape (num_test_files, NumClasses)




## === cell 17
sample_sub = pd.read_csv("../input/sample_submission.csv", nrows=0)
ordered_cols = list(sample_sub.columns)  # includes 'fname' as first column

submission_df = pd.DataFrame(preds, columns=ordered_cols[1:])
submission_df.insert(0, "fname", all_fnames)




## === cell 18
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
