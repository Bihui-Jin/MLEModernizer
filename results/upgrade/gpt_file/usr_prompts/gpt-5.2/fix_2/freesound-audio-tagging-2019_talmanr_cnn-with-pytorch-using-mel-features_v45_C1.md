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

0.03515

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.03515) has done: 'I fix the runtime errors caused by API changes in librosa (deprecated `waveplot`, keyword-only `melspectrogram`), NumPy (`np.int` removal), and a few DataLoader/dataset issues that prevent training/inference from running. I also make the label one-hot encoding deterministic and aligned to `sample_submission.csv` column order, so the model outputs map to the correct 80 labels and the submission columns match exactly. Finally, I correct inference batching/collection logic and ensure the script always writes a valid `submission.csv` with `fname` plus the 80 label columns. These changes preserve your core MFCC→CNN training/inference approach, but make it run end-to-end and produce a valid submission.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import librosa
import librosa.display

import IPython.display as ipd

import matplotlib.pyplot as plt

import torchaudio
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import models, transforms
import torch.optim as optim

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_on_gpu = torch.cuda.is_available()

BASE_DIR = (
    "../input/freesound-audio-tagging-2019"
    if os.path.exists("../input/freesound-audio-tagging-2019")
    else "../input"
)

print("BASE_DIR:", BASE_DIR)
print("Listing ../input:", os.listdir("../input")[:20])



## === cell 1
Labels = pd.read_csv(os.path.join(BASE_DIR, "train_curated.csv"))
WavPath = os.path.join(BASE_DIR, "train_curated") + os.sep
Fils = os.listdir(WavPath)

sound, sample_rate = torchaudio.load(os.path.join(WavPath, Fils[5]))
ipd.Audio(data=sound[0, :], rate=sample_rate)



## === cell 2
x, sr = librosa.load(os.path.join(WavPath, Fils[4]), sr=None)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(np.abs(X))
plt.figure(figsize=(14, 5))
print("STFT db shape:", Xdb.shape)

S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=23)
delta2_mfcc = librosa.feature.delta(MFCC, order=2)

plt.figure(figsize=(10, 4))
librosa.display.specshow(delta2_mfcc, x_axis="time")
plt.colorbar()
plt.tight_layout()



## === cell 3
x



## === cell 4
FilesS = np.zeros(len(Fils))
for i, File in enumerate(Fils):
    FilesS[i] = os.path.getsize(os.path.join(WavPath, File))

plt.figure(figsize=(20, 8))
plt.hist(FilesS, bins=50)



## === cell 5
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
label_cols = [c for c in sample_sub.columns if c != "fname"]

Fils_2 = Labels["fname"].reset_index(drop=True)

NumClasses = len(label_cols)
OneHot_All = np.zeros((len(Fils_2), NumClasses), dtype=np.float32)

label_to_idx = {lab: i for i, lab in enumerate(label_cols)}
for i, labs in enumerate(Labels["labels"].fillna("").astype(str).values):
    for lab in labs.split(","):
        lab = lab.strip()
        if lab in label_to_idx:
            OneHot_All[i, label_to_idx[lab]] = 1.0

print("NumClasses:", NumClasses)
print("OneHot_All shape:", OneHot_All.shape)



## === cell 6
split_frac = 0.92
batch_size = 32

split_idx = int(len(Fils_2) * split_frac)
split_idx1 = int(batch_size * np.floor(split_idx / batch_size))
split_idx2 = int(batch_size * np.floor((len(Fils_2) - split_idx1) / batch_size))

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
        self,
        list_IDs,
        labels_dict,
        DataPath,
        RecLen,
        DecNum=5,
        fft_Samp=256,
        Im_3D=False,
    ):
        self.labels_dict = labels_dict  # dict: fname -> multi-hot vector
        self.list_IDs = (
            list_IDs if isinstance(list_IDs, (list, tuple)) else list(list_IDs)
        )
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
        fp = os.path.join(self.DataPath, ID)

        fs, sig = wavfile.read(fp)

        sig = np.asarray(sig)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        sig = sig.astype(np.float32)

        S = librosa.feature.melspectrogram(y=sig, sr=fs, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=self.NFCC_Num)

        if self.labels_dict is None or ID not in self.labels_dict:
            y = np.zeros((NumClasses,), dtype=np.float32)
        else:
            y = self.labels_dict[ID].astype(np.float32)
        LabelOut = torch.from_numpy(y).float()

        Im = torch.zeros((self.NFCC_Num, self.TimeSamp), dtype=torch.float32)

        if MFCC.shape[1] > self.TimeSamp:
            Im = torch.from_numpy(MFCC[:, : self.TimeSamp]).float()
        else:
            start = int(self.TimeSamp / 2 - int(MFCC.shape[1] / 2))
            Im[:, start : start + MFCC.shape[1]] = torch.from_numpy(MFCC).float()

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
class CnnTransferNet(nn.Module):
    def __init__(self):
        super(CnnTransferNet, self).__init__()

        self.vgg = models.vgg16_bn().cuda()
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
model = CnnAudioNet(NumClasses)
if train_on_gpu:
    model.cuda()
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.005, amsgrad=False)



## === cell 12
labelsDict_train = dict(zip(train_x.tolist(), train_y))
labelsDict_val = dict(zip(val_x.tolist(), val_y))

params = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": 2,
    "pin_memory": train_on_gpu,
}

RecLen = 176400

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

training_set = Dataset(
    train_x.tolist(), labelsDict_train, WavPath, RecLen, transforms.Compose([normalize])
)
training_generator = data.DataLoader(training_set, **params)

val_set = Dataset(
    val_x.reset_index(drop=True).tolist(),
    labelsDict_val,
    WavPath,
    RecLen,
    transforms.Compose([normalize]),
)
val_generator = data.DataLoader(val_set, **params)



## === cell 13
import time

start_time = time.time()

n_epochs = 12
valid_loss_min = np.inf
print("Start training:")
idx = 0

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0

    model.train()
    for dataBatch, target, _ in training_generator:
        idx += 1
        dataBatch = dataBatch.unsqueeze(1).float()
        target = target.float()
        if train_on_gpu:
            dataBatch = dataBatch.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        output = model(dataBatch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * dataBatch.size(0)

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for dataBatch_v, target_v, _ in val_generator:
            dataBatch_v = dataBatch_v.unsqueeze(1).float()
            target_v = target_v.float()
            if train_on_gpu:
                dataBatch_v = dataBatch_v.cuda(non_blocking=True)
                target_v = target_v.cuda(non_blocking=True)

            output_v = model(dataBatch_v)
            loss_v = criterion(output_v, target_v)
            valid_loss += loss_v.item() * dataBatch_v.size(0)

    train_loss = train_loss / len(training_generator.dataset)
    valid_loss = valid_loss / len(val_generator.dataset)

    print(
        f"Epoch: {epoch}\tTraining Loss: {train_loss:.6f}\tValidation Loss: {valid_loss:.6f}"
    )
    print("--- %s seconds ---" % (time.time() - start_time))



## === cell 14
try:
    plt.figure(figsize=(12, 3))
    plt.plot(target_v[0, :].detach().cpu().numpy(), label="target")
    plt.plot(torch.sigmoid(output_v[0, :]).detach().cpu().numpy(), label="pred")
    plt.legend()
    plt.tight_layout()
except Exception as e:
    print("Skipping plot:", repr(e))



## === cell 15
WavPath_test = os.path.join(BASE_DIR, "test") + os.sep
Fils_test = os.listdir(WavPath_test)

one_hot_test = np.zeros((len(Fils_test), NumClasses), dtype=np.float32)
labelsDict_test = dict(zip(Fils_test, one_hot_test))

params_test = {
    "batch_size": 16,
    "shuffle": False,
    "num_workers": 2,
    "pin_memory": train_on_gpu,
}
test_set = Dataset(
    Fils_test, labelsDict_test, WavPath_test, RecLen, transforms.Compose([normalize])
)
test_generator = data.DataLoader(test_set, **params_test)



## === cell 16
model.eval()
Output_all = []
BatchRecs_all = []

with torch.no_grad():
    for dataBatch, _, BatchRecs in test_generator:
        dataBatch = dataBatch.unsqueeze(1).float()
        if train_on_gpu:
            dataBatch = dataBatch.cuda(non_blocking=True)

        output = model(dataBatch)
        outP = torch.sigmoid(output).detach().cpu().numpy()

        Output_all.append(outP)
        BatchRecs_all.extend(list(BatchRecs))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/2325841604.py in <cell line: 0>()
      4 
      5 with torch.no_grad():
----> 6     for dataBatch, _, BatchRecs in test_generator:
      7         dataBatch = dataBatch.unsqueeze(1).float()
      8         if train_on_gpu:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

IsADirectoryError: Caught IsADirectoryError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_11/2335159604.py", line 34, in __getitem__
    fs, sig = wavfile.read(fp)
              ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/scipy/io/wavfile.py", line 674, in read
    fid = open(filename, 'rb')
          ^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '../input/freesound-audio-tagging-2019/test/test'


## === cell 17
Dataout = (
    np.vstack(Output_all)
    if len(Output_all) > 0
    else np.zeros((0, NumClasses), dtype=np.float32)
)
Names = BatchRecs_all

print("Pred matrix:", Dataout.shape, "Names:", len(Names))

n = min(len(Names), Dataout.shape[0])
Dataout = Dataout[:n, :]
Names = Names[:n]



## === cell 18
Output_all_DF = pd.DataFrame(Dataout, columns=label_cols)
Output_all_DF.insert(0, "fname", Names)

Output_all_DF = (
    sample_sub[["fname"]].merge(Output_all_DF, on="fname", how="left").fillna(0.0)
)



## === cell 19
print(Output_all_DF.head())
out_path = "submission.csv"
Output_all_DF.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", Output_all_DF.shape)
print(
    "Columns match sample_submission:",
    list(Output_all_DF.columns) == list(sample_sub.columns),
)
