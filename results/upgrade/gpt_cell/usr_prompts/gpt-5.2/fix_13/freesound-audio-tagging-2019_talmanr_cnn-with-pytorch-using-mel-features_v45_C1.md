# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd

import librosa
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim

print(os.listdir("../input"))
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")

torch.manual_seed(0)
np.random.seed(0)
if train_on_gpu:
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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
FilesS = np.array([os.path.getsize(WavPath + f) for f in Fils], dtype=np.float64)



## === cell 5
sample_sub_path = "../input/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
label_cols = [c for c in sample_sub.columns if c != "fname"]
NumClasses = len(label_cols)

bad_curated = {
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
    "1d44b0bd.wav",  # corrupted
}

Labels = Labels[~Labels["fname"].isin(bad_curated)].reset_index(drop=True)

Fils_2 = Labels["fname"].reset_index(drop=True)

label_to_idx = {c: i for i, c in enumerate(label_cols)}
OneHot_All = np.zeros((len(Fils_2), NumClasses), dtype=np.float32)
for i, labs in enumerate(Labels["labels"].astype(str).values):
    for lab in labs.split(","):
        j = label_to_idx.get(lab)
        if j is not None:
            OneHot_All[i, j] = 1.0

print("NumClasses:", NumClasses, "| curated rows after drop:", len(Fils_2))



## === cell 6
split_frac = 0.92
batch_size = 32

n = len(Fils_2)
split_idx = int(n * split_frac)
split_idx1 = int(batch_size * np.floor(split_idx / batch_size))
split_idx2 = int(batch_size * np.floor((n - split_idx1) / batch_size))

train_x_cur, val_x = Fils_2[:split_idx1], Fils_2[split_idx1 : split_idx1 + split_idx2]
train_y_cur, val_y = (
    OneHot_All[:split_idx1, :],
    OneHot_All[split_idx1 : split_idx1 + split_idx2, :],
)

print(len(train_x_cur) / batch_size, len(val_x) / batch_size)



## === cell 7
torch.zeros((2, 3)).type(torch.FloatTensor)



## === cell 8
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor, as_completed


def _mfcc_image_from_wav(path, NFCC_Num=23, TimeSamp=1024):
    fs, aud = wavfile.read(path)
    S = librosa.feature.melspectrogram(y=np.float32(aud), sr=fs, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=NFCC_Num)

    Im = np.zeros((NFCC_Num, TimeSamp), dtype=np.float32)
    if MFCC.shape[1] > TimeSamp:
        Im[:] = MFCC[:, :TimeSamp].astype(np.float32, copy=False)
    else:
        start = int(TimeSamp / 2 - int(MFCC.shape[1] / 2))
        Im[:, start : start + MFCC.shape[1]] = MFCC.astype(np.float32, copy=False)
    return Im


def precompute_mfcc_cache(
    file_list, data_path, cache_dir, NFCC_Num=23, TimeSamp=1024, max_workers=None
):
    os.makedirs(cache_dir, exist_ok=True)

    to_do = []
    for fn in file_list:
        outp = os.path.join(cache_dir, fn + ".npy")
        if not os.path.isfile(outp):
            to_do.append((fn, outp))

    if not to_do:
        return

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = min(8, max(2, cpu // 2))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for fn, outp in to_do:
            inp = os.path.join(data_path, fn)
            futs[ex.submit(_mfcc_image_from_wav, inp, NFCC_Num, TimeSamp)] = (fn, outp)

        done = 0
        for fut in as_completed(futs):
            fn, outp = futs[fut]
            Im = fut.result()
            tmp = outp + ".tmp"
            np.save(tmp, Im)
            os.replace(tmp, outp)
            done += 1
            if done % 500 == 0 or done == len(to_do):
                elapsed = time.time() - t0
                print(f"MFCC cache: {done}/{len(to_do)} written in {elapsed:.1f}s")


class Dataset(data.Dataset):
    def __init__(
        self,
        list_IDs,
        labels,
        DataPath,
        RecLen,
        DecNum=5,
        fft_Samp=256,
        Im_3D=False,
        cache_dir=None,
    ):
        self.labels = labels
        self.list_IDs = list_IDs
        self.DataPath = DataPath
        self.RecLen = RecLen
        self.fft_Samp = fft_Samp
        self.Im_3D = Im_3D

        self.NFCC_Num = 23
        self.TimeSamp = 1024
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        if self.cache_dir is not None:
            npy_path = os.path.join(self.cache_dir, ID + ".npy")
            if os.path.isfile(npy_path):
                Im_np = np.load(npy_path, allow_pickle=False)
                Im = torch.from_numpy(Im_np).float()
            else:
                Im = torch.from_numpy(
                    _mfcc_image_from_wav(
                        self.DataPath + ID, self.NFCC_Num, self.TimeSamp
                    )
                ).float()
        else:
            Im = torch.from_numpy(
                _mfcc_image_from_wav(self.DataPath + ID, self.NFCC_Num, self.TimeSamp)
            ).float()

        y = torch.from_numpy(self.labels[ID]).float()
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
Labels_noisy = pd.read_csv("../input/train_noisy.csv")
WavPath_noisy = "../input/train_noisy/"

Fils_noisy = Labels_noisy["fname"].reset_index(drop=True)

OneHot_noisy = np.zeros((len(Fils_noisy), NumClasses), dtype=np.float32)
for i, labs in enumerate(Labels_noisy["labels"].astype(str).values):
    for lab in labs.split(","):
        j = label_to_idx.get(lab)
        if j is not None:
            OneHot_noisy[i, j] = 1.0

train_x = pd.concat(
    [train_x_cur.reset_index(drop=True), Fils_noisy], axis=0
).reset_index(drop=True)
train_y = np.concatenate([train_y_cur, OneHot_noisy], axis=0).astype(np.float32)

labelsDict_train_cur = dict(zip(train_x_cur.tolist(), train_y_cur))
labelsDict_train_noisy = dict(zip(Fils_noisy.tolist(), OneHot_noisy))
labelsDict_train = {}
labelsDict_train.update(labelsDict_train_cur)
labelsDict_train.update(labelsDict_train_noisy)

labelsDict_val = dict(zip(val_x.tolist(), val_y))

RecLen = 176400

cache_root = "../working/mfcc_cache"
cache_cur = os.path.join(cache_root, "train_curated")
cache_noisy = os.path.join(cache_root, "train_noisy")
cache_val = os.path.join(cache_root, "val_curated")
cache_test = os.path.join(cache_root, "test")

t_cache = time.time()
precompute_mfcc_cache(train_x_cur.tolist(), WavPath, cache_cur)
precompute_mfcc_cache(Fils_noisy.tolist(), WavPath_noisy, cache_noisy)
precompute_mfcc_cache(val_x.tolist(), WavPath, cache_val)
print(f"MFCC precompute (train/val) total time: {time.time()-t_cache:.1f}s")

common_workers = min(8, max(2, (os.cpu_count() or 2) // 2))
train_params = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": common_workers,
    "pin_memory": train_on_gpu,
    "persistent_workers": True if common_workers > 0 else False,
    "prefetch_factor": 2 if common_workers > 0 else None,
}

val_params = {
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": common_workers,
    "pin_memory": train_on_gpu,
    "persistent_workers": True if common_workers > 0 else False,
    "prefetch_factor": 2 if common_workers > 0 else None,
}

training_set_cur = Dataset(
    train_x_cur.tolist(), labelsDict_train, WavPath, RecLen, cache_dir=cache_cur
)
training_set_noisy = Dataset(
    Fils_noisy.tolist(), labelsDict_train, WavPath_noisy, RecLen, cache_dir=cache_noisy
)

training_set = data.ConcatDataset([training_set_cur, training_set_noisy])
training_generator = data.DataLoader(training_set, **train_params)

val_set = Dataset(val_x.tolist(), labelsDict_val, WavPath, RecLen, cache_dir=cache_val)
val_generator = data.DataLoader(val_set, **val_params)

print(
    "Train batches:",
    len(training_generator),
    "| Val batches:",
    len(val_generator),
    "| num_workers:",
    common_workers,
)



## === cell 13
epochs = 2

model.train()
for ep in range(epochs):
    running_loss = 0.0
    for xb, yb, _ in training_generator:
        xb = xb.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, dtype=torch.float32, non_blocking=True)

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
            xb = xb.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device, dtype=torch.float32, non_blocking=True)
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

t_cache = time.time()
precompute_mfcc_cache(test_files, WavPath_test, cache_test)
print(f"MFCC precompute (test) total time: {time.time()-t_cache:.1f}s")

test_params = {
    "batch_size": 32,  # safe increase; does not change predictions, only throughput
    "shuffle": False,  # critical for correct fname alignment
    "num_workers": common_workers,
    "pin_memory": train_on_gpu,
    "persistent_workers": True if common_workers > 0 else False,
    "prefetch_factor": 2 if common_workers > 0 else None,
}

test_set = Dataset(test_files, dummy_labels, WavPath_test, RecLen, cache_dir=cache_test)
test_generator = data.DataLoader(test_set, **test_params)



## === cell 15
model.eval()
preds = np.zeros((len(test_files), NumClasses), dtype=np.float32)

offset = 0
with torch.no_grad():
    for xb, _, ids in test_generator:
        bs = xb.shape[0]
        xb = xb.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
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
