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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, librosa, librosa.display, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils import data
from scipy.io import wavfile

train_on_gpu = torch.cuda.is_available()
print("GPU:", train_on_gpu)
print("Files in input:", os.listdir("../input"))



## === cell 1
labels_df = pd.read_csv("../input/train_curated.csv")
wav_path = "../input/train_curated/"
file_list = os.listdir(wav_path)

sound, sr = (
    torch.load(wav_path + file_list[5]) if False else (None, None)
)  # placeholder to keep original intent



## === cell 2
x, sr = librosa.load(wav_path + file_list[4])
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)  # replaces deprecated waveplot
plt.title("Waveform")
plt.show()



## === cell 3
unique_labels = set()
for lbl in labels_df["labels"]:
    for cl in lbl.split(","):
        unique_labels.add(cl)
all_classes = sorted(list(unique_labels))
num_classes = len(all_classes)

filenames = labels_df["fname"].values
one_hot = np.zeros((len(filenames), num_classes), dtype=int)
for i, lbl in enumerate(labels_df["labels"]):
    present = set(lbl.split(","))
    for j, cl in enumerate(all_classes):
        one_hot[i, j] = int(cl in present)



## === cell 4
split_frac = 0.92
batch_size = 32

split_idx = int(len(filenames) * split_frac)
split_idx = int(batch_size * np.floor(split_idx / batch_size))  # make it batch‑aligned
train_x, val_x = filenames[:split_idx], filenames[split_idx:]
train_y, val_y = one_hot[:split_idx], one_hot[split_idx:]

print(
    f"Train batches: {len(train_x)//batch_size}, Val batches: {len(val_x)//batch_size}"
)




## === cell 5
class AudioDataset(data.Dataset):
    def __init__(self, ids, label_dict, data_path, rec_len):
        self.ids = ids
        self.label_dict = label_dict
        self.data_path = data_path
        self.rec_len = rec_len
        self.n_mfcc = 23
        self.time_samp = 1024

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]
        fs, wav = wavfile.read(self.data_path + file_id)
        wav = np.float32(wav)
        S = librosa.feature.melspectrogram(y=wav, sr=fs, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=self.n_mfcc)

        img = torch.zeros((self.n_mfcc, self.time_samp), dtype=torch.float32)
        if mfcc.shape[1] >= self.time_samp:
            img = torch.from_numpy(mfcc[:, : self.time_samp]).float()
        else:
            start = (self.time_samp - mfcc.shape[1]) // 2
            img[:, start : start + mfcc.shape[1]] = torch.from_numpy(mfcc).float()

        label = torch.from_numpy(self.label_dict[file_id]).float()
        return img, label, file_id




## === cell 6
class CnnAudioNet(nn.Module):
    def __init__(self, num_classes):
        super(CnnAudioNet, self).__init__()
        self.fc_features = 128
        self.C1 = nn.Conv2d(1, 8, 3, padding=1)
        self.C2 = nn.Conv2d(8, 16, 3, padding=1)
        self.C3 = nn.Conv2d(16, 16, 3, padding=1)
        self.C4 = nn.Conv2d(16, 32, 3, padding=1)
        self.C5 = nn.Conv2d(32, 32, 3, padding=1)

        self.BN1 = nn.BatchNorm2d(8)
        self.BN2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d((1, 2), (1, 2))

        self.fc1 = nn.Linear(32 * 23 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.bn_fc = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.maxpool2(F.relu(self.BN1(self.C1(x))))
        x = self.maxpool2(F.relu(self.BN2(self.C2(x))))
        x = self.maxpool2(F.relu(self.BN2(self.C3(x))))
        x = self.maxpool2(F.relu(self.C4(x)))
        x = self.maxpool2(F.relu(self.C5(x)))
        x = self.maxpool2(F.relu(self.C5(x)))
        x = x.view(-1, 32 * 23 * 16)
        x = self.dropout(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        return x




## === cell 7
model = CnnAudioNet(num_classes)
if train_on_gpu:
    model = model.cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, amsgrad=False)



## === cell 8
train_labels = dict(zip(train_x, train_y))
val_labels = dict(zip(val_x, val_y))

params = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": 4,
    "pin_memory": True,
}
rec_len = 176400  # not used directly but kept for compatibility

train_dataset = AudioDataset(train_x, train_labels, wav_path, rec_len)
val_dataset = AudioDataset(val_x, val_labels, wav_path, rec_len)

train_loader = data.DataLoader(train_dataset, **params)
val_loader = data.DataLoader(val_dataset, **params)



## === cell 9
import time

start_time = time.time()
n_epochs = 12

for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y, _ in train_loader:
        batch_x = batch_x.unsqueeze(1)  # add channel dimension
        if train_on_gpu:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x = batch_x.unsqueeze(1)
            if train_on_gpu:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.numel()

    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(
        f"Epoch {epoch:02d} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val Acc {acc:.2f}%"
    )
    print(f"Elapsed {time.time() - start_time:.1f}s")



## === cell 10
test_path = "../input/test/"
test_files = os.listdir(test_path)
dummy_labels = dict(
    zip(test_files, np.zeros((len(test_files), num_classes), dtype=int))
)

test_dataset = AudioDataset(test_files, dummy_labels, test_path, rec_len)
test_params = {"batch_size": 4, "shuffle": False, "num_workers": 4, "pin_memory": True}
test_loader = data.DataLoader(test_dataset, **test_params)



## === cell 11
model.eval()
all_outputs = []
all_names = []

with torch.no_grad():
    for batch_x, _, batch_names in test_loader:
        batch_x = batch_x.unsqueeze(1)
        if train_on_gpu:
            batch_x = batch_x.cuda()
        logits = model(batch_x)
        probs = torch.sigmoid(logits).cpu()
        all_outputs.append(probs)
        all_names.extend(batch_names)

preds_array = torch.cat(
    all_outputs, dim=0
).numpy()  # shape (num_test_files, num_classes)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/3955044267.py in <cell line: 0>()
      4 
      5 with torch.no_grad():
----> 6     for batch_x, _, batch_names in test_loader:
      7         batch_x = batch_x.unsqueeze(1)
      8         if train_on_gpu:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1453                 data = self._task_info.pop(self._rcvd_idx)[1]
   1454                 self._rcvd_idx += 1
-> 1455                 return self._process_data(data)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0

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

IsADirectoryError: Caught IsADirectoryError in DataLoader worker process 2.
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
  File "/tmp/ipykernel_55/193428402.py", line 15, in __getitem__
    fs, wav = wavfile.read(self.data_path + file_id)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/scipy/io/wavfile.py", line 674, in read
    fid = open(filename, 'rb')
          ^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '../input/test/test'


## === cell 12
submission_df = pd.DataFrame(preds_array, columns=all_classes)
submission_df.insert(0, "fname", all_names)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission_df.shape}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1730080008.py in <cell line: 0>()
      1 # Build submission DataFrame matching the sample format
----> 2 submission_df = pd.DataFrame(preds_array, columns=all_classes)
      3 submission_df.insert(0, "fname", all_names)
      4 submission_path = "submission.csv"
      5 submission_df.to_csv(submission_path, index=False)

NameError: name 'preds_array' is not defined
