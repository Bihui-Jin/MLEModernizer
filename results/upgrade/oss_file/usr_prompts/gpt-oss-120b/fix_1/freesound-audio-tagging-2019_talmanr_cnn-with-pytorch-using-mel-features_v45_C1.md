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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import numpy as np 
import pandas as pd
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
from torchvision import datasets, models, transforms
import torch.optim as optim

train_on_gpu=torch.cuda.is_available()



## === cell 1
Labels = pd.read_csv("../input/train_curated.csv")
WavPath = "../input/train_curated/"
Fils = os.listdir(WavPath)
sound, sample_rate = torchaudio.load(WavPath+Fils[5])
ipd.Audio(data=sound[0,:],rate=sample_rate) # load a local WAV file


## === cell 3
x, sr = librosa.load(WavPath+Fils[4])

plt.figure(figsize=(14, 5))



librosa.display.waveplot(x, sr=sr)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
Xdb.shape

S = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=23)
delta2_mfcc = librosa.feature.delta(MFCC, order=2)

librosa.display.specshow(delta2_mfcc)
plt.colorbar()
plt.tight_layout()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4046069515.py in <cell line: 0>()
      5 
      6 
----> 7 librosa.display.waveplot(x, sr=sr)
      8 X = librosa.stft(x)
      9 Xdb = librosa.amplitude_to_db(abs(X))

AttributeError: module 'librosa.display' has no attribute 'waveplot'

## === cell 4
x


## === cell 5
FilesS = np.zeros(len(Fils))
for i,File in enumerate(Fils):
    FilesS[i] = os.path.getsize(WavPath+File)

plt.figure(figsize=(20,8))
plt.hist(FilesS,bins=50)


## === cell 6
Fils_2 = Labels['fname']
Fils_2

Class =set(Labels['labels'])
All_class= [] 
for i in Class:
    for j  in i.split(','):
        All_class.append(j)

All_class = set(All_class)

NumClasses = len(All_class)
OneHot_All = np.zeros((len(Fils_2),NumClasses))

for  i,file in enumerate(Labels['labels']):
    for j,clas in enumerate(All_class):
        OneHot_All[i,j] = np.int(clas in file)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3354820903.py in <cell line: 0>()
     15 for  i,file in enumerate(Labels['labels']):
     16     for j,clas in enumerate(All_class):
---> 17         OneHot_All[i,j] = np.int(clas in file)

/usr/local/lib/python3.11/dist-packages/numpy/__init__.py in __getattr__(attr)
    322 
    323         if attr in __former_attrs__:
--> 324             raise AttributeError(__former_attrs__[attr])
    325 
    326         if attr == 'testing':

AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

## === cell 7

print(NumClasses)
split_frac = 0.92
batch_size = 32

split_idx = int(len(Fils)*split_frac)
split_idx1 = int(batch_size*np.floor(split_idx/batch_size))
split_idx2 = int(batch_size*np.floor( (len(Fils) - split_idx1)/batch_size ))
train_x, val_x = Fils_2[:split_idx1], Fils_2[split_idx1:split_idx1+split_idx2]
train_y, val_y = OneHot_All[:split_idx1,:], OneHot_All[split_idx1:split_idx1+split_idx2,:]
print(len(train_x)/batch_size, len(val_x)/batch_size )


## === cell 8
torch.zeros((2,3)).type(torch.FloatTensor)


## === cell 9
from scipy.io import wavfile
from librosa.feature import mfcc
class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels,DataPath,RecLen,DecNum=5,fft_Samp= 256,Im_3D= False):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.DataPath = DataPath
        self.RecLen = RecLen # length of most records
        self.fft_Samp = fft_Samp 
        self.Im_3D = Im_3D
        
        self.NFCC_Num = 23
        self.TimeSamp = 1024
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        fs, data = wavfile.read(self.DataPath + ID)
        S = librosa.feature.melspectrogram(np.float32(data), sr=fs, n_mels=128)

        log_S = librosa.power_to_db(S, ref=np.max)
        MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=23)
        LabelOut = torch.from_numpy(self.labels[ID]).double()
        
        
        Im = torch.zeros((self.NFCC_Num,self.TimeSamp)).type(torch.FloatTensor)
        
        if MFCC.shape[1] > self.TimeSamp :
            Im = torch.from_numpy(MFCC[:,:self.TimeSamp]).float()
        else: 
            Im[:,int(self.TimeSamp/2-int(MFCC.shape[1]/2)) : int(self.TimeSamp/2-int(MFCC.shape[1]/2)+MFCC.shape[1] ) ] = torch.from_numpy(MFCC).float()
        
        
        

            
       
                
        


        return Im, LabelOut,ID


## === cell 10
class CnnAudioNet(nn.Module):
    def __init__(self,NumClasses):
        super(CnnAudioNet,self).__init__()
        self.NumClasses = NumClasses
        self.Fc_features = 128
        self.C1 = nn.Conv2d(1,8,3,padding=1)
        self.C2 = nn.Conv2d(8,16,3,padding=1)
        self.C3 = nn.Conv2d(16,16,3,padding=1)
        self.C4 = nn.Conv2d(16,32,3,padding=1)
        self.C5 = nn.Conv2d(32,32,3,padding=1)
        
        self.BN1 = nn.BatchNorm2d(8)
        self.BN2 = nn.BatchNorm2d(16)
        self.maxpoll = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d((1,2),(1,2))
        
        
        self.fc1 = nn.Linear(32*23* 16,128)
        self.fc2 = nn.Linear(128,self.NumClasses )
        self.dropout = nn.Dropout(0.25)
        self.Bat1 = nn.BatchNorm1d(128)

        
        
    def forward(self,x):
        x = self.maxpool2(F.relu(self.BN1(self.C1(x))))
        x = self.maxpool2(F.relu(self.BN2(self.C2(x))))
        x = self.maxpool2(F.relu(self.BN2(self.C3(x))))
        x = self.maxpool2(F.relu(self.C4(x)))
        x = self.maxpool2(F.relu(self.C5(x)))
        x = self.maxpool2(F.relu(self.C5(x)))
        x = x.view(-1, 32*23* 16)
        x = self.dropout(self.Bat1(self.fc1(x)))
        x = self.fc2(x)
        return x
        


## === cell 11
from torchvision import datasets, models, transforms




class CnnTransferNet(nn.Module):
    def __init__(self):
        super(CnnTransferNet,self).__init__()
        
        self.vgg =  models.vgg16_bn().cuda()
        for param in self.vgg.features.parameters():
            param.require_grad = False

        
        self.fc1 = nn.Linear(1000,128)
        self.fc2 = nn.Linear(128,NumClasses)
        self.dropout = nn.Dropout(0.25)

        
        
    def forward(self,x):
        Features = self.dropout(self.vgg(x))
        Features = F.relu(self.fc1(Features))
        Features = self.fc2(Features)
        return Features


## === cell 12
model = CnnAudioNet(NumClasses)
if train_on_gpu:
    model.cuda()
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.005, amsgrad=False)# specify optimizer


## === cell 13
labelsDict_train = dict(zip(train_x,train_y))
labelsDict_val = dict(zip(val_x,val_y))

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 10}

RecLen = 176400

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

training_set = Dataset(train_x, labelsDict_train,WavPath,RecLen,transforms.Compose(normalize))
training_generator = data.DataLoader(training_set, **params)

val_set = Dataset(val_x.reset_index(drop=True), labelsDict_val,WavPath,RecLen,transforms.Compose(normalize))
val_generator = data.DataLoader(val_set, **params)


## === cell 14
import time
start_time = time.time()

n_epochs = 12

valid_loss_min = np.Inf # track change in validation loss
print("Start training:")
idx = 0 
for epoch in range(1, n_epochs+1):

    train_loss = 0.0

    
    model.train()
    for dataBatch, target,_ in training_generator:
        
        idx+=1
        if train_on_gpu:
            dataBatch, target = dataBatch.unsqueeze(1).float().cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(dataBatch)
        loss = criterion(output,target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*dataBatch.size(0)
        _,pred = torch.max(output,1)
        Correct = torch.sum(torch.pow(torch.sigmoid(output)-target.float(),2))#

        
        
    print(idx)
    model.eval()
    valid_loss = 0.0
    SumCorrectVal = 0
    TotVal =0 
    for dataBatch_v, target ,_ in val_generator  :

        if train_on_gpu:
            dataBatch_v, target = dataBatch_v.unsqueeze(1).float().cuda(),target.cuda()
        output = model(dataBatch_v)
        loss = criterion(output,target.float())

        output.shape
        _,pred = torch.max(output,1)
        Correct = torch.sum(pred ==torch.squeeze(torch.argmax(target,dim=-1)))
        SumCorrectVal += Correct
        valid_loss += loss.item()*dataBatch.size(0)
        TotVal += dataBatch.size(0)

    train_loss = train_loss/len(training_generator.dataset)
    valid_loss = valid_loss/len(val_generator.dataset)

    print('Epoch: {} \t Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    print('Epoch: {} \t Validation Correct: {:d}  Out of: {:d}, Accurcy: {:d}%'.format(
        epoch, SumCorrectVal,TotVal,SumCorrectVal.cpu().detach().numpy()/np.float32(TotVal)))
    print("--- %s seconds ---" % (time.time() - start_time))


         


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1667014267.py in <cell line: 0>()
     18     ###################
     19     model.train()
---> 20     for dataBatch, target,_ in training_generator:
     21 
     22         idx+=1

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

TypeError: Caught TypeError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_11/1342097330.py", line 24, in __getitem__
    S = librosa.feature.melspectrogram(np.float32(data), sr=fs, n_mels=128)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: melspectrogram() takes 0 positional arguments but 1 positional argument (and 1 keyword-only argument) were given


## === cell 15
plt.plot(target[5,:].detach().cpu().numpy())
plt.plot(torch.sigmoid(output[5,:]).detach().cpu().numpy())


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3870437432.py in <cell line: 0>()
----> 1 plt.plot(target[5,:].detach().cpu().numpy())
      2 plt.plot(torch.sigmoid(output[5,:]).detach().cpu().numpy())

NameError: name 'target' is not defined

## === cell 16
WavPath_test =  '../input/test/'

Fils_test = os.listdir(WavPath_test)


one_hot_test = OneHot_All = np.zeros((len(Fils_test),NumClasses))

labelsDict = dict(zip(Fils_test,one_hot_test))

params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 8}

test_set = Dataset(Fils_test, labelsDict,WavPath_test,RecLen,transforms.Compose(normalize))
test_generator = data.DataLoader(test_set, **params)


## === cell 17
model.eval()
SoftM = torch.nn.Softmax()
Output_all = [] 
BatchRecs_all = [] 
with torch.no_grad():                   # operations inside don't track history

    for dataBatch, Lab,BatchRecs in test_generator:
        if train_on_gpu:
            dataBatch, target = dataBatch.unsqueeze(1).float().cuda(), target.cuda()
        output = model(dataBatch)
        outP = torch.sigmoid(output)
        Output_all.append(outP)
        BatchRecs_all.append(BatchRecs)

  


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2294827566.py in <cell line: 0>()
      5 with torch.no_grad():                   # operations inside don't track history
      6 
----> 7     for dataBatch, Lab,BatchRecs in test_generator:
      8         if train_on_gpu:
      9             dataBatch, target = dataBatch.unsqueeze(1).float().cuda(), target.cuda()

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

TypeError: Caught TypeError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_11/1342097330.py", line 24, in __getitem__
    S = librosa.feature.melspectrogram(np.float32(data), sr=fs, n_mels=128)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: melspectrogram() takes 0 positional arguments but 1 positional argument (and 1 keyword-only argument) were given


## === cell 18
Dataout = np.zeros((4*len(Output_all)-3,80))
Names = []
for i in range(len(Output_all)):
    Dataout[i*4:(i+1)*4,:] = Output_all[i].cpu().detach().numpy()
    Names.append(BatchRecs_all[i][0])
    if i<840:
        Names.append(BatchRecs_all[i][1])
        Names.append(BatchRecs_all[i][2])
        Names.append(BatchRecs_all[i][3])
    


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2797307800.py in <cell line: 0>()
----> 1 Dataout = np.zeros((4*len(Output_all)-3,80))
      2 Names = []
      3 for i in range(len(Output_all)):
      4     Dataout[i*4:(i+1)*4,:] = Output_all[i].cpu().detach().numpy()
      5     Names.append(BatchRecs_all[i][0])

ValueError: negative dimensions are not allowed

## === cell 19
Cl = list(All_class)

Output_all_DF =pd.DataFrame(columns=Cl,data = Dataout)
Output_all_DF['fname'] = Names


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3639822742.py in <cell line: 0>()
      2 #Cl.append('fname')
      3 
----> 4 Output_all_DF =pd.DataFrame(columns=Cl,data = Dataout)
      5 Output_all_DF['fname'] = Names

NameError: name 'Dataout' is not defined

## === cell 20
Output_all_DF.head()
Output_all_DF.to_csv('submission.csv', index=False)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3422454894.py in <cell line: 0>()
----> 1 Output_all_DF.head()
      2 Output_all_DF.to_csv('submission.csv', index=False)

NameError: name 'Output_all_DF' is not defined
