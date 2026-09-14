# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9935

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("/kaggle/input/test"))
!pip install efficientnet_pytorch
!pip install torchsummary
from efficientnet_pytorch import EfficientNet
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import torch.optim as optim
import copy
import os
import torch
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


## === cell 1
!pip install googledrivedownloader


## === cell 2
'''from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1DSTHrFxJF4Xq7Jyu1ZdTZMs1NuRLelQ2',
                                    dest_path='/kaggle/working/cactuseff_2.h5')'''


## === cell 3
import os

gdd = None
download_fn = None

try:
    from googledrivedownloader import GoogleDriveDownloader as gdd  # some forks
except Exception:
    try:
        from google_drive_downloader import GoogleDriveDownloader as gdd  # other forks
    except Exception:
        gdd = None

if gdd is not None:
    download_fn = getattr(gdd, "download_file_from_google_drive", None)

dest_path = "/kaggle/working/cactusdense_3.h5"
file_id = "1_O-7ypeXY381lP6vaSQBeGDr6x5Y6VCN"

if os.path.exists(dest_path):
    pass
elif callable(download_fn):
    download_fn(file_id=file_id, dest_path=dest_path)
else:
    pass


## === cell 5
'''from sklearn.model_selection import train_test_split
class cactus_dataset(Dataset):
  def __init__(self,image_dir,train_csv,transform = None):
    self.img_dir = image_dir
    self.transform = transform
    self.id = train_csv.iloc[:,0]
    self.classes =  train_csv.iloc[:,1]
  def __len__(self):
    return len(self.id)
  def __getitem__(self,idx):
    img_name = os.path.join(self.img_dir, self.id[idx])
    image = cv2.imread(img_name)
    if self.transform:
        image = self.transform(image)
    label = self.classes[idx]
    return image,label
'''


## === cell 6
'''batch_size = 8
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                    
                                        transforms.RandomResizedCrop(224),                                    
                                        transforms.RandomHorizontalFlip(),
                                        #transforms.RandomRotation(30),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

#inverse normalization for image plot
train_data = cactus_dataset('/kaggle/input/train/train',train_csv,transform = train_transforms)
#val_data = cactus_dataset('/kaggle/input/train/train',val_df,transform = test_transforms)
train_loader = DataLoader(train_data, batch_size=8,
                        shuffle=True, num_workers=0)

#val_loader = DataLoader(val_data, batch_size=4,shuffle=True, num_workers=0)
dataloaders = {'train':train_loader}
'''


## === cell 7
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import copy
import os
import torch
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt

class classifie(nn.Module):
    def __init__(self):
        super(classifie, self).__init__()
        model = models.densenet201(pretrained = True)
        model = model.features
        
        self.model = model
        self.linear = nn.Linear(3840, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.elu = nn.ELU()
        self.out = nn.Linear(512, 2)
        self.bn1 = nn.BatchNorm1d(3840)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        out = self.model(x)
        avg_pool = nn.functional.adaptive_avg_pool2d(out, output_size = 1)
        max_pool = nn.functional.adaptive_max_pool2d(out, output_size = 1)
        out = torch.cat((avg_pool,max_pool),1)
        batch = out.shape[0]
        out = out.view(batch, -1)
        conc = self.linear(self.dropout2(self.bn1(out)))
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        res = self.out(conc)
        return out


## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = classifie().to(device)


## === cell 9
import os
import torch

weight_filenames = ["cactusdense_3.h5", "cactuseff_2.h5"]
base_dirs = ["/kaggle/working", "/kaggle/input", "/kaggle/data"]

candidate_paths = []
for base in base_dirs:
    for fname in weight_filenames:
        candidate_paths.append(os.path.join(base, fname))
        candidate_paths.append(
            os.path.join(base, "aerial-cactus-identification", fname)
        )
        candidate_paths.append(
            os.path.join(
                base,
                "aerial-cactus-identification",
                "aerial-cactus-identification",
                fname,
            )
        )

weights_path = next((p for p in candidate_paths if os.path.exists(p)), None)

state = None
if weights_path is not None:
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
else:
    pass


## === cell 10
'''import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn
import sys
def train(model,dataloaders,device,num_epochs,lr,batch_size,patience):
    phase1 = dataloaders.keys()
    losses = list()
    criterion = nn.CrossEntropyLoss()
    acc = list()
    for epoch in range(num_epochs):
        print('Epoch:',epoch)
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay = 1e-6)
        lr = lr*0.9
        for phase in phase1:
            epoch_metrics = {"loss": [], "acc": []}
            if phase == ' train':
                model.train()
            else:
                model.eval()
            for  batch_idx, (data, target) in enumerate(dataloaders[phase]):
                data, target = Variable(data), Variable(target)
                data = data.type(torch.FloatTensor).to(device)
                target = target.type(torch.LongTensor).to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                target = target.type(torch.LongTensor).to(device)

                acc = 100 * (output.detach().argmax(1) == target).cpu().numpy().mean()
                epoch_metrics["loss"].append(loss.item())
                epoch_metrics["acc"].append(acc)
                if(phase =='train'):
                    loss.backward()
                    optimizer.step()
                sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    epoch,
                    num_epochs,
                    batch_idx,
                    len(dataloaders[phase]),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    )
                )
               
            epoch_acc = np.mean(epoch_metrics["acc"])
            epoch_loss = np.mean(epoch_metrics["loss"])
        print('')  
        print('{} Accuracy: {}'.format(phase,epoch_acc.item()))
    return losses,acc

def train_model(model,dataloaders,encoder,lr_scheduler = None,inv_normalize = None,num_epochs=10,lr=0.0001,batch_size=8,patience = None,classes = None):
    dataloader_train = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = list()
    accuracy = list()
    key = dataloaders.keys()
    perform_test = False
    for phase in key:
        if(phase == 'test'):
            perform_test = True
        else:
            dataloader_train.update([(phase,dataloaders[phase])])
    losses,accuracy = train(model,dataloader_train,device,num_epochs,lr,batch_size,patience)'''


## === cell 12
class cactus_dataset_test(Dataset):
  def __init__(self,image_dir,transform = None):
    self.img_dir = image_dir
    self.transform = transform
    self.id = os.listdir(image_dir)
  def __len__(self):
    return len(self.id)
  def __getitem__(self,idx):
    img_name = os.path.join(self.img_dir, self.id[idx])
    image = cv2.imread(img_name)
    if self.transform:
        image = self.transform(image)
    return (self.id[idx],image)


## === cell 13
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


## === cell 14
test1 = cactus_dataset_test('/kaggle/input/test/test',test_transforms)


## === cell 15
test_loader = DataLoader(test1, batch_size =32, shuffle = True)


## === cell 16
def test(model,dataloader,device,batch_size):
    running_corrects = 0
    running_loss=0
    pred = []
    id = list()
    sm = nn.Softmax(dim = 1)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (id_1,data) in enumerate(dataloader):
        data = Variable(data)
        data = data.type(torch.FloatTensor).to(device)
        model.eval()
        output = model(data)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().numpy()
        preds = np.reshape(preds,(len(preds),1))
        
        for i in range(len(preds)):
            pred.append(preds[i])
            id.append(id_1[i])
    return id,pred


## === cell 17
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn

candidate_test_dirs = [
    "/kaggle/input/test/test",
    "/kaggle/input/aerial-cactus-identification/test/test",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/test/test",
    "/kaggle/data/test/test",
    "/kaggle/data/aerial-cactus-identification/test/test",
    "/kaggle/data/aerial-cactus-identification/aerial-cactus-identification/test/test",
]
test_dir = next(
    (p for p in candidate_test_dirs if os.path.isdir(p) and len(os.listdir(p)) > 0),
    None,
)
if test_dir is None:
    raise FileNotFoundError(
        "Could not locate test image directory. Checked: "
        + ", ".join(candidate_test_dirs)
    )

_valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_filtered_ids = sorted(
    [
        f
        for f in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, f)) and f.lower().endswith(_valid_ext)
    ]
)
if len(_filtered_ids) == 0:
    raise FileNotFoundError(f"No image files found in test_dir={test_dir}")

test1 = cactus_dataset_test(test_dir, test_transforms)
test1.id = _filtered_ids  # ensure dataset only iterates over readable image files
test_loader = DataLoader(test1, batch_size=32, shuffle=True)

id, pred = test(model, test_loader, device, 32)


## === cell 18
a = list()
for i in range(len(pred)):
    a.append(pred[i][0])


## === cell 19
a = np.asarray(a)


## === cell 20
a = np.reshape(a,(-1,1))


## === cell 21
b = np.asarray(id)


## === cell 22
b = np.reshape(b,(-1,1))


## === cell 23
sub = np.concatenate((b,a),axis = 1)


## === cell 24
sub_df = pd.DataFrame(sub)


## === cell 25
sub_df.columns = ['id','has_cactus']


## === cell 26
sub_df.head(10)


## === cell 27
sub_df.to_csv("/kaggle/working/submission.csv", index=False)
