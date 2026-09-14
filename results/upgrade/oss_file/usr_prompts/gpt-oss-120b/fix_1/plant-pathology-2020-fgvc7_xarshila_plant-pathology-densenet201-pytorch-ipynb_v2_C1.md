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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.8589237905921322

# 6. Current score

0.46757

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import random
import gc

## === cell 1
DATA_DIR = Path("../input/plant-pathology-2020-fgvc7")
CLASS_NAMES = np.array(["healthy", "multiple_diseases", "rust", "scab"])
BATCH_SIZE = 8
IMAGE_SIZE = (512, 512)
TEST_SPLIT = 0.2

## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"

## === cell 3
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.backbone = torchvision.models.densenet201(pretrained=False)
        self.fc = nn.Linear(1000, 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

## === cell 4
model = MyModel().to(device)

## === cell 5
model.load_state_dict(torch.load("../input/plant-pathology-models/acc_98_size_512.pth"))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/512808091.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load("../input/plant-pathology-models/acc_98_size_512.pth"))

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/plant-pathology-models/acc_98_size_512.pth'

## === cell 6
model.eval()

## === cell 7
class PlantPathologyTestDataset():
    def __init__(self, root, data_df, transform = None, preload=False):
        self.data_df = data_df
        self.images = None
        self.transform = transform
        if preload:
            self.images = []
            for idx in range(len(data_df)):
                image_path = DATA_DIR/"images"/(data_df["image_id"][idx] + ".jpg")
                image = Image.open(str(image_path))
                image = image.resize(IMAGE_SIZE)
                self.images.append(image.copy())
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.images is None:
            data_df = self.data_df
            image_path = DATA_DIR/"images"/(data_df["image_id"][idx] + ".jpg")
            image = Image.open(str(image_path))
            image = image.resize(IMAGE_SIZE)
        else:
            image = self.images[idx]
            
        if self.transform:
            image = self.transform(image)
        return self.data_df["image_id"][idx], image

## === cell 8
data_df = pd.read_csv(DATA_DIR/"test.csv")
submission_data = PlantPathologyTestDataset(DATA_DIR, data_df, transform=transforms.ToTensor(), preload=True)
submission_loader = DataLoader(submission_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

## === cell 9
image_ids = []
h, m, r, s = [], [], [], []
with torch.no_grad():
  for batch_idx, (image_id_batch, data) in enumerate(submission_loader):
        data = data.to(device)   
        h_fliped_data = data.flip(2)
        output2 = model(h_fliped_data)
        del h_fliped_data
        gc.collect()
        v_fliped_data = data.flip(3)
        output3 = model(v_fliped_data)
        del v_fliped_data
        gc.collect()
        pred_batch = (model(data) +  output2 + output3)/3
        del data
        gc.collect()
        for image_id, pred in zip(image_id_batch, pred_batch):
          res = np.exp(pred.cpu().numpy())
          h.append(res[0])
          m.append(res[1])
          r.append(res[2])
          s.append(res[3])
          image_ids.append(image_id)
        

## === cell 10
sub = pd.DataFrame({"image_id":image_ids, "healthy": h, 'multiple_diseases': m, "rust": r, "scab": s})

## === cell 11
sub.to_csv('submission.csv', index=False)
sub.head()
