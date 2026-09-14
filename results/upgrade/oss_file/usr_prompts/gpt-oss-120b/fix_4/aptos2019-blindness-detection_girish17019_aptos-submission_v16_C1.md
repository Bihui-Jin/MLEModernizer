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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.4200751392428454

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented fixes to handle missing model checkpoint and ensure a valid submission is generated. The model now loads pretrained EfficientNet‑B0 weights, adjusts the classifier for 5 classes, and gracefully initializes when the checkpoint file is absent. Inference uses softmax + argmax to produce class predictions, and the submission CSV is correctly written.'
- What this solution (achieved 0.0) has done: 'Implemented two lightweight adjustments to move the QWK score toward the target while keeping the core model unchanged:  
1. **Removed random augmentations during inference** – test transforms now contain only deterministic operations, preventing stochastic image changes that can degrade predictions.  
2. **Used expected‑value rounding instead of arg‑max** – the model’s softmax outputs are converted to a weighted average class index and rounded, which often aligns better with the quadratic weighted kappa metric.

These changes are minimal, preserve the original architecture, and ensure a valid `submission.csv` is produced.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import cv2 as cv
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import albumentations as A
from torchvision import models
from copy import deepcopy




## === cell 1
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Device : {device.upper()}")




## === cell 2
TEST_PATH = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG = "../input/aptos2019-blindness-detection/test_images"
SAMPLE_SUB_PATH = "../input/aptos2019-blindness-detection/sample_submission.csv"




## === cell 3
class AptosDataset(Dataset):
    def __init__(self, data_path, img_dir, name, transforms, resize=(512, 512)):
        self.data_path = data_path
        self.img_dir = img_dir
        self.resize = resize
        self.transforms = transforms
        self.df = pd.read_csv(self.data_path)
        self.name = name

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_name = self.df["id_code"][idx] + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        img_vector = cv.imread(img_path)
        if img_vector is None:
            img_vector = np.zeros((self.resize[0], self.resize[1], 3), dtype=np.uint8)
        if self.transforms:
            transformed = self.transforms(image=img_vector)
            img_vector = transformed["image"]
        if self.resize:
            img_vector = cv.resize(img_vector, self.resize)
            img_vector = img_vector.reshape((3, self.resize[0], self.resize[1]))
        img_tensor = torch.tensor(img_vector, dtype=torch.float32) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor

    def show(self, idx):
        img_tensor = self.__getitem__(idx)
        plt.imshow(img_tensor.permute(1, 2, 0).numpy())
        plt.title("Image")
        plt.show()




## === cell 4
class LogisticCumulativeLink(nn.Module):
    def __init__(self, num_classes: int, init_cutpoints: str = "ordered") -> None:
        assert num_classes > 2, "Only use this model if you have 3 or more classes"
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        if init_cutpoints == "ordered":
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            self.cutpoints = nn.Parameter(cutpoints)
        elif init_cutpoints == "random":
            cutpoints = torch.rand(self.num_classes - 1).sort()[0]
            self.cutpoints = nn.Parameter(cutpoints)
        else:
            raise ValueError(f"{init_cutpoints} is not a valid init_cutpoints " f"type")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sigmoids = torch.sigmoid(self.cutpoints - X)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat(
            (sigmoids[:, [0]], link_mat, (1 - sigmoids[:, [-1]])), dim=1
        )
        return link_mat


class OrdinalLogisticModel(nn.Module):
    def __init__(
        self, predictor: nn.Module, num_classes: int, init_cutpoints: str = "ordered"
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.predictor = deepcopy(predictor)
        self.link = LogisticCumulativeLink(
            self.num_classes, init_cutpoints=init_cutpoints
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.link(self.predictor(X))




## === cell 5
BATCH_SIZE = 16
IMG_DIM = 512

transforms = A.Compose([])

test_dataset = AptosDataset(
    TEST_PATH, TEST_IMG, "test", transforms, resize=(IMG_DIM, IMG_DIM)
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

base_model = models.efficientnet_b0(pretrained=True)
base_model.classifier = nn.Linear(in_features=1280, out_features=5, bias=True)

checkpoint_path = "../input/7epochs/7epochs.bin"
if os.path.isfile(checkpoint_path):
    try:
        base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded fine‑tuned checkpoint.")
    except Exception as e:
        print(f"Failed to load checkpoint ({e}); using pretrained ImageNet weights.")
else:
    print("Checkpoint not found; using pretrained ImageNet weights.")

model = base_model.to(device)
model.eval()

y_pred = []
num_classes = 5
class_range = torch.arange(num_classes, device=device, dtype=torch.float32)

with torch.no_grad():
    for x in tqdm(test_loader, desc="Predicting"):
        outputs = model(x.float().to(device))  # [B,5]
        probs = torch.softmax(outputs, dim=1)  # [B,5]
        expected = (probs * class_range).sum(dim=1)  # [B]
        preds = torch.round(expected).long().clamp(0, num_classes - 1)
        y_pred.extend(preds.cpu().numpy().tolist())




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2829004933.py in <cell line: 0>()
     30 
     31 with torch.no_grad():
---> 32     for x in tqdm(test_loader, desc="Predicting"):
     33         outputs = model(x.float().to(device))  # [B,5]
     34         probs = torch.softmax(outputs, dim=1)  # [B,5]

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/4034478919.py in __getitem__(self, idx)
     28         mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
     29         std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
---> 30         img_tensor = (img_tensor - mean) / std
     31         # -------------------------------------------------------------------------
     32 

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["diagnosis"] = y_pred
sample_sub.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created with", len(y_pred), "predictions.")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2064043010.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
----> 2 sample_sub["diagnosis"] = y_pred
      3 sample_sub.to_csv("submission.csv", index=False)
      4 print("Submission file 'submission.csv' created with", len(y_pred), "predictions.")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (367)
