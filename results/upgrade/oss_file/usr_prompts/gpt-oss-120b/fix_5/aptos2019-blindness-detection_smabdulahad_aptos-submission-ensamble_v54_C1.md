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

3.12

# 3. Installed packages

geopandas==0.14.4
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
timm==1.0.19
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

0.8486945958951383

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -0.04736) has done: 'I adjust the script to avoid missing model files by loading pretrained models directly, handle any model‑loading failures gracefully, recompute weights only for the successfully loaded models, and fix the inference aggregation so that a non‑empty tensor list is always produced. These changes ensure the pipeline runs end‑to‑end and writes a valid `submission.csv` while keeping the original architecture and processing logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassQuadraticWeightedKappa

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_56/1759836507.py in <cell line: 0>()
     10 import timm
     11 from sklearn.model_selection import train_test_split
---> 12 from torchmetrics.classification import MulticlassQuadraticWeightedKappa
     13 
     14 torch.backends.cudnn.benchmark = True

ImportError: cannot import name 'MulticlassQuadraticWeightedKappa' from 'torchmetrics.classification' (/usr/local/lib/python3.11/dist-packages/torchmetrics/classification/__init__.py)

## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self._cache = {}  # cache for loaded tensors

        if not self.test:
            for idx in range(len(self.annotations)):
                img_path = os.path.join(
                    self.root_dir, self.annotations.iloc[idx, 0] + ".png"
                )
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                self._cache[idx] = image

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = self._cache[idx]
        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 4
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
full_train_dataset = BlindnessDataset(
    train_csv_file, train_root_dir, transform=transform, test=False
)

train_idx, val_idx = train_test_split(
    range(len(full_train_dataset)),
    test_size=0.2,
    stratify=full_train_dataset.annotations["diagnosis"],
    random_state=42,
)
train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_train_dataset, val_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 5
model_names_to_load = ["resnet18", "seresnext50_32x4d", "seresnext101_32x4d"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_model_names = []

for name in model_names_to_load:
    try:
        model = timm.create_model(name, pretrained=True, num_classes=5)
        model.to(device)
        model.train()
        models_list.append(model)
        loaded_model_names.append(name)
    except Exception as e:
        print(f"Warning: could not load model {name}: {e}")

if not loaded_model_names:
    raise RuntimeError("No models were successfully loaded.")




## === cell 6
criterion = nn.CrossEntropyLoss()
optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in models_list]
epochs = 2  # keep lightweight

for epoch in range(epochs):
    for model, optimizer in zip(models_list, optimizers):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # mixed‑precision for speed
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print(f"Completed epoch {epoch+1}")

for model in models_list:
    model.eval()




## === cell 7
kappa_metric = MulticlassQuadraticWeightedKappa(num_classes=5)
validation_scores = {}

for name, model in zip(loaded_model_names, models_list):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validating {name}"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(labels.cpu())
    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    kappa = kappa_metric(preds_cat, targets_cat).item()
    validation_scores[name] = kappa
    print(f"Model {name} validation QWK: {kappa:.4f}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3314937260.py in <cell line: 0>()
----> 1 kappa_metric = MulticlassQuadraticWeightedKappa(num_classes=5)
      2 validation_scores = {}
      3 
      4 for name, model in zip(loaded_model_names, models_list):
      5     all_preds = []

NameError: name 'MulticlassQuadraticWeightedKappa' is not defined

## === cell 8
total_score = sum(validation_scores.get(k, 0) for k in loaded_model_names)
if total_score == 0:
    weights = {k: 1.0 / len(loaded_model_names) for k in loaded_model_names}
else:
    weights = {k: validation_scores.get(k, 0) / total_score for k in loaded_model_names}




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2795187572.py in <cell line: 0>()
----> 1 total_score = sum(validation_scores.get(k, 0) for k in loaded_model_names)
      2 if total_score == 0:
      3     weights = {k: 1.0 / len(loaded_model_names) for k in loaded_model_names}
      4 else:
      5     weights = {k: validation_scores.get(k, 0) / total_score for k in loaded_model_names}

/tmp/ipykernel_56/2795187572.py in <genexpr>(.0)
----> 1 total_score = sum(validation_scores.get(k, 0) for k in loaded_model_names)
      2 if total_score == 0:
      3     weights = {k: 1.0 / len(loaded_model_names) for k in loaded_model_names}
      4 else:
      5     weights = {k: validation_scores.get(k, 0) / total_score for k in loaded_model_names}

NameError: name 'validation_scores' is not defined

## === cell 9
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        weighted_preds = []
        for name, model in zip(loaded_model_names, models_list):
            w = weights.get(name, 1.0 / len(models_list))
            with torch.cuda.amp.autocast():
                preds = nn.functional.softmax(model(images), dim=1) * w
            weighted_preds.append(preds)
        ensemble_output = torch.stack(weighted_preds).sum(dim=0)  # (batch, 5)
        all_outputs.extend(ensemble_output.cpu().numpy())

all_outputs = np.array(all_outputs)
final_predictions = np.argmax(all_outputs, axis=1)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_56/2308033700.py in <cell line: 0>()
      2 
      3 with torch.no_grad():
----> 4     for images in tqdm(test_loader, desc="Inference"):
      5         images = images.to(device, non_blocking=True)
      6         weighted_preds = []

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

KeyError: Caught KeyError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_56/1627875532.py", line 25, in __getitem__
    image = self._cache[idx]
            ~~~~~~~~~~~^^^^^
KeyError: 0


## === cell 10
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1898926955.py in <cell line: 0>()
      1 submission_df = pd.DataFrame(
----> 2     {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
      3 )
      4 submission_df.to_csv("submission.csv", index=False)

NameError: name 'final_predictions' is not defined
