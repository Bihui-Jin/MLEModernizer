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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.1578947368421052

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
KAGGLE = True  # ensure the flag exists before any use

import os, gc, json, time
import cv2, pandas as pd, numpy as np
import torch, torch.nn as nn, torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler

try:
    from efficientnet_pytorch import model as enet

    _USE_EFFICIENTNET_PYTORCH = True
except ModuleNotFoundError:
    _USE_EFFICIENTNET_PYTORCH = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
TEST = True
VER = "v1"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.5
VOTERS = 1
TTAS = [0, 1, 2]
FOLDS = [0, 1]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()



## === cell 2
params_path = f"{MDLS_PATH}/params.json"
if os.path.isfile(params_path):
    with open(params_path) as file:
        params = json.load(file)
else:
    params = {
        "img_size": 224,
        "batch_size": 32,
        "workers": 4,
        "dropout": 0.2,
        "backbone": "efficientnet-b0",
    }

train_csv_path = f"{DATA_PATH}/train.csv" if KAGGLE else "./data/train.csv"
train_df = pd.read_csv(train_csv_path)
all_labels = set()
for lbls in train_df["labels"].astype(str):
    all_labels.update(lbls.split())
LABELS_ = sorted(list(all_labels))
LABELS = {str(i): label for i, label in enumerate(LABELS_)}

WORKERS = 4 if KAGGLE else params.get("workers", 4)
print("Loaded params:", params)
print("Number of classes:", len(LABELS_))



## === cell 3
df_sub = pd.DataFrame(os.listdir(IMGS_PATH))
df_sub.columns = ["image"]
df_sub["labels"] = "healthy"  # placeholder; will be overwritten if inference runs
print("Submission template preview:")
display(df_sub.head())




## === cell 4
def flip(img, axis=0):
    if axis == 1:
        return img[
            ::-1,
            :,
        ]
    elif axis == 2:
        return img[
            :,
            ::-1,
        ]
    elif axis == 3:
        return img[
            ::-1,
            ::-1,
        ]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        if _USE_EFFICIENTNET_PYTORCH:
            self.enet = enet.EfficientNet.from_name(params["backbone"])
            nc = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
        else:
            self.enet = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
            )
            nc = self.enet.classifier[1].in_features
            self.enet.classifier = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.ELU(),
            nn.BatchNorm1d(int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 5
models = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.isfile(path):
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {path}")
    else:
        print(f"Weight file not found at {path}; using untrained model.")
    model = model.to(DEVICE).float()
    model.eval()
    models.append(model)
del model
gc.collect()
print(f"Total models loaded: {len(models)}")



## === cell 6
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=True,
    )
    loaders.append(loader)




## === cell 7
def get_labels(row, labels_dict, th):
    idxs = [i for i, x in enumerate(row) if x > th]
    lbls = [labels_dict[str(i)] for i in idxs if str(i) in labels_dict]
    return "healthy" if ("healthy" in lbls or len(lbls) == 0) else " ".join(lbls)


if not models:
    df_sub["labels"] = "healthy"
else:
    all_logits = []
    with torch.no_grad():
        for model in models:
            model_logits = []
            for loader in loaders:
                tta_logits = []
                for batch in loader:
                    batch = batch.to(DEVICE)
                    preds = model(batch).sigmoid().cpu().numpy()
                    tta_logits.append(preds)
                tta_mean = np.mean(np.concatenate(tta_logits, axis=0), axis=0)
                model_logits.append(tta_mean)
            model_logits_mean = np.mean(np.stack(model_logits, axis=0), axis=0)
            all_logits.append(model_logits_mean)
    ensemble_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    df_sub["labels"] = [get_labels(row, LABELS, TH) for row in ensemble_logits]

elapsed_time = time.time() - start_time
print(f"Time elapsed: {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2558003112.py in <cell line: 0>()
     16             for loader in loaders:
     17                 tta_logits = []
---> 18                 for batch in loader:
     19                     batch = batch.to(DEVICE)
     20                     preds = model(batch).sigmoid().cpu().numpy()

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

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 1.
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
  File "/tmp/ipykernel_55/2383343383.py", line 38, in __getitem__
    raise FileNotFoundError(f"Image not found: {img_path}")
FileNotFoundError: Image not found: ../input/plant-pathology-2021-fgvc8/test_images/test_images


## === cell 8
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
display(df_sub.head())



## === cell 9
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
