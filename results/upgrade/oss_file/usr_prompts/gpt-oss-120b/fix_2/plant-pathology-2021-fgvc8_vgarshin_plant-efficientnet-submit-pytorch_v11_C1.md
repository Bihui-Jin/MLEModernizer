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

0.831154201292706

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, json, time
import cv2, pandas as pd, numpy as np
import torch, torch.nn as nn, torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms

KAGGLE = True
if not KAGGLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = (
    "../input/plant-pathology-2021-fgvc8/train.csv" if KAGGLE else "./data/train.csv"
)
train_df = pd.read_csv(train_path)
all_labels = set()
for lab_str in train_df["labels"].astype(str):
    all_labels.update(lab_str.split())
all_labels = sorted(all_labels)
LABELS_ = {lbl: idx for idx, lbl in enumerate(all_labels)}
LABELS = {str(idx): lbl for lbl, idx in LABELS_.items()}

WORKERS = 2
print(f"Loaded {len(train_df)} training rows, {len(LABELS_)} unique labels.")



## === cell 1
TEST = True
VER = "v4"  # kept for compatibility; not used further
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
else:
    DATA_PATH = "./data"
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
TH = 0.4  # threshold for label activation
TTAS = [0, 1, 2]  # test‑time augmentations (indices used in flip)
FOLDS = [0]  # we will use a single dummy model instead of external checkpoints

print("Data path:", DATA_PATH)
print("Image folder:", IMGS_PATH)




## === cell 2
class DummyModel(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        batch = x.shape[0]
        return torch.zeros(batch, self.out_dim, device=x.device)


models = []
for n_fold in FOLDS:
    model = DummyModel(out_dim=len(LABELS_))
    model.eval()
    model.to(DEVICE)
    models.append(model)
    print(f"Initialized dummy model for fold {n_fold}")
del n_fold, model
gc.collect()



## === cell 3
df_sub = pd.DataFrame(os.listdir(IMGS_PATH))
df_sub.columns = ["image"]
df_sub["labels"] = "healthy"  # placeholder; will be overwritten after inference
print("Submission dataframe preview:")
print(df_sub.head())




## === cell 4
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels  # None for inference
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        img_path = os.path.join(IMGS_PATH, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = flip(img, axis=self.tta)
        img = img.transpose(2, 0, 1)  # C,H,W
        return torch.tensor(img.copy())




## === cell 5
datasets, loaders = [], []
img_size = 224  # reasonable default size for EfficientNet‑B0 style models
batch_size = 32

for tta in TTAS:
    ds = PlantDataset(df=df_sub, size=img_size, labels=None, transform=None, tta=tta)
    datasets.append(ds)
    loader = data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=SequentialSampler(ds),
        num_workers=WORKERS,
        pin_memory=True,
    )
    loaders.append(loader)

print(f"Created {len(loaders)} loaders for TTAs {TTAS}")




## === cell 6
def get_labels(row, idx2label, th):
    try:
        idxs = [i for i, x in enumerate(row) if x > th]
        labs = [idx2label[str(i)] for i in idxs]
        if "healthy" in labs or len(labs) == 0:
            return "healthy"
        return " ".join(labs)
    except Exception as e:
        print("Error in get_labels:", e, row)
        return "healthy"


logits_per_tta = []  # will hold list of (num_images, num_labels) arrays per TTA
start_time = time.time()

with torch.no_grad():
    for tta_idx, loader in enumerate(loaders):
        batch_logits = []
        for imgs in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            model_preds = []
            for model in models:
                preds = model(imgs).sigmoid().cpu().numpy()
                model_preds.append(preds)
            avg_preds = np.mean(model_preds, axis=0)  # shape (batch, num_labels)
            batch_logits.append(avg_preds)
        tta_logits = (
            np.vstack(batch_logits) if batch_logits else np.empty((0, len(LABELS_)))
        )
        logits_per_tta.append(tta_logits)
        print(f"TTA {tta_idx} done, shape {tta_logits.shape}")

if logits_per_tta:
    logits = np.mean(
        np.stack(logits_per_tta, axis=0), axis=0
    )  # (num_images, num_labels)
else:
    logits = np.empty((len(df_sub), len(LABELS_)))

df_sub["labels"] = [get_labels(row, LABELS, TH) for row in logits]

elapsed = time.time() - start_time
print(f"Inference completed in {int(elapsed // 60)}m {int(elapsed % 60)}s")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2788427074.py in <cell line: 0>()
     19     for tta_idx, loader in enumerate(loaders):
     20         batch_logits = []
---> 21         for imgs in loader:
     22             imgs = imgs.to(DEVICE, non_blocking=True)
     23             # Average predictions over all dummy models (identical)

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
  File "/tmp/ipykernel_55/1841842385.py", line 30, in __getitem__
    raise FileNotFoundError(f"Image not found: {img_path}")
FileNotFoundError: Image not found: ../input/plant-pathology-2021-fgvc8/test_images/test_images


## === cell 7
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
print("\nSubmission preview:")
print(df_sub.head())



## === cell 8
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
