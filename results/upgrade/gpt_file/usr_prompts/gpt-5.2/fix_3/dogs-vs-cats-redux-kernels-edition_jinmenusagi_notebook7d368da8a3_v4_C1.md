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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

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
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.0840549903524859

# 6. Current score

0.92839

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I fix the path and extraction logic so the script can reliably find the competition zip files in this environment and build `train_list/test_list` without crashing. I also correct dataset image loading (force RGB) and label dtype to avoid `CrossEntropyLoss`/shape issues, and make the train/val split reproducible for stability. Finally, I ensure inference runs in batches and produces a properly sorted `submission.csv` with the exact `id,label` columns, written to `/kaggle/working/submission.csv`. These changes are execution-blocking bug fixes and should also improve logloss consistency by ensuring correct probability output alignment.'
- What this solution (achieved 0.92839) has done: 'I fix the dataset extraction/path logic so `train_list` and `test_list` are populated (right now they’re empty because the zip extracts into nested folders). Then I make the train/val split and dataloaders work reliably by pointing them at the actual extracted image directories and ensuring deterministic ordering. Finally, I keep your exact model/training core logic intact, but correct the layer-unfreezing bug (currently it never unfreezes anything because ResNet params don’t start with `"layer4"`) so training can actually learn and move logloss down toward the target. The rest of the pipeline (batch inference, softmax dog prob, sorted `id,label` submission) stays the same and write `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os, glob, copy, zipfile
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models, transforms




## === cell 2
def _pick_existing_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


dir_zip = _pick_existing_dir(
    [
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition",
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
        "../input/dogs-vs-cats-redux-kernels-edition",
    ]
)

if dir_zip is None:
    raise FileNotFoundError(
        "Could not locate dataset directory. Tried: "
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition, /kaggle/input/dogs-vs-cats-redux-kernels-edition, ../input/..."
    )

train_zip_path = os.path.join(dir_zip, "train.zip")
test_zip_path = os.path.join(dir_zip, "test.zip")
sample_sub_path = os.path.join(dir_zip, "sample_submission.csv")

for p in [train_zip_path, test_zip_path, sample_sub_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected file: {p}")

print("Using dataset dir:", dir_zip)



## === cell 3
mean = (0.485, 0.456, 0.406)  # ImageNet mean
std = (0.229, 0.224, 0.225)  # ImageNet std
batch_size = 32  # batch size
lr = 0.001  # learning rate
epochs = 3  # epochs



## === cell 4
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



## === cell 5
extract_root = "/kaggle/working/data"
os.makedirs(extract_root, exist_ok=True)




## === cell 6
def _ensure_extracted(zip_path, extract_root):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)


def _find_jpg_dir(root, prefer_leaf_name=None):
    candidates = []
    for d, _, _ in os.walk(root):
        if len(glob.glob(os.path.join(d, "*.jpg"))) > 0:
            candidates.append(d)
    if not candidates:
        return None
    if prefer_leaf_name is not None:
        for d in candidates:
            if os.path.basename(d) == prefer_leaf_name:
                return d
    candidates = sorted(
        candidates, key=lambda x: len(glob.glob(os.path.join(x, "*.jpg"))), reverse=True
    )
    return candidates[0]


if len(glob.glob(os.path.join(extract_root, "**", "*.jpg"), recursive=True)) == 0:
    _ensure_extracted(train_zip_path, extract_root)
    _ensure_extracted(test_zip_path, extract_root)
else:
    if (
        len(glob.glob(os.path.join(extract_root, "**", "cat.*.jpg"), recursive=True))
        == 0
        and len(
            glob.glob(os.path.join(extract_root, "**", "dog.*.jpg"), recursive=True)
        )
        == 0
    ):
        _ensure_extracted(train_zip_path, extract_root)
    if (
        len(glob.glob(os.path.join(extract_root, "**", "[0-9]*.jpg"), recursive=True))
        == 0
    ):
        _ensure_extracted(test_zip_path, extract_root)

train_dir = _find_jpg_dir(extract_root, prefer_leaf_name="train")
test_dir = _find_jpg_dir(extract_root, prefer_leaf_name="test")

if train_dir is None or test_dir is None:
    raise FileNotFoundError(
        f"Could not find extracted image dirs under {extract_root}. "
        f"Found train_dir={train_dir}, test_dir={test_dir}"
    )

train_list = sorted(glob.glob(os.path.join(train_dir, "*.jpg")))
test_list = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))

if len(train_list) == 0:
    raise FileNotFoundError(f"train_list is empty. train_dir={train_dir}")
if len(test_list) == 0:
    raise FileNotFoundError(f"test_list is empty. test_dir={test_dir}")

train_list, val_list = train_test_split(
    train_list, test_size=0.2, random_state=SEED, shuffle=True
)

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)
print(f"Train Data:{len(train_list)}")
print(f"Validation Data:{len(val_list)}")
print(f"Test Data:{len(test_list)}")



## === cell 7
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)




## === cell 8
class CatDogDataset(data.Dataset):
    def __init__(self, file_list, transform=None, is_test=False):
        self.file_list = file_list
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img) if self.transform is not None else img

        if self.is_test:
            img_id = int(os.path.basename(img_path).split(".")[0])
            return img_transformed, img_id

        label_str = os.path.basename(img_path).split(".")[0]
        if label_str == "dog":
            label = 1
        elif label_str == "cat":
            label = 0
        else:
            raise ValueError(f"Unexpected label prefix in filename: {img_path}")

        return img_transformed, int(label)




## === cell 9
train_dataset = CatDogDataset(train_list, transform=train_transforms, is_test=False)
val_dataset = CatDogDataset(val_list, transform=val_transforms, is_test=False)
test_dataset = CatDogDataset(test_list, transform=val_transforms, is_test=True)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_loader = data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 11
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)

update_params = "layer4"  # only update layer4 (original intent)
for name, param in model.named_parameters():
    if name.startswith(update_params + "."):
        param.requires_grad = True
    else:
        param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("fc."):
        param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
param_list = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(param_list, lr=lr)




## === cell 12
def _to_device(batch, device):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y




## === cell 13
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

best_model = copy.deepcopy(model.state_dict())
best_accuracy = 0.0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0

    for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{epochs}"):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()

        output = model(xb)
        loss = criterion(output, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == yb).float().mean()
        epoch_accuracy += acc.item()
        epoch_loss += loss.item()

    epoch_accuracy /= len(train_loader)
    epoch_loss /= len(train_loader)

    with torch.no_grad():
        model.eval()
        epoch_val_accuracy = 0.0
        epoch_val_loss = 0.0

        for xb, yb in tqdm(
            val_loader, desc=f"Val epoch {epoch+1}/{epochs}", leave=False
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()

            val_output = model(xb)
            val_loss = criterion(val_output, yb)

            acc = (val_output.argmax(dim=1) == yb).float().mean()
            epoch_val_accuracy += acc.item()
            epoch_val_loss += val_loss.item()

        epoch_val_accuracy /= len(val_loader)
        epoch_val_loss /= len(val_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - "
        f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

    if epoch_val_accuracy > best_accuracy:
        best_accuracy = epoch_val_accuracy
        best_model = copy.deepcopy(model.state_dict())

    train_acc_list.append(epoch_accuracy)
    val_acc_list.append(epoch_val_accuracy)
    train_loss_list.append(epoch_loss)
    val_loss_list.append(epoch_val_loss)

os.makedirs("/kaggle/working", exist_ok=True)
torch.save(best_model, "/kaggle/working/model.pth")
print("Saved best model to /kaggle/working/model.pth with val_acc:", best_accuracy)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3749690284.py in <cell line: 0>()
     12     epoch_accuracy = 0.0
     13 
---> 14     for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{epochs}"):
     15         xb = xb.to(device, non_blocking=True)
     16         yb = yb.to(device, non_blocking=True).long()

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

ValueError: Caught ValueError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/1565605897.py", line 25, in __getitem__
    raise ValueError(f"Unexpected label prefix in filename: {img_path}")
ValueError: Unexpected label prefix in filename: /kaggle/working/data/1057.jpg


## === cell 14
import matplotlib.pyplot as plt

epochs_x = range(1, len(train_acc_list) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_x, train_loss_list, "bo-", label="Train Loss")
plt.plot(epochs_x, val_loss_list, "ro-", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_x, train_acc_list, "bo-", label="Train Accuracy")
plt.plot(epochs_x, val_acc_list, "ro-", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()



## === cell 15
param = torch.load("/kaggle/working/model.pth", map_location=device)
model.load_state_dict(param)
model = model.to(device).eval()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2128389778.py in <cell line: 0>()
----> 1 param = torch.load("/kaggle/working/model.pth", map_location=device)
      2 model.load_state_dict(param)
      3 model = model.to(device).eval()
      4 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/model.pth'

## === cell 16
id_list = []
pred_list = []

with torch.no_grad():
    for xb, ids in tqdm(test_loader, desc="Predict"):
        xb = xb.to(device, non_blocking=True)
        outputs = model(xb)
        probs_dog = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()

        id_list.extend([int(i) for i in ids])
        pred_list.extend([float(p) for p in probs_dog])



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4011450331.py in <cell line: 0>()
      3 
      4 with torch.no_grad():
----> 5     for xb, ids in tqdm(test_loader, desc="Predict"):
      6         xb = xb.to(device, non_blocking=True)
      7         outputs = model(xb)

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

ValueError: Caught ValueError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/1565605897.py", line 16, in __getitem__
    img_id = int(os.path.basename(img_path).split(".")[0])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: 'cat'


## === cell 17
submit = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame({"id": id_list, "label": pred_list}).sort_values("id")
submit = submit.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))

if "label_pred" in submit.columns:
    submit["label"] = submit["label_pred"].combine_first(submit["label"])
    submit = submit.drop(columns=["label_pred"])

submit = submit[["id", "label"]].sort_values("id")
submit.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv")
print(submit.head())



## === cell 18
assert os.path.exists("/kaggle/working/submission.csv")
chk = pd.read_csv("/kaggle/working/submission.csv")
assert list(chk.columns) == ["id", "label"]
assert chk["label"].between(0.0, 1.0).all()
print("Submission OK. Rows:", len(chk))



## === cell 19
weights = models.ResNet18_Weights.DEFAULT
_model_tmp = models.resnet18(weights=weights)
print(_model_tmp)
