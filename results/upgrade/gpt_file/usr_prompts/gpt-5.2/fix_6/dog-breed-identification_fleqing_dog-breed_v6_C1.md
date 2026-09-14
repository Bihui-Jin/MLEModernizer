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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.12

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
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.5506

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import copy
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split, KFold
import cv2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

DATA_DIR = "/kaggle/input/dog-breed-identification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)



## === cell 1
train_data = pd.read_csv(LABELS_CSV)

labels = sorted(train_data["breed"].unique().tolist())
label_to_idx = {b: i for i, b in enumerate(labels)}
train_data["number"] = train_data["breed"].map(label_to_idx).astype(np.int64)

train_data.shape



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_data = sample_sub[["id"]].copy()
test_data.head()



## === cell 3
transforms_train = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transforms_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 4
import torchvision.transforms.functional as TF


class Dog_Breed(Dataset):
    def __init__(self, train_csv, transform=None, test=False):
        super().__init__()
        self.train_csv = train_csv.reset_index(drop=True)

        if "id" in self.train_csv.columns:
            ids = self.train_csv["id"].astype(str).values.tolist()
        else:
            ids = []

        ids = [p for p in ids if isinstance(p, str) and len(p) > 0]

        self.test = test
        if not self.test:
            self.label_nums = self.train_csv["number"].to_numpy(dtype=np.int64)

        self.transform = transform

        if self.test and len(ids) == 0:
            fns = sorted(
                [fn for fn in os.listdir(TEST_DIR) if fn.lower().endswith(".jpg")]
            )
            ids = [fn[:-4] for fn in fns]

        base_dir = TEST_DIR if self.test else TRAIN_DIR
        self.image_ids = ids
        self.image_paths = [os.path.join(base_dir, img_id + ".jpg") for img_id in ids]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image

            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = TF.to_tensor(img)
            if self.transform is not None:
                image = self.transform(image)

        if not self.test:
            label = int(self.label_nums[idx])
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image_paths)




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
device



## === cell 6
train, valid = train_test_split(
    train_data, test_size=0.2, random_state=2021, stratify=train_data["breed"]
)
trainset = Dog_Breed(train, transform=transforms_train)
validset = Dog_Breed(valid, transform=transforms_test)




## === cell 7
def seed_worker(worker_id):
    worker_seed = 2021 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(2021)

NUM_WORKERS = min(8, os.cpu_count() or 2)
PIN_MEMORY = torch.cuda.is_available()

_COMMON_LOADER_KW = dict(
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)


def make_loader(ds, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        **_COMMON_LOADER_KW,
    )


train_loader = make_loader(trainset, batch_size=32, shuffle=True)
valid_loader = make_loader(validset, batch_size=32, shuffle=False)



## === cell 8
DO_VIS = False
if DO_VIS:
    import matplotlib.pyplot as plt

    try:
        dataiter = iter(train_loader)
        images, label = next(dataiter)
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))

        for j, ax in enumerate(axes):
            image = images[j].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            ax.imshow(image)
            ax.set_title(f"Label: {labels[int(label[j].item())]}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Visualization skipped:", repr(e))




## === cell 9
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        try:
            self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.net(x)
        return x




## === cell 10
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=120):
        super(EfficientNetCustom, self).__init__()
        raise ImportError(
            "EfficientNetCustom requires the 'efficientnet_pytorch' package, which is not available here."
        )

    def forward(self, x):
        raise RuntimeError("EfficientNetCustom is unavailable in this environment.")




## === cell 11
testset_global = Dog_Breed(test_data, transform=transforms_test, test=True)
test_loader_global = make_loader(testset_global, batch_size=64, shuffle=False)




## === cell 12
def train_model(
    model,
    train_loader,
    valid_loader,
    loss,
    optimizer,
    epoch,
    device=torch.device("cuda:0"),
    test_loader=None,
):
    device = torch.device(device) if not isinstance(device, torch.device) else device

    net = model.to(device)
    if device.type != "cpu":
        net = net.to(memory_format=torch.channels_last)

    best_epoch = 0
    best_score = -1.0
    best_model_state = None
    early_stopping_round = 3
    losses = []

    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)

    n_train = len(train_loader.dataset)
    n_valid = len(valid_loader.dataset)

    for i in range(epoch):
        correct_train = 0
        loss_sum = 0.0
        net.train()
        for x, y in tqdm(train_loader, leave=False):
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type != "cpu":
                x = x.contiguous(memory_format=torch.channels_last)

            y_hat = net(x)
            loss_temp = loss(y_hat, y)

            loss_sum += float(loss_temp.detach().cpu().item())

            loss_temp.backward()
            optimizer.step()

            correct_train += (y_hat.argmax(dim=1) == y).sum().item()

        scheduler.step()
        losses.append(loss_sum / max(1, len(train_loader)))
        train_acc = correct_train / max(1, n_train)
        print(
            "epoch:",
            i,
            "loss=",
            loss_sum / max(1, n_train),
            "训练集准确度=",
            train_acc,
            end="",
        )

        correct_val = 0
        net.eval()
        with torch.inference_mode():
            for x, y in tqdm(valid_loader, leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if device.type != "cpu":
                    x = x.contiguous(memory_format=torch.channels_last)
                y_hat = net(x)
                correct_val += (y_hat.argmax(dim=1) == y).sum().item()

        val_acc = correct_val / max(1, n_valid)
        print(" 验证集准确度", val_acc)

        if correct_val > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = correct_val
            best_epoch = i
            print("best epoch save!")
        if i - best_epoch >= early_stopping_round:
            break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    if test_loader is None:
        raise ValueError("test_loader must be provided (reused global loader).")

    predictions = []
    net.eval()
    with torch.inference_mode():
        for x in tqdm(test_loader, leave=False):
            x = x.to(device, non_blocking=True)
            if device.type != "cpu":
                x = x.contiguous(memory_format=torch.channels_last)
            y_hat = net(x)
            predictions.append(y_hat.detach().cpu())
    predictions = torch.cat(predictions, dim=0)  # logits [N, 120]
    return predictions




## === cell 13
learn_rate = 0.001
momentum = 0.9
epoch = 10



## === cell 14
base_model = MyResNet50()
base_state = copy.deepcopy(base_model.state_dict())

kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

all_fold_probs = []
for fold, (train_index, val_index) in enumerate(kfold.split(train_data), start=1):
    print(f"\n==== Fold {fold}/5 ====")
    train, valid = train_data.iloc[train_index], train_data.iloc[val_index]
    trainset = Dog_Breed(train, transform=transforms_train)
    validset = Dog_Breed(valid, transform=transforms_test)

    train_loader = make_loader(trainset, batch_size=32, shuffle=True)
    valid_loader = make_loader(validset, batch_size=32, shuffle=False)

    model = MyResNet50()
    model.load_state_dict(base_state)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

    fold_logits = train_model(
        model,
        train_loader,
        valid_loader,
        loss,
        optimizer,
        epoch,
        device,
        test_loader=test_loader_global,
    )
    fold_probs = F.softmax(fold_logits, dim=1).cpu().numpy()
    all_fold_probs.append(fold_probs)

mean_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)  # [Ntest, 120]

result = pd.DataFrame(mean_probs, columns=labels)
result = pd.concat([test_data.reset_index(drop=True), result], axis=1)

result = result.reindex(columns=sample_sub.columns, fill_value=0.0)

probs = result.iloc[:, 1:].to_numpy(dtype=np.float64)
probs[np.isnan(probs)] = 0.0
row_sums = probs.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
probs = probs / row_sums
result.iloc[:, 1:] = probs

out_path = "/kaggle/working/dog_breed.csv"
result.to_csv(out_path, index=False)
print("Saved submission to", out_path, "with shape:", result.shape)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1305188018.py in <cell line: 0>()
     20     optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)
     21 
---> 22     fold_logits = train_model(
     23         model,
     24         train_loader,

/tmp/ipykernel_11/2447874289.py in train_model(model, train_loader, valid_loader, loss, optimizer, epoch, device, test_loader)
     34         loss_sum = 0.0
     35         net.train()
---> 36         for x, y in tqdm(train_loader, leave=False):
     37             optimizer.zero_grad(set_to_none=True)
     38             x = x.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

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
  File "/tmp/ipykernel_11/1947175444.py", line 53, in __getitem__
    image = self.transform(image)
            ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
          ^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 137, in __call__
    return F.to_tensor(pic)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py", line 142, in to_tensor
    raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")
TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>


## === cell 15
df = pd.read_csv("/kaggle/working/dog_breed.csv")
probs = df.iloc[:, 1:].to_numpy(dtype=np.float64)
probs[np.isnan(probs)] = 0.0
row_sums = probs.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
probs = probs / row_sums
df.iloc[:, 1:] = probs
df = df.reindex(columns=sample_sub.columns, fill_value=0.0)
df.to_csv("/kaggle/working/dog_breed.csv", index=False)
print("Submission verified and rewritten to /kaggle/working/dog_breed.csv")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3744189427.py in <cell line: 0>()
----> 1 df = pd.read_csv("/kaggle/working/dog_breed.csv")
      2 probs = df.iloc[:, 1:].to_numpy(dtype=np.float64)
      3 probs[np.isnan(probs)] = 0.0
      4 row_sums = probs.sum(axis=1, keepdims=True)
      5 row_sums[row_sums == 0] = 1.0

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/dog_breed.csv'
