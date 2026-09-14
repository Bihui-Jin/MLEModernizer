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

3.9

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.995

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the dataset extraction/path logic so `./train` and `./test` are actually created where the Dataset expects them, which removes the cascading `FileNotFoundError`/undefined variables. I also fix a bug in `train()` where it incorrectly references `net` instead of the passed `model`, which currently prevents training from running. Finally, I make the submission generation robust and aligned to `sample_submission.csv` ordering (and ensure the file is written as `submission.csv` with correct columns), without changing the model architecture or training loop semantics.'

# 9. Code solution

## === cell 0
import os
import zipfile
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from skimage import io
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_DIR = "../input/aerial-cactus-identification/"
TRAIN_ZIP_DIR = DATA_DIR + "train.zip"
TEST_ZIP_DIR = DATA_DIR + "test.zip"
SAMPLE_SUBMIS = DATA_DIR + "sample_submission.csv"
ANNOTATIONS_DIR = DATA_DIR + "train.csv"

TRAIN_DIR = "./train"
TEST_DIR = "./test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LABELS = 2
N_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
LABELS_MAP = {0: "No Cactus", 1: "Cactus"}


def init_weights(layer):
    if type(layer) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)


def display_data(data, n=10, classes=None):
    fig, ax = plt.subplots(1, n, figsize=(15, 3))
    indices = np.random.randint(0, len(data), size=n)
    for i, j in enumerate(indices):
        ax[i].imshow(np.transpose(data[j][0], (1, 2, 0)))
        ax[i].axis("off")
        if classes:
            ax[i].set_title(classes[data[j][1]])


def train_epoch(
    model, dataloader, lr=LEARNING_RATE, optimizer=None, loss_fn=nn.NLLLoss()
):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss, accuracy, count = 0.0, 0.0, 0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        predicted = torch.max(out, 1)[1]
        accuracy += float((predicted == y).sum().item())
        count += len(y)
    return total_loss / count, accuracy / count


def validate(model, dataloader, loss_fn=nn.NLLLoss()):
    model.eval()
    total_loss, accuracy, count = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = loss_fn(out, y)
            total_loss += float(loss.item()) * len(y)
            predicted = torch.max(out, 1)[1]
            accuracy += float((predicted == y).sum().item())
            count += len(y)
    return total_loss / count, accuracy / count


def train(
    model,
    train_loader,
    valid_loader=None,
    optimizer=None,
    lr=LEARNING_RATE,
    epochs=N_EPOCHS,
    loss_fn=nn.NLLLoss(),
):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_accuracy": []}
    if valid_loader is not None:
        history["validation_loss"] = []
        history["validation_accuracy"] = []
    for epoch in range(epochs):
        tl, ta = train_epoch(
            model, train_loader, lr=lr, optimizer=optimizer, loss_fn=loss_fn
        )
        history["train_loss"].append(tl)
        history["train_accuracy"].append(ta)
        if valid_loader is not None:
            vl, va = validate(model, valid_loader, loss_fn=loss_fn)
            print(
                f"Epoch {epoch:2}, Train Acc = {ta:.3f}, Val Acc = {va:.3f}, Train Loss = {tl:.3f}, Val Loss={vl:.3f}"
            )
            history["validation_loss"].append(vl)
            history["validation_accuracy"].append(va)
        else:
            print(f"Epoch {epoch:2}, Train Acc = {ta:.3f}, Train Loss = {tl:.3f}")
    return history


def plot_history(history, validation=False):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(history["train_accuracy"], label="Training")
    if validation and "validation_accuracy" in history:
        plt.plot(history["validation_accuracy"], label="Validation")
    plt.legend()
    plt.subplot(122)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(history["train_loss"], label="Training")
    if validation and "validation_loss" in history:
        plt.plot(history["validation_loss"], label="Validation")
    plt.legend()


def submission(dataset, model, sample_path=SAMPLE_SUBMIS, out_path="./submission.csv"):
    model.eval()

    preds = {}
    with torch.no_grad():
        for img_tensor, img_id in dataset:
            X = img_tensor[None, ...].to(DEVICE)
            out = model(X)
            prob_cactus = float(torch.exp(out)[0, 1].item())
            preds[img_id] = prob_cactus

    sample = pd.read_csv(sample_path)
    sample["has_cactus"] = sample["id"].map(preds).fillna(0.5).astype(float)
    sample.to_csv(out_path, index=False)
    return sample




## === cell 1
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

if len(os.listdir(TRAIN_DIR)) == 0:
    with zipfile.ZipFile(TRAIN_ZIP_DIR, "r") as zip_ref:
        zip_ref.extractall("./")
    if not os.path.isdir(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
        pass

if len(os.listdir(TEST_DIR)) == 0:
    with zipfile.ZipFile(TEST_ZIP_DIR, "r") as zip_ref:
        zip_ref.extractall("./")

if not os.path.isdir(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
    raise FileNotFoundError(
        f"TRAIN_DIR not populated: {TRAIN_DIR}. Check zip extraction."
    )
if not os.path.isdir(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
    raise FileNotFoundError(
        f"TEST_DIR not populated: {TEST_DIR}. Check zip extraction."
    )




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2181013449.py in <cell line: 0>()
     20 # Safety: assert directories exist with some images
     21 if not os.path.isdir(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
---> 22     raise FileNotFoundError(
     23         f"TRAIN_DIR not populated: {TRAIN_DIR}. Check zip extraction."
     24     )

FileNotFoundError: TRAIN_DIR not populated: ./train. Check zip extraction.

## === cell 2
class ACIDataset(Dataset):
    def __init__(
        self, img_dir, annotations_file=None, transform=None, target_transform=None
    ):
        self.img_dir = img_dir
        self.is_labeled = False
        if annotations_file is not None:
            self.is_labeled = True
            self.img_labels = pd.read_csv(annotations_file)
        else:
            self.img_labels = pd.DataFrame(sorted(os.listdir(img_dir)))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        if self.is_labeled:
            label = int(self.img_labels.iloc[idx, 1])
            if self.target_transform:
                label = self.target_transform(label)
            sample = [image, label]
        else:
            sample = [image, img_name]
        return sample


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

data = ACIDataset(
    img_dir=TRAIN_DIR,
    annotations_file=ANNOTATIONS_DIR,
    transform=transform,
    target_transform=None,
)

test_data = ACIDataset(img_dir=TEST_DIR, transform=transform)

gen = torch.Generator().manual_seed(SEED)
train_len = len(data) * 8 // 10
val_len = len(data) - train_len
train_data, val_data = random_split(data, [train_len, val_len], generator=gen)

train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
all_dl = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)



## === cell 3
try:
    display_data(data, n=12, classes=LABELS_MAP)
    plt.show()
except Exception as e:
    print("Skipping display_data due to:", repr(e))




## === cell 4
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3))
        self.fc = nn.Linear(in_features=20 * 6 * 6, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 6 * 6)
        x = F.log_softmax(self.fc(x), dim=1)
        return x




## === cell 5
model_ = Net().to(DEVICE)
model_.apply(init_weights)



## === cell 6
optimizer = torch.optim.Adam(model_.parameters(), lr=LEARNING_RATE)
history = train(
    model_,
    all_dl,
    optimizer=optimizer,
    lr=LEARNING_RATE,
    epochs=N_EPOCHS,
    loss_fn=nn.NLLLoss(),
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3729573013.py in <cell line: 0>()
      1 optimizer = torch.optim.Adam(model_.parameters(), lr=LEARNING_RATE)
      2 # Keep original approach: train on all data (no validation used)
----> 3 history = train(
      4     model_,
      5     all_dl,

/tmp/ipykernel_11/653031448.py in train(model, train_loader, valid_loader, optimizer, lr, epochs, loss_fn)
    106         history["validation_accuracy"] = []
    107     for epoch in range(epochs):
--> 108         tl, ta = train_epoch(
    109             model, train_loader, lr=lr, optimizer=optimizer, loss_fn=loss_fn
    110         )

/tmp/ipykernel_11/653031448.py in train_epoch(model, dataloader, lr, optimizer, loss_fn)
     61     model.train()
     62     total_loss, accuracy, count = 0.0, 0.0, 0
---> 63     for X, y in dataloader:
     64         X, y = X.to(DEVICE), y.to(DEVICE)
     65         optimizer.zero_grad()

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

/tmp/ipykernel_11/3915838115.py in __getitem__(self, idx)
     20         img_name = self.img_labels.iloc[idx, 0]
     21         img_path = os.path.join(self.img_dir, img_name)
---> 22         image = io.imread(img_path)
     23         if self.transform:
     24             image = self.transform(image)

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in fixed_func(*args, **kwargs)
    326                     kwargs[self.new_name] = deprecated_value
    327 
--> 328             return func(*args, **kwargs)
    329 
    330         if self.modify_docstring and func.__doc__ is not None:

/usr/local/lib/python3.11/dist-packages/skimage/io/_io.py in imread(fname, as_gray, plugin, **plugin_args)
     80 
     81     with file_or_url_context(fname) as fname, _hide_plugin_deprecation_warnings():
---> 82         img = call_plugin('imread', fname, plugin=plugin, **plugin_args)
     83 
     84     if not hasattr(img, 'ndim'):

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in wrapped(*args, **kwargs)
    536             stacklevel = 1 + self.get_stack_length(func) - stack_rank
    537             warnings.warn(message, category=FutureWarning, stacklevel=stacklevel)
--> 538             return func(*args, **kwargs)
    539 
    540         # modify docstring to display deprecation warning

/usr/local/lib/python3.11/dist-packages/skimage/io/manage_plugins.py in call_plugin(kind, *args, **kwargs)
    252             raise RuntimeError(f'Could not find the plugin "{plugin}" for {kind}.')
    253 
--> 254     return func(*args, **kwargs)
    255 
    256 

/usr/local/lib/python3.11/dist-packages/skimage/io/_plugins/imageio_plugin.py in imread(*args, **kwargs)
      9 @wraps(imageio_imread)
     10 def imread(*args, **kwargs):
---> 11     out = np.asarray(imageio_imread(*args, **kwargs))
     12     if not out.flags['WRITEABLE']:
     13         out = out.copy()

/usr/local/lib/python3.11/dist-packages/imageio/v3.py in imread(uri, index, plugin, extension, format_hint, **kwargs)
     51         call_kwargs["index"] = index
     52 
---> 53     with imopen(uri, "r", **plugin_kwargs) as img_file:
     54         return np.asarray(img_file.read(**call_kwargs))
     55 

/usr/local/lib/python3.11/dist-packages/imageio/core/imopen.py in imopen(uri, io_mode, plugin, extension, format_hint, legacy_mode, **kwargs)
    111         request.format_hint = format_hint
    112     else:
--> 113         request = Request(uri, io_mode, format_hint=format_hint, extension=extension)
    114 
    115     source = "<bytes>" if isinstance(uri, bytes) else uri

/usr/local/lib/python3.11/dist-packages/imageio/core/request.py in __init__(self, uri, mode, extension, format_hint, **kwargs)
    247 
    248         # Parse what was given
--> 249         self._parse_uri(uri)
    250 
    251         # Set extension

/usr/local/lib/python3.11/dist-packages/imageio/core/request.py in _parse_uri(self, uri)
    407                 # Reading: check that the file exists (but is allowed a dir)
    408                 if not os.path.exists(fn):
--> 409                     raise FileNotFoundError("No such file: '%s'" % fn)
    410             else:
    411                 # Writing: check that the directory to write to does exist

FileNotFoundError: No such file: '/kaggle/working/train/ece4386a7aba77e608712e56c94ac528.jpg'

## === cell 7
plot_history(history, validation=False)
plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3158810250.py in <cell line: 0>()
----> 1 plot_history(history, validation=False)
      2 plt.show()
      3 

NameError: name 'history' is not defined

## === cell 8
_ = submission(
    test_data, model_, sample_path=SAMPLE_SUBMIS, out_path="./submission.csv"
)
print("Wrote submission to ./submission.csv")
print(pd.read_csv("./submission.csv").head())
