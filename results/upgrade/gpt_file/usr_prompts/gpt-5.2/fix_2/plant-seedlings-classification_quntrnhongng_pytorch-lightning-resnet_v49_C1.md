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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
lightning-utilities==0.15.2
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
seaborn==0.12.2
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
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.70654

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os



## === cell 1
import lightning  # noqa: F401



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2622086582.py in <cell line: 0>()
      1 # Kaggle environment already has lightning installed per package list.
      2 # Avoid pip-install to prevent runtime/network issues.
----> 3 import lightning  # noqa: F401
      4 

ModuleNotFoundError: No module named 'lightning'

## === cell 2
import math
import os
import random
import time
import urllib.request
from functools import partial
from urllib.error import HTTPError
from types import SimpleNamespace

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns

from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.transforms import v2 as tforms
import torchmetrics

import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from tqdm.notebook import tqdm



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3327375023.py in <cell line: 0>()
      8 from types import SimpleNamespace
      9 
---> 10 import lightning as L
     11 from lightning.pytorch.loggers import CSVLogger
     12 

ModuleNotFoundError: No module named 'lightning'

## === cell 3
DATASET_PATH = os.environ.get(
    "PATH_DATASETS", "/kaggle/input/plant-seedlings-classification"
)
TRAIN_DATASET_PATH = os.environ.get(
    "PATH_TRAIN_DATASETS", "/kaggle/input/plant-seedlings-classification/train"
)
TEST_DATASET_PATH = os.environ.get(
    "PATH_TEST_DATASETS", "/kaggle/input/plant-seedlings-classification/test"
)

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "/kaggle/working")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

num_workers = os.cpu_count()



## === cell 4
pass



## === cell 5
pass



## === cell 6
plt.set_cmap("cividis")
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

seed = 13
L.seed_everything(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/902037345.py in <cell line: 0>()
----> 1 plt.set_cmap("cividis")
      2 # In .py execution on Kaggle, IPython magics may fail; keep plotting config without magics.
      3 matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
      4 matplotlib.rcParams["lines.linewidth"] = 2.0
      5 sns.reset_orig()

NameError: name 'plt' is not defined

## === cell 7
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 8
def make_fname_class_df(train_dir):
    classes = sorted(os.listdir(train_dir))

    file_lst = []
    class_lst = []
    class_idx_lst = []
    for i, cl in enumerate(classes):
        path = train_dir + f"/{cl}"
        files = os.listdir(path)
        file_lst = file_lst + files
        class_lst = class_lst + [cl] * len(files)
        class_idx_lst = class_idx_lst + [i] * len(files)
    full_df = pd.DataFrame(
        {"file": file_lst, "class": class_lst, "class_idx": class_idx_lst}
    )
    return full_df


full_df = make_fname_class_df(TRAIN_DATASET_PATH)



## === cell 9
plt.figure(figsize=(12, 8))
g = sns.countplot(
    data=full_df,
    x="class",
    order=full_df["class"].value_counts().index,
    palette="Greens_r",
)
plt.xticks(rotation=45)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/256660352.py in <cell line: 0>()
----> 1 plt.figure(figsize=(12, 8))
      2 g = sns.countplot(
      3     data=full_df,
      4     x="class",
      5     order=full_df["class"].value_counts().index,

NameError: name 'plt' is not defined

## === cell 10
def visualize_input_images(root_dir, images_per_row=7):
    classes = sorted(os.listdir(root_dir))
    fig = plt.figure(1, figsize=(len(classes) * 3, images_per_row * 3))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(len(classes), images_per_row), axes_pad=0.05
    )
    for row, class_name in enumerate(classes):
        path = root_dir + f"/{class_name}"
        fnames = os.listdir(path)
        k = min(images_per_row, len(fnames))
        image_fname_samples = random.sample(fnames, k=k) if k > 0 else []
        for col in range(images_per_row):
            ax = grid[row * images_per_row + col]
            ax.set_axis_off()
            if col < len(image_fname_samples):
                img_fname = image_fname_samples[col]
                img_path = root_dir + f"/{class_name}" + f"/{img_fname}"
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                ax.imshow(img)
            if col == images_per_row - 1:
                ax.text(
                    250,
                    110,
                    class_name,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
    plt.show()


visualize_input_images(TRAIN_DATASET_PATH)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2164837470.py in <cell line: 0>()
     32 
     33 # Keep call but safe now
---> 34 visualize_input_images(TRAIN_DATASET_PATH)
     35 
     36 

/tmp/ipykernel_11/2164837470.py in visualize_input_images(root_dir, images_per_row)
      2     # BUGFIX: random.sample() k must be <= population; clamp k by available images.
      3     classes = sorted(os.listdir(root_dir))
----> 4     fig = plt.figure(1, figsize=(len(classes) * 3, images_per_row * 3))
      5     grid = ImageGrid(
      6         fig, 111, nrows_ncols=(len(classes), images_per_row), axes_pad=0.05

NameError: name 'plt' is not defined

## === cell 11
def visualize_input_images_2(root_dir, images_per_row=5):
    classes = sorted(os.listdir(root_dir))
    fig, axes = plt.subplots(nrows=len(classes), ncols=images_per_row, figsize=(15, 10))
    fig.tight_layout(pad=0.0)
    for row, class_name in enumerate(classes):
        path = root_dir + f"/{class_name}"
        fnames = os.listdir(path)
        k = min(images_per_row, len(fnames))
        image_fname_samples = random.sample(fnames, k=k) if k > 0 else []
        for col in range(images_per_row):
            axes[row][col].set_axis_off()
            if col < len(image_fname_samples):
                img_fname = image_fname_samples[col]
                img_path = root_dir + f"/{class_name}" + f"/{img_fname}"
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                axes[row][col].imshow(img)
            if col == images_per_row - 1:
                axes[row][col].text(
                    250,
                    110,
                    class_name,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
    plt.show()


visualize_input_images_2(TRAIN_DATASET_PATH)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1162252636.py in <cell line: 0>()
     27 
     28 
---> 29 visualize_input_images_2(TRAIN_DATASET_PATH)
     30 
     31 

/tmp/ipykernel_11/1162252636.py in visualize_input_images_2(root_dir, images_per_row)
      2     # BUGFIX: same clamp as above.
      3     classes = sorted(os.listdir(root_dir))
----> 4     fig, axes = plt.subplots(nrows=len(classes), ncols=images_per_row, figsize=(15, 10))
      5     fig.tight_layout(pad=0.0)
      6     for row, class_name in enumerate(classes):

NameError: name 'plt' is not defined

## === cell 12
class PlantTrainDataset(Dataset):

    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.fname_class_df = make_fname_class_df(root_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.fname_class_df.shape[0]

    def __getitem__(self, idx):
        path = (
            self.root_dir
            + f"/{self.fname_class_df.loc[self.fname_class_df.index[idx], 'class']}"
            + f"/{self.fname_class_df.loc[self.fname_class_df.index[idx], 'file']}"
        )
        image = Image.open(path).convert("RGB")
        target = int(
            self.fname_class_df.loc[self.fname_class_df.index[idx], "class_idx"]
        )

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


class PlantTestDataset(Dataset):

    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.fname_class_df = pd.DataFrame({"file": os.listdir(root_dir)})
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.fname_class_df.shape[0]

    def __getitem__(self, idx):
        path = (
            self.root_dir
            + f"/{self.fname_class_df.loc[self.fname_class_df.index[idx], 'file']}"
        )
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3375883472.py in <cell line: 0>()
----> 1 class PlantTrainDataset(Dataset):
      2 
      3     def __init__(self, root_dir, transform=None, target_transform=None):
      4         self.root_dir = root_dir
      5         self.fname_class_df = make_fname_class_df(root_dir)

NameError: name 'Dataset' is not defined

## === cell 13
def compute_mean_std(dataset, batch_size=8):
    rgb_values = torch.cat([img.reshape(3, -1) for img, target in dataset], dim=-1)
    rgb_values_chunks = rgb_values.split(224 * 224 * batch_size, dim=1)
    rgb_mean_chunks = [
        torch.mean(chunk.float(), dim=-1) for chunk in rgb_values_chunks
    ]  # list of tensor shape (3,)
    rgb_mean = torch.mean(torch.stack(rgb_mean_chunks, dim=0), dim=0).reshape(
        (3, 1)
    )  # shape (3, 1)

    rgb_var_chunks = [
        torch.mean((chunk - rgb_mean).float() ** 2, dim=-1)
        for chunk in rgb_values_chunks
    ]
    rgb_std = torch.sqrt(torch.mean(torch.stack(rgb_var_chunks, dim=0), dim=0))

    return rgb_mean.squeeze() / 255.0, rgb_std / 255.0




## === cell 14
class PlantDataModule(L.LightningDataModule):
    def __init__(self, root_dir, test_dir, batch_size):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        backbone_mean, backbone_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.transform = tforms.Compose(
            [
                tforms.ToImage(),
                tforms.ToDtype(torch.uint8, scale=True),
                tforms.RandomAffine(degrees=10, translate=(0.2, 0.2)),
                tforms.RandomHorizontalFlip(),
                tforms.RandomVerticalFlip(),
                tforms.Resize((224, 224), antialias=True),
                tforms.ToDtype(torch.float32, scale=True),
                tforms.Normalize(backbone_mean, backbone_std),
            ]
        )
        self.test_transform = tforms.Compose(
            [
                tforms.ToImage(),
                tforms.ToDtype(torch.uint8, scale=True),
                tforms.Resize((224, 224), antialias=True),
                tforms.ToDtype(torch.float32, scale=True),
                tforms.Normalize(backbone_mean, backbone_std),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            full_dataset = PlantTrainDataset(self.root_dir, transform=self.transform)

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(seed)
            )
        if stage == "test":
            full_dataset = PlantTrainDataset(self.root_dir, transform=self.transform)

            _, self.test_dataset = random_split(
                full_dataset,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(torch.initial_seed()),
            )
        if stage == "predict":
            self.predict_dataset = PlantTestDataset(
                self.test_dir, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(torch.initial_seed()),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(torch.initial_seed()),
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(torch.initial_seed()),
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(torch.initial_seed()),
            persistent_workers=True,
        )




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3164196018.py in <cell line: 0>()
----> 1 class PlantDataModule(L.LightningDataModule):
      2     def __init__(self, root_dir, test_dir, batch_size):
      3         super().__init__()
      4         self.root_dir = root_dir
      5         self.test_dir = test_dir

NameError: name 'L' is not defined

## === cell 15
class PlantSeedModule(L.LightningModule):
    def __init__(
        self,
        model_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        metrics_name,
        metrics_hparams,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        self._configure_metrics()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=-1)
        loss = self.loss_module(outputs, targets)
        self.train_metrics.update(preds, targets)
        self.log("train_loss", loss, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=-1)
        self.valid_metrics.update(preds, targets)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=-1)
        self.test_metrics.update(preds, targets)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)
        return torch.argmax(outputs, dim=-1)

    def _configure_metrics(self):
        metric_lst = []
        for metric_name, metric_hprams in zip(
            self.hparams.metrics_name, self.hparams.metrics_hparams
        ):
            if metric_name == "Accuracy":
                metric_lst.append(torchmetrics.Accuracy(**metric_hprams))
            elif metric_name == "Precision":
                metric_lst.append(torchmetrics.Precision(**metric_hprams))
            elif metric_name == "Recall":
                metric_lst.append(torchmetrics.Recall(**metric_hprams))
            elif metric_name == "F1Score":
                metric_lst.append(torchmetrics.F1Score(**metric_hprams))
            else:
                pass
        metrics = torchmetrics.MetricCollection(metric_lst)
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.hparams.optimizer_hparams,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.hparams.optimizer_hparams,
            )
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3714796806.py in <cell line: 0>()
----> 1 class PlantSeedModule(L.LightningModule):
      2     def __init__(
      3         self,
      4         model_name,
      5         model_hparams,

NameError: name 'L' is not defined

## === cell 16
model_dict = {}


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert (
            False
        ), f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'




## === cell 17
class TransferLearningEfficientnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.hparams = SimpleNamespace(num_classes=num_classes)
        self._create_network()
        self._init_params()

    def _create_network(self):
        backbone = torchvision.models.efficientnet_b4(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))
        num_ftrs = backbone.classifier[1].in_features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, self.hparams.num_classes),
        )
        for param in self.features.parameters():
            param.requires_grad = False

    def _init_params(self):
        pass

    def forward(self, x):
        ftrs = self.features(x)
        ftrs = ftrs.view(ftrs.size(0), -1)
        y = self.classifier(ftrs)
        return y


model_dict["TLEfficientnet"] = TransferLearningEfficientnet




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/918445582.py in <cell line: 0>()
----> 1 class TransferLearningEfficientnet(nn.Module):
      2     def __init__(self, num_classes):
      3         super().__init__()
      4         self.hparams = SimpleNamespace(num_classes=num_classes)
      5         self._create_network()

NameError: name 'nn' is not defined

## === cell 18
def train_model(model_name, datamodule, max_epochs, save_name=None, **kwargs):
    """Train model.

    Args:
        model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    logger = CSVLogger(
        save_dir=os.path.join(CHECKPOINT_PATH, save_name),
        name="logs",
        version="version_1",
    )

    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                mode="max",
                monitor="val_MulticlassAccuracy",
                verbose=True,
                save_last=True,
                save_top_k=2,
                enable_version_counter=True,
            ),
            LearningRateMonitor("epoch"),
        ],
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = PlantSeedModule.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(seed)  # To be reproducible
        model = PlantSeedModule(model_name=model_name, **kwargs)
        trainer.fit(model, datamodule=datamodule, ckpt_path="last")
        model = PlantSeedModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    test_result = trainer.test(model, datamodule=datamodule, verbose=True)

    return trainer, model, test_result




## === cell 19
batch_size = 32
plantdata = PlantDataModule(TRAIN_DATASET_PATH, TEST_DATASET_PATH, batch_size)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/196443773.py in <cell line: 0>()
      1 batch_size = 32
----> 2 plantdata = PlantDataModule(TRAIN_DATASET_PATH, TEST_DATASET_PATH, batch_size)
      3 

NameError: name 'PlantDataModule' is not defined

## === cell 20
trainer, model, test_result = train_model(
    model_name="TLEfficientnet",
    datamodule=plantdata,
    max_epochs=1,
    model_hparams={"num_classes": 12},
    optimizer_name="Adam",
    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    metrics_name=["Accuracy", "Precision", "Recall", "F1Score"],
    metrics_hparams=[
        {"task": "multiclass", "num_classes": 12, "average": "macro"},
        {"task": "multiclass", "num_classes": 12, "average": "macro"},
        {"task": "multiclass", "num_classes": 12, "average": "macro"},
        {"task": "multiclass", "num_classes": 12, "average": "macro"},
    ],
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/464348967.py in <cell line: 0>()
      1 trainer, model, test_result = train_model(
      2     model_name="TLEfficientnet",
----> 3     datamodule=plantdata,
      4     max_epochs=1,
      5     model_hparams={"num_classes": 12},

NameError: name 'plantdata' is not defined

## === cell 21
pass



## === cell 22
preds = trainer.predict(model, datamodule=plantdata)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2707842685.py in <cell line: 0>()
----> 1 preds = trainer.predict(model, datamodule=plantdata)
      2 

NameError: name 'trainer' is not defined

## === cell 23
classes = sorted(os.listdir(TRAIN_DATASET_PATH))
pred_idx = torch.cat(preds, 0).detach().cpu().numpy().astype(int)
species = np.vectorize(lambda idx: classes[idx])(pred_idx)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4076811247.py in <cell line: 0>()
      1 # BUGFIX: ensure class index -> species mapping matches make_fname_class_df ordering (sorted).
      2 classes = sorted(os.listdir(TRAIN_DATASET_PATH))
----> 3 pred_idx = torch.cat(preds, 0).detach().cpu().numpy().astype(int)
      4 species = np.vectorize(lambda idx: classes[idx])(pred_idx)
      5 

NameError: name 'torch' is not defined

## === cell 24
pass



## === cell 25
pass



## === cell 26
sample_path = os.path.join(DATASET_PATH, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

test_files_in_loader_order = plantdata.predict_dataset.fname_class_df["file"].tolist()

pred_map = dict(zip(test_files_in_loader_order, species.tolist()))
submission = sample_sub.copy()
submission["species"] = submission["file"].map(pred_map)

if submission["species"].isna().any():
    fallback = classes[0]
    submission["species"] = submission["species"].fillna(fallback)

submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4114081561.py in <cell line: 0>()
      6 # Align predicted labels to sample_sub order by creating a test filename list in the same order as prediction dataloader.
      7 # PlantTestDataset uses os.listdir(root_dir) order; enforce same order here too.
----> 8 test_files_in_loader_order = plantdata.predict_dataset.fname_class_df["file"].tolist()
      9 
     10 # Map filename -> predicted species

NameError: name 'plantdata' is not defined

## === cell 27
submission.head()

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
