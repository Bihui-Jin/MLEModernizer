# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.12

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os



## === cell 2
!pip install lightning


## === cell 3
import math
import os
import random
import time
import urllib.request
from functools import partial
from urllib.error import HTTPError
from types import SimpleNamespace

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

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


## === cell 4
DATASET_PATH = os.environ.get("PATH_DATASETS", "/kaggle/input/plant-seedlings-classification")
TRAIN_DATASET_PATH = os.environ.get("PATH_TRAIN_DATASETS", "/kaggle/input/plant-seedlings-classification/train")
TEST_DATASET_PATH = os.environ.get("PATH_TEST_DATASETS", "/kaggle/input/plant-seedlings-classification/test")

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "/kaggle/working")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

num_workers = os.cpu_count()


## === cell 7
plt.set_cmap("cividis")
%matplotlib inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

seed = 13
L.seed_everything(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


## === cell 8
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


## === cell 9
def make_fname_class_df(train_dir):
    classes = os.listdir(train_dir)

    file_lst = []
    class_lst = []
    class_idx_lst = []
    for i, cl in enumerate(classes):
        path = train_dir + f"/{cl}"
        file_lst = file_lst + os.listdir(path)
        class_lst = class_lst + [cl]* len(os.listdir(path))
        class_idx_lst = class_idx_lst + [i]* len(os.listdir(path))
    full_df = pd.DataFrame({"file": file_lst, "class": class_lst,\
                              "class_idx": class_idx_lst})
    return full_df

full_df = make_fname_class_df(TRAIN_DATASET_PATH)


## === cell 10
plt.figure(figsize=(12, 8))
g = sns.countplot(data=full_df, x="class", order=full_df['class'].value_counts().index, palette='Greens_r')
plt.xticks(rotation=45);


## === cell 11
def visualize_input_images(root_dir, images_per_row=7):
    classes = os.listdir(root_dir)
    fig = plt.figure(1, figsize=(len(classes) * 3, images_per_row * 3))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(len(classes), images_per_row), axes_pad=0.05
    )
    for row, class_name in enumerate(classes):
        path = root_dir + f"/{class_name}"
        fnames = os.listdir(path)
        k = min(images_per_row, len(fnames))
        if k == 0:
            continue
        image_fname_samples = random.sample(fnames, k=k)
        for img_fname, col in zip(image_fname_samples, range(images_per_row)):
            img_path = root_dir + f"/{class_name}" + f"/{img_fname}"
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            grid[row * images_per_row + col].imshow(img)
            grid[row * images_per_row + col].set_axis_off()
            if col == images_per_row - 1:
                grid[row * images_per_row + col].text(
                    250,
                    110,
                    class_name,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
    fig.show()


visualize_input_images(TRAIN_DATASET_PATH)


## === cell 12
def visualize_input_images_2(root_dir, images_per_row=5):
    classes = os.listdir(root_dir)
    fig, axes = plt.subplots(nrows=len(classes), ncols=images_per_row, figsize=(15, 10))
    fig.tight_layout(pad=0.0)
    for row, class_name in enumerate(classes):
        path = root_dir + f"/{class_name}"
        fnames = os.listdir(path)
        k = min(images_per_row, len(fnames))
        if k == 0:
            continue
        image_fname_samples = random.sample(fnames, k=k)
        for img_fname, col in zip(image_fname_samples, range(k)):
            img_path = root_dir + f"/{class_name}" + f"/{img_fname}"
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            axes[row][col].imshow(img)
            axes[row][col].set_axis_off()
            if col == images_per_row - 1:
                axes[row][col].text(
                    250,
                    110,
                    class_name,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
    fig.show()


visualize_input_images_2(TRAIN_DATASET_PATH)


## === cell 13
class PlantTrainDataset(Dataset):

    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.fname_class_df = make_fname_class_df(root_dir)
        self.transform = transform
        self.target_transform = target_transform



    def __len__(self):
        return self.fname_class_df.shape[0]

    def __getitem__(self, idx):
        path = self.root_dir + f"/{self.fname_class_df.loc[self.fname_class_df.index[idx], 'class']}" \
            + f"/{self.fname_class_df.loc[self.fname_class_df.index[idx], 'file']}"
        image = Image.open(path).convert("RGB")
        target = self.fname_class_df.loc[self.fname_class_df.index[idx], "class_idx"]

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
        path = self.root_dir + f"/{self.fname_class_df.loc[self.fname_class_df.index[idx], 'file']}"
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image


## === cell 14
def compute_mean_std(dataset, batch_size=8):
    rgb_values = torch.cat([img.reshape(3, -1) for img, target in dataset], dim=-1)
    rgb_values_chunks = rgb_values.split(224*224*batch_size, dim=1)
    rgb_mean_chunks = [torch.mean(chunk.float(), dim=-1) for chunk in rgb_values_chunks] #list of tensor shape (3,)
    rgb_mean = torch.mean(torch.stack(rgb_mean_chunks, dim=0), dim=0).reshape((3, 1)) #shape (3, 1)

    rgb_var_chunks = [torch.mean((chunk - rgb_mean).float()** 2, dim=-1) for chunk in rgb_values_chunks]
    rgb_std = torch.sqrt(torch.mean(torch.stack(rgb_var_chunks, dim=0), dim=0))

    return rgb_mean.squeeze()/ 255.0, rgb_std/ 255.0


## === cell 15
class PlantDataModule(L.LightningDataModule):
    def __init__(self, root_dir, test_dir, batch_size):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        backbone_mean, backbone_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.transform = tforms.Compose([
            tforms.ToImage(),
            tforms.ToDtype(torch.uint8, scale=True),
            tforms.RandomAffine(degrees=10, translate=(0.2, 0.2)),
            tforms.RandomHorizontalFlip(),
            tforms.RandomVerticalFlip(),
            tforms.Resize((224, 224), antialias=True),
            tforms.ToDtype(torch.float32, scale=True),
            tforms.Normalize(backbone_mean, backbone_std)
        ])
        self.test_transform = tforms.Compose([
            tforms.ToImage(),
            tforms.ToDtype(torch.uint8, scale=True),
            tforms.Resize((224, 224), antialias=True),
            tforms.ToDtype(torch.float32, scale=True),
            tforms.Normalize(backbone_mean, backbone_std)
        ])

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            full_dataset = PlantTrainDataset(self.root_dir,transform=self.transform)
            
            self.train_dataset, self.val_dataset = random_split(full_dataset, [0.8, 0.2],\
                generator=torch.Generator().manual_seed(seed)
            )
        if stage == "test":
            full_dataset = PlantTrainDataset(self.root_dir,transform=self.transform)
            
            _, self.test_dataset = random_split(full_dataset, [0.8, 0.2],\
                generator=torch.Generator().manual_seed(torch.initial_seed())
            )
        if stage == "predict":
            self.predict_dataset = PlantTestDataset(self.test_dir, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,\
                    num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker,\
                    generator=torch.Generator().manual_seed(torch.initial_seed()), persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,\
                    num_workers=num_workers, worker_init_fn=seed_worker,\
                    generator=torch.Generator().manual_seed(torch.initial_seed()), persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,\
                    num_workers=num_workers, worker_init_fn=seed_worker,\
                    generator=torch.Generator().manual_seed(torch.initial_seed()), persistent_workers=True
        )

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,\
                    num_workers=num_workers, worker_init_fn=seed_worker,\
                    generator=torch.Generator().manual_seed(torch.initial_seed()), persistent_workers=True
        )


## === cell 16
class PlantSeedModule(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, metrics_name, metrics_hparams):
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
        for metric_name, metric_hprams in zip(self.hparams.metrics_name, self.hparams.metrics_hparams):
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
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]


## === cell 17
model_dict = {}

def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'


## === cell 18
class TransferLearningEfficientnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.hparams = SimpleNamespace(num_classes=num_classes)
        self._create_network()
        self._init_params()

    def _create_network(self):
        backbone = torchvision.models.efficientnet_b4(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))
        num_ftrs = backbone.classifier[1].in_features
        self.classifier = torch.nn.Sequential(
                            torch.nn.Linear(num_ftrs, 128),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.2),
                            torch.nn.Linear(128, self.hparams.num_classes)
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


## === cell 19
def train_model(model_name, datamodule, max_epochs, save_name=None, **kwargs):
    """Train model.

    Args:
        model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        logger=TensorBoardLogger(save_dir=os.path.join(CHECKPOINT_PATH, save_name), name="logs", version="version_1"),
        callbacks=[
            ModelCheckpoint(
                mode="max", monitor="val_MulticlassAccuracy", verbose=True, save_last=True, save_top_k=2, enable_version_counter=True
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
        log_every_n_steps=50,
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

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
        )  # Load best checkpoint after training

    test_result = trainer.test(model, datamodule=datamodule, verbose=True)

    return trainer, model, test_result


## === cell 20
batch_size = 32
plantdata = PlantDataModule(TRAIN_DATASET_PATH, TEST_DATASET_PATH, batch_size)


## === cell 21
from lightning.pytorch.loggers import CSVLogger as _SafeLogger

TensorBoardLogger = _SafeLogger  # monkey-patch the symbol used inside train_model()


def make_fname_class_df(train_dir):
    classes = sorted(os.listdir(train_dir))

    file_lst = []
    class_lst = []
    class_idx_lst = []
    for i, cl in enumerate(classes):
        path = train_dir + f"/{cl}"
        fnames = os.listdir(path)
        file_lst = file_lst + fnames
        class_lst = class_lst + [cl] * len(fnames)
        class_idx_lst = class_idx_lst + [i] * len(fnames)
    full_df = pd.DataFrame(
        {"file": file_lst, "class": class_lst, "class_idx": class_idx_lst}
    )
    return full_df


full_df = make_fname_class_df(TRAIN_DATASET_PATH)

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


## === cell 22


# display(IPython.display.HTML('''
# Open TensorBoard
# Hide TensorBoard
# document.querySelector('#open_tb').onclick = () => { window.open(document.querySelector('iframe').src, "__blank") }


## === cell 23
preds = trainer.predict(model, plantdata)


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIsADirectoryError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2750698618.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mpreds[0m [0;34m=[0m [0mtrainer[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mplantdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/trainer/trainer.py[0m in [0;36mpredict[0;34m(self, model, dataloaders, datamodule, return_predictions, ckpt_path, weights_only)[0m
[1;32m    939[0m         [0mself[0m[0;34m.[0m[0mstate[0m[0;34m.[0m[0mstatus[0m [0;34m=[0m [0mTrainerStatus[0m[0;34m.[0m[0mRUNNING[0m[0;34m[0m[0;34m[0m[0m
[1;32m    940[0m         [0mself[0m[0;34m.[0m[0mpredicting[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 941[0;31m         return call._call_and_handle_interrupt(
[0m[1;32m    942[0m             [0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    943[0m             [0mself[0m[0;34m.[0m[0m_predict_impl[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/trainer/call.py[0m in [0;36m_call_and_handle_interrupt[0;34m(trainer, trainer_fn, *args, **kwargs)[0m
[1;32m     47[0m         [0;32mif[0m [0mtrainer[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mlauncher[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m             [0;32mreturn[0m [0mtrainer[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mlauncher[0m[0;34m.[0m[0mlaunch[0m[0;34m([0m[0mtrainer_fn[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0mtrainer[0m[0;34m=[0m[0mtrainer[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 49[0;31m         [0;32mreturn[0m [0mtrainer_fn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;34m[0m[0m
[1;32m     51[0m     [0;32mexcept[0m [0m_TunerExitException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/trainer/trainer.py[0m in [0;36m_predict_impl[0;34m(self, model, dataloaders, datamodule, return_predictions, ckpt_path, weights_only)[0m
[1;32m    988[0m             [0mself[0m[0;34m.[0m[0mstate[0m[0;34m.[0m[0mfn[0m[0;34m,[0m [0mckpt_path[0m[0;34m,[0m [0mmodel_provided[0m[0;34m=[0m[0mmodel_provided[0m[0;34m,[0m [0mmodel_connected[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlightning_module[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    989[0m         )
[0;32m--> 990[0;31m         [0mresults[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_run[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mckpt_path[0m[0;34m=[0m[0mckpt_path[0m[0;34m,[0m [0mweights_only[0m[0;34m=[0m[0mweights_only[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    991[0m [0;34m[0m[0m
[1;32m    992[0m         [0;32massert[0m [0mself[0m[0;34m.[0m[0mstate[0m[0;34m.[0m[0mstopped[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/trainer/trainer.py[0m in [0;36m_run[0;34m(self, model, ckpt_path, weights_only)[0m
[1;32m   1077[0m         [0;31m# RUN THE TRAINER[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1078[0m         [0;31m# ----------------------------[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1079[0;31m         [0mresults[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_run_stage[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1080[0m [0;34m[0m[0m
[1;32m   1081[0m         [0;31m# ----------------------------[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/trainer/trainer.py[0m in [0;36m_run_stage[0;34m(self)[0m
[1;32m   1116[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_evaluation_loop[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1117[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mpredicting[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1118[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mpredict_loop[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1119[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mtraining[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1120[0m             [0;32mwith[0m [0misolate_rng[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/loops/utilities.py[0m in [0;36m_decorator[0;34m(self, *args, **kwargs)[0m
[1;32m    177[0m             [0mcontext_manager[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m[0m[0;34m[0m[0m
[1;32m    178[0m         [0;32mwith[0m [0mcontext_manager[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 179[0;31m             [0;32mreturn[0m [0mloop_run[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    180[0m [0;34m[0m[0m
[1;32m    181[0m     [0;32mreturn[0m [0m_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/loops/prediction_loop.py[0m in [0;36mrun[0;34m(self)[0m
[1;32m    120[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m                     [0mdataloader_iter[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m                     [0mbatch[0m[0;34m,[0m [0mbatch_idx[0m[0;34m,[0m [0mdataloader_idx[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mdata_fetcher[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m                 [0mself[0m[0;34m.[0m[0mbatch_progress[0m[0;34m.[0m[0mis_last_batch[0m [0;34m=[0m [0mdata_fetcher[0m[0;34m.[0m[0mdone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m                 [0;31m# run step hooks[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/loops/fetchers.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    132[0m         [0;32melif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mdone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m             [0;31m# this will run only when no pre-fetching was done.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 134[0;31m             [0mbatch[0m [0;34m=[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__next__[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    135[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    136[0m             [0;31m# the iterator is empty[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/loops/fetchers.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m     59[0m         [0mself[0m[0;34m.[0m[0m_start_profiler[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 61[0;31m             [0mbatch[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0miterator[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     62[0m         [0;32mexcept[0m [0mStopIteration[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m             [0mself[0m[0;34m.[0m[0mdone[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/utilities/combined_loader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    339[0m     [0;32mdef[0m [0m__next__[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0m_ITERATOR_RETURN[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    340[0m         [0;32massert[0m [0mself[0m[0;34m.[0m[0m_iterator[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 341[0;31m         [0mout[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_iterator[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    342[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_iterator[0m[0;34m,[0m [0m_Sequential[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    343[0m             [0;32mreturn[0m [0mout[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning/pytorch/utilities/combined_loader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    140[0m [0;34m[0m[0m
[1;32m    141[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 142[0;31m             [0mout[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0miterators[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mexcept[0m [0mStopIteration[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m             [0;31m# try the next iterator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m   1453[0m                 [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_task_info[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_rcvd_idx[0m[0;34m)[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1454[0m                 [0mself[0m[0;34m.[0m[0m_rcvd_idx[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1455[0;31m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_process_data[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1456[0m [0;34m[0m[0m
[1;32m   1457[0m             [0;32massert[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0m_shutdown[0m [0;32mand[0m [0mself[0m[0;34m.[0m[0m_tasks_outstanding[0m [0;34m>[0m [0;36m0[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_process_data[0;34m(self, data)[0m
[1;32m   1503[0m         [0mself[0m[0;34m.[0m[0m_try_put_index[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1504[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mExceptionWrapper[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1505[0;31m             [0mdata[0m[0;34m.[0m[0mreraise[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1506[0m         [0;32mreturn[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1507[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_utils.py[0m in [0;36mreraise[0;34m(self)[0m
[1;32m    731[0m             [0;31m# instantiate since we don't know how to[0m[0;34m[0m[0;34m[0m[0m
[1;32m    732[0m             [0;32mraise[0m [0mRuntimeError[0m[0;34m([0m[0mmsg[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 733[0;31m         [0;32mraise[0m [0mexception[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    734[0m [0;34m[0m[0m
[1;32m    735[0m [0;34m[0m[0m

[0;31mIsADirectoryError[0m: Caught IsADirectoryError in DataLoader worker process 1.
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
  File "/tmp/ipykernel_11/321412940.py", line 43, in __getitem__
    image = Image.open(path).convert("RGB")
            ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/plant-seedlings-classification/test/test'


## === cell 24
classes = os.listdir(TRAIN_DATASET_PATH)
species = np.vectorize(lambda idx: classes[idx])(torch.cat(preds, 0))
