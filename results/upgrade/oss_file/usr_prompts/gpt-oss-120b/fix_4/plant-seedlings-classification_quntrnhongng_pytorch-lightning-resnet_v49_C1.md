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
import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_inline import backend_inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms, models
import torchvision
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from PIL import Image
from types import SimpleNamespace
from torchvision.transforms import v2 as tforms

plt.set_cmap("cividis")
backend_inline.set_matplotlib_formats("svg", "pdf")
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

seed = 13
pl.seed_everything(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)




## === cell 1
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 2
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


def make_fname_class_df(train_dir):
    classes = sorted(os.listdir(train_dir))
    file_lst, class_lst, class_idx_lst = [], [], []
    for i, cl in enumerate(classes):
        path = os.path.join(train_dir, cl)
        files = os.listdir(path)
        file_lst += files
        class_lst += [cl] * len(files)
        class_idx_lst += [i] * len(files)
    return pd.DataFrame(
        {"file": file_lst, "class": class_lst, "class_idx": class_idx_lst}
    )


full_df = make_fname_class_df(TRAIN_DATASET_PATH)




## === cell 3
plt.figure(figsize=(12, 8))
sns.countplot(
    data=full_df,
    x="class",
    order=full_df["class"].value_counts().index,
    palette="Greens_r",
)
plt.xticks(rotation=45)
plt.show()




## === cell 4
class PlantTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.fname_class_df = make_fname_class_df(root_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.fname_class_df)

    def __getitem__(self, idx):
        row = self.fname_class_df.iloc[idx]
        path = os.path.join(self.root_dir, row["class"], row["file"])
        image = Image.open(path).convert("RGB")
        target = row["class_idx"]
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
        return len(self.fname_class_df)

    def __getitem__(self, idx):
        file_name = self.fname_class_df.iloc[idx]["file"]
        path = os.path.join(self.root_dir, file_name)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image




## === cell 5
def compute_mean_std(dataset, batch_size=8):
    rgb_values = torch.cat([img.reshape(3, -1) for img, _ in dataset], dim=-1)
    chunk_size = 224 * 224 * batch_size
    rgb_chunks = rgb_values.split(chunk_size, dim=1)
    means = [torch.mean(chunk.float(), dim=-1) for chunk in rgb_chunks]
    rgb_mean = torch.mean(torch.stack(means), dim=0).reshape(3, 1)
    vars_ = [
        torch.mean(((chunk - rgb_mean) ** 2).float(), dim=-1) for chunk in rgb_chunks
    ]
    rgb_std = torch.sqrt(torch.mean(torch.stack(vars_), dim=0))
    return rgb_mean.squeeze() / 255.0, rgb_std / 255.0




## === cell 6
class PlantDataModule(pl.LightningDataModule):
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

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            full_dataset = PlantTrainDataset(self.root_dir, transform=self.transform)
            n_total = len(full_dataset)
            n_train = int(0.8 * n_total)
            n_val = n_total - n_train
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(seed),
            )
        if stage == "test":
            full_dataset = PlantTrainDataset(self.root_dir, transform=self.transform)
            n_total = len(full_dataset)
            n_train = int(0.8 * n_total)
            n_val = n_total - n_train
            _, self.test_dataset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(seed),
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
            generator=torch.Generator().manual_seed(seed),
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
            generator=torch.Generator().manual_seed(seed),
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
            generator=torch.Generator().manual_seed(seed),
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
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=True,
        )




## === cell 7
model_dict = {}


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        raise ValueError(
            f'Unknown model name "{model_name}". Available models are: {list(model_dict.keys())}'
        )




## === cell 8
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
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.hparams.num_classes),
        )
        for param in self.features.parameters():
            param.requires_grad = False

    def _init_params(self):
        pass  # default initialization is sufficient

    def forward(self, x):
        ftrs = self.features(x)
        ftrs = ftrs.view(ftrs.size(0), -1)
        return self.classifier(ftrs)


model_dict["TLEfficientnet"] = TransferLearningEfficientnet




## === cell 9
class PlantSeedModule(pl.LightningModule):
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
        for name, hpr in zip(self.hparams.metrics_name, self.hparams.metrics_hparams):
            if name == "Accuracy":
                metric_lst.append(torchmetrics.Accuracy(**hpr))
            elif name == "Precision":
                metric_lst.append(torchmetrics.Precision(**hpr))
            elif name == "Recall":
                metric_lst.append(torchmetrics.Recall(**hpr))
            elif name == "F1Score":
                metric_lst.append(torchmetrics.F1Score(**hpr))
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
            raise ValueError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]




## === cell 10
def train_model(model_name, datamodule, max_epochs, save_name=None, **kwargs):
    if save_name is None:
        save_name = model_name
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        logger=CSVLogger(
            save_dir=os.path.join(CHECKPOINT_PATH, save_name),
            name="logs",
            version="version_1",
        ),
        callbacks=[
            ModelCheckpoint(
                mode="max",
                monitor="val_MulticlassAccuracy",
                verbose=True,
                save_last=True,
                save_top_k=2,
                enable_version_counter=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=50,
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = PlantSeedModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(seed)
        model = PlantSeedModule(model_name=model_name, **kwargs)
        trainer.fit(model, datamodule=datamodule, ckpt_path="last")
        model = PlantSeedModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
    test_result = trainer.test(model, datamodule=datamodule, verbose=True)
    return trainer, model, test_result




## === cell 11
batch_size = 32
plantdata = PlantDataModule(TRAIN_DATASET_PATH, TEST_DATASET_PATH, batch_size)




## === cell 12
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




## === cell 13
preds = trainer.predict(model, datamodule=plantdata)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/3262873873.py in <cell line: 0>()
----> 1 preds = trainer.predict(model, datamodule=plantdata)
      2 
      3 

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in predict(self, model, dataloaders, datamodule, return_predictions, ckpt_path)
    884         self.state.status = TrainerStatus.RUNNING
    885         self.predicting = True
--> 886         return call._call_and_handle_interrupt(
    887             self, self._predict_impl, model, dataloaders, datamodule, return_predictions, ckpt_path
    888         )

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
     47         if trainer.strategy.launcher is not None:
     48             return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
---> 49         return trainer_fn(*args, **kwargs)
     50 
     51     except _TunerExitException:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _predict_impl(self, model, dataloaders, datamodule, return_predictions, ckpt_path)
    925             self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
    926         )
--> 927         results = self._run(model, ckpt_path=ckpt_path)
    928 
    929         assert self.state.stopped

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run(self, model, ckpt_path)
   1009         # RUN THE TRAINER
   1010         # ----------------------------
-> 1011         results = self._run_stage()
   1012 
   1013         # ----------------------------

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_stage(self)
   1048             return self._evaluation_loop.run()
   1049         if self.predicting:
-> 1050             return self.predict_loop.run()
   1051         if self.training:
   1052             with isolate_rng():

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/utilities.py in _decorator(self, *args, **kwargs)
    177             context_manager = torch.no_grad
    178         with context_manager():
--> 179             return loop_run(self, *args, **kwargs)
    180 
    181     return _decorator

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/prediction_loop.py in run(self)
    120                 else:
    121                     dataloader_iter = None
--> 122                     batch, batch_idx, dataloader_idx = next(data_fetcher)
    123                 self.batch_progress.is_last_batch = data_fetcher.done
    124                 # run step hooks

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
    132         elif not self.done:
    133             # this will run only when no pre-fetching was done.
--> 134             batch = super().__next__()
    135         else:
    136             # the iterator is empty

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
     59         self._start_profiler()
     60         try:
---> 61             batch = next(self.iterator)
     62         except StopIteration:
     63             self.done = True

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
    339     def __next__(self) -> _ITERATOR_RETURN:
    340         assert self._iterator is not None
--> 341         out = next(self._iterator)
    342         if isinstance(self._iterator, _Sequential):
    343             return out

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
    140 
    141         try:
--> 142             out = next(self.iterators[0])
    143         except StopIteration:
    144             # try the next iterator

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

IsADirectoryError: Caught IsADirectoryError in DataLoader worker process 1.
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
  File "/tmp/ipykernel_55/1118067831.py", line 36, in __getitem__
    image = Image.open(path).convert("RGB")
            ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/plant-seedlings-classification/test/test'


## === cell 14
class_names = make_fname_class_df(TRAIN_DATASET_PATH)["class"].unique().tolist()
pred_indices = torch.cat(preds, dim=0).cpu().numpy()
species = np.vectorize(lambda idx: class_names[int(idx)])(pred_indices)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2166557736.py in <cell line: 0>()
      1 class_names = make_fname_class_df(TRAIN_DATASET_PATH)["class"].unique().tolist()
----> 2 pred_indices = torch.cat(preds, dim=0).cpu().numpy()
      3 species = np.vectorize(lambda idx: class_names[int(idx)])(pred_indices)
      4 
      5 

NameError: name 'preds' is not defined

## === cell 15
submission = pd.DataFrame(
    {"file": sorted(os.listdir(TEST_DATASET_PATH)), "species": species}
)
submission_path = os.path.join(CHECKPOINT_PATH, "submission.csv")
submission.to_csv(submission_path, index=False, header=True)
print(f"Submission written to {submission_path}")




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4030717343.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"file": sorted(os.listdir(TEST_DATASET_PATH)), "species": species}
      3 )
      4 submission_path = os.path.join(CHECKPOINT_PATH, "submission.csv")
      5 submission.to_csv(submission_path, index=False, header=True)

NameError: name 'species' is not defined

## === cell 16
submission.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
