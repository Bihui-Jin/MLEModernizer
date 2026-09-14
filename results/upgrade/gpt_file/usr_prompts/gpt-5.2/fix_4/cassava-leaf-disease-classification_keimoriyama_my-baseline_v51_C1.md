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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8860682985796313

# 6. Current score

0.08782

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11173) has done: 'I remove the hard dependency on missing local pretrained weight files by switching to `pretrained=True` for the same timm model names, preserving the exact architectures while unblocking execution. I also fix inference-time issues by filtering `test_images/` to only real image files (avoiding nested directories) and aligning the submission order to `sample_submission.csv` so `image_id` and `label` lengths always match. Finally, I fix the validation utility to avoid Pandas index pitfalls and ensure predictions are constrained to 5 classes, and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.08782) has done: 'The timeout is dominated by extremely heavy training (3-fold KFold × 5 epochs × two large pretrained CNNs at 512px) and by slow per-image Python loops during validation/test inference. Without changing the training logic or models, the biggest speedups come from eliminating per-sample PIL loading inside loops by switching evaluation/inference to batched DataLoaders, and from reducing DataLoader overhead via persistent workers, higher worker count, prefetching, and pinned-memory usage. I also remove non-essential notebook display work (plots/image show) that can block/slow execution, and I enable deterministic settings while using cuDNN benchmarking only when it won’t change results (kept deterministic). These changes preserve the exact training procedure/architecture while cutting constant-factor overhead enough to fit the 600s budget.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")



## === cell 1
import pandas as pd
import timm



## === cell 2
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)



## === cell 3
df = pd.read_csv(path + "/train.csv")



## === cell 4
_ = df.head(0)



## === cell 5
df["path"] = df["image_id"].map(lambda x: path + "/train_images/" + x)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 6
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 7
pass



## === cell 8
pass



## === cell 9
train_df = train_df.reset_index(drop=True)
_ = train_df.head(0)



## === cell 10
valid_df = valid_df.reset_index(drop=True)
_ = valid_df.head(0)



## === cell 11
pass



## === cell 12
pass



## === cell 13
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from PIL import Image, ImageDraw

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from sklearn.model_selection import KFold




## === cell 14
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


seed_everything(42)




## === cell 15
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df["path"])

    def __getitem__(self, index):
        path = self.df.loc[index, "path"]
        label = int(self.df.loc[index, "label"])
        with open(path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 16
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        start_width, start_height = [], []
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            mw = min(self.mask_size, max(1, width - 1))
            mh = min(self.mask_size, max(1, height - 1))
            for _ in range(10):
                if width - mw <= 0 or height - mh <= 0:
                    break
                start_width.append(random.randrange(0, width - mw))
                start_height.append(random.randrange(0, height - mh))
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + mw, y + mh),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image




## === cell 17
image_size = 512
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        make_mask_image(p=0.3, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 18
pass



## === cell 19
dataset = CassavaDataset(train_df, train_transform)



## === cell 20
epoch = 5
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 21
resNet = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
resNet = resNet.to(device)



## === cell 22
ef_model = timm.create_model(
    "tf_efficientnet_b2_ns", pretrained=True, num_classes=num_classes
)
ef_model = ef_model.to(device)



## === cell 23
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 24
class CassavaEvalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        y = int(self.df.loc[idx, "label"])
        with open(p, "rb") as f:
            im = Image.open(f).convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, y


def calc_correction(model, df_eval, batch_size=64, num_workers=None):
    model.eval()
    df_eval = df_eval.reset_index(drop=True)

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 2)

    eval_ds = CassavaEvalDataset(df_eval, transform=valid_transform)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    correct = 0
    total = 0
    pred_list = [0, 0, 0, 0, 0]

    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(1).clamp(0, 4)
            binc = torch.bincount(preds.detach().cpu(), minlength=5).tolist()
            pred_list = [a + b for a, b in zip(pred_list, binc)]
            correct += (preds == yb).sum().item()
            total += yb.numel()

    return (correct / total) if total else 0.0, pred_list




## === cell 25
def plot_losses(epoch, title, train_losses, valid_losses):
    try:
        from matplotlib import pyplot as plt

        y = list(range(len(train_losses)))
        train_loss = plt.plot(y, train_losses)
        valid_loss = plt.plot(y, valid_losses)
        plt.title(title)
        plt.ylabel("loss")
        plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
        plt.show()
    except Exception:
        pass




## === cell 26
import time
import copy




## === cell 27
def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_state_dict = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    num_workers = min(4, os.cpu_count() or 2)
    pin = torch.cuda.is_available()

    for fold, (train_index, valid_index) in enumerate(kf.split(range(len(dataset)))):
        print("fold: ", fold)
        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        valid_dataset = Subset(dataset, valid_index)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        for ep in range(1, epoch + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for data, target in train_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                for data, target in valid_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    output = model(data)
                    pred = output.argmax(1) == target
                    acc.append((pred.float().mean()).item())

                    loss = criterion(output, target)
                    valid_loss += loss.item() * len(data)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state_dict = copy.deepcopy(model.state_dict())

            scheduler.step()

            collection = sum(acc) / len(acc) if len(acc) else 0.0
            valid_loss = valid_loss / len(valid_loader.dataset)
            valid_losses.append(valid_loss)
            print(
                "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                    time.time() - epoch_start_time,
                    ep,
                    train_loss,
                    valid_loss,
                    collection,
                )
            )

    if best_state_dict is not None:
        torch.save(best_state_dict, model_title)

    return model, train_losses, valid_losses




## === cell 28
model_title = "./res_model.pth"



## === cell 29
title = "resNet losses"



## === cell 30
resNet, res_train_losses, res_valid_losses = train_model(
    resNet,
    dataset,
    batch_size,
    resNet_optimizer,
    criterion,
    resNet_scheduler,
    epoch,
    model_title,
)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3386759480.py in <cell line: 0>()
----> 1 resNet, res_train_losses, res_valid_losses = train_model(
      2     resNet,
      3     dataset,
      4     batch_size,
      5     resNet_optimizer,

/tmp/ipykernel_55/3567165254.py in train_model(model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title)
     49                 output = model(data)
     50                 loss = criterion(output, target)
---> 51                 loss.backward()
     52                 optimizer.step()
     53                 train_loss += loss.item() * len(data)

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 31
ef_model_title = "./ef_model.pth"
ef_model, ef_train_losses, ef_valid_losses = train_model(
    ef_model,
    dataset,
    batch_size,
    ef_optimizer,
    criterion,
    ef_scheduler,
    epoch,
    ef_model_title,
)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2249299136.py in <cell line: 0>()
      1 ef_model_title = "./ef_model.pth"
----> 2 ef_model, ef_train_losses, ef_valid_losses = train_model(
      3     ef_model,
      4     dataset,
      5     batch_size,

/tmp/ipykernel_55/3567165254.py in train_model(model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title)
     49                 output = model(data)
     50                 loss = criterion(output, target)
---> 51                 loss.backward()
     52                 optimizer.step()
     53                 train_loss += loss.item() * len(data)

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 32
if os.path.isfile(model_title):
    resNet.load_state_dict(torch.load(model_title, map_location=device))
if os.path.isfile(ef_model_title):
    ef_model.load_state_dict(torch.load(ef_model_title, map_location=device))




## === cell 33
class CassaveClassifier(nn.Module):
    def __init__(self, model, ef_model):
        super().__init__()
        self.model = model
        self.ef_model = ef_model

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return 0.5 * x1 + 0.5 * x2




## === cell 34
classifier = CassaveClassifier(resNet, ef_model)
classifier = classifier.to(device)



## === cell 35
try:
    val_acc, pred_hist = calc_correction(classifier, valid_df, batch_size=64)
    val_acc, pred_hist
except Exception as e:
    print("Validation check skipped due to:", repr(e))



## === cell 36
path_test = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 37
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(path_test, img_id) for img_id in image_id]

missing = [p for p in image_path if not os.path.isfile(p)]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test image files. Example: {missing[0]}"
    )




## === cell 38
class CassavaTestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        with open(p, "rb") as f:
            im = Image.open(f).convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im


classifier.eval()
pred = []

num_workers = min(4, os.cpu_count() or 2)
test_ds = CassavaTestDataset(image_path, transform=valid_transform)
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = classifier(xb)
        preds = logits.argmax(1).clamp(0, 4).detach().cpu().tolist()
        pred.extend([int(p) for p in preds])

len(pred), len(image_id)



## === cell 39
pred[:10]



## === cell 40
sub = pd.DataFrame({"image_id": image_id, "label": pred})
sub.head()



## === cell 41
sub.shape



## === cell 42
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
