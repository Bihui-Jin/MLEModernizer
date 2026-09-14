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

0.5945

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import tqdm


def _find_base_path():
    possible_paths = [
        "./input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "./data/cassava-leaf-disease-classification",
    ]
    for p in possible_paths:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Base data folder not found among expected locations.")


BASE_PATH = _find_base_path()

train_data_path = os.path.join(BASE_PATH, "train_images")
test_data_path = os.path.join(BASE_PATH, "test_images")
train_csv_path = os.path.join(BASE_PATH, "train.csv")

img_resize = (224, 224)  # VGG16 input size
batch_size = 64
train_size = 0.9  # proportion of images used for training split
model_pretrained = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class get_dataset(Dataset):
    def __init__(self, data_path, csv_label_path, train, train_size, transforms=None):
        self.train = train
        self.data_path = data_path
        self.transforms = transforms

        self.label_csv = pd.read_csv(csv_label_path)
        self.label_dict = self.label_csv.set_index("image_id")["label"].to_dict()

        images_name_list = os.listdir(data_path)
        train_image, test_image = train_test_split(
            images_name_list, train_size=train_size, random_state=0
        )
        self.image_list = train_image if self.train else test_image

    def __getitem__(self, index):
        image_name = self.image_list[index]
        label = self.label_dict[image_name]
        image = Image.open(os.path.join(self.data_path, image_name)).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.image_list)


class get_test_dataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.image_list = os.listdir(data_path)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(os.path.join(self.data_path, image_name)).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, image_name

    def __len__(self):
        return len(self.image_list)


train_transforms = transforms.Compose(
    [
        transforms.Resize(img_resize),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(),
    ]
)

val_test_transforms = transforms.Compose(
    [
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = get_dataset(
    train_data_path, train_csv_path, True, train_size, train_transforms
)
validation_dataset = get_dataset(
    train_data_path, train_csv_path, False, train_size, val_test_transforms
)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

test_dataset = get_test_dataset(test_data_path, val_test_transforms)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)




## === cell 2
model = models.vgg16_bn(pretrained=model_pretrained)
sequential = list(model.classifier[:3])
sequential.append(nn.Linear(4096, 5))
model.classifier = nn.Sequential(*sequential)
model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5
)
criterion = nn.CrossEntropyLoss()




## === cell 3
def train(cur_epoch, dataloader, compute_grid=True):
    tq_description = f"epoch {cur_epoch}"
    tqbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.0
    preds_list = []
    labels_list = []

    for i, item in tqbar:
        tqbar.set_description(tq_description)
        images, labels = item
        images = images.to(device)
        labels = labels.to(device)

        model_out = model(images)
        loss = criterion(model_out, labels)
        _, preds = torch.max(model_out, 1)

        if compute_grid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds_list += preds.tolist()
        labels_list += labels.tolist()
    return preds_list, labels_list, total_loss


def generate_submission_csv():
    tq_description = "generate csv"
    tqbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    if os.path.isfile("model.pkl"):
        model.load_state_dict(torch.load("model.pkl"))
    else:
        print("Warning: model.pkl not found – using current model weights.")

    names_list = []
    preds_list = []
    for i, item in tqbar:
        tqbar.set_description(tq_description)
        images, names = item
        images = images.to(device)
        model_out = model(images)
        _, preds = torch.max(model_out, 1)

        names_list += list(names)
        preds_list += preds.tolist()

    submission = pd.DataFrame({"image_id": names_list, "label": preds_list})
    submission.to_csv("submission.csv", index=False)
    print("submission.csv generated.")


def compute_recall(preds_list, labels_list, class_num):
    preds_arr = np.array(preds_list)
    labels_arr = np.array(labels_list)
    recall_arr = np.zeros(class_num)
    for i in range(class_num):
        i_labels_mask = labels_arr == i
        i_preds_mask = preds_arr == i
        total_i_class_num = np.sum(i_labels_mask)
        preds_i_class_num = np.sum(i_preds_mask & i_labels_mask)
        recall_arr[i] = (
            preds_i_class_num / total_i_class_num if total_i_class_num != 0 else 0
        )
    return recall_arr


def compute_accuracy(preds_list, labels_list):
    preds_arr = np.array(preds_list)
    labels_arr = np.array(labels_list)
    return np.sum(preds_arr == labels_arr) / len(labels_arr)


def do_train(epoch):
    train_loss_list = []
    train_accuracy_list = []
    train_recall_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_recall_list = []

    best_accuracy = [-1, -1]  # (epoch, value)

    print("info:")
    print("train image number: ", len(train_dataset))
    print("validation image number:", len(validation_dataset))
    print("train on:", device)
    print("train epochs:", epoch)

    for i in range(epoch):
        preds_list, labels_list, total_loss = train(i, train_dataloader, True)
        accuracy = compute_accuracy(preds_list, labels_list)
        recall = compute_recall(preds_list, labels_list, 5)
        train_loss_list.append(total_loss)
        train_accuracy_list.append(accuracy)
        train_recall_list.append(recall)
        print(f"train loss: {total_loss:.4f}")
        print(f"train accuracy: {accuracy:.4f}")
        print("train recall:", recall)

        preds_list, labels_list, total_loss = train(i, validation_dataloader, False)
        accuracy = compute_accuracy(preds_list, labels_list)
        recall = compute_recall(preds_list, labels_list, 5)
        val_loss_list.append(total_loss)
        val_accuracy_list.append(accuracy)
        val_recall_list.append(recall)
        print(f"validation loss: {total_loss:.4f}")
        print(f"validation accuracy: {accuracy:.4f}")
        print("validation recall:", recall)

        if best_accuracy[1] < accuracy:
            best_accuracy = [i, accuracy]
            torch.save(model.state_dict(), "model.pkl")

    plt.figure()
    plt.plot(train_loss_list, label="train")
    plt.plot(val_loss_list, label="validation")
    plt.title("Loss")
    plt.legend()

    plt.figure()
    plt.plot(train_accuracy_list, label="train")
    plt.plot(val_accuracy_list, label="validation")
    plt.title("Accuracy")
    plt.legend()

    plt.figure()
    plt.plot(train_recall_list, label="train")
    plt.plot(val_recall_list, label="validation")
    plt.title("Recall")
    plt.legend()
    plt.show()




## === cell 4
do_train(5)  # quick training
generate_submission_csv()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2271764135.py in <cell line: 0>()
----> 1 do_train(5)  # quick training
      2 generate_submission_csv()

/tmp/ipykernel_55/2867032725.py in do_train(epoch)
     92 
     93     for i in range(epoch):
---> 94         preds_list, labels_list, total_loss = train(i, train_dataloader, True)
     95         accuracy = compute_accuracy(preds_list, labels_list)
     96         recall = compute_recall(preds_list, labels_list, 5)

/tmp/ipykernel_55/2867032725.py in train(cur_epoch, dataloader, compute_grid)
      6     labels_list = []
      7 
----> 8     for i, item in tqbar:
      9         tqbar.set_description(tq_description)
     10         images, labels = item

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

KeyError: Caught KeyError in DataLoader worker process 1.
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
  File "/tmp/ipykernel_55/1503327674.py", line 18, in __getitem__
    label = self.label_dict[image_name]
            ~~~~~~~~~~~~~~~^^^^^^^^^^^^
KeyError: 'train_images'
