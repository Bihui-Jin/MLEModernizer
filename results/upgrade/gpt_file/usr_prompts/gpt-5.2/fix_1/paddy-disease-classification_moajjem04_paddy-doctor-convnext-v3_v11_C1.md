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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.2292626728110599

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import torch
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from tqdm.notebook import tqdm
from torchmetrics import Accuracy
import pandas as pd
from skimage import io

## === cell 1
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

## === cell 3
Prob = 0.5
train_tf = transforms.Compose([transforms.RandomHorizontalFlip(Prob),
                              transforms.RandomVerticalFlip(Prob),
                              transforms.RandomResizedCrop((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                              ])
val_tf = transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                              ])

## === cell 4
train_ds = ImageFolder("../input/paddy-doctor-make-train-and-val-folder/Train", transform = train_tf)
val_ds = ImageFolder("../input/paddy-doctor-make-train-and-val-folder/Val", transform = val_tf)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1199185727.py in <cell line: 0>()
----> 1 train_ds = ImageFolder("../input/paddy-doctor-make-train-and-val-folder/Train", transform = train_tf)
      2 val_ds = ImageFolder("../input/paddy-doctor-make-train-and-val-folder/Val", transform = val_tf)

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, transform, target_transform, loader, is_valid_file, allow_empty)
    326         allow_empty: bool = False,
    327     ):
--> 328         super().__init__(
    329             root,
    330             loader,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)
    147     ) -> None:
    148         super().__init__(root, transform=transform, target_transform=target_transform)
--> 149         classes, class_to_idx = self.find_classes(self.root)
    150         samples = self.make_dataset(
    151             self.root,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in find_classes(self, directory)
    232             (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
    233         """
--> 234         return find_classes(directory)
    235 
    236     def __getitem__(self, index: int) -> Tuple[Any, Any]:

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in find_classes(directory)
     39     See :class:`DatasetFolder` for details.
     40     """
---> 41     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
     42     if not classes:
     43         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

FileNotFoundError: [Errno 2] No such file or directory: '../input/paddy-doctor-make-train-and-val-folder/Train'

## === cell 5
train_ds.class_to_idx

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2619565017.py in <cell line: 0>()
----> 1 train_ds.class_to_idx

NameError: name 'train_ds' is not defined

## === cell 6
val_ds.class_to_idx

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/196948309.py in <cell line: 0>()
----> 1 val_ds.class_to_idx

NameError: name 'val_ds' is not defined

## === cell 7
train_loader    = DataLoader(train_ds,batch_size= 64*8, shuffle = True, pin_memory=True, drop_last=True)
val_loader      = DataLoader(val_ds,  batch_size= 64*8, shuffle = True, pin_memory=True)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1375171687.py in <cell line: 0>()
----> 1 train_loader    = DataLoader(train_ds,batch_size= 64*8, shuffle = True, pin_memory=True, drop_last=True)
      2 val_loader      = DataLoader(val_ds,  batch_size= 64*8, shuffle = True, pin_memory=True)

NameError: name 'train_ds' is not defined

## === cell 8
model_dir = '../input/pytorch-model-weights/convnext_models'
model_name = 'convnext_tiny.pt'
model = torch.load(os.path.join(model_dir, model_name))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/544413583.py in <cell line: 0>()
      2 model_dir = '../input/pytorch-model-weights/convnext_models'
      3 model_name = 'convnext_tiny.pt'
----> 4 model = torch.load(os.path.join(model_dir, model_name))

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/pytorch-model-weights/convnext_models/convnext_tiny.pt'

## === cell 9
for param in model.parameters():
        param.requires_grad = False

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4124871757.py in <cell line: 0>()
----> 1 for param in model.parameters():
      2         param.requires_grad = False

NameError: name 'model' is not defined

## === cell 10
model.classifier

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1355538586.py in <cell line: 0>()
----> 1 model.classifier

NameError: name 'model' is not defined

## === cell 11
model.classifier[-1] = torch.nn.Linear(in_features = 768, out_features = 10) #tiny


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2067110718.py in <cell line: 0>()
----> 1 model.classifier[-1] = torch.nn.Linear(in_features = 768, out_features = 10) #tiny
      2 #model.classifier[-1] = torch.nn.Linear(in_features = 1024, out_features = 10) #base

NameError: name 'model' is not defined

## === cell 12
for param in model.parameters():
        if param.requires_grad:
            print(param.shape)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3504242534.py in <cell line: 0>()
----> 1 for param in model.parameters():
      2         if param.requires_grad:
      3             print(param.shape)

NameError: name 'model' is not defined

## === cell 13
def train(data, target, model, optimizer, criterion, TRAIN):

    if TRAIN:
        optimizer.zero_grad()

    output = model(data)
    
    norms = torch.norm(output, p=2, dim=-1, keepdim=True) + 1e-7
    logit_norm = torch.div(output, norms)
    loss = criterion(logit_norm , target)

    if TRAIN:
        loss.backward()  
        optimizer.step()     

    return output, loss

## === cell 14
optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4) 
criterion = torch.nn.CrossEntropyLoss() 

device = 'cuda'
model.to(device)

n_epochs = 1


acc = Accuracy().to(device)
print(f"\nTraining Model")
best_val_ = 1000
failure_count = 0
for epoch in tqdm(range(n_epochs), desc = "# Epochs", position= 0):
    train_loss = 0
    val_loss = 0

    train_acc = 0
    val_acc = 0

    model.train()

    for i, (data, target) in enumerate(tqdm(train_loader, desc = "Training", leave= True, position= 1)):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        output, loss = train(data, target, model, optimizer, criterion, TRAIN= True)

        train_loss += loss.item() * data.size(0)
        _, output = torch.max(output, dim=1) 
        acc(output, target.squeeze())

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = acc.compute().item()
    acc.reset()

    with torch.no_grad():
        model.eval()

        for i, (data, target) in enumerate(tqdm(val_loader, desc = "Validation", leave= True, position= 2)):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output, loss = train(data, target, model, optimizer, criterion, TRAIN= False)
            val_loss += loss.item() * data.size(0)
            _, output = torch.max(output, dim=1) 
            acc(output, target.squeeze())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = acc.compute().item()
        acc.reset()
        
    if val_loss < best_val_:
        best_val_ = val_loss
        best_acc = val_acc
        failure_count = 0
        torch.save(model, 'best_model.pt')
    else:
        failure_count += 1
    
    if failure_count >= 10:
        break
        
    print(f"Epoch # {epoch+1:04d}")
    print(f"Train Loss: {train_loss: .4f},\t Val Loss: {val_loss: .4f}")
    print(f"Train Acc : {train_acc: .4f},\t Val Acc : {val_acc: .4f}")  
    print(f"Best Val Loss : {best_val_: .4f},\t Val Acc : {best_acc: .4f}")
    print(f"Failure Count = {failure_count}")


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2609814312.py in <cell line: 0>()
      1 # Training Options
----> 2 optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
      3 criterion = torch.nn.CrossEntropyLoss()
      4 
      5 # training device  ('cpu', 'cuda')

NameError: name 'model' is not defined

## === cell 16
train_ds.class_to_idx

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2619565017.py in <cell line: 0>()
----> 1 train_ds.class_to_idx

NameError: name 'train_ds' is not defined

## === cell 17
class_dict = {v: k for k, v in train_ds.class_to_idx.items()}
class_dict

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2994551531.py in <cell line: 0>()
----> 1 class_dict = {v: k for k, v in train_ds.class_to_idx.items()}
      2 class_dict

NameError: name 'train_ds' is not defined

## === cell 18
sample_df = pd.read_csv("../input/paddy-disease-classification/sample_submission.csv")
sample_df.head()

## === cell 19
model = torch.load('best_model.pt')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/427499462.py in <cell line: 0>()
      1 # Load Best Model
----> 2 model = torch.load('best_model.pt')

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

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pt'

## === cell 20
test_dir = "../input/paddy-disease-classification/test_images"
file_list = os.listdir(test_dir)
file_list[:5]

## === cell 21
test_tfs = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                ])

## === cell 22
result_df = pd.DataFrame(columns = sample_df.columns)
print("Begin Inference\n")
for ii, file_name in enumerate(tqdm(file_list)):
    image = io.imread(os.path.join(test_dir, file_name))
    image = test_tfs(image)
    image = image.unsqueeze(0)
    out = model(image.to('cuda', non_blocking=True))
    out = torch.argmax(out).item()
    label = class_dict[out]
    result_df.loc[ii,:] = [file_name, label]
    
    if (i+1) % 200 == 0:
        print(f"{i+1:04d}/{len(file_list)}")
else:
    print("All Test Images are processed")

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/899750933.py in <cell line: 0>()
      5     image = test_tfs(image)
      6     image = image.unsqueeze(0)
----> 7     out = model(image.to('cuda', non_blocking=True))
      8     out = torch.argmax(out).item()
      9     label = class_dict[out]

NameError: name 'model' is not defined

## === cell 23
result_df.to_csv('submission.csv', index = False, columns = result_df.columns)
print("\nSubmission File Created!")

## --- ERROR in outputing the csv:
Invalid submission: Expected 2602 rows, but got 0
