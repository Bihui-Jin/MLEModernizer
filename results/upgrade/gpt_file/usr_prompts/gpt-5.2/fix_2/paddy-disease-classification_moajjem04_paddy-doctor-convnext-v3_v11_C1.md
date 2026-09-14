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

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the missing input paths by building the train/val datasets directly from `train.csv` and `train_images/` (instead of a non-existent pre-split folder), and I replace the missing external `.pt` weight load with a torchvision ConvNeXt-Tiny initialized from built-in ImageNet weights. I keep your same transforms, training loop structure, and “normalize logits then CrossEntropyLoss” semantics, but make it run reliably on CPU/GPU and with correct torchmetrics usage for multiclass. I also fix inference bugs (device handling, undefined `i` vs `ii`, and model eval/no_grad) and ensure the submission has exactly the sample rows and correct column names/order. This should run end-to-end in the Kaggle environment and produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from tqdm.notebook import tqdm
from torchmetrics import Accuracy
import pandas as pd
from skimage import io




## === cell 1
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 2
Prob = 0.5
train_tf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(Prob),
        transforms.RandomVerticalFlip(Prob),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 3
DATA_DIR = "../input/paddy-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_ROOT = os.path.join(DATA_DIR, "train_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

labels = sorted(train_df["label"].unique().tolist())
label2idx = {lab: i for i, lab in enumerate(labels)}
idx2label = {i: lab for lab, i in label2idx.items()}

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=train_df["label"]
)
tr_df = tr_df.reset_index(drop=True)
va_df = va_df.reset_index(drop=True)


class PaddyCSVDataset(Dataset):
    def __init__(self, df, img_root, transform=None, label2idx=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = row["label"]

        img_path = os.path.join(self.img_root, label, image_id)
        img = io.imread(img_path)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        if self.transform is not None:
            img = self.transform(img)

        target = self.label2idx[label] if self.label2idx is not None else -1
        return img, torch.tensor(target, dtype=torch.long)


train_ds = PaddyCSVDataset(
    tr_df, TRAIN_IMG_ROOT, transform=train_tf, label2idx=label2idx
)
val_ds = PaddyCSVDataset(va_df, TRAIN_IMG_ROOT, transform=val_tf, label2idx=label2idx)

print("Num classes:", len(labels))
print("Train size:", len(train_ds), "Val size:", len(val_ds))



## === cell 4
train_ds.class_to_idx = label2idx
train_ds.class_to_idx



## === cell 5
val_ds.class_to_idx = label2idx
val_ds.class_to_idx



## === cell 6
BATCH_SIZE = 64
NUM_WORKERS = 2

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=NUM_WORKERS,
)



## === cell 7
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)



## === cell 8
for param in model.parameters():
    param.requires_grad = False



## === cell 9
model.classifier



## === cell 10
model.classifier[-1] = torch.nn.Linear(in_features=768, out_features=10)  # tiny



## === cell 11
for param in model.parameters():
    if param.requires_grad:
        print(param.shape)




## === cell 12
def train(data, target, model, optimizer, criterion, TRAIN):

    if TRAIN:
        optimizer.zero_grad()

    output = model(data)

    norms = torch.norm(output, p=2, dim=-1, keepdim=True) + 1e-7
    logit_norm = torch.div(output, norms)
    loss = criterion(logit_norm, target)

    if TRAIN:
        loss.backward()
        optimizer.step()

    return output, loss




## === cell 13
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

n_epochs = 1

acc = Accuracy(task="multiclass", num_classes=10).to(device)
print(f"\nTraining Model on device={device}")
best_val_ = 1000
failure_count = 0

for epoch in tqdm(range(n_epochs), desc="# Epochs", position=0):
    train_loss = 0
    val_loss = 0

    train_acc = 0
    val_acc = 0

    model.train()

    for i, (data, target) in enumerate(
        tqdm(train_loader, desc="Training", leave=True, position=1)
    ):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output, loss = train(data, target, model, optimizer, criterion, TRAIN=True)

        train_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, dim=1)
        acc(pred, target)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = acc.compute().item()
    acc.reset()

    with torch.no_grad():
        model.eval()

        for i, (data, target) in enumerate(
            tqdm(val_loader, desc="Validation", leave=True, position=2)
        ):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output, loss = train(data, target, model, optimizer, criterion, TRAIN=False)
            val_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, dim=1)
            acc(pred, target)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = acc.compute().item()
        acc.reset()

    if val_loss < best_val_:
        best_val_ = val_loss
        best_acc = val_acc
        failure_count = 0
        torch.save(model, "best_model.pt")
    else:
        failure_count += 1

    if failure_count >= 10:
        break

    print(f"Epoch # {epoch+1:04d}")
    print(f"Train Loss: {train_loss: .4f},\t Val Loss: {val_loss: .4f}")
    print(f"Train Acc : {train_acc: .4f},\t Val Acc : {val_acc: .4f}")
    print(f"Best Val Loss : {best_val_: .4f},\t Val Acc : {best_acc: .4f}")
    print(f"Failure Count = {failure_count}")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/821439364.py in <cell line: 0>()
     22     model.train()
     23 
---> 24     for i, (data, target) in enumerate(
     25         tqdm(train_loader, desc="Training", leave=True, position=1)
     26     ):

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
  File "/tmp/ipykernel_55/2248674014.py", line 51, in __getitem__
    img = self.transform(img)
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
          ^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 713, in forward
    return F.hflip(img)
           ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py", line 669, in hflip
    return F_pil.hflip(img)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/_functional_pil.py", line 54, in hflip
    raise TypeError(f"img should be PIL Image. Got {type(img)}")
TypeError: img should be PIL Image. Got <class 'numpy.ndarray'>


## === cell 14
pass



## === cell 15
train_ds.class_to_idx



## === cell 16
class_dict = {v: k for k, v in train_ds.class_to_idx.items()}
class_dict



## === cell 17
sample_df = pd.read_csv("../input/paddy-disease-classification/sample_submission.csv")
sample_df.head()



## === cell 18
if os.path.exists("best_model.pt"):
    model = torch.load("best_model.pt", map_location=device)
model.to(device)
model.eval()



## === cell 19
test_dir = "../input/paddy-disease-classification/test_images"
file_list = os.listdir(test_dir)
file_list[:5], len(file_list)



## === cell 20
test_tfs = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 21
result_df = sample_df.copy()

pred_map = {}

print("Begin Inference\n")
with torch.no_grad():
    for ii, file_name in enumerate(tqdm(file_list)):
        image = io.imread(os.path.join(test_dir, file_name))

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        image = test_tfs(image).unsqueeze(0).to(device, non_blocking=True)
        out = model(image)
        out = torch.argmax(out, dim=1).item()
        label = class_dict[out]
        pred_map[file_name] = label

        if (ii + 1) % 200 == 0:
            print(f"{ii+1:04d}/{len(file_list)}")
    else:
        print("All Test Images are processed")

result_df["label"] = result_df["image_id"].map(pred_map)

if result_df["label"].isna().any():
    fallback = train_df["label"].mode().iloc[0]
    result_df["label"] = result_df["label"].fillna(fallback)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/2389765322.py in <cell line: 0>()
     12 with torch.no_grad():
     13     for ii, file_name in enumerate(tqdm(file_list)):
---> 14         image = io.imread(os.path.join(test_dir, file_name))
     15 
     16         if image.ndim == 2:

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
    221             "Specify the plugin explicitly using the `plugin` kwarg, e.g. `plugin='DICOM'`"
    222         )
--> 223         raise err_type(err_msg)
    224 
    225     # close the current request here and use fresh/new ones while trying each

OSError: ImageIO does not generally support reading folders. Limited support may be available via specific plugins. Specify the plugin explicitly using the `plugin` kwarg, e.g. `plugin='DICOM'`

## === cell 22
result_df = result_df[["image_id", "label"]]
assert len(result_df) == len(
    sample_df
), f"Expected {len(sample_df)} rows, got {len(result_df)}"
result_df.to_csv("submission.csv", index=False)
print("\nSubmission File Created! -> submission.csv")
print(result_df.head())
