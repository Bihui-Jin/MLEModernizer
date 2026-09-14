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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8555454820187368

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10613) has done: 'I make the script reliably create a submission.csv by loading the sample‑submission file and iterating over its image list (ensuring the order matches Kaggle’s expected format). I add a safe‑guard when loading the pretrained weights: if the provided *.pth file is missing, the model falls back to the default torchvision weights so the code can still run. Minor error handling and deterministic sorting are introduced, but the core model architecture and inference logic stay unchanged. These changes guarantee a valid CSV output and keep the performance close to the original model’s behavior, moving the solution toward the target score.'
- What this solution (achieved 0.12369) has done: 'The fix removes the TensorFlow import that caused a protobuf‑related crash, keeping only the libraries actually used for image loading, preprocessing, and PyTorch inference. No core modeling logic is altered, so the original architecture and prediction pipeline remain intact while allowing the script to run and generate a valid `submission.csv`.'
- What this solution (achieved 0.11921) has done: 'The script was failing because essential libraries weren’t imported, the image‑preprocessing transform was missing, and variables were referenced before they were defined.  
I added the required imports, set up a simple validation transform that resizes images to the selected model’s input size, ensured the model‑selection logic runs after the imports, and kept the original architecture unchanged. These fixes let the notebook run end‑to‑end and generate a correctly‑named `submission.csv` while preserving the original prediction logic.'
- What this solution (achieved 0.24963) has done: 'The changes focus on eliminating the costly training step (which is unnecessary because pretrained weights are already loaded) and modestly speeding up data loading. By setting `num_epochs = 0` and directly putting the model in evaluation mode we keep the same architecture and weight loading logic while removing the expensive forward‑backward pass. Minor tweaks such as larger batch sizes, enabling cuDNN benchmarking, and reducing the number of workers further cut overhead without altering any predictions or evaluation semantics.'
- What this solution (achieved 0.05531) has done: 'The change enables a short fine‑tuning phase (rather than skipping training) by setting `num_epochs` to a small positive value. This keeps the original model architecture, loss, optimizer, and data pipeline unchanged while allowing the pretrained network to adapt to the Cassava data, which should raise validation accuracy and move the Kaggle score toward the target.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T, models
from PIL import Image
from tqdm import tqdm

torch.backends.cudnn.benchmark = True  # enable faster GPU kernel selection

base_path = "/kaggle/input/cassava-leaf-disease-classification"

train_csv_path = os.path.join(base_path, "train.csv")
train_data_directory = os.path.join(base_path, "train_images")
test_data_directory = os.path.join(base_path, "test_images")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/5/vit_h_14_518_8369_base.pth"
)
vit_image_size = 518

model_select = "en"  # keep existing choice

if model_select == "vit":
    model_image_size = vit_image_size
elif model_select == "en":
    model_image_size = en_image_size
else:
    raise ValueError("model_select must be 'vit' or 'en'")

if model_select == "en":
    en_model = models.efficientnet_v2_l(pretrained=False)
    in_features = en_model.classifier[1].in_features
    en_model.classifier[1] = nn.Linear(in_features, num_classes)
    try:
        state = torch.load(en_model_path, map_location=device)
        en_model.load_state_dict(state)
        print("Loaded EfficientNetV2‑L weights from custom checkpoint.")
    except Exception as e:
        print(
            f"Custom EfficientNet checkpoint not loaded ({e}), using ImageNet pretrained weights."
        )
        en_model = models.efficientnet_v2_l(pretrained=True)
        en_model.classifier[1] = nn.Linear(in_features, num_classes)
    en_model = en_model.to(device)
else:
    vit_model = models.vit_h_14(pretrained=False)
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, num_classes)
    try:
        state = torch.load(vit_model_path, map_location=device)
        vit_model.load_state_dict(state)
        print("Loaded ViT weights from custom checkpoint.")
    except Exception as e:
        print(
            f"Custom ViT checkpoint not loaded ({e}), using ImageNet pretrained weights."
        )
        vit_model = models.vit_h_14(pretrained=True)
        vit_model.heads.head = nn.Linear(in_features, num_classes)
    vit_model = vit_model.to(device)

if model_select == "en":

    class CassavaDataset(Dataset):
        def __init__(self, csv_path, img_dir, transform):
            self.df = pd.read_csv(csv_path)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = int(row["label"])
            return image, label

    train_transforms = T.Compose(
        [
            T.Resize((model_image_size, model_image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = CassavaDataset(
        train_csv_path, train_data_directory, train_transforms
    )

    train_len = int(0.9 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # increased from 8
        shuffle=True,
        num_workers=2,  # reduced to avoid oversubscription
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,  # increased from 16
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(en_model.parameters(), lr=1e-4)

    en_model.train()
    num_epochs = 3  # modest number of epochs to improve accuracy
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = en_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / train_len
        en_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = en_model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}"
        )
        en_model.train()
    en_model.eval()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/3822662360.py in <cell line: 0>()
    135 
    136             optimizer.zero_grad()
--> 137             outputs = en_model(images)
    138             loss = criterion(outputs, labels)
    139             loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/efficientnet.py in forward(self, x)
    341 
    342     def forward(self, x: Tensor) -> Tensor:
--> 343         return self._forward_impl(x)
    344 
    345 

/usr/local/lib/python3.11/dist-packages/torchvision/models/efficientnet.py in _forward_impl(self, x)
    331 
    332     def _forward_impl(self, x: Tensor) -> Tensor:
--> 333         x = self.features(x)
    334 
    335         x = self.avgpool(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/efficientnet.py in forward(self, input)
    162 
    163     def forward(self, input: Tensor) -> Tensor:
--> 164         result = self.block(input)
    165         if self.use_res_connect:
    166             result = self.stochastic_depth(result)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/ops/misc.py in forward(self, input)
    258     def forward(self, input: Tensor) -> Tensor:
    259         scale = self._scale(input)
--> 260         return scale * input
    261 
    262 

OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 30.88 MiB is free. Process 501185 has 47.49 GiB memory in use. Of the allocated memory 46.91 GiB is allocated by PyTorch, and 284.96 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 1
val_transforms = T.Compose(
    [
        T.Resize((model_image_size, model_image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 2
class TestDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        image_path = os.path.join(self.img_dir, image_name)
        if not os.path.exists(image_path):
            dummy = torch.zeros(3, model_image_size, model_image_size)
            return dummy, idx, True  # missing flag
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, idx, False


sample_df = pd.read_csv(sample_submission_path)
image_ids = sample_df["image_id"].tolist()

test_dataset = TestDataset(image_ids, test_data_directory, val_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,  # fewer workers, sufficient for I/O
    pin_memory=torch.cuda.is_available(),
)

predictions = [0] * len(image_ids)  # pre‑allocate

with torch.no_grad():
    for batch_imgs, batch_idxs, missing_flags in tqdm(test_loader, desc="Test"):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        outputs = (
            en_model(batch_imgs) if model_select == "en" else vit_model(batch_imgs)
        )
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().tolist()
        for idx, pred, miss in zip(batch_idxs.tolist(), preds, missing_flags.tolist()):
            if not miss:
                predictions[idx] = pred
            else:
                predictions[idx] = 0  # default class for missing files




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/2448081939.py in <cell line: 0>()
     35 with torch.no_grad():
     36     for batch_imgs, batch_idxs, missing_flags in tqdm(test_loader, desc="Test"):
---> 37         batch_imgs = batch_imgs.to(device, non_blocking=True)
     38         outputs = (
     39             en_model(batch_imgs) if model_select == "en" else vit_model(batch_imgs)

OutOfMemoryError: CUDA out of memory. Tried to allocate 86.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 30.88 MiB is free. Process 501185 has 47.49 GiB memory in use. Of the allocated memory 46.91 GiB is allocated by PyTorch, and 284.96 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 3
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
