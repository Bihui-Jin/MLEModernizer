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

geopandas==0.14.4
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

0.8845572680568148

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11659) has done: 'The fix filters out sub‑directories and non‑image files when iterating over the test folder, preventing the `IsADirectoryError`. With this change the script runs end‑to‑end, creates a correctly formatted `submission.csv`, and can be evaluated for the target metric.'
- What this solution (achieved 0.29821) has done: 'We replace the manual transform (Resize → ToTensor → Normalize with 0.5/0.5) by the official ImageNet preprocessing supplied with the ViT‑H weights. Using the correct mean and std aligns the fine‑tuned model with its training data and is expected to raise accuracy substantially while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.23879) has done: 'The fix changes the ViT model image size to the expected 224 px (matching the pretrained ImageNet weights) so the model can be instantiated without error. This keeps the original architecture, loading of fine‑tuned weights, and prediction pipeline intact while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.1917) has done: 'We add a quick fine‑tuning stage that freezes the pretrained ViT‑L backbone and only trains the final linear head on the provided training set for a few epochs. This keeps the original architecture unchanged while giving the model knowledge of the cassava classes, which should raise the validation‑style accuracy from the near‑random level (~0.24) toward the target (~0.88). The rest of the pipeline – loading the model, applying the official ImageNet transforms, and writing a correctly formatted `submission.csv` – remains the same.'

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_l_16, ViT_L_16_Weights

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 224
num_classes = 5

pretrained_weight = (
    ViT_L_16_Weights.IMAGENET1K_V1
    if hasattr(ViT_L_16_Weights, "IMAGENET1K_V1")
    else ViT_L_16_Weights.DEFAULT
)

val_transforms = pretrained_weight.transforms()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vit_l_16(weights=pretrained_weight, image_size=image_size)

if isinstance(model.heads, torch.nn.Sequential):
    if hasattr(model.heads[1], "in_features"):
        in_features = model.heads[1].in_features
    else:
        in_features = 1024  # fallback for ViT‑L
    model.heads[1] = torch.nn.Linear(in_features, num_classes)
elif isinstance(model.heads, torch.nn.Linear):
    in_features = model.heads.in_features
    model.heads = torch.nn.Linear(in_features, num_classes)
else:
    in_features = getattr(model, "hidden_dim", 1024)
    model.heads = torch.nn.Linear(in_features, num_classes)

if os.path.isfile(model_path):
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine‑tuned weights from {model_path}")
    except Exception as e:
        print(f"Warning: could not load checkpoint ({e}). Using pretrained weights.")
else:
    print(f"Warning: checkpoint not found at {model_path}. Using pretrained weights.")

model.to(device)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/656218150.py in <cell line: 0>()
     31 if isinstance(model.heads, torch.nn.Sequential):
     32     # Sequential usually: dropout -> linear
---> 33     if hasattr(model.heads[1], "in_features"):
     34         in_features = model.heads[1].in_features
     35     else:

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in __getitem__(self, idx)
    141             return self.__class__(OrderedDict(list(self._modules.items())[idx]))
    142         else:
--> 143             return self._get_item_by_idx(self._modules.values(), idx)
    144 
    145     def __setitem__(self, idx: int, module: Module) -> None:

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in _get_item_by_idx(self, iterator, idx)
    132         idx = operator.index(idx)
    133         if not -size <= idx < size:
--> 134             raise IndexError(f"index {idx} is out of range")
    135         idx %= size
    136         return next(islice(iterator, idx, None))

IndexError: index 1 is out of range

## === cell 1
class CassavaDataset(torch.utils.data.Dataset):
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
        image = self.transform(image)
        label = int(row["label"])
        return image, label


train_dataset = CassavaDataset(train_csv_path, train_data_directory, val_transforms)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
)




## === cell 2
for param in model.parameters():
    param.requires_grad = False

if isinstance(model.heads, torch.nn.Sequential):
    model.heads[1].requires_grad_(True)
else:
    model.heads.requires_grad_(True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

model.train()
epochs = 10  # increased epochs to improve accuracy
for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for imgs, lbls in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{epochs}"):
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    epoch_acc = correct / total
    print(f"Epoch {epoch+1}: loss={epoch_loss/total:.4f}, accuracy={epoch_acc:.4f}")

model.eval()  # Switch back to eval mode for inference




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/1148725925.py in <cell line: 0>()
      4 
      5 if isinstance(model.heads, torch.nn.Sequential):
----> 6     model.heads[1].requires_grad_(True)
      7 else:
      8     model.heads.requires_grad_(True)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in __getitem__(self, idx)
    141             return self.__class__(OrderedDict(list(self._modules.items())[idx]))
    142         else:
--> 143             return self._get_item_by_idx(self._modules.values(), idx)
    144 
    145     def __setitem__(self, idx: int, module: Module) -> None:

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in _get_item_by_idx(self, iterator, idx)
    132         idx = operator.index(idx)
    133         if not -size <= idx < size:
--> 134             raise IndexError(f"index {idx} is out of range")
    135         idx %= size
    136         return next(islice(iterator, idx, None))

IndexError: index 1 is out of range

## === cell 3
test_predictions = []
image_ids = []

for image_name in tqdm(sorted(os.listdir(test_data_directory)), desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)

    if not os.path.isfile(image_path):
        continue
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image = Image.open(image_path).convert("RGB")
    image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, dim=1)

    test_predictions.append(predicted_class.item())
    image_ids.append(image_name)

submission_df = pd.DataFrame({"image_id": image_ids, "label": test_predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3025218537.py in <cell line: 0>()
     14 
     15     with torch.no_grad():
---> 16         output = model(image)
     17         _, predicted_class = torch.max(output, dim=1)
     18 

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in forward(self, x)
    289     def forward(self, x: torch.Tensor):
    290         # Reshape and permute the input tensor
--> 291         x = self._process_input(x)
    292         n = x.shape[0]
    293 

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _process_input(self, x)
    275 
    276         # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
--> 277         x = self.conv_proj(x)
    278         # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    279         x = x.reshape(n, self.hidden_dim, n_h * n_w)

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
